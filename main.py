from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage, SystemMessage
import uuid
from pinecone import Pinecone, ServerlessSpec
import os


# Initialize Pinecone and OpenAi
os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"
pc = Pinecone(api_key="YOUR_API_KEY")
llm = ChatOpenAI(api_key=os.environ["OPENAI_API_KEY"])


# Create an index (if it doesn't exist)
index_name = "chat-history"
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384, # Replace with your model dimensions
        metric="cosine", # Replace with your model metric\
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        ) 
    )  # Adjust dimension as needed


# Connect to the index
index = pc.Index(index_name)


system_prompt = SystemMessage(content="""
    You will be provided with some user context on the basis of that give appropriate response
""")

class ChatBot:
    def __init__(self, llm):
        self.llm = llm
        self.system_prompt = system_prompt
        self.model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

    def save_message(self, phone_number, message, sender):
        # Create a unique ID for the message using UUID
        message_id = str(uuid.uuid4())

        # Convert the message to a vector
        message_vector = self.model.encode(message).tolist()

        # Store the message in Pinecone with namespace
        namespace = phone_number
        
        index.upsert([(message_id, message_vector, {"user_id": phone_number, "message": message, "sender": sender})], namespace=namespace)

    def get_chat_history(self, phone_number, message):
        
        namespace = phone_number
        
        message_vector = self.model.encode(message).tolist()
        # Fetch all vectors for the user
        response = index.query(
            vector=message_vector,
            top_k=2,  # Adjust as needed
            include_metadata=True,
            namespace=namespace
        )

        # Extract messages from the response
        chat_history = [{"timestamp": vector["id"], "message": vector["metadata"]["message"], "sender": vector["metadata"]["sender"]} for vector in response["matches"]]

        # Sort by timestamp (assuming UUIDs will roughly sort by creation time)
        return chat_history

    def get_response(self, phone_number, user_input):
        # Load context from Pinecone
        chat_history = self.get_chat_history(phone_number, user_input)
        print(chat_history)
        self.context = [HumanMessage(content=msg["message"]) if msg["sender"] == "user" else AIMessage(content=msg["message"]) for msg in chat_history]

        # Add user input to context
        self.context.append(HumanMessage(content=user_input))
        messages = [self.system_prompt] + self.context

        # Generate response using the language model
        response = self.llm(messages=messages)
        generated_text = response.content.strip()

        # Add response to context
        self.context.append(AIMessage(content=generated_text))

        # Save context to Pinecone
        self.save_message(phone_number, user_input, "user")
        self.save_message(phone_number, generated_text, "assistant")

        # Remove everything before and including the first colon
        if ':' in generated_text:
            generated_text = generated_text.split(':', 1)[1].strip()

        return generated_text

chatbot = ChatBot(llm)

# App initialise
app = FastAPI()

class Message(BaseModel):
    phone_number: str
    message: str

@app.post("/chat")
def chat(message: Message):
    user_input = message.message
    phone_number = message.phone_number
    if user_input:
        response = chatbot.get_response(phone_number, user_input)
        return {"response": response}
    else:
        raise HTTPException(status_code=400, detail="No message provided")
