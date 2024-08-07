## Simple ChatBot
This project is a simple approach towards how we can create a Chatbot using GPT, langchain and pinecone

## Installation & Setup 

First setup a virtual environment<br />
```bash
py -m  venv myenv
```
<br />

Activate the environment<br />
```bash
myenv\Scripts\Activate
```
<br />

Install the dependencies<br />
```bash
pip install fastapi uvicorn pydantic sentence-transformers langchain langchain-openai pinecone-client
```
<br />
Before setup replace YOUR_OPENAI_API_KEY and YOUR_API_KEY with the keys<br />

Run the server<br />
```bash
uvicorn main:app --reload
```
<br />
