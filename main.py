import gr
from langchain import PromptTemplate
from langchain_community.llms import LlamaCpp
from langchain.chains import RetrievalQA
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import SystemMessagePromptTemplate
from langchain_community.embeddings import SentenceTransformerEmbeddings
from fastapi import FastAPI, Request, Form, Response
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.encoders import jsonable_encoder
from qdrant_client import QdrantClient
from langchain_community.vectorstores import Qdrant
from langchain.chains import RetrievalQA
import os
#pip install ctransformers>=0.2.24
import json
import gradio as gr
from langchain_community.llms import CTransformers

llm = CTransformers(model="MaziyarPanahi/BioMistral-7B-GGUF", model_file="BioMistral-7B.Q4_K_M.gguf", temperature=0.7,
                      max_tokens=2048, top_p=1, n_ctx=2048)

print("LLM Initialized....")

prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer. Answer must be detailed and well explained.
Helpful answer:
"""

embeddings = SentenceTransformerEmbeddings(model_name="NeuML/pubmedbert-base-embeddings")
url = "http://localhost:6333"
client = QdrantClient(url=url, prefer_grpc=False)
db = Qdrant(client=client, embeddings=embeddings, collection_name="vector_db")
print(db)
retriever = db.as_retriever(search_kwargs={"k": 1})
chat_history = []

def predict(message, history):
  # Create the custom chain
  context = "\n".join(chat_history)  # Efficiently build context from history

  prompt = PromptTemplate(template=prompt_template, input_variables=["context", 'message'])
  chain_type_kwargs = {"prompt": prompt}
  if llm is not None and db is not None:
    chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever,chain_type="stuff", chain_type_kwargs=chain_type_kwargs, verbose=True)
  else:
    print("LLM or Vector Database not initialized")

  history_langchain_format = []
  response = chain(message)
  print(response)
  answer = response['result']
  chat_history.append((message, answer))
  temp = []
  for input_question, bot_answer in history:
          temp.append(input_question)
          temp.append(bot_answer)
          history_langchain_format.append(temp)
  temp.clear()
  temp.append(message)
  temp.append(answer)
  history_langchain_format.append(temp)
  return answer





gr.ChatInterface(predict).launch()