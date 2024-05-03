import os
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.document_loaders import UnstructuredFileLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.vectorstores import Chroma
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_community.llms import LlamaCpp
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory
# from langchain import PromptTemplate
from langchain.prompts import PromptTemplate
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
import gradio as gr

# initializes an embeddings model
embeddings = SentenceTransformerEmbeddings(model_name="NeuML/pubmedbert-base-embeddings")

print(embeddings)

# loader = TextLoader("health_report.txt")
loader = DirectoryLoader("health_reports", glob="**/*.txt", show_progress=True)

docs = loader.load()

print(len(docs), "*******************")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=70)

pages = loader.load_and_split(text_splitter)

db = Chroma.from_documents(pages, embeddings, persist_directory='/content/db')

llm = CTransformers(model="MaziyarPanahi/BioMistral-7B-GGUF", model_file="BioMistral-7B.Q8_0.gguf", temperature=0.7,
                    max_tokens=2048, top_p=1, n_ctx=2048)

prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer. Answer must be detailed and well explained.
Helpful answer:
"""

retriever = db.as_retriever(search_kwargs={"k": 1})

chat_history = []
# Create the custom chain
if llm is not None and db is not None:
    chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever)
else:
    print("LLM or Vector Database not initialized")

prompt = PromptTemplate(template=prompt_template,
                            input_variables=['message'])

qa_chain = RetrievalQA.from_chain_type(
    llm, retriever=retriever, chain_type_kwargs={"prompt": prompt}
)
# gr.ChatInterface(predict).launch()

print("Bot: Hello There, Welcome! I am a disease prediction bot, here to help you.\n"
      "Caution: I am only suggesting based on my current knowledge. In no way it is an alternative to consulting a doctor.")

# Define questions
questions = [
    "How may I help you today?",
    "How long are you facing this?",
    "Do you have any other symptoms?",
    "Have you used any medication?"
]

# Initialize total_input
print("Bot:", questions[0])
total_input = input("User: ") + ". "

# Loop through questions
for question in questions[1:]:
    print("Bot:", question)
    user_input = input("User: ")
    total_input += question + " " + user_input + ". "  # Concatenate user input with a period and space

main_question = "Can you diagnose and recommend some precautionary measures?"

final_question = total_input + " " + main_question
print(final_question)

result = qa_chain.invoke({"query": final_question})
print("Bot:", result["result"])

# Ask if the user is satisfied
user_input = input("Bot: Are you satisfied with my answer? (Yes/No): ")

# If not satisfied, ask for additional information
if user_input.lower() == "no":
    print("Bot: Please provide me with some additional information to predict better.")
    additional_info = input("User: ")
    final_question += ". Additional information: " + additional_info + " " + main_question
    result = qa_chain.invoke({"query": final_question})
    print("Bot:", result["result"])

print("Bot: Thank you! Get well soon :)")
