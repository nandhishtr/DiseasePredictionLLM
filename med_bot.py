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


def predict(message, history):
    history_langchain_format = []
    prompt = PromptTemplate(template=prompt_template,
                            input_variables=["chat_history", 'message'])

    response = chain.invoke({"question": message, "chat_history": chat_history})
    answer = response['answer']
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
# def predict(message):
#     prompt = PromptTemplate(template=prompt_template, input_variables=["question"])
#     response = chain.invoke({"question": message})
#     return response['answer']


# gr.ChatInterface(predict).launch()

# # Get user input
## user_input = input("User: ")
#Get user input
print("Bot: Hello There, Welcome! I am a disease prediction bot, to help you. \nCaution: I am only "
      "suggesting this based on my current knowledge.\nIn no way it is an alternative to consulting a doctor.")
question_1 = "How may I help you today?"
print("Bot: ", question_1)
user_input = input("User: ")
total_input = user_input
question_2 = ". How long are you facing this?"
print("Bot: ", question_2)
user_input = input("User: ")
total_input = total_input + question_2 + " " + user_input
question_3 = ". Do you have any other symptoms?"
print("Bot: ", question_3)
user_input = input("User: ")
total_input = total_input + question_3 + " " + user_input
question_4 = ". Have you used any medication?"
print("Bot: ", question_4)
user_input = input("User: ")
total_input = total_input + question_4 + " " + user_input
main_question = ". Can you diagnose and recommend some precautionary measures?"
total_input = total_input + main_question
print(total_input)
# Use the predict function to get LLM's response
response = predict(total_input,chat_history)

# Print LLM's response
print("Bot:", response)


# Update chat history
chat_history.append((user_input, response))
