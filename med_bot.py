import os
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.document_loaders import UnstructuredFileLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
#from langchain.vectorstores import Chroma
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_community.llms import LlamaCpp
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory
#from langchain import PromptTemplate
from langchain.prompts import PromptTemplate
from langchain_community.llms import CTransformers

# initializes an embeddings model
embeddings = SentenceTransformerEmbeddings(model_name="NeuML/pubmedbert-base-embeddings")

print(embeddings)

loader = TextLoader("health_report.txt")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=70)

pages = loader.load_and_split(text_splitter)

db = Chroma.from_documents(pages, embeddings, persist_directory='/content/db')

llm = CTransformers(model="MaziyarPanahi/BioMistral-7B-GGUF", model_file="BioMistral-7B.Q8_0.gguf", temperature=0.7,
                    max_tokens=2048, top_p=1, n_ctx=2048)

prompt_template = """In the field of medicine, considering the provided medical report and your conversation history,
answer the user's question to the best of your ability.
If the answer is not found within the medical report or you are unsure, inform the user.
If the user's question is not related to the field of medicine, tell that it is out of scope.

Chat History: {chat_history}
Question: {question}

**Focus on providing medically accurate and informative answers.**

Answer:
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


while True:
    # Get user input
    user_input = input("User: ")

    # Use the predict function to get LLM's response
    response = predict(user_input, chat_history)

    # Print LLM's response
    print("Bot:", response)

    # Update chat history
    chat_history.append((user_input, response))
