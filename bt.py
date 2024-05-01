import streamlit as st
import time
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


# Streamed response emulator
def response_generator(question):
    for word in question.split():
        yield word + " "
        time.sleep(0.05)


def generate_response(llm_input):
    # initializes an embeddings model
    embeddings = SentenceTransformerEmbeddings(model_name="NeuML/pubmedbert-base-embeddings")


    # loader = TextLoader("health_report.txt")
    loader = DirectoryLoader("health_reports", glob="**/*.txt", show_progress=True)

    docs = loader.load()


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

    main_question = "Can you diagnose and recommend some precautionary measures?"

    llm_query = llm_input + " " + main_question

    result = qa_chain.invoke({"query": llm_query})

    return result["result"]


st.title("Medical Chatbot")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.questions = ["How may I help you today?", "How long are you facing this?",
                                  "Do you have any other symptoms?", "Have you used any medication?"]
    st.session_state.current_question_index = 0
    st.session_state.llm_input = ""

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Check if assistant needs to start the conversation
if st.session_state.current_question_index == 0:
    with st.chat_message("assistant"):
        st.write_stream(response_generator(st.session_state.questions[0]))
    st.session_state.messages.append({"role": "assistant", "content": st.session_state.questions[0]})
    st.session_state.llm_input += f"Doctor: {st.session_state.questions[0]}\n"
    st.session_state.current_question_index += 1

    # Accept user input

if prompt := st.chat_input("What is up?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt + "."})
    st.session_state.llm_input += f"Patient: {prompt}.\n"
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Ask the current question from the list
    if st.session_state.current_question_index < len(st.session_state.questions):
        current_question = st.session_state.questions[st.session_state.current_question_index]
        with st.chat_message("assistant"):
            st.write_stream(response_generator(current_question))
        # Add assistant question to chat history
        st.session_state.messages.append({"role": "assistant", "content": current_question})
        st.session_state.llm_input += f"Doctor: {current_question}\n"

        # Move to the next question
        st.session_state.current_question_index += 1
    else:
        # Generate response based on llm_input
        response = generate_response(st.session_state.llm_input)
        st.session_state.messages.append({"role": "assistant", "content": response})
        # Display the response
        with st.chat_message("assistant"):
            st.markdown(response)


