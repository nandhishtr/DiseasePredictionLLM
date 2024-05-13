import streamlit as st
import time
from langchain.prompts import PromptTemplate
import ingest as ij
from langchain.schema.runnable import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

#Streamed response emulator
def response_generator(question):
    for word in question.split():
        yield word + " "
        time.sleep(0.05)

db = ij.db
llm = ij.llm
def generate_response(llm_input):
    # initialize a prompt template

    global qa_chain
    prompt_template = """ You are an AI medical health assistance. The user will provide you his/her health condition and based on the context and the question you have to predict the disease and give necessary suggestions.
            If you don't know the answer , just say that you don't know, don't try to make up an answer.Only answer healthcare related questions.
            Please refer to the below examples to answer user query:

            Example 1 :

            question : Hello, there is a pain around the navel, I don’t know what's going on (female, 29 years old).. How long are you facing this? It has been there for three days.. Do you have any other symptoms? No symptoms. Have you used any medication? No.  Can you diagnose and recommend some precautionary measures?
            answer : The pain in the navel area could be related to gastric dysfunction. Gastric dysfunction is a common symptom, which can be caused by food digestion problems. The symptoms of gastric dysfunction include abdominal pain, loss of appetite, loose stools or constipation, diarrhea, bloating, etc. In this case, it's best to drink enough water and eat light food. It is recommended not to drink tea in hot condition as well as avoid drinking soda. You should also limit the consumption of spicy food and alcoholic beverages.

            Example 2 :

            question : Hello, I have continuous head ache. How long are you facing this ? It has been there for over a week. Do you have any other symptoms? I have runny nose and cold. Have you used any medication? I have taken paracetamol. Can you diagnose and recommend some precautionary measures?
            answer : Headache may have different causes and this one lasts for a week, so it is advisable to checkup for brain CT or MRI and in some cases do a lumbar puncture as well as blood tests.

            Only return the helpful answer. Answer must be detailed and well explained.

            Context: {context}
            Question: {question}


            Helpful answer:
            """

    retriever = db.as_retriever(search_kwargs={"k": 1})

    prompt = PromptTemplate(template=prompt_template,
                            input_variables=["context", "question"])
    
    # Create the custom chain
    if llm is not None and db is not None:
        qa_chain = ({"context": retriever, "question": RunnablePassthrough()} | prompt | llm | StrOutputParser())
    else:
        print("LLM or Vector Database not initialized")

    main_question = "Can you diagnose and recommend some precautionary measures?"

    question = llm_input + " " + main_question

    result = qa_chain.invoke(question)

    return result


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

