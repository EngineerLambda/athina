
import streamlit as st
import time
from langchain.prompts import PromptTemplate
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_pinecone import PineconeVectorStore

st.set_page_config(page_title="Your car insurance policy booklet", page_icon="🏎️")
st.title("Your car insurance policy booklet")

def stream_data(response):
    for word in response.split(" "):
        yield word + " "
        time.sleep(0.02)

llm = GoogleGenerativeAI(model="gemini-pro")
embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

store = PineconeVectorStore(index_name="athina", embedding=embedding)
retriever = store.as_retriever()

prompt_template = """
You are a question and answer agent, you are provided with this question
question: {question}
and you are to use the provided context below to provide a context-aware answer, if the provided information does not
contain the answer, tell the user you dont know. instead of saying accoring to the context, you can say according to my source
context: {context}
"""

prompt = PromptTemplate.from_template(prompt_template)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)


# chat interface for consistent queries
if "messages" not in st.session_state:
    welcome_message = "Hello there, I am a context aware AI model based on ChurChill's car insurance policy booklet, ask your questions and I will be happy\
        to answer to the best of my ability"
    st.chat_message("ai").write_stream(stream_data(welcome_message))
    st.session_state.messages = []

# Display for all the messages
for message, kind in st.session_state.messages:
    with st.chat_message(kind):
        st.markdown(message)

prompt = st.chat_input("Ask your questions ...")

if prompt:
    # Handling prompts and rendering to the chat interface
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append([prompt, "user"])  # updating the list of prompts

    with st.spinner("Generating response"):
        answer = rag_chain.invoke(prompt)
        if answer:
            st.chat_message("ai").write_stream(stream_data(answer))
            st.session_state.messages.append([answer, "ai"])