import streamlit as st
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.memory import ConversationBufferMemory
import os
from langchain_groq import ChatGroq
from langchain_community.vectorstores import Chroma
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
google_api_key = os.getenv("GOOGLE_API_KEY")
from dotenv import load_dotenv
load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=google_api_key) 
llm = ChatGroq(temperature=0.7, model_name="llama-3.1-8b-instant", api_key=groq_api_key)



st.set_page_config(page_title="RAG Chatbot", page_icon="📘", layout="wide")
st.markdown("""
    <style>
        .main-title {text-align: center; color: #4CAF50; font-size: 40px; font-weight: bold; margin-bottom: 0;}
        .subtitle {text-align: center; font-size: 18px; color: gray; margin-top: 0;}
        .stChatMessage {border-radius: 12px; padding: 10px; margin: 5px 0; color: black !important;}
        .user-msg {background-color: #E8F5E9;}
        .bot-msg {background-color: #E3F2FD;}
    </style>
""", unsafe_allow_html=True)



st.markdown("<h1 class='main-title'>📘 RAG Chatbot with Memory</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Upload multiple PDFs and chat with them in real-time!</p>", unsafe_allow_html=True)
uploaded_files = st.file_uploader("📂 Upload PDF files", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    docs = []
    for uploaded_file in uploaded_files:
        with open(uploaded_file.name, "wb") as f:
            f.write(uploaded_file.getbuffer())
        loader = PyPDFLoader(uploaded_file.name)
        data = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs.extend(text_splitter.split_documents(data))


    vectorstore = Chroma.from_documents(documents=docs, embedding=embeddings)
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})

    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    
    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer the question. "
        "If you don't know the answer, say that you don't know. Keep answers concise.\n\n{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

  
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    query = st.chat_input("💬 Ask me anything about the documents...")

    if query:
        response = rag_chain.invoke({"input": query})
        answer = response["answer"]

        st.session_state.chat_history.append((query, answer, response.get("context", [])))
        st.session_state.memory.chat_memory.add_user_message(query)
        st.session_state.memory.chat_memory.add_ai_message(answer)

    for q, a, ctx in st.session_state.chat_history:
        with st.chat_message("user"):
            st.markdown(f"<div class='stChatMessage user-msg'>{q}</div>", unsafe_allow_html=True)
        with st.chat_message("assistant"):
            st.markdown(f"<div class='stChatMessage bot-msg'>{a}</div>", unsafe_allow_html=True)
            if st.checkbox(f"🔍 Show retrieved context for: '{q[:20]}...'", key=q):
                st.write(ctx)

    with st.sidebar:
        st.subheader("🧠 Conversation Memory")
        if st.button("Clear Memory"):
            st.session_state.chat_history = []
            st.session_state.memory.clear()
            st.success("Memory cleared!")

        st.write("### Memory Contents")
        st.json(st.session_state.memory.chat_memory.messages)

else:
    st.info("👆 Please upload one or more PDFs to start chatting.")
