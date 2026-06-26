import os
import streamlit as st
from dotenv import load_dotenv

from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)

from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.tools import create_retriever_tool
from langgraph.prebuilt import create_react_agent

# -------------------------------
# Load Environment Variables
# -------------------------------
load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(
    page_title="Enterprise Intel Agent",
    page_icon="🤖"
)

st.title("💼 Enterprise Document Intelligence Agent")
st.markdown(
    "Upload a text document and ask questions about it using Gemini."
)

if not api_key:
    st.error("GOOGLE_API_KEY not found in .env")
    st.stop()

# -------------------------------
# Sidebar
# -------------------------------
with st.sidebar:
    st.header("📄 Upload Document")

    uploaded_file = st.file_uploader(
        "Upload a .txt file",
        type=["txt"]
    )

    if uploaded_file:
        file_text = uploaded_file.read().decode("utf-8")
        st.success("Document Loaded!")

# -------------------------------
# Persistent Memory Architecture
# -------------------------------
if "agent_executor" not in st.session_state:
    st.session_state.agent_executor = None

if uploaded_file and st.session_state.agent_executor is None:
    with st.spinner("Processing document embeddings..."):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
        )

        docs = splitter.create_documents([file_text])

        embeddings = GoogleGenerativeAIEmbeddings(
            model="gemini-embedding-2-preview",
            google_api_key=api_key,
        )

        vector_db = Chroma.from_documents(
            documents=docs,
            embedding=embeddings,
        )

        retriever = vector_db.as_retriever(
            search_kwargs={"k": 2}
        )

        tool = create_retriever_tool(
            retriever,
            "search_enterprise_documents",
            "Searches uploaded enterprise documents."
        )

        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0,
            google_api_key=api_key,
        )

        st.session_state.agent_executor = create_react_agent(
            llm,
            [tool]
        )

if not uploaded_file:
    st.session_state.agent_executor = None

# -------------------------------
# Chat History Panel
# -------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "👋 Upload a document and ask me anything about it!"
        }
    ]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# -------------------------------
# Chat Input & Invocation Block
# -------------------------------
user_query = st.chat_input("Ask a question...")

if user_query:

    if not uploaded_file or st.session_state.agent_executor is None:
        st.error("Please upload a document first.")
        st.stop()

    with st.chat_message("user"):
        st.markdown(user_query)

    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_query
        }
    )

    with st.spinner("Thinking..."):
        # Format history context correctly for LangChain
        formatted_history = [
            (msg["role"], msg["content"]) for msg in st.session_state.messages[:-1]
        ]
        formatted_history.append(("user", user_query))

        # Pass context seamlessly into the agent invocation
        response = st.session_state.agent_executor.invoke(
            {
                "messages": formatted_history
            }
        )

        final_message = response["messages"][-1]
        if isinstance(final_message.content, list):
            assistant_reply = final_message.content[0].get("text", str(final_message.content))
        else:
            assistant_reply = final_message.content

    with st.chat_message("assistant"):
        st.markdown(assistant_reply)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": assistant_reply
        }
    )