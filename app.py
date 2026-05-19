import streamlit as st
from dotenv import load_dotenv
import os

from langchain_groq import ChatGroq
from langchain.agents import initialize_agent, AgentType

from langchain_community.tools import (
    DuckDuckGoSearchRun,
    WikipediaQueryRun,
    ArxivQueryRun,
)

from langchain_community.utilities import (
    WikipediaAPIWrapper,
    ArxivAPIWrapper,
)

from langchain_community.callbacks.streamlit import StreamlitCallbackHandler

# Load environment variables
load_dotenv()

# ---------------- PAGE CONFIG ---------------- #
st.set_page_config(
    page_title="LangChain Chat with Search",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 LangChain Chat with Search")

# ---------------- SIDEBAR ---------------- #
st.sidebar.title("Settings")

api_key = st.sidebar.text_input(
    "Enter your Groq API Key",
    type="password"
)

# ---------------- TOOLS ---------------- #

# Arxiv Tool
arxiv_wrapper = ArxivAPIWrapper(
    top_k_results=1,
    doc_content_chars_max=300
)

arxiv_tool = ArxivQueryRun(api_wrapper=arxiv_wrapper)

# Wikipedia Tool
wiki_wrapper = WikipediaAPIWrapper(
    top_k_results=1,
    doc_content_chars_max=300
)

wiki_tool = WikipediaQueryRun(api_wrapper=wiki_wrapper)

# DuckDuckGo Search Tool
search_tool = DuckDuckGoSearchRun(name="Search")

tools = [search_tool, wiki_tool, arxiv_tool]

# ---------------- SESSION STATE ---------------- #

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hi! I am an AI assistant with web search capabilities. How can I help you?"
        }
    ]

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# ---------------- CHAT INPUT ---------------- #

prompt = st.chat_input("Ask anything...")

if prompt:

    # Check API key
    if not api_key:
        st.error("Please enter your Groq API key.")
        st.stop()

    # Store user message
    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt
        }
    )

    # Display user message
    with st.chat_message("user"):
        st.write(prompt)

    # ---------------- LLM ---------------- #

    llm = ChatGroq(
        groq_api_key=api_key,
        model_name="llama3-8b-8192",
        streaming=True
    )

    # ---------------- AGENT ---------------- #

    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True
    )

    # ---------------- ASSISTANT RESPONSE ---------------- #

    with st.chat_message("assistant"):

        st_cb = StreamlitCallbackHandler(
            parent_container=st.container(),
            expand_new_thoughts=False
        )

        try:
            # IMPORTANT FIX:
            # agent.run() expects STRING input not message list
            response = agent.run(
                prompt,
                callbacks=[st_cb]
            )

            st.write(response)

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": response
                }
            )

        except Exception as e:
            st.error(f"Error: {str(e)}")