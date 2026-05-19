import streamlit as st
from dotenv import load_dotenv

from langchain_groq import ChatGroq

from langchain_community.tools import (
    DuckDuckGoSearchRun,
    WikipediaQueryRun,
    ArxivQueryRun,
)

from langchain_community.utilities import (
    WikipediaAPIWrapper,
    ArxivAPIWrapper,
)

from langgraph.prebuilt import create_react_agent

# ---------------- LOAD ENV ---------------- #
load_dotenv()

# ---------------- PAGE CONFIG ---------------- #
st.set_page_config(
    page_title="AI Search Agent",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 AI Search Agent")

# ---------------- SIDEBAR ---------------- #
st.sidebar.title("Settings")

api_key = st.sidebar.text_input(
    "Enter Groq API Key",
    type="password"
)

# ---------------- TOOLS ---------------- #

search_tool = DuckDuckGoSearchRun(name="Search")

wiki_tool = WikipediaQueryRun(
    api_wrapper=WikipediaAPIWrapper(
        top_k_results=1,
        doc_content_chars_max=300
    )
)

arxiv_tool = ArxivQueryRun(
    api_wrapper=ArxivAPIWrapper(
        top_k_results=1,
        doc_content_chars_max=300
    )
)

tools = [search_tool, wiki_tool, arxiv_tool]

# ---------------- SESSION STATE ---------------- #

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hi! I am an AI search agent. Ask me anything."
        }
    ]

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# ---------------- USER INPUT ---------------- #

prompt = st.chat_input("Ask anything...")

if prompt:

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
        model_name="llama3-8b-8192"
    )

    # ---------------- AGENT ---------------- #

    agent = create_react_agent(
        model=llm,
        tools=tools
    )

    # ---------------- RESPONSE ---------------- #

    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):

            try:

                response = agent.invoke(
                    {
                        "messages": [
                            {
                                "role": "user",
                                "content": prompt
                            }
                        ]
                    }
                )

                output = response["messages"][-1].content

                st.write(output)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": output
                    }
                )

            except Exception as e:
                st.error(f"Error: {str(e)}")