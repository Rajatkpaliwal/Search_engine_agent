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

from langchain_community.callbacks.streamlit import StreamlitCallbackHandler

from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub

# ---------------- LOAD ENV ---------------- #
load_dotenv()

# ---------------- PAGE CONFIG ---------------- #
st.set_page_config(
    page_title="LangChain Search Agent",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 LangChain Search Agent")

# ---------------- SIDEBAR ---------------- #
st.sidebar.title("Settings")

api_key = st.sidebar.text_input(
    "Enter Groq API Key",
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

# Search Tool
search_tool = DuckDuckGoSearchRun(name="Search")

tools = [search_tool, wiki_tool, arxiv_tool]

# ---------------- CHAT HISTORY ---------------- #

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hi! I can search the web. Ask me anything."
        }
    ]

# Display messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# ---------------- USER INPUT ---------------- #

prompt = st.chat_input("Ask anything...")

if prompt:

    if not api_key:
        st.error("Please enter your Groq API key.")
        st.stop()

    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt
        }
    )

    with st.chat_message("user"):
        st.write(prompt)

    # ---------------- LLM ---------------- #

    llm = ChatGroq(
        groq_api_key=api_key,
        model_name="llama3-8b-8192",
        streaming=True
    )

    # ---------------- PROMPT TEMPLATE ---------------- #

    react_prompt = hub.pull("hwchase17/react")

    # ---------------- CREATE AGENT ---------------- #

    agent = create_react_agent(
        llm=llm,
        tools=tools,
        prompt=react_prompt
    )

    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True
    )

    # ---------------- RESPONSE ---------------- #

    with st.chat_message("assistant"):

        st_cb = StreamlitCallbackHandler(
            st.container(),
            expand_new_thoughts=False
        )

        try:

            response = agent_executor.invoke(
                {
                    "input": prompt
                },
                {
                    "callbacks": [st_cb]
                }
            )

            output = response["output"]

            st.write(output)

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": output
                }
            )

        except Exception as e:
            st.error(f"Error: {str(e)}")