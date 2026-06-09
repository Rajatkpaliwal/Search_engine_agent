import warnings
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

warnings.filterwarnings("ignore", category=DeprecationWarning)

load_dotenv()

st.set_page_config(
    page_title="AI Search Agent",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 AI Search Agent")
st.sidebar.title("Settings")

api_key = st.sidebar.text_input(
    "Enter Groq API Key",
    type="password"
)

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

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hi! I am an AI search agent. Ask me anything."
        }
    ]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

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

    llm = ChatGroq(
        groq_api_key=api_key,
        model_name="llama-3.3-70b-versatile",
        temperature=0
    )

    agent = create_react_agent(model=llm, tools=tools)

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

                last_message = response["messages"][-1]

                if isinstance(last_message.content, list):
                    output = " ".join(
                        block.get("text", "") for block in last_message.content
                        if isinstance(block, dict)
                    )
                else:
                    output = last_message.content

                st.write(output)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": output
                    }
                )

            except Exception as e:
                st.error(f"Error: {str(e)}")