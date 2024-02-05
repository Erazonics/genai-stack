import os

import streamlit as st
from dotenv import load_dotenv
from langchain.callbacks.base import BaseCallbackHandler
from langchain.graphs import Neo4jGraph
from streamlit.logger import get_logger
import pdfplumber

from chains import (
    load_embedding_model,
    load_llm,
    configure_llm_only_chain,
    configure_qa_rag_chain,
    configure_rulecheck_chain,
)
from utils import create_vector_index_pdf

load_dotenv(".env")

# Neo4j and embedding configurations
url = os.getenv("NEO4J_URI")
username = os.getenv("NEO4J_USERNAME")
password = os.getenv("NEO4J_PASSWORD")
ollama_base_url = os.getenv("OLLAMA_BASE_URL")
embedding_model_name = os.getenv("EMBEDDING_MODEL")
llm_name = os.getenv("LLM")
os.environ["NEO4J_URL"] = url

logger = get_logger(__name__)

neo4j_graph = Neo4jGraph(url=url, username=username, password=password)
embeddings, dimension = load_embedding_model(
    embedding_model_name, config={"ollama_base_url": ollama_base_url}, logger=logger
)
create_vector_index_pdf(neo4j_graph, dimension)

llm = load_llm(llm_name, logger=logger, config={"ollama_base_url": ollama_base_url})
llm_chain = configure_llm_only_chain(llm)

rag_chain = configure_qa_rag_chain(llm, embeddings, embeddings_store_url=url, username=username, password=password)
rulecheck_chain = configure_rulecheck_chain(llm, neo4j_graph)


def convert_pdf_tables_to_json(pdf_file):
    # Open the PDF file
    with pdfplumber.open(pdf_file) as pdf:
        json_data = []

        # Iterate over the pages in the PDF
        for page in pdf.pages:
            # Extract tables from the page
            tables = page.extract_tables()

            # Iterate over the tables
            for table in tables:
                # Convert table to JSON
                json_table = [dict(zip(table[0], row)) for row in table[1:]]
                json_data.append(json_table)

        return json.dumps(json_data)


class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)


# Streamlit UI
styl = f"""
<style>
    /* not great support for :has yet (hello FireFox), but using it for now */
    .element-container:has([aria-label="Select RAG mode"]) {{
      position: fixed;
      bottom: 33px;
      background: white;
      z-index: 101;
    }}
    .stChatFloatingInputContainer {{
        bottom: 20px;
    }}

    /* Generate ticket text area */
    textarea[aria-label="Description"] {{
        height: 200px;
    }}
</style>
"""
st.markdown(styl, unsafe_allow_html=True)


def chat_input():
    user_input = st.chat_input("What can I help you with today?")

    if user_input:
        with st.chat_message("user"):
            st.write(user_input)
        with st.chat_message("assistant"):
            st.caption(f"RAG: {name}")
            stream_handler = StreamHandler(st.empty())
            result = output_function(
                {"question": user_input, "chat_history": []}, callbacks=[stream_handler]
            )["answer"]
            output = result
            st.session_state[f"user_input"].append(user_input)
            st.session_state[f"generated"].append(output)
            st.session_state[f"rag_mode"].append(name)


def display_chat():
    # Session state
    if "generated" not in st.session_state:
        st.session_state[f"generated"] = []

    if "user_input" not in st.session_state:
        st.session_state[f"user_input"] = []

    if "rag_mode" not in st.session_state:
        st.session_state[f"rag_mode"] = []

    if st.session_state[f"generated"]:
        size = len(st.session_state[f"generated"])
        # Display only the last three exchanges
        for i in range(max(size - 3, 0), size):
            with st.chat_message("user"):
                st.write(st.session_state[f"user_input"][i])

            with st.chat_message("assistant"):
                st.caption(f"RAG: {st.session_state[f'rag_mode'][i]}")
                st.write(st.session_state[f"generated"][i])
        with st.container():
            st.write("&nbsp;")


def mode_select() -> str:
    options = ["Disabled", "Enabled", "Rule"]
    return st.radio("Select mode", options, horizontal=True)


name = mode_select()
if name == "LLM only" or name == "Disabled":
    output_function = llm_chain
elif name == "Vector + Graph" or name == "Enabled":
    output_function = rag_chain
elif name == "Rule":
    pdf_file = st.file_uploader("Upload your PDF", type="pdf")
    if pdf_file is not None:
        json_data = convert_pdf_tables_to_json(pdf_file)
    else:
        json_data = st.text_input("Enter JSON data")

    # Check if json_data is not undefined
    if json_data:
        output_function = rulecheck_chain(json_data)
    else:
        raise ValueError("json_data is undefined")


def open_sidebar():
    st.session_state.open_sidebar = True


def close_sidebar():
    st.session_state.open_sidebar = False


display_chat()
chat_input()
