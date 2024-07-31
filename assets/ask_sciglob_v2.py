import streamlit as st
import logging
import os
import ollama

from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_community.llms import Ollama
from typing import List, Tuple, Dict, Any, Optional

# Streamlit page configuration
st.set_page_config(
    page_title="SciGlobAI",
    page_icon="assets/sciglob_symbol.webp",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
.big-font {
    font-size:40px !important;
    color: #87CEEB;
    font-family: Times New Roman;
}
</style>
""", unsafe_allow_html=True)

c1, c2 = st.columns([1, 7.7])

with c1:
    st.image('assets/sciglob_logoRGB.png', width=180)
with c2:
    st.markdown('<p class="big-font">AI</p>', unsafe_allow_html=True)

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)

def extract_model_names(
    models_info: Dict[str, List[Dict[str, Any]]],
) -> Tuple[str, ...]:
    logger.info("Extracting model names from models_info")
    model_names = tuple(model["name"] for model in models_info["models"])
    logger.info(f"Extracted model names: {model_names}")
    return model_names

def create_vector_db_from_local_path(local_path: str) -> Chroma:
    logger.info(f"Creating vector DB from files in local path: {local_path}")
    data = []
    for root, _, files in os.walk(local_path):
        for file in files:
            if file.endswith(".pdf"):
                path = os.path.join(root, file)
                loader = UnstructuredPDFLoader(path)
                data.extend(loader.load())

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    chunks = text_splitter.split_documents(data)
    logger.info("Documents split into chunks")

    embeddings = OllamaEmbeddings(model="nomic-embed-text", show_progress=True)
    vector_db = Chroma.from_documents(
        documents=chunks, embedding=embeddings, collection_name="myRAG"
    )
    logger.info("Vector DB created")
    return vector_db

def process_question(question: str, vector_db: Chroma, selected_model: str) -> str:
    logger.info(f"Processing question: {question} using model: {selected_model}")
    llm = ChatOllama(model=selected_model, temperature=0)
    QUERY_PROMPT = PromptTemplate(
        input_variables=["question"],
        template="""You are an AI language model assistant for the company SciGlob. Your task is to generate 3
        different versions of the given user question to retrieve relevant documents from
        a vector database. By generating multiple perspectives on the user question, your
        goal is to help the user overcome some of the limitations of the distance-based
        similarity search. Provide these alternative questions separated by newlines.
        Original question: {question}""",
    )

    retriever = MultiQueryRetriever.from_llm(
        vector_db.as_retriever(), llm, prompt=QUERY_PROMPT
    )

    template = """Answer the question based ONLY on the following context:
    {context}
    Question: {question}
    If you do not know the answer, just say that you do not know and suggest to send email to info@sciglob.com address.
    Do not try to make up an answer.
    Only provide the answer from the {context}, nothing else.
    Add snippets of the context you used to answer the question.
    """

    prompt = ChatPromptTemplate.from_template(template)

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    response = chain.invoke(question)
    logger.info("Question processed and response generated")
    return response

def delete_vector_db(vector_db: Optional[Chroma]) -> None:
    logger.info("Deleting vector DB")
    if vector_db is not None:
        vector_db.delete_collection()
        st.session_state.pop("vector_db", None)
        st.success("Stopped and deleted successfully.")
        logger.info("Vector DB and related session state cleared")
        st.rerun()
    else:
        st.error("No vector database found to delete.")
        logger.warning("Attempted to delete vector DB, but none was found")

def clear_all_messages() -> None:
    st.session_state["messages"] = []
    st.success("All messages cleared successfully.")
    logger.info("All messages cleared successfully")
    st.rerun()

def main_page():
    #st.subheader("Global Science for Global Solutions", divider="rainbow", anchor=False)
    st.subheader('', divider="rainbow", anchor=False)

    models_info = ollama.list()
    available_models = extract_model_names(models_info)

    col1, col2 = st.columns([1.5, 2])

    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    if "vector_db" not in st.session_state:
        st.session_state["vector_db"] = None

    if "button_clicked" not in st.session_state:
        st.session_state["button_clicked"] = False

    if available_models:
        selected_model = col2.selectbox(
            "Pick a model available locally on your system ↓", available_models, key="main_page_model"
        )

    with col1:
        with st.container(height=500, border=True):
            try:
                st.markdown('<img src="https://sciglob.com/wp-content/uploads/2023/07/PGN_Deployments_map.gif" alt="Map GIF" style="width:600px; height:460px;">', unsafe_allow_html=True)
            except:
                st.image('assets/Pandora-Deployments.png')

    delete_collection = col1.button("Stop SciGlob Chatbot", type="secondary", help="Delete the Vector Database")
    delete_messages = col1.button("Clear All Messages", type="secondary", help="Clear all questions and responses")

    if delete_collection:
        delete_vector_db(st.session_state["vector_db"])
    
    if delete_messages:
        clear_all_messages()
    
    with col2:
        message_container = st.container(height=415, border=True)

        button_label = "Update SciGlob Chatbot" if st.session_state["button_clicked"] else "Start SciGlob Chatbot"

        if st.button(button_label, type="primary", help="Start the model to begin chat"):
            with st.spinner("Starting ..."):
                try:
                    local_path = "/docs"
                    st.session_state["vector_db"] = create_vector_db_from_local_path(local_path)
                    st.session_state["button_clicked"] = True
                    st.success("Started successfully.")
                except Exception as e:
                    st.error(f"Failed to start chatbot: {e}")
                    logger.error(f"Error starting chatbot: {e}")

        for message in st.session_state["messages"]:
            avatar = "🤖" if message["role"] == "assistant" else "✋"
            with message_container.chat_message(message["role"], avatar=avatar):
                st.markdown(message["content"])

        if prompt := st.chat_input("Enter a prompt here..."):
            try:
                if st.session_state["vector_db"] is None:
                    st.warning("Please start the chatbot to begin chat.")
                else:
                    st.session_state["messages"].append({"role": "user", "content": prompt})
                    message_container.chat_message("user", avatar="✋").markdown(prompt)

                    with message_container.chat_message("assistant", avatar="🤖"):
                        with st.spinner(":green[generating response...]"):
                            response = process_question(
                                prompt, st.session_state["vector_db"], selected_model
                            )
                            st.markdown(response)

                    st.session_state["messages"].append(
                        {"role": "assistant", "content": response}
                    )

            except Exception as e:
                st.error(e, icon="⛔️")
                logger.error(f"Error processing prompt: {e}")
        else:
            if st.session_state["vector_db"] is None:
                st.warning("Please start/update the chatbot to begin chat...")

### Updated Second Code (Second Page Function)
def second_page():
    import streamlit as st
    from langchain_community.llms import Ollama

    st.title("Local Chatbot")
    models_info = ollama.list()
    available_models = extract_model_names(models_info)

    if available_models:
        selected_model = st.selectbox(
            "Pick a model available locally on your system ↓", available_models, key="second_page_model"
        )
        llm = Ollama(model=selected_model)
    else:
        st.warning("You have not pulled any model from Ollama yet!", icon="⚠️")
        return

    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    message_container = st.container()

    prompt2 = st.text_area("Enter your prompt:")

    if st.button("Generate"):
        if prompt2:
            with st.spinner("Generating response..."):
                response = llm(prompt2)
                st.session_state["messages"].append({"role": "user", "content": prompt2})
                st.session_state["messages"].append({"role": "assistant", "content": response})
                st.experimental_rerun()

    for message in st.session_state["messages"]:
        avatar = "🤖" if message["role"] == "assistant" else "✋"
        with message_container.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])

    if st.button("Clear Chat"):
        st.session_state["messages"] = []
        st.experimental_rerun()

### Tabs Implementation
tabs = st.tabs(["Main Page", "Local Chatbot"])

with tabs[0]:
    main_page()

with tabs[1]:
    second_page()
