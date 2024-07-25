import os
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever

local_path = "4. SciGlob/Manuals/"

# Function to load all PDF documents from the directory
def load_documents(path):
    documents = []
    for file_name in os.listdir(path):
        file_path = os.path.join(path, file_name)
        if file_name.endswith('.pdf'):
            loader = UnstructuredPDFLoader(file_path=file_path)
            documents.extend(loader.load())
    return documents

# Load all PDF documents
data = load_documents(local_path)

# Split and chunk the documents
text_splitter = RecursiveCharacterTextSplitter(chunk_size=7500, chunk_overlap=100)
chunks = text_splitter.split_documents(data)

# Define the embedding function
embedding_function = OllamaEmbeddings(model="nomic-embed-text", show_progress=True)

# Add to vector database with persistent storage
vector_db = Chroma.from_documents(
    documents=chunks, 
    embedding=embedding_function,
    collection_name="local-rag"
)

# LLM from Ollama
local_model = "mistral"
llm = ChatOllama(model=local_model)

QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""You are an AI language model assistant. Your task is to generate three
    different versions of the given user question to retrieve relevant documents from
    a vector database. By generating multiple perspectives on the user question, your
    goal is to help the user overcome some of the limitations of the distance-based
    similarity search. Provide these alternative questions separated by newlines.
    Original question: {question}""",
)

retriever = MultiQueryRetriever.from_llm(
    vector_db.as_retriever(), 
    llm,
    prompt=QUERY_PROMPT
)

# RAG prompt
template = """Answer the question based ONLY on the following context:
{context}
Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

chain.invoke(input(""))

#chain.invoke("Your Question?")