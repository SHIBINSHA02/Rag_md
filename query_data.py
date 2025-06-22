from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document
from langchain_community.llms import Ollama
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
import streamlit as st
import os

# --- RAG Pipeline Initialization (from your provided code) ---
# Ensure the data directory exists and alice_in_wonderland.md is there
# You might want to make this path dynamic or configurable in a real app
data_file_path = "./data/books/alice_in_wonderland.md"

# Check if the file exists before attempting to load
if not os.path.exists(data_file_path):
    st.error(f"Error: Data file not found at {data_file_path}. Please ensure it exists.")
    st.stop()

@st.cache_resource # Cache the RAG pipeline to avoid re-initializing on every rerun
def initialize_rag_pipeline():
    # Step 1: Load
    loader = TextLoader(data_file_path)
    data = loader.load()

    if data and hasattr(data[0], 'page_content'):
        markdown_document_content = data[0].page_content
    else:
        raise ValueError("Could not load markdown content from the file.")

    # Step 2: Split
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    document = Document(page_content=markdown_document_content)
    split_docs = text_splitter.split_documents([document])

    # Step 3: Embed & Store
    embedding_model = OllamaEmbeddings(model="nomic-embed-text")
    vector_store = InMemoryVectorStore(embedding=embedding_model)
    vector_store.add_documents(documents=split_docs)

    # Step 4: Create a Retriever
    retriever = vector_store.as_retriever(k=5)

    # Step 5: Define the LLM for generation
    llm = Ollama(model="llama3")

    # Step 6: Create a prompt template for the LLM
    prompt = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context. If the answer is not in the context, say "I don't have enough information to answer that."

<context>
{context}
</context>

Question: {input}""")

    # Step 7: Create the RAG chain
    document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    return retrieval_chain

# Initialize the RAG pipeline
retrieval_chain = initialize_rag_pipeline()

# --- Streamlit UI ---
st.title("Alice in Wonderland RAG Chatbot")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("Ask about Alice in Wonderland..."):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.spinner("Thinking..."):
        # Invoke the RAG chain
        response = retrieval_chain.invoke({"input": prompt})
        answer = response["answer"]
        context_docs = response["context"]

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.markdown(answer)
            with st.expander("Show Retrieved Context"):
                for i, doc in enumerate(context_docs):
                    st.write(f"**Document {i+1}**")
                    st.write(doc.page_content)
                    st.markdown("---") # Separator between documents

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": answer})

st.markdown("---")
st.markdown("This chatbot uses a Retrieval Augmented Generation (RAG) pipeline to answer questions about 'Alice in Wonderland' based on the provided text.")