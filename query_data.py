from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document
from langchain_community.llms import Ollama
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Step 1: Load
loader = TextLoader("./data/books/alice_in_wonderland.md")
data = loader.load()

if data and hasattr(data[0], 'page_content'):
    markdown_document_content = data[0].page_content
else:
    raise ValueError("Could not load markdown content from the file.")

# Step 2: Split
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # Increased chunk size
    chunk_overlap=200 # Increased chunk overlap
)

document = Document(page_content=markdown_document_content)
split_docs = text_splitter.split_documents([document])

print(f"Number of chunks: {len(split_docs)}")
# Optional: Inspect a few chunks to understand the splitting
# for i, doc in enumerate(split_docs[:3]):
#     print(f"\n--- Chunk {i+1} (Length: {len(doc.page_content)}) ---")
#     print(doc.page_content)

# Step 3: Embed & Store
# Using a dedicated embedding model (ensure it's pulled with `ollama pull nomic-embed-text`)
embedding_model = OllamaEmbeddings(model="nomic-embed-text")
vector_store = InMemoryVectorStore(embedding=embedding_model)
vector_store.add_documents(documents=split_docs)

# Step 4: Create a Retriever
retriever = vector_store.as_retriever(k=5) # Retrieve top 5 documents

# Step 5: Define the LLM for generation
# Ensure llama3 is pulled with `ollama pull llama3`
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

# Step 8: Invoke the RAG chain with your query
query = "No, no! The adventures first who said"
response = retrieval_chain.invoke({"input": query})

# Step 9: Display the final answer and retrieved context
print(f"\n--- Final Answer ---")
print(response["answer"])

print(f"\n--- Retrieved Documents (used for answer generation) ---")
for i, doc in enumerate(response["context"]):
    print(f"\n--- Document {i+1} ---")
    print(doc.page_content[:500] + "...") # Print first 500 characters