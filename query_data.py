# query_data.py
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore



# Step 1: Load

# Assuming your markdown file is named 'your_file.md'
loader = TextLoader("./data/books/alice_in_wonderland.md")
data = loader.load()

# 'data' will be a list containing a single Document object
# with the entire content of the markdown file as its page_content.
if data and hasattr(data[0], 'page_content'):
    markdown_document_content = data[0].page_content
else:
    raise ValueError("Could not load markdown content from the file.")

# print(f"Content loaded (first 500 chars): {markdown_document_content[:500]}...")
# print(f"Metadata: {data[0].metadata}")

# Step 2: Split
headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
]

markdown_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=headers_to_split_on, strip_headers=False
)

# Pass the extracted markdown content (string) to split_text
md_header_splits = markdown_splitter.split_text(markdown_document_content)

# Step 3: Embed & Store
embedding_model = OllamaEmbeddings(model="llama3")
vector_store = InMemoryVectorStore(embedding=embedding_model)
vector_store.add_documents(documents=md_header_splits)

query = "What is the rabbit doing?"
results = vector_store.similarity_search(query, k=3)

for doc in results:
    print(doc.page_content)