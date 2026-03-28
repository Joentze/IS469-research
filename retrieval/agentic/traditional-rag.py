"""
use chroma db as vector store
https://docs.langchain.com/oss/python/integrations/vectorstores/chroma

"""

from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


# dummy data (same as bm25)
raw_docs = [
    "Apple Inc. reported a 10% increase in quarterly revenue, driven by strong iPhone sales.",
    "Tesla announced the launch of its new Model Y, targeting the mid-range EV market.",
    "Federal Reserve raised interest rates by 0.25%, citing inflation concerns.",
    "Goldman Sachs forecasts a bullish trend in the stock market for 2026.",
    "Amazon's cloud division, AWS, saw a 15% growth in Q1 2026.",
    "JP Morgan reported higher profits due to increased trading activity.",
    "Microsoft's Azure revenue grew by 20% year-over-year.",
    "Meta Platforms faces regulatory scrutiny over data privacy issues.",
    "Bank of America expects mortgage rates to remain stable throughout 2026.",
    "Nvidia's GPU sales surge as demand for AI chips increases."
]

# --- Step 1: Wrap raw strings into LangChain Document objects ---
# Documents can carry metadata (e.g. source, page number) — useful later with real PDFs
documents = [Document(page_content=text) for text in raw_docs]

# --- Step 2: Split documents into chunks ---
# For short dummy data this won't split much, but essential for real PDFs
splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
chunks = splitter.split_documents(documents)

# --- Step 3: Embed chunks and store in ChromaDB ---
# Using a lightweight local HuggingFace model — no API key needed
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

vectorstore = Chroma.from_documents(
    chunks,
    embedding=embedding_model,
    collection_name="traditional_rag",
    # persist_directory="./database",  # uncomment to persist to disk later on
)

# --- Step 4: Create a retriever ---
#return top 2 most similar
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# --- Step 5: Query ---
query = "Which company reported the highest cloud revenue growth in 2026?"
results = retriever.invoke(query)

# --- Print results ---
for doc in results:
    print(doc.page_content)