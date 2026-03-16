"""
use bm25 in-memory store
https://docs.langchain.com/oss/python/integrations/retrievers/bm25
"""
from rank_bm25 import BM25Okapi

# Your documents
docs = [
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

# Tokenize documents
tokenized_docs = [doc.lower().split() for doc in docs]

# Build BM25 index
bm25 = BM25Okapi(tokenized_docs)

# Example query
query = "Which company reported the highest cloud revenue growth in 2026?"
tokenized_query = query.lower().split()

# Get scores for all docs
scores = bm25.get_scores(tokenized_query)

# Retrieve top 2 documents
import numpy as np
top_n = np.argsort(scores)[::-1][:2]
results = [docs[i] for i in top_n]

# Print results
for doc in results:
    print(doc)