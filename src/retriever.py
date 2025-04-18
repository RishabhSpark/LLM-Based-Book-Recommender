import pandas as pd
from langchain_chroma import Chroma

def retrieve_semantic_recommendations(query: str, db: Chroma, books_df: pd.DataFrame, top_k: int = 10) -> pd.DataFrame:
    """
    Retrieves semantically similar books to a query using vector search.

    Args:
        query (str): User query or book description.
        db (Chroma): Chroma vector DB instance.
        books_df (pd.DataFrame): Full DataFrame with book metadata.
        top_k (int): Number of top recommendations to return.

    Returns:
        pd.DataFrame: Filtered DataFrame of recommended books.
    """

    recs = db.similarity_search(query, k=top_k)
    books_list = []

    for i in range(0, len(recs)):
        books_list += [int(recs[i].page_content.split()[0].strip())]
    
    return books_df[books_df['isbn13'].isin(books_list)]