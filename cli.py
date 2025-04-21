import typer
import pandas as pd
from src.retriever import retrieve_semantic_recommendations
from src.vectorstore import build_vectorstore

app = typer.Typer()

# Load cleaned and preprocessed books
input_path = "data/preprocessed/books_with_emotions.csv"
description_txt = "data/preprocessed/tagged_descriptions.txt"

# Check if vector store exists or rebuild it if necessary
VECTORSTORE_DIR = "chroma_db"

# Build vectorstore (once) if not already built
def load_or_build_vectorstore():
    try:
        # Attempt to load the existing vector store
        print(f"Loading vectorstore from {VECTORSTORE_DIR}...")
        vector_db = build_vectorstore(
            csv_path=input_path,
            description_txt_path=description_txt,
            persist_directory=VECTORSTORE_DIR
        )
        print(f"Vectorstore loaded from {VECTORSTORE_DIR}")
    except Exception as e:
        # If loading fails, rebuild the vector store
        print(f"Error loading vector store: {e}. Rebuilding...")
        vector_db = build_vectorstore(
            csv_path=input_path,
            description_txt_path=description_txt,
            persist_directory=VECTORSTORE_DIR
        )
        print(f"Vectorstore rebuilt and saved to {VECTORSTORE_DIR}")
    return vector_db

# Load books data
books_df = pd.read_csv(input_path)

@app.command()
def recommend(
    query: str = typer.Option(..., help="Search query to recommend books based on"),
    category: str = typer.Option("All", help="Category filter for recommendations"),
    emotion: str = typer.Option("All", help="Emotion filter for recommendations"),
    top_k: int = typer.Option(5, help="Number of top recommendations to show")
):
    """
    Recommend books based on a search query, optional category, and emotional tone.
    """
    # Load or rebuild vector store
    vector_db = load_or_build_vectorstore()

    # Retrieve semantic recommendations
    recs = retrieve_semantic_recommendations(query, vector_db, books_df, top_k=top_k)

    # Debugging: Print raw recommendations before filtering
    print(f"Raw recommendations (before filtering): {len(recs)}")
    print(recs.head())  # Print top few rows to check content

    # Filter by category if necessary
    if category != "All":
        recs = recs[recs['simple_categories'].str.contains(category, case=False, na=False)]

    # Filter by emotion if necessary
    if emotion != "All":
        recs = recs[recs[emotion].notnull() & (recs[emotion] > 0)]
    
    # Debugging: Print recommendations after filtering
    print(f"Filtered recommendations (after applying category and emotion filters): {len(recs)}")

    # Display the recommendations
    if recs.empty:
        print(f"No recommendations found for the query '{query}' with category '{category}' and emotion '{emotion}'.")
    else:
        for index, row in recs.iterrows():
            description = row['description']
            truncated_desc = " ".join(description.split()[:30]) + "..."
            print(f"Title: {row['title_and_subtitle']}")
            print(f"Category: {row['simple_categories']}")
            print(f"Emotion Scores (Joy, Fear, Sadness, etc.): {row[['joy', 'fear', 'sadness', 'anger', 'surprise']]}")
            print(f"Description: {truncated_desc}")
            print("-" * 40)

if __name__ == "__main__":
    app()