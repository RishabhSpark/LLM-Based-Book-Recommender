from src.preprocessing import clean_books_dataset
from src.vectorstore import build_vectorstore
from src.retriever import retrieve_semantic_recommendations
from src.classifier import ZeroShotBookClassifier
from src.category_mapper import map_categories
from src.sentiment.analyzer import analyze_book_emotions

def run_pipeline():
    print("Starting book recommender pipeline...")

    input_path = "data/raw/books.csv"
    output_path = "data/preprocessed/books_cleaned.csv"
    description_txt = "data/preprocessed/tagged_descriptions.txt"

    print("Cleaning dataset...")
    cleaned_df = clean_books_dataset(input_path, output_path)

    print(f"Dataset cleaned. {len(cleaned_df)} books ready.")

    print("Building vectorstore...")
    vector_db = build_vectorstore(
        csv_path=output_path,
        description_txt_path=description_txt,
        persist_directory="chroma_db"
    )
    print("Vectorstore database created")

    print("Testing semantic search...")
    sample_query = "A magical school where students learn spells and secrets"
    recs = retrieve_semantic_recommendations(sample_query, vector_db, cleaned_df, top_k=5)
    print(recs[['title_and_subtitle', 'average_rating']])

    cleaned_df = map_categories(cleaned_df)

    print("Running zero-shot classification for missing categories...")
    classifier = ZeroShotBookClassifier()
    cleaned_df = classifier.fill_missing_categories(cleaned_df)

    cats_path = 'data/preprocessed/books_with_cats.csv'
    cleaned_df.to_csv(cats_path, index=False)

    print("Running sentiment analysis...")
    emotion_output_path = 'data/preprocessed/books_with_emotions.csv'
    final_df = analyze_book_emotions(cats_path, emotion_output_path)
    print("Sentiment analysis complete.")

    print("Pipeline completed successfully.")

if __name__ == "__main__":
    run_pipeline()