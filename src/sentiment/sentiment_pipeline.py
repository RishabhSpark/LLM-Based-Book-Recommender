import pandas as pd
from analyzer import analyze_book_emotions

input_path = "data/preprocessed/books_with_cats.csv"
output_path = "data/preprocessed/books_with_emotions.csv"

books_df = pd.read_csv(input_path)
books_with_emotions = analyze_book_emotions(books_df)
books_with_emotions.to_csv(output_path, index=False)

print("Sentiment analysis completed and saved.")
