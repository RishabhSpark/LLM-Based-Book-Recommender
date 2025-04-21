import pandas as pd
from tqdm import tqdm

from .config import get_emotion_classifier, EMOTION_LABELS
from .utils import calculate_max_emotion_scores

def analyze_book_emotions(books_df: pd.DataFrame) -> pd.DataFrame:
    """Analyzes the emotional tone of each book description in the provided DataFrame and appends the emotion scores to the original DataFrame.

    This function uses an emotion classifier to evaluate the emotional sentiment of each book's description (e.g., joy, fear, sadness, anger, etc.). It processes the description of each book, calculates the maximum emotion score for each predefined label, and returns the modified DataFrame with added emotion scores.

    Args:
        books_df (pd.DataFrame): A DataFrame containing the metadata and descriptions of books.

    Returns:
        pd.DataFrame: A DataFrame with additional columns for each emotion score (e.g., 'joy', 'fear', 'sadness', 'anger', 'surprise') for each book.
    """
    classifier = get_emotion_classifier()
    isbn = []
    emotion_scores = {label: [] for label in EMOTION_LABELS}

    for i in tqdm(range(len(books_df)), desc="Analyzing book emotions"):
        isbn.append(books_df["isbn13"][i])
        sentences = books_df["description"][i].split(".")
        predictions = classifier(sentences)
        max_scores = calculate_max_emotion_scores(predictions)
        for label in EMOTION_LABELS:
            emotion_scores[label].append(max_scores[label])

    emotions_df = pd.DataFrame(emotion_scores)
    emotions_df["isbn13"] = isbn

    return pd.merge(books_df, emotions_df, on="isbn13") 
