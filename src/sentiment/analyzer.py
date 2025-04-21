import pandas as pd
from tqdm import tqdm

from config import get_emotion_classifier, EMOTION_LABELS
from utils import calculate_max_emotion_scores

def analyze_book_emotions(books_df: pd.DataFrame) -> pd.DataFrame:
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
