import numpy as np
from typing import List, Dict
from .config import EMOTION_LABELS

def calculate_max_emotion_scores(predictions: List[List[Dict]]) -> Dict[str, float]:
    """Calculate the maximum emotion scores from the predictions for each emotion label.

    This function processes the predictions returned by the emotion classifier, which contain the predicted emotion labels and their associated scores. It then extracts the maximum score for each emotion label (e.g., joy, sadness, fear, etc.) across all predictions for a single book.

    Args:
        predictions (List[List[Dict]]): A list of predictions, where each prediction is a list of dictionaries containing a "label" (emotion) and its corresponding "score". This list represents the classification output for a book description.

    Returns:
        Dict[str, float]: A dictionary where each key is an emotion label (e.g., "joy", "sadness", etc.) and the corresponding value is the maximum score for that emotion across all predictions.
    """
    per_emotion_scores = {label: [] for label in EMOTION_LABELS}
    
    for prediction in predictions:
        sorted_predictions = sorted(prediction, key=lambda x: x["label"])
        for index, label in enumerate(EMOTION_LABELS):
            per_emotion_scores[label].append(sorted_predictions[index]["score"])

    return {label: np.max(scores) for label, scores in per_emotion_scores.items()}