import numpy as np
from typing import List, Dict
from .config import EMOTION_LABELS

def calculate_max_emotion_scores(predictions: List[List[Dict]]) -> Dict[str, float]:
    per_emotion_scores = {label: [] for label in EMOTION_LABELS}
    
    for prediction in predictions:
        sorted_predictions = sorted(prediction, key=lambda x: x["label"])
        for index, label in enumerate(EMOTION_LABELS):
            per_emotion_scores[label].append(sorted_predictions[index]["score"])

    return {label: np.max(scores) for label, scores in per_emotion_scores.items()}