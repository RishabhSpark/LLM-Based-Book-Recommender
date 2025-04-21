from transformers import pipeline

def get_emotion_classifier():
    """Loads and returns an emotion classification pipeline for analyzing text sentiment.

    Returns:
        pipeline: A Hugging Face `pipeline` object configured for text classification using the `j-hartmann/emotion-english-distilroberta-base` model.
    """
    return pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        top_k=None
    )

EMOTION_LABELS = ["anger", "disgust", "fear", "joy", "sadness", "surprise", "neutral"]