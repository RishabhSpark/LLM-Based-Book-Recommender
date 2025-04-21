import pandas as pd
import numpy as np
from transformers import pipeline
from tqdm import tqdm
from src.config import FICTION_CATEGORIES


class ZeroShotBookClassifier:
    def __init__(self, model_name: str = "facebook/bart-large-mnli"):
        self.pipe = pipeline("zero-shot-classification", model=model_name)

    def predict_category(self, sequence: str, categories: list[str] = FICTION_CATEGORIES) -> str:
        """Predicts the most likely category for a given text sequence using a zero-shot classification pipeline.
        
        This method applies the Hugging Face `pipeline` for zero-shot classification to determine  hich category from the provided list best fits the input text. It selects the category with the highest confidence score.

        Args:
            sequence (str): The input text (e.g., a book description) to classify.
            categories (list[str], optional): A list of possible category labels to compare against the sequence. Defaults to FICTION_CATEGORIES.

        Returns:
            str: The label from `categories` that best matches the input text based on the model's prediction.
        """
        prediction = self.pipe(sequence, categories)
        max_index = np.argmax(prediction["scores"])
        return prediction["labels"][max_index]

    def evaluate_model(self, books_df: pd.DataFrame, sample_size: int = 300) -> pd.DataFrame:
        """Evaluates the zero-shot classification model on a balanced subset of Fiction and Nonfiction books.

        Args:
            books_df (pd.DataFrame): The DataFrame containing book metadata, including descriptions and category labels.
            sample_size (int, optional): Number of samples to evaluate from each category (Fiction and Nonfiction). Defaults to 300.

        Returns:
            pd.DataFrame: A DataFrame with actual vs. predicted categories and a correctness flag for each prediction.

        Prints:
            Accuracy score (%) based on the selected sample.
        """
        actual, predicted = [], []

        for label in FICTION_CATEGORIES:
            subset = books_df[books_df["simple_categories"] == label].reset_index(drop=True)
            for i in tqdm(range(min(sample_size, len(subset)))):
                desc = subset["description"][i]
                actual.append(label)
                predicted.append(self.predict_category(desc))

        predictions_df = pd.DataFrame({
            "actual_categories": actual,
            "predicted_categories": predicted
        })

        predictions_df["correct_prediction"] = (
            predictions_df["actual_categories"] == predictions_df["predicted_categories"]
        ).astype(int)

        accuracy = predictions_df["correct_prediction"].mean()
        print(f"Accuracy on {2 * sample_size} samples: {accuracy:.2%}")
        return predictions_df

    def fill_missing_categories(self, books_df: pd.DataFrame) -> pd.DataFrame:
        """Predicts and fills missing values in the 'simple_categories' column using a zero-shot classification model.

        Args:
            books_df (pd.DataFrame): The DataFrame containing book metadata, including 'isbn13', 'description', and potentially missing 'simple_categories'.

        Returns:
            pd.DataFrame: Updated DataFrame with missing 'simple_categories' filled in using predicted values. The temporary 'predicted_categories' column is dropped after merging.
        """
        missing = books_df[books_df["simple_categories"].isna()][["isbn13", "description"]].reset_index(drop=True)
        predictions = []

        for i in tqdm(range(len(missing))):
            pred = self.predict_category(missing["description"][i])
            predictions.append(pred)

        predicted_df = pd.DataFrame({
            "isbn13": missing["isbn13"],
            "predicted_categories": predictions
        })

        books_df = pd.merge(books_df, predicted_df, on="isbn13", how="left")
        books_df["simple_categories"] = books_df["simple_categories"].fillna(books_df["predicted_categories"])
        return books_df.drop(columns=["predicted_categories"])
    
