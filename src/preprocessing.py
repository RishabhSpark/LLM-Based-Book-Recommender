from typing import Optional
import pandas as pd
import numpy as np

def clean_books_dataset(input_csv_path: str, output_csv_path: Optional[str] = None) -> pd.DataFrame:
    """Cleans the book dataset by applying filtering, transformations, and feature engineering.

    Args:
        input_csv_path (str): Path to the input CSV file.
        output_csv_path (Optional[str], optional): If provided, saves the cleaned dataset to this path. Defaults to None.

    Returns:
        pd.DataFrame: The cleaned and filtered books DataFrame.
    """
    df = pd.read_csv(input_csv_path)

    # Create missing description and age_of_books columns
    df['missing_description'] = np.where(df['description'].isnull(), 1, 0)
    df['age_of_books'] = 2025 - df['published_year']

    # Filter out books with missing values in specific columns
    books = df.dropna(subset=['description', 'num_pages', 'average_rating', 'published_year']).copy()

    # Filter out books with less than 30 words in the description
    books['words_in_description'] = books['description'].apply(lambda x: len(str(x).split()))
    filtered_books = books[books['words_in_description'] > 30].copy()

    # Combine title and subtitle
    filtered_books['title_and_subtitle'] = np.where(
        filtered_books['subtitle'].isna(),
        filtered_books['title'],
        filtered_books[['title', 'subtitle']].astype(str).agg(": ".join, axis=1)
    )

    # Create tagged_description
    filtered_books['tagged_description'] = filtered_books[['isbn13', 'description']].astype(str).agg(" ".join, axis=1)

    # Drop unneeded columns
    cleaned_books = filtered_books.drop(
        ['subtitle', 'missing_description', 'age_of_books', 'words_in_description'],
        axis=1
    )

    # Save if output path provided
    if output_csv_path:
        cleaned_books.to_csv(output_csv_path, index=False)

    return cleaned_books