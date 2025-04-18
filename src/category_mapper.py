import pandas as pd

CATEGORY_MAPPING = {
    'Fiction': "Fiction",
    'Juvenile Fiction': "Children's Fiction",
    'Biography & Autobiography': "Nonfiction",
    'History': "Nonfiction",
    'Literary Criticism': "Nonfiction",
    'Philosophy': "Nonfiction",
    'Religion': "Nonfiction",
    'Comics & Graphic Novels': "Fiction",
    'Drama': "Fiction",
    'Juvenile Nonfiction': "Children's Nonfiction",
    'Science': "Nonfiction",
    'Poetry': "Fiction"
}

FICTION_CATEGORIES = ["Fiction", "Nonfiction"]

def map_categories(books_df: pd.DataFrame) -> pd.DataFrame:
    """Maps the original book categories to simplified categories.
    
    This function uses a predefined dictionary (CATEGORY_MAPPING) to convert the riginal, possibly complex or varied 'categories' column in the dataset into simplified and consistent set of categories. The new mapped values are added to a new column called 'simple_categories'.

    Args:
        books_df (pd.DataFrame): The input DataFrame containing a 'categories' column with raw category labels.

    Returns:
        pd.DataFrame: A copy of the input DataFrame with an added 'simple_categories' column containing the mapped category values.
    """
    books_df["simple_categories"] = books_df["categories"].map(CATEGORY_MAPPING)
    return books_df