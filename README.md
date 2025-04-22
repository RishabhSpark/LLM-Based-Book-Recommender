# LLM-Based Book Recommender

This is an AI-powered Book Recommendation System that leverages semantic search, zero-shot classification, and emotion analysis to suggest books based on natural language queries. It provides intelligent recommendations not just by keywords or genres, but by understanding the deeper meaning and emotional tone of the query.


---

## Features

- **Semantic Search**: Uses vector embeddings (via LangChain + Chroma) to retrieve books similar to a user’s description.
- **Category Filtering**: Uses a Zero-Shot Classifier to predict book genres using Hugging Face models.
- **Emotion Filtering**: Applies sentiment analysis on book descriptions to infer emotional tones like joy, sadness, fear, etc.
- **Efficient CLI**: Fast, simple command-line access for quick recommendations.
- **Gradio Dashboard**: Clean UI for interactive exploration and search.

---

## Installation

```bash
git clone https://github.com/your-username/llm-novel-recommendation-system.git
poetry install
```

---

## pipeline.py
The pipeline.py script is the core of the preprocessing and data preparation process for the book recommender system. It automates the entire workflow from raw dataset cleaning to categorization and sentiment analysis, ensuring that the data is in a ready-to-use format for building and querying the recommender system.

Key Functions:
- Data Cleaning:
The script starts by cleaning the raw book dataset (books.csv). It removes any unnecessary columns, handles missing values, and ensures the text data (such as book descriptions) is formatted properly for further analysis.
- Vectorstore Creation:
After cleaning, the script creates a vector store using the book descriptions. This is crucial for semantic search functionality, allowing the system to retrieve the most relevant books based on a query. The vector store is built using the LangChain and Chroma libraries, which provide powerful embeddings and vector search capabilities.
- Category Mapping:
The dataset is enriched with categories assigned to each book. The script maps the books to predefined categories to help filter recommendations based on user preferences.
- Sentiment Analysis:
The sentiment analysis step assesses the emotional tone of each book's description using the Transformers library. This step analyzes emotions like joy, fear, sadness, and anger to give users more personalized recommendations based on the emotional tone they prefer.

Final Output:
- The output is a cleaned, categorized, and sentiment-analyzed dataset that is ready for use in the recommendation pipeline. This enriched dataset is saved to a file (books_with_emotions.csv), which is used by the recommender system during the retrieval of recommendations.

This workflow ensures that the dataset is consistently prepared, and the recommender system is working with up-to-date, relevant data.

How to Run pipeline.py:
```bash
python pipeline.py
```
This will run all steps of the pipeline in sequence. The dataset will be cleaned, vector store will be created, categories will be mapped, and sentiment analysis will be performed.

---

## Usage
```bash
python cli.py --query "a young wizard learning magic" --category "Fiction" --emotion "joy" --top-k 5
```

### CLI Options

| Option       | Type   | Description                                                                  | Default |
|--------------|--------|------------------------------------------------------------------------------|---------|
| `--query`    | `str`  | **(Required)** The input query to find similar book descriptions             | —       |
| `--category` | `str`  | Filter recommendations by book category (e.g., `Fiction`, `Fantasy`, etc.)   | `"All"` |
| `--emotion`  | `str`  | Filter by emotional tone (e.g., `joy`, `fear`, `sadness`, `anger`, etc.)     | `"All"` |
| `--top-k`    | `int`  | Number of recommendations to display      

To see help:
```bash
python cli.py recommend --help
```

---

## Gradio Dashboard
To run the Gradio app and get an interactive UI for querying the system:
```bash
python gradio_app.py
```

---

## Tech Stack
| **Purpose**                | **Tool/Library**                          |
|----------------------------|-------------------------------------------|
| Vector Embeddings           | LangChain, Chroma                         |
| Sentiment Analysis          | transformers                             |
| Classification (Zero-Shot)  | transformers, HuggingFace                 |
| CLI Interface               | Typer                                     |
| UI / Dashboard              | Gradio                                    |
| Data Processing             | pandas, sklearn                           |
| Visualization               | matplotlib, seaborn                       |
