# LLM-Based Book Recommender CLI

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

## Usage
```bash
python cli.py --query "a young wizard learning magic" --category Fantasy --emotion joy --top-k 5
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
