# AI Assistant

## Locally Hosted
No calls required after you install Ollama and pull the model.

## Features
- Parses .txt and .pdf files from a folder provided
- Chunks text into small pieces
- Embeds with sentence-transformers
- Indexes into FAISS
- search and summarize key concepts
- Retrieves the top-k relevant chunks
- Builds a simple prompt with citations
- Calls Ollama (mistral)
- Returns a structured JSON response

## Quickstart
download ollama

```
ollama pull mistral

pip install -r requirements.txt
```

In project folder add a folder called "docs" and put pdf or txt files into it.

```
python src/main.py ingest docs
python src/main.py ask "INSERT YOUR QUESTION HERE" --k 5 --json
python src/main.py search "YOUR SEMANTIC SEARCH HERE" --k 5
python src/main.py summarize --json
```