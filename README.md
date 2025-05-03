# IU South Bend Library QA Chatbot

A BERT-based question-answering chatbot that helps users get answers about IU South Bend Library services using natural language.

## Features
- Auto-generates Q&A pairs from structured JSON data
- Fine-tunes BERT for extractive QA
- Uses FAISS for fast semantic context retrieval
- Interactive command-line interface

## Quick Start

1. **Install dependencies**  
```bash
pip install -r requirements.txt
```

2. **Run the chatbot**  
```bash
python main.py
```

3. **Ask your questions**  
Type questions like _"What are the borrowing rules?"_ or _"Who do I contact for fines?"_

## Notes
- Dataset files go in the `datasets/` folder
- Large model files are excluded due to GitHub limits
- Use `ai_chatbot_package.zip` if included, to get a lightweight project snapshot

