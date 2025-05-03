import os
import json
import random
import numpy as np
from transformers import BertTokenizerFast, BertForQuestionAnswering, Trainer, TrainingArguments
from datasets import Dataset
from sentence_transformers import SentenceTransformer
import faiss
import torch
import logging
import re

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Determine device (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Suppress tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Directory containing JSON files
JSON_DIR = "datasets/"  # Update this to your folder path
MODEL_DIR = "./qa_model/"  # Directory to save/load the trained model

# Function to load JSON files from a directory
def load_json_files(directory):
    datasets = []
    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            filepath = os.path.join(directory, filename)
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    if os.path.getsize(filepath) == 0:
                        logger.error(f"File {filename} is empty")
                        continue
                    data = json.load(f)
                    datasets.append(data)
                logger.info(f"Loaded {filename}")
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON in {filename}: {e}")
                continue
            except Exception as e:
                logger.error(f"Error loading {filename}: {e}")
                continue
    return datasets

# Function to generate synthetic question-answer pairs
def generate_qa_pairs(datasets):
    qa_pairs = []
    question_templates = [
        "What is {keyword}?",
        "How do I {keyword}?",
        "Can you tell me about {keyword}?",
        "What are the rules for {keyword}?",
        "How does {keyword} work?",
        "Where can I find {keyword}?",
        "What are the requirements for {keyword}?",
    ]

    for dataset in datasets:
        keywords = dataset.get("keywords", [])
        keyword_mapping = dataset.get("keyword_mapping", {})
        page_link = dataset.get("page_link", "")

        if not keywords:
            logger.warning(f"No keywords found in dataset: {dataset}")
        if not keyword_mapping:
            logger.warning(f"No keyword_mapping found in dataset: {dataset}")

        for keyword in keywords:
            mapping = keyword_mapping.get(keyword, "")
            if not mapping:
                logger.debug(f"Skipping keyword '{keyword}' due to empty or missing mapping")
                continue

            # Generate multiple questions for each keyword
            for template in random.sample(question_templates, min(3, len(question_templates))):
                question = template.format(keyword=keyword.lower())
                answer = mapping
                qa_pairs.append({
                    "context": mapping,
                    "question": question,
                    "answer": answer,
                    "link": page_link
                })

            # Add a specific question for contact-related keywords
            if "contact" in keyword.lower():
                qa_pairs.append({
                    "context": mapping,
                    "question": f"Who do I contact for {keyword.lower()}?",
                    "answer": answer,
                    "link": page_link
                })

    logger.info(f"Generated {len(qa_pairs)} question-answer pairs")
    return qa_pairs

# Function to preprocess data for BERT
def preprocess_data(qa_pairs, tokenizer, max_length=512):
    questions = [pair["question"] for pair in qa_pairs]
    contexts = [pair["context"] for pair in qa_pairs]
    answers = [pair["answer"] for pair in qa_pairs]
    links = [pair["link"] for pair in qa_pairs]

    encodings = tokenizer(
        questions,
        contexts,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_offsets_mapping=True,
        return_tensors="pt"
    )

    start_positions = []
    end_positions = []

    for i in range(len(qa_pairs)):
        context = contexts[i]
        answer = answers[i]
        offset_mapping = encodings["offset_mapping"][i].tolist()

        # Find the start and end character positions of the answer in the context
        start_char = context.find(answer)
        if start_char == -1:
            # If answer not found, mark as invalid (CLS token)
            start_positions.append(0)
            end_positions.append(0)
            continue

        end_char = start_char + len(answer)

        # Convert character positions to token positions
        start_token = None
        end_token = None
        for idx, (start, end) in enumerate(offset_mapping):
            if start_token is None and start <= start_char < end:
                start_token = idx
            if end_token is None and start <= end_char <= end:
                end_token = idx
                break

        if start_token is None or end_token is None:
            # If token positions not found, mark as invalid
            start_positions.append(0)
            end_positions.append(0)
        else:
            start_positions.append(start_token)
            end_positions.append(end_token)

    encodings_dict = {
        "input_ids": encodings["input_ids"],
        "attention_mask": encodings["attention_mask"],
        "token_type_ids": encodings["token_type_ids"],
        "start_positions": start_positions,
        "end_positions": end_positions,
        "links": links
    }

    return encodings_dict

# Function to create FAISS index for retrieval
def create_faiss_index(datasets):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    mappings = []
    metadata = []

    for dataset in datasets:
        keyword_mapping = dataset.get("keyword_mapping", {})
        page_link = dataset.get("page_link", "")
        for keyword, mapping in keyword_mapping.items():
            mappings.append(mapping)
            metadata.append({"keyword": keyword, "mapping": mapping, "link": page_link})

    embeddings = model.encode(mappings)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    return index, model, metadata

# Function to train the BERT model
def train_model(encodings, output_dir=MODEL_DIR):
    # Convert encodings to Dataset
    dataset = Dataset.from_dict(encodings)

    # Split into train and validation
    train_size = int(0.8 * len(dataset))
    train_dataset = dataset.select(range(train_size))
    val_dataset = dataset.select(range(train_size, len(dataset)))

    # Initialize model and tokenizer
    model_name = "bert-base-uncased"
    tokenizer = BertTokenizerFast.from_pretrained(model_name)
    model = BertForQuestionAnswering.from_pretrained(model_name)
    model.to(device)  # Move model to the appropriate device

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",  # Changed from eval_strategy
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset
    )

    # Train the model
    trainer.train()

    # Save the model
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info(f"Model saved to {output_dir}")

    return model, tokenizer

# Function to answer a question using the trained model
def answer_question(question, context, model, tokenizer, link, max_length=512):
    try:
        logger.info(f"Processing question: {question}")
        logger.info(f"Context: {context}")

        inputs = tokenizer(
            question,
            context,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt"
        )

        # Move inputs to the same device as the model
        inputs = {key: val.to(device) for key, val in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            start_scores = outputs.start_logits
            end_scores = outputs.end_logits

            start_idx = torch.argmax(start_scores)
            end_idx = torch.argmax(end_scores) + 1

            answer_tokens = inputs["input_ids"][0][start_idx:end_idx]
            answer = tokenizer.decode(answer_tokens, skip_special_tokens=True)
            logger.info(f"Raw answer: {answer}")

        # Post-process: Remove the question text if it appears in the answer
        question_lower = question.lower().strip('?')
        answer_clean = re.sub(r'^' + re.escape(question_lower) + r'\W*', '', answer.lower(), flags=re.IGNORECASE).strip()
        if not answer_clean:
            return f"Sorry, I couldn't find a specific answer. For more details, visit {link}"

        # Capitalize the first letter of the cleaned answer
        answer_clean = answer_clean[0].upper() + answer_clean[1:] if answer_clean else answer_clean
        return f"{answer_clean}. For more details, visit {link}"
    except Exception as e:
        logger.error(f"Error answering question: {e}")
        return f"An error occurred while processing your question. Please try again or visit {link} for more information."

# Main function
def main():
    # Load JSON files
    datasets = load_json_files(JSON_DIR)
    if not datasets:
        logger.error("No JSON files found in the directory")
        return

    # Generate QA pairs
    qa_pairs = generate_qa_pairs(datasets)
    if not qa_pairs:
        logger.error("No question-answer pairs generated. Please check the JSON data structure.")
        return

    # Initialize tokenizer
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

    # Check if model exists, otherwise train it
    if os.path.exists(MODEL_DIR) and os.path.exists(os.path.join(MODEL_DIR, "pytorch_model.bin")):
        logger.info(f"Loading pre-trained model from {MODEL_DIR}")
        model = BertForQuestionAnswering.from_pretrained(MODEL_DIR)
        tokenizer = BertTokenizerFast.from_pretrained(MODEL_DIR)
        model.to(device)
    else:
        logger.info("Training new model...")
        encodings = preprocess_data(qa_pairs, tokenizer)
        model, tokenizer = train_model(encodings)

    # Create FAISS index for retrieval
    faiss_index, sent_model, metadata = create_faiss_index(datasets)

    # Interactive question-answering loop
    print("\nWelcome to the IU South Bend Library QA System!")
    print("Type your question below or enter 'exit' to quit.")
    while True:
        question = input("\nYour question: ").strip()
        if question.lower() in ["exit", "quit"]:
            print("Thank you for using the IU South Bend Library QA System!")
            break
        if not question or question.lower() in ["bye", "hello", "hi"]:
            print("Please enter a valid question or type 'exit' to quit.")
            continue

        try:
            # Retrieve relevant context using FAISS
            question_embedding = sent_model.encode([question])
            distances, indices = faiss_index.search(question_embedding, k=1)
            retrieved_metadata = metadata[indices[0][0]]
            context = retrieved_metadata["mapping"]
            link = retrieved_metadata["link"]

            # Get answer from the model
            answer = answer_question(question, context, model, tokenizer, link)
            print(f"Answer: {answer}")
        except Exception as e:
            logger.error(f"Error processing question: {e}")
            print("An error occurred. Please try again or contact the library at 574-520-4440.")

if __name__ == "__main__":
    main()
