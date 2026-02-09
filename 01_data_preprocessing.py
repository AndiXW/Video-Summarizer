# Data Preprocessing for MeetingBank Summarization

# In this .py, we will:
# 1. Load the MeetingBank dataset.
# 2. Preprocess the data (tokenize transcripts and summaries).
# 3. Save the processed dataset to disk for reuse in our training experiments.
# We'll use the Hugging Face Datasets library along with a BART tokenizer
# (using the `"facebook/bart-large-xsum"` checkpoint) for our tokenization.

from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer
import torch
import os

# Check if a GPU is available (useful for later when training)
print("CUDA available?", torch.cuda.is_available())

# 1. Load the MeetingBank Dataset
# The dataset contains pre-defined splits. We'll load all three splits: train, validation, and test.

# Load the MeetingBank dataset from Hugging Face Hub
meetingbank = load_dataset("huuuyeah/meetingbank")
print("Dataset splits:", meetingbank.keys())

# Inspect one sample from the training set
print("Sample training example:")
print(meetingbank["train"][0])

# 2. Preprocessing: Tokenize Transcripts and Summaries
# We'll use a pre-trained tokenizer from BART to tokenize the transcripts (input) and summaries (target).
# We also set maximum lengths:
# max_input_length: Maximum token length for the transcript.
# max_target_length: Maximum token length for the summary.

# Define model checkpoint and load the tokenizer
model_checkpoint = "facebook/bart-large-xsum"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

# Define tokenization parameters
max_input_length = 1024
max_target_length = 128

def preprocess_function(examples):
    # Extract transcripts and summaries
    inputs = examples["transcript"]
    targets = examples["summary"]
    # Tokenize inputs (transcripts)
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)
    # Tokenize targets using `text_target` for label creation
    labels = tokenizer(text_target=targets, max_length=max_target_length, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Apply the Preprocessing Function.
# We apply the mapping function over the train, validation, and test splits.

# Tokenize the dataset splits
tokenized_train = meetingbank["train"].map(preprocess_function, batched=True)
tokenized_val   = meetingbank["validation"].map(preprocess_function, batched=True)
tokenized_test  = meetingbank["test"].map(preprocess_function, batched=True)

# Inspect one preprocessed training example
print("Tokenized training example:")
print(tokenized_train[0])

# 3. Save the Processed Dataset to Disk
# Saving our processed data allows us to quickly load it in our training notebook
# without repeating the tokenization step each time.
# We'll combine the splits into a `DatasetDict` and save them into a folder (./data/processed/tokenized_meetingbank).

# Create a DatasetDict with the tokenized splits
tokenized_dataset = DatasetDict({
    "train": tokenized_train,
    "validation": tokenized_val,
    "test": tokenized_test
})

# Define save path
save_path = "data/processed/tokenized_meetingbank"
os.makedirs(os.path.dirname(save_path), exist_ok=True)

# Save the tokenized dataset to disk
tokenized_dataset.save_to_disk(save_path)
print(f"Tokenized dataset saved to {save_path}")
