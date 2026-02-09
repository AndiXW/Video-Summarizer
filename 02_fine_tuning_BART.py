# Fine-Tuning BART for Meeting Summarization
# In this .py, we will:
# 1. Load the preprocessed (tokenized) MeetingBank dataset from disk.
# 2. Set up the Facebook BART model and tokenizer.
# 3. Define training arguments and create the Trainer.
# 4. Fine-tune the model on the training data.
# 5. Evaluate on the validation split and save the model.
from itertools import product

from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Trainer, TrainingArguments, DataCollatorForSeq2Seq
import torch
import pandas as pd
import csv
import os
from transformers import TrainingArguments, Trainer

# Helps with memory allocation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# To set working directory in config
print("Current working directory:", os.getcwd())

# Check if CUDA is available
print("CUDA available:", torch.cuda.is_available())


# 1. Load the Preprocessed Dataset
# We load the tokenized data from disk. This dataset has already been split into training, validation, and test sets.

# Define the path where the tokenized dataset is saved

data_path = "./data/processed/tokenized_meetingbank"
data_path_SSH = "/home/ubuntu/tmp/pycharm_project_460/notebooks/data/processed/tokenized_meetingbank" # used for SSH

# Load the dataset
tokenized_dataset = load_from_disk(data_path)
print("Available splits:", tokenized_dataset.keys())

# Retrieve the splits
train_data = tokenized_dataset["train"]
val_data   = tokenized_dataset["validation"]
test_data  = tokenized_dataset["test"]

print("Length of train, eval and test data: ", len(train_data),len(val_data), len(test_data))

# Optional: use only half, tenth or hundredth of the training data to speed up training:
half_train_data = train_data.select(range(len(train_data) // 2))
tenth_train_data = train_data.select(range(len(train_data) // 10))
hundredth_train_data = train_data.select(range(len(train_data) // 100))

# 2. Set Up the Model and Tokenizer
# We'll use the "facebook/bart-large-xsum" checkpoint as our starting point.

model_checkpoint = "facebook/bart-large-xsum"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)

# 3. Define Training Arguments and Data Collator
# The `TrainingArguments` control the training configurations


# Create a data collator that dynamically pads data
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# Here are v1, v2, v2.1 and v2.2 of our training arguments.
# v2.2 is the best one currently

#v1
training_args_v1 = TrainingArguments(
    output_dir="./models/fine_tuned_v1",   # Directory to save checkpoints and final model
    eval_strategy="epoch",                 # Evaluate at the end of each epoch
    learning_rate=5e-5,
    per_device_train_batch_size=32,         # Adjust batch size based on GPU memory
    per_device_eval_batch_size=2,
    num_train_epochs=10,
    weight_decay=0.01,
    save_total_limit=2,                    # Only keep the 2 most recent checkpoints
    logging_dir="./logs",                  # Directory for logs
    logging_steps=50,
    fp16=True,
    dataloader_num_workers=4,
)

#v2
training_args_v2 = TrainingArguments(
    output_dir="./models/fine_tuned_v2",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=1,
    num_train_epochs=4,
    weight_decay=0.01,
    save_total_limit=2,
    logging_dir="./logs",
    logging_strategy="epoch",
    fp16=True,
    dataloader_num_workers=4,
    # gradient_checkpointing=True,
    prediction_loss_only=True,            # True: disable metric tracking and avoid OOM
    # eval_accumulation_steps = 1
)

#v2.1 (used for Hyperparameter tuning)

'''
# Set up search space
batch_sizes = [8, 16, 32, 64]
learning_rates = [2e-5, 3e-5, 5e-5]
epochs = [2, 3, 4]

# Prepare CSV log file
with open("loss_summary.csv", mode="a", newline="") as f:
    writer = csv.writer(f)
    if f.tell() == 0:  # only if file is empty
        writer.writerow(["batch_size", "learning_rate", "epochs", "epoch", "train_loss", "eval_loss"])


for bs, lr, ep in product(batch_sizes, learning_rates, epochs):
    print(f"Running config: bs={bs}, lr={lr}, epochs={ep}")

    training_args_v2_1 = TrainingArguments(
        output_dir=f"./results/bs{bs}_lr{lr}_ep{ep}",
        per_device_train_batch_size=bs,
        per_device_eval_batch_size=1,
        learning_rate=lr,
        num_train_epochs=ep,
        eval_strategy="epoch",
        save_strategy="no",
        logging_strategy="epoch",
        prediction_loss_only=True,
        dataloader_num_workers=4,
        weight_decay=0.01,
        fp16=True,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args_v2_1,
        train_dataset=train_data,
        eval_dataset=val_data,
        data_collator=data_collator

    )

    trainer.train()

    # Extract last train/eval loss & log it
    log_history = trainer.state.log_history

    train_loss = next((l["loss"] for l in reversed(log_history) if "loss" in l), None)
    eval_loss = next((l["eval_loss"] for l in reversed(log_history) if "eval_loss" in l), None)
    epoch = next((l["epoch"] for l in reversed(log_history) if "loss" in l or "eval_loss" in l), None)

    with open("/home/ubuntu/tmp/pycharm_project_460/notebooks/results/loss_summary.csv", mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([bs, lr, ep, round(epoch, 2), train_loss, eval_loss])
'''

#v2.2
training_args_v2_2 = TrainingArguments(
    output_dir="./models/fine_tuned_v2.2",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=1,
    num_train_epochs=4,
    weight_decay=0.01,
    save_total_limit=2,
    logging_dir="./logs",
    logging_strategy="epoch",
    fp16=True,
    dataloader_num_workers=4,
    prediction_loss_only=True,            # True: disable metric tracking and avoid OOM
)

# This was supposed to be a real-time ROUGE calculator, which turned out to produce consistent OOM errors.

'''ROUGE Calculator

rouge = evaluate.load("rouge")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred

    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

    # Replace -100 in labels for padding (used by HuggingFace)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Light cleanup (optional)
    decoded_preds = ["\n".join(p.strip().split(". ")) for p in decoded_preds]
    decoded_labels = ["\n".join(l.strip().split(". ")) for l in decoded_labels]

    result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    torch.cuda.empty_cache()  # just in case
    return {k: round(v * 100, 2) for k, v in result.items()}
'''

# This was supposed to be a fix for our constant OOM.
# It is not currently functional

'''
# OOM "fix"
def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        logits = logits[0]  # Ignore any extra outputs (like past_key_values)

    # logits shape = [batch_size, sequence_length, vocab_size]
    # take argmax over vocab dimension to get predicted token IDs
    predicted_token_ids = torch.argmax(logits, dim=-1)

    # Ensure it's a CPU tensor with int values
    return predicted_token_ids.detach().cpu()
'''

# 4. Create the Trainer and Fine-Tune the Model
trainer = Trainer(
    model=model,
    args=training_args_v2_2, # choose from different hyperparameters here
    train_dataset=train_data,  # Use smaller splits here for faster experimentation
    eval_dataset=val_data,
    data_collator=data_collator,
    # compute_metrics=compute_metrics,
    # preprocess_logits_for_metrics=preprocess_logits_for_metrics
)

# Start training
trainer.train()

# Eval output
log_file = "results/epoch_loss_summary.csv"
logs = trainer.state.log_history

epoch_data = {}

for log in logs:
    epoch = int(round(log.get("epoch", -1)))
    if epoch <= 0:
        continue
    if epoch not in epoch_data:
        epoch_data[epoch] = {"train_loss": None, "eval_loss": None}
    if "loss" in log:
        epoch_data[epoch]["train_loss"] = log["loss"]
    if "eval_loss" in log:
        epoch_data[epoch]["eval_loss"] = log["eval_loss"]

with open(log_file, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["Epoch", "Training Loss", "Validation Loss"])
    for epoch, losses in sorted(epoch_data.items()):
        writer.writerow([epoch, losses["train_loss"], losses["eval_loss"]])

df = pd.read_csv(log_file)
print(df)


# 5. Evaluate and Save the Model
# After training, we evaluate the model on the validation set and then save the model and tokenizer.

eval_results = trainer.evaluate()
print("Evaluation results:", eval_results)

# Save the fine-tuned model and tokenizer
model_save_path = "./models/fine_tuned_v1"
model.save_pretrained(model_save_path)
tokenizer.save_pretrained(model_save_path)
print(f"Model and tokenizer saved to {model_save_path}")