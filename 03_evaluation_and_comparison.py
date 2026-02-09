# Evaluation & Comparison
# In this .py, we:
# 1. Load the saved tokenized test data.
# 2. Load the fine-tuned model from saved experiment and the base (pre-trained) model.
# 3. Generate predictions for both models on the test set.
# 4. Compute ROUGE scores for both sets of predictions.
# 5. Export side-by-side results to a CSV file so we can visualize them in Excel for the presentation.

import csv
from datasets import load_from_disk
from evaluate import load as load_metric
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import pandas as pd
import torch

print("CUDA available:", torch.cuda.is_available())

# 1. Load the Tokenized Test Dataset
data_path = "./data/processed/tokenized_meetingbank"
tokenized_dataset = load_from_disk(data_path)
test_data = tokenized_dataset["test"]

print("Total test examples:", len(test_data))

# For demonstration limit predictions to a manageable subset
sample_size = len(test_data)
sample_test = test_data.select(range(sample_size))

# 2.Load Fine-Tuned Model and Base Model
# Load the fine-tuned model and tokenizer from experiment directory

fine_tuned_model_path = "./models/fine_tuned_v2.3" # this only exists after running 02_fine_tuning_BART.py
ft_model = AutoModelForSeq2SeqLM.from_pretrained(fine_tuned_model_path)
ft_tokenizer = AutoTokenizer.from_pretrained(fine_tuned_model_path)
ft_model.to("cuda")

# Load the base (pre-trained) model and tokenizer
base_model_checkpoint = "facebook/bart-large-xsum"
base_model = AutoModelForSeq2SeqLM.from_pretrained(base_model_checkpoint)
base_tokenizer = AutoTokenizer.from_pretrained(base_model_checkpoint)
base_model.to("cuda")

# 3. Generate Predictions for both models on the test set
# Helper Function: Generate Predictions for a Single Example
# Could've been solved better by using a HuggingFace pipeline and not do this manually!
def generate_prediction(model, tokenizer, example, max_length=128, num_beams=4):
    # Convert input ids and attention mask to tensors and add a batch dimension
    # We add attention_mask here to tell the model which tokens are real input and which are just padding.
    input_ids = torch.tensor(example["input_ids"]).unsqueeze(0).to("cuda")
    attention_mask = torch.tensor(example["attention_mask"]).unsqueeze(0).to("cuda")

    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=max_length,
        num_beams=num_beams,
        early_stopping=True
    )
    prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return prediction

# Generate Predictions on the Sample Test Set
ft_predictions = []
base_predictions = []
gold_summaries = []  # from the test set
transcripts = []     # original transcript text
print("Sample test transcript:", tokenized_dataset["test"][0]["transcript"])
print("Sample test summary:", tokenized_dataset["test"][0]["summary"])
# Iterate over the sample examples
for example in tqdm(sample_test, desc="Generating summaries"):
    # For ROUGE evaluation, use the raw text target
    gold_summaries.append(example["summary"])
    # The original transcript text
    transcripts.append(example["transcript"])

    # Generate fine-tuned prediction
    ft_pred = generate_prediction(ft_model, ft_tokenizer, example)
    ft_predictions.append(ft_pred)

    # Generate base model prediction.
    base_pred = generate_prediction(base_model, base_tokenizer, example)
    base_predictions.append(base_pred)

# 4. Compute ROUGE scores for both sets of predictions.

# Compute ROUGE Metrics for Both Models
rouge_metric = load_metric("rouge")

ft_rouge = rouge_metric.compute(predictions=ft_predictions, references=gold_summaries)
base_rouge = rouge_metric.compute(predictions=base_predictions, references=gold_summaries)

print("Fine-Tuned Model ROUGE Scores:")
print(ft_rouge)
print("\nBase Model ROUGE Scores:")
print(base_rouge)

# 5. Export side-by-side results to a CSV

# Prepare Data for Export to CSV
# Create a DataFrame for side-by-side comparison
comparison_df = pd.DataFrame({
    "Transcript": transcripts,
    "Gold Summary": gold_summaries,
    "Fine-Tuned Prediction": ft_predictions,
    "Base Model Prediction": base_predictions
})

# Export the DataFrame to a CSV file so we can open it in Excel for further visualization
csv_path = "results/evaluation_results.csv"
comparison_df.to_csv(
    "evaluation_results.csv",
    sep=";",
    index=False,
    quoting=csv.QUOTE_ALL  # Ensures all fields are quoted
)
print(f"Side-by-side evaluation data saved to {csv_path}")