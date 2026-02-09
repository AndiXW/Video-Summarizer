
# ONLY FOR REFERENCE
# THIS IS THE FIRST DRAFT OF OUR TRAINING PIPELINE BEFORE MOVING TO A MORE STRUCTURED APPROACH

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import Trainer, TrainingArguments
import torch

print(torch.cuda.is_available()) # test if training is done using NVIDIA GPU

# Load the MeetingBank dataset
meetingbank = load_dataset("huuuyeah/meetingbank")
train_data = meetingbank["train"]
val_data = meetingbank["validation"]
test_data = meetingbank["test"]

# print(train_data[0])

model_checkpoint = "facebook/bart-large-xsum"  # facebook/bart-base or "facebook/bart-large-cnn" for a larger model
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)

max_input_length = 1024
max_target_length = 128


def preprocess_function(examples):
    inputs = examples["transcript"]
    targets = examples["summary"]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)
    labels = tokenizer(text_target=targets, max_length=max_target_length, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs



# Apply the preprocessing to the training, validation and test sets
tokenized_train = train_data.map(preprocess_function, batched=True)
tokenized_val = val_data.map(preprocess_function, batched=True)
tokenized_test = test_data.map(preprocess_function, batched=True)

half_train_data = tokenized_train.select(range(len(tokenized_train) // 2))

training_args = TrainingArguments(
    output_dir="./meeting_summarizer",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=2,    # Adjust based on your GPU memory
    per_device_eval_batch_size=2,
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=2,
)
from transformers import DataCollatorForSeq2Seq

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=half_train_data, # use tokenized_train for entire data set
    eval_dataset=tokenized_val,
    data_collator=data_collator
)

# Start training
trainer.train()

# Evaluate on the validation set
eval_results = trainer.evaluate()
print("Evaluation results:", eval_results)

# Save the model and tokenizer
model.save_pretrained("meeting_summarizer_model")
tokenizer.save_pretrained("meeting_summarizer_model")


# Generate predictions on the test set using the fine-tuned model
predictions_output = trainer.predict(tokenized_test)
decoded_preds = tokenizer.batch_decode(predictions_output.predictions, skip_special_tokens=True)
decoded_labels = tokenizer.batch_decode(predictions_output.label_ids, skip_special_tokens=True)

# Calculate ROUGE scores (or any other metric) to quantify performance
from evaluate import load as load_metric
rouge = load_metric("rouge")
rouge_results = rouge.compute(predictions=decoded_preds, references=decoded_labels)
print("ROUGE scores for the fine-tuned model:", rouge_results)

