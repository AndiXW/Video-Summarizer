from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer

# Path to our fine-tuned model
# CAUTION: this ONLY exists after running 02_fine_tuning_BART.py
fine_tuned_model_path = "./models/fine_tuned_v2.3"

# Load model and tokenizer
ft_model = AutoModelForSeq2SeqLM.from_pretrained(fine_tuned_model_path)
ft_tokenizer = AutoTokenizer.from_pretrained(fine_tuned_model_path)

# Send model to GPU
ft_model.to("cuda")

# Create summarization pipeline
summarizer = pipeline(
    "summarization",
    device=0,  # GPU index, should already be default

    # you can remove "model and "tokenizer" for testing purposes, if fine  tuned model not available (it will use the default model then and not our fine-tuned one!)
    model=ft_model, 
    tokenizer=ft_tokenizer, 
)

# Your input from the GUI
with open("transcript.txt", "r") as file:
    text = file.read()

# Generate summary
summary = summarizer(text, max_length=60, min_length=20, do_sample=False)
print(summary[0]['summary_text']) # pass it back to the GUI here
