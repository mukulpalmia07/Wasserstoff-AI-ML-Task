from transformers import BartTokenizer, BartForConditionalGeneration
import csv

def load_summarization_model():
    """Loading the pre-trained BART model for summarization."""

    model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
    return model, tokenizer

def summarize_text(model, tokenizer, text):
    """Generating a summary of the given text."""

    inputs = tokenizer(text, return_tensors = "pt", max_length = 1024, truncation = True)
    summary_ids = model.generate(inputs["input_ids"], max_length = 150, min_length = 30, length_penalty = 2.0, num_beams = 4, early_stopping = True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens = True)
    return summary

def summarize_object_attributes(object_id, extracted_text):
    """Summarizing attributes of the object using the extracted text."""

    model, tokenizer = load_summarization_model()
    summary = summarize_text(model, tokenizer, extracted_text)
    return object_id, summary

def save_summaries_to_csv(output_path, summaries):
    """Save summarized attributes of objects to a CSV file."""

    with open(output_path, mode = 'w', newline = '') as file:
        writer = csv.writer(file)
        writer.writerow(["Object ID", "Summary"])
        for entry in summaries:
            writer.writerow(entry)
