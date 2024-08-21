import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.summarization import load_summarization_model, summarize_text, summarize_object_attributes, save_summaries_to_csv

def test_summarization():
    """Testing the summarization functions."""

    # Sample data
    object_attributes = {
        "object_1": "Data for Object 1",
        "object_2": "Data for Object 2",
        "object_3": "Data for Object 3"
    }

    # Loading model and tokenizer
    model, tokenizer = load_summarization_model()

    # Summarize object attributes
    summaries = []
    for object_id, text in object_attributes.items():
        summary = summarize_text(model, tokenizer, text)
        summaries.append(summarize_object_attributes(object_id, summary))
    
    # Defining output path for CSV file
    output_path = "data\\output\\test_summarization.csv"

    # Saving summaries to CSV
    save_summaries_to_csv(output_path, summaries)
    print(f"Summaries have been saved to {output_path}")

if __name__ == "__main__":
    test_summarization()
