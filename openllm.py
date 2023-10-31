# Imports for models
from transformers import pipeline

# Initialize the summarization models
model_names = ["t5-small", "facebook/bart-base", "google/pegasus-xsum"]

# Example function to summarize using transformers pipeline
def summarize_with_pipeline(model_name, text):
    summarizer = pipeline("summarization", model=model_name)
    summary = summarizer(text, max_length=100, min_length=30, do_sample=False)
    return summary[0]['summary_text']

# Example code to load dataset (this step depends on the dataset)
# Assuming you have a list of articles and their corresponding summaries
articles = [...]  # List of articles
summaries = [...]  # List of corresponding summaries

# Loop through models and evaluate summaries
for model_name in model_names:
    rouge_scores = []
    for article, gold_summary in zip(articles, summaries):
        # Generate summaries using the model
        generated_summary = summarize_with_pipeline(model_name, article)
        # Compute ROUGE scores
        rouge_scores.append(compute_rouge(generated_summary, gold_summary))
    # Calculate average ROUGE scores for the model
    average_rouge = sum(rouge_scores) / len(rouge_scores)
    print(f"Model: {model_name}, Average ROUGE: {average_rouge}")
