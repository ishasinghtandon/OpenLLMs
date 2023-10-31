import pandas as pd

# Example scores for each model
model_scores = {
    "Model": ["t5-small", "facebook/bart-base", "google/pegasus-xsum"],
    "Average ROUGE": [0.4, 0.45, 0.48]  # Replace with actual ROUGE scores
}

df = pd.DataFrame(model_scores)
print(df)
