import evaluate
import json
import torch
from load_finetuned_model import model,tokenizer
# Load ROUGE metric
rouge = evaluate.load("rouge")

# Load your test data
with open('test.json') as f:
    test_data = json.load(f)

# Assuming you have a function to generate predictions
def generate_predictions(model, tokenizer, data, batch_size=12):
    predictions = []
    keys = list(data.keys())
    for i in range(0, len(keys), batch_size):
        batch_keys = keys[i:i+batch_size]
        batch_inputs = tokenizer([data[key]['disfluent'] for key in batch_keys], padding=True, return_tensors="pt", truncation=True).to('cuda')
        with torch.no_grad():
            batch_outputs = model.generate(**batch_inputs)
        batch_predictions = [tokenizer.decode(output, skip_special_tokens=True) for output in batch_outputs]
        predictions.extend(batch_predictions)
    return predictions

# Generate predictions
predictions = generate_predictions(model, tokenizer, test_data)

# Prepare references
references = [test_data[key]['original'] for key in test_data]

# Compute ROUGE scores
results = rouge.compute(predictions=predictions, references=references)

print(results)
