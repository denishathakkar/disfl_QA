import pandas as pd
import json
from datasets import Dataset


from load_original_model import load_model

model,tokenizer = load_model()
# Load data from JSON files
def load_data(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return pd.DataFrame({
        'id': list(data.keys()),
        'original': [v['original'] for v in data.values()],
        'disfluent': [v['disfluent'] for v in data.values()]
    })



# Prepare dataset for training
def tokenize_function(examples,tokenizer=tokenizer):
    inputs = tokenizer(examples['disfluent'], truncation=True, padding='max_length', max_length=128,
                            return_tensors='pt')
    targets = tokenizer(examples['original'], truncation=True, padding='max_length', max_length=128,
                             return_tensors='pt')

    # Ensure that targets are converted to lists of integers
    inputs['labels'] = targets['input_ids'].tolist()

    return inputs


def preprocess_data(df):
    dataset = Dataset.from_pandas(df[['disfluent', 'original']])
    return dataset.map(tokenize_function, batched=True, remove_columns=['disfluent', 'original'])

