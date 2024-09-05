from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def load_model():
    # Load the pre-trained model
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large")
    tokenizer.pad_token_id = tokenizer.eos_token_id


    # Add pad token if not already present
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))

    return model, tokenizer


