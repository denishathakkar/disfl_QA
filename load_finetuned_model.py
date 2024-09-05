from transformers import T5ForConditionalGeneration, T5Tokenizer


model_path = 'result'
# Load the tokenizer
tokenizer = T5Tokenizer.from_pretrained(model_path)
# Load the model
model = T5ForConditionalGeneration.from_pretrained(model_path)

prompt = "In what country is Norse found no wait Normandy not Norse?"
input_ids= tokenizer(prompt, return_tensors="pt").input_ids.to("cpu")
outputs = model.generate(input_ids=input_ids)
tokenizer.decode(outputs[0])