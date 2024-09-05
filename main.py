from data_loader import load_data,preprocess_data
from load_original_model import load_model
from fine_tune import trainer_run, loss_logger,training_args

# Load model and tokenizer
model, tokenizer = load_model()

#load data
train_path = 'Data/train.json'
test_path = 'Data/test.json'
dev_path = 'Data/dev.json'

train_df = load_data(file_path=train_path)
dev_df = load_data(file_path=dev_path)
test_df = load_data(file_path=test_path)

# Preprocess
train_dataset = preprocess_data(df=train_df)
dev_dataset = preprocess_data(df=dev_df)
test_dataset = preprocess_data(df=test_df)


# Fine_tune the model
trainer = trainer_run(train_dataset=train_dataset, dev_dataset=dev_dataset).train()

# Evaluate the model
eval_results = trainer.evaluate()

# Print the evaluation results
print("Evaluation Results:", eval_results)
# Save the losses to CSV files
loss_logger.save_losses(training_args.output_dir)

# save model
model.save_pretrained("result")

