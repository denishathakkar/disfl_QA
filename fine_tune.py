from peft import LoraConfig, get_peft_model
from transformers import Trainer, TrainingArguments
from transformers import TrainerCallback
import pandas as pd

from load_original_model import load_model
model,tokenizer = load_model()


class LossLoggerCallback(TrainerCallback):
    def __init__(self):
        self.train_losses = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            if 'loss' in logs:
                self.train_losses.append(logs['loss'])

    def save_losses(self, output_dir):
        train_losses_df = pd.DataFrame(self.train_losses, columns=['train_loss'])
        train_losses_df.to_csv(f'{output_dir}/train_losses.csv', index=False)

loss_logger = LossLoggerCallback()

# Configure LoRA
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules=["q", "k", "v", "o"]
)
model_lora_config = get_peft_model(model, lora_config)

# Define training arguments
training_args = TrainingArguments(
    output_dir='/t5-large-lora',
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=2,
    logging_dir='/logs',
    logging_steps=100,
    evaluation_strategy="steps",
    save_strategy="epoch",
    learning_rate=1e-5,
    weight_decay=0.01,
    remove_unused_columns=False
)

# Initialize Trainer
def trainer_run(train_dataset, dev_dataset):
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        tokenizer=tokenizer,
        callbacks=[loss_logger]
    )
    return trainer