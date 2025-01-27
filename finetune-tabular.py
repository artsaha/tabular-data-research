from transformers import TextDataset, DataCollatorForLanguageModeling
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import Trainer, TrainingArguments

def load_dataset(file_path, tokenizer, block_size = 128):
    dataset = TextDataset(tokenizer=tokenizer, block_size=block_size, file_path=file_path)
    return dataset

def load_data_collator(tokenizer, mlm=False):
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=mlm)
    return data_collator

def train(train_file_path, model_name, output_dir, overwrite_output_dir, per_device_train_batch_size,
          num_train_epochs, save_steps):
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    train_dataset = load_dataset(train_file_path, tokenizer)
    data_collator = load_data_collator(tokenizer)

    tokenizer.save_pretrained(output_dir)
    model = GPT2LMHeadModel.from_pretrained(output_dir)
    model.save_pretrained(output_dir)

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=overwrite_output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        num_train_epochs=num_train_epochs,
    )

    trainer = Trainer(
        model = model,
        args = training_args,
        data_collator = data_collator,
        train_dataset = train_dataset,
    )

    trainer.train()
    trainer.save_model()



train_file_path = "tables.txt"
model_name = 'gpt2'
output_dir = 'models/'
overwrite_output_dir = False
per_device_train_batch_size = 8
num_train_epochs = 2.0
save_steps = 3


# It takes about 30 minutes to train in colab.
train(
    train_file_path=train_file_path,
    model_name=model_name,
    output_dir=output_dir,
    overwrite_output_dir=overwrite_output_dir,
    per_device_train_batch_size=per_device_train_batch_size,
    num_train_epochs=num_train_epochs,
    save_steps=save_steps
)








