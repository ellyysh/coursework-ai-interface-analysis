
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
import torch

model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct",
    torch_dtype=torch.float16,
    device_map="auto"
)

lora_config = LoraConfig(
    r=64,
    lora_alpha=128,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
print(f"Trainable params: {model.num_parameters(only_trainable=True):,}")

dataset = load_dataset("json", data_files="data/dataset.json")
dataset = dataset["train"].train_test_split(test_size=0.1)
train_dataset = dataset["train"]
eval_dataset = dataset["test"]

processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

def preprocess_function(examples):
    texts = []
    for instruction, output in zip(examples["instruction"], examples["output"]):
        text = f"Вопрос: {instruction}\nОтвет: {output}"
        texts.append(text)
    inputs = processor(text=texts, images=examples["image"], return_tensors="pt", padding=True, truncation=True)
    inputs["labels"] = inputs["input_ids"].clone()
    return inputs

tokenized_dataset = train_dataset.map(preprocess_function, batched=True)

training_args = TrainingArguments(
    output_dir="./lora_checkpoint",
    num_train_epochs=3,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    warmup_steps=10,
    logging_steps=10,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    fp16=True,
    report_to="tensorboard",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    eval_dataset=eval_dataset.map(preprocess_function, batched=True),
)

trainer.train()

model.save_pretrained("./lora_checkpoint")
processor.save_pretrained("./lora_checkpoint")
print("LoRA adapters saved to ./lora_checkpoint")