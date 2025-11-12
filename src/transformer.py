from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import numpy as np
import evaluate

MODEL = 'distilbert-base-uncased'
NUM_LABELS = 3

raw = load_dataset('csv', data_files={'train':'data/processed/train.csv','validation':'data/processed/test.csv'})

tokenizer = AutoTokenizer.from_pretrained(MODEL)

def preprocess(examples):
    return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=256)

tokenized = raw.map(preprocess, batched=True)

model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=NUM_LABELS)

metric = evaluate.load('f1')

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {'f1_macro': metric.compute(predictions=preds, references=labels, average='macro')['f1']}

training_args = TrainingArguments(
    output_dir='checkpoints',
    evaluation_strategy='epoch',
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    num_train_epochs=3,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model='f1_macro',
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized['train'],
    eval_dataset=tokenized['validation'],
    compute_metrics=compute_metrics
)

trainer.train()