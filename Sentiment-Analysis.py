from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import precision_recall_fscore_support
import random
from nltk.corpus import wordnet
import optuna
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Load dataset (adjust path to your dataset)
dataset = load_dataset('csv', data_files={'train': 'train_150.txt'})

# Load BERT tokenizer and model
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=3)

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=128)

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Split into training and evaluation sets
tokenized_dataset = tokenized_dataset['train'].train_test_split(test_size=0.2)

# Define metrics for evaluation
def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='weighted')
    return {"accuracy": acc, "f1": f1}

# Set up training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,                  
    per_device_train_batch_size=4,      
    per_device_eval_batch_size=4,
    warmup_steps=500,                    # Number of warmup steps for learning rate scheduler
    weight_decay=0.01,                   # Strength of weight decay
    logging_dir='./logs',
    evaluation_strategy="epoch"
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset['train'],
    eval_dataset=tokenized_dataset['test'],
    compute_metrics=compute_metrics
)

# Train the model
trainer.train()

# Load the model and tokenizer
model = BertForSequenceClassification.from_pretrained("./sentiment_analysis_model")
tokenizer = BertTokenizer.from_pretrained("./sentiment_analysis_model")

# Function to classify sentiment of new text
def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()
    label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
    return label_map[predicted_class]

# Synonym Replacement Function
def synonym_replacement(text, n=1):
    words = text.split()
    new_words = words.copy()
    random_word_list = list(set([word for word in words if wordnet.synsets(word)]))
    random.shuffle(random_word_list)

    num_replaced = 0
    for random_word in random_word_list:
        synonyms = wordnet.synsets(random_word)
        if synonyms:
            synonym = synonyms[0].lemmas()[0].name()
            new_words = [synonym if word == random_word else word for word in new_words]
            num_replaced += 1
        if num_replaced >= n:
            break

    return ' '.join(new_words)

# Augment Dataset
augmented_texts = [synonym_replacement(text) for text in dataset['train']['text']]
augmented_labels = dataset['train']['label']

def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

# Update trainer with new metrics
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset['train'],
    eval_dataset=tokenized_dataset['test'],
    compute_metrics=compute_metrics
)

def objective(trial):
    # Suggest values for hyperparameters
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 5e-5)
    batch_size = trial.suggest_categorical('batch_size', [4, 8, 16])

    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        evaluation_strategy="epoch",
        learning_rate=learning_rate
    )

    # Initialize Trainer with new training_args
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset['train'],
        eval_dataset=tokenized_dataset['test'],
        compute_metrics=compute_metrics
    )

    # Train and evaluate
    trainer.train()
    eval_result = trainer.evaluate()

    return eval_result['eval_accuracy']

# Run the hyperparameter search
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=10)

# Print the best hyperparameters
print("Best hyperparameters:", study.best_params)

# Predict on test dataset
preds_output = trainer.predict(tokenized_dataset['test'])
preds = np.argmax(preds_output.predictions, axis=1)
labels = preds_output.label_ids

# Confusion Matrix
cm = confusion_matrix(labels, preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Negative", "Neutral", "Positive"])
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()

# Save the model and tokenizer
model.save_pretrained("./sentiment_analysis_model")
tokenizer.save_pretrained("./sentiment_analysis_model")

