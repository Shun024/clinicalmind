"""
NER model trainer: BERT baseline vs BioMedBERT fine-tuning.
Demonstrates why domain-adapted models outperform general BERT
on biomedical text — the core argument for clinical NLP.
"""

import json
import mlflow
import numpy as np
from pathlib import Path
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
)
from seqeval.metrics import (
    f1_score,
    precision_score,
    recall_score,
    classification_report,
)


def load_label_info(path: str = "data/processed/label_info.json") -> dict:
    with open(path) as f:
        return json.load(f)


def compute_metrics(eval_pred, label_list: list):
    """Compute seqeval metrics for NER evaluation."""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label)
         if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label)
         if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    return {
        "precision": precision_score(true_labels, true_predictions),
        "recall": recall_score(true_labels, true_predictions),
        "f1": f1_score(true_labels, true_predictions),
    }


def train_ner_model(
    model_name: str,
    run_name: str,
    num_epochs: int = 5,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
) -> dict:
    """Train a NER model and return evaluation metrics."""
    mlflow.set_experiment("clinicalmind-ner")

    print(f"\n{'='*50}")
    print(f"Training: {run_name}")
    print(f"Model: {model_name}")
    print(f"{'='*50}")

    # Load data and labels
    dataset = load_from_disk("data/processed/ner_dataset")
    label_info = load_label_info()
    label_list = label_info["label_list"]
    label2id = label_info["label2id"]
    id2label = {int(k): v for k, v in label_info["id2label"].items()}

    # Tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=len(label_list),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )

    data_collator = DataCollatorForTokenClassification(tokenizer)

    def metrics_fn(eval_pred):
        return compute_metrics(eval_pred, label_list)

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            "model": model_name,
            "epochs": num_epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "train_size": len(dataset["train"]),
        })

        training_args = TrainingArguments(
            output_dir=f"data/processed/{run_name}",
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=0.01,
            eval_strategy="epoch",
            save_strategy="no",
            load_best_model_at_end=False,
            report_to="none",
            logging_steps=50,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["validation"],
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=metrics_fn,
        )

        print("Training...")
        trainer.train()

        print("Evaluating on test set...")
        results = trainer.evaluate(dataset["test"])

        metrics = {
            "test_f1": round(results.get("eval_f1", 0), 4),
            "test_precision": round(results.get("eval_precision", 0), 4),
            "test_recall": round(results.get("eval_recall", 0), 4),
            "test_loss": round(results.get("eval_loss", 0), 4),
        }

        mlflow.log_metrics(metrics)

        print(f"\nResults for {run_name}:")
        print(f"  F1:        {metrics['test_f1']}")
        print(f"  Precision: {metrics['test_precision']}")
        print(f"  Recall:    {metrics['test_recall']}")

        # Save model for inference
        model_path = f"data/processed/{run_name}_model"
        trainer.save_model(model_path)
        tokenizer.save_pretrained(model_path)
        print(f"  Model saved to {model_path}")

    return metrics, trainer, tokenizer


def run_comparison():
    """Train BERT baseline and BioMedBERT, compare results."""
    results = {}

    # Model 1: General BERT baseline
    print("\n[1/2] Training BERT baseline (general domain)...")
    bert_metrics, _, _ = train_ner_model(
        model_name="bert-base-uncased",
        run_name="bert_baseline",
        num_epochs=3,
        batch_size=16,
        learning_rate=2e-5,
    )
    results["bert"] = bert_metrics

    # Model 2: BioMedBERT (domain-adapted)
    print("\n[2/2] Rebuilding dataset for BioBERT tokenizer...")
    prepare_dataset_for_model("dmis-lab/biobert-base-cased-v1.2")

    print("Training BioBERT...")
    biobert_metrics, trainer, tokenizer = train_ner_model(
        model_name="dmis-lab/biobert-base-cased-v1.2",
        run_name="biomedbert",
        num_epochs=3,
        batch_size=16,
        learning_rate=2e-5,
    )
    results["biomedbert"] = biobert_metrics

    # Save comparison
    with open("data/processed/model_comparison.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 50)
    print("Model Comparison — Biomedical NER")
    print("=" * 50)
    print(f"{'Model':<25} {'F1':<10} {'Precision':<12} {'Recall'}")
    print("-" * 55)
    print(
        f"{'BERT (general)':<25} "
        f"{results['bert']['test_f1']:<10} "
        f"{results['bert']['test_precision']:<12} "
        f"{results['bert']['test_recall']}"
    )
    print(
        f"{'BioBERT (biomedical)':<25} "
        f"{results['biomedbert']['test_f1']:<10} "
        f"{results['biomedbert']['test_precision']:<12} "
        f"{results['biomedbert']['test_recall']}"
    )
    delta = results['biomedbert']['test_f1'] - results['bert']['test_f1']
    print(f"\nBioBERT improvement: {delta:+.4f} F1 points")
    print("=" * 50)

    return results

def prepare_dataset_for_model(model_name: str) -> None:
    """Rebuild dataset with model-specific tokenizer."""
    from src.data.loader import build_dataset
    print(f"  Rebuilding dataset for {model_name}...")
    build_dataset(
        model_name=model_name,
        max_samples=3000,
        output_dir="data/processed",
    )


if __name__ == "__main__":
    run_comparison()