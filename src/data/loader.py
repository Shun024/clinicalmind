"""
Data loader and preprocessor for ClinicalMind.
Source: knowledgator/biomed_NER (4,840 PubMed abstracts, 24 entity types)
Converts character-span annotations to token-level IOB2 format
for sequence labelling with transformers.
"""

import json
import pandas as pd
from pathlib import Path
from datasets import load_dataset, Dataset, DatasetDict
from transformers import AutoTokenizer
from collections import Counter


# Focus on clinically relevant entity types
CLINICAL_LABELS = [
    "DISEASE_DISORDER",
    "CLINICAL_DRUG",
    "CHEMICALS",
    "DIAGNOSTIC_PROCEDURE",
    "THERAPEUTIC_PROCEDURE",
    "BODY_PART_ORGAN_OR_ORGAN_COMPONENT",
    "SIGN_SYMPTOM",
]

# IOB2 label list
LABEL_LIST = ["O"] + [
    f"{prefix}-{label}"
    for label in CLINICAL_LABELS
    for prefix in ["B", "I"]
]
LABEL2ID = {label: i for i, label in enumerate(LABEL_LIST)}
ID2LABEL = {i: label for label, i in LABEL2ID.items()}


def spans_to_iob(text: str, entities: list, tokenizer, label_list: list) -> dict:
    """
    Convert character-level spans to token-level IOB2 tags.
    Handles subword tokenization alignment.
    """
    # Tokenize with offset mapping
    encoding = tokenizer(
        text,
        return_offsets_mapping=True,
        truncation=True,
        max_length=256,
        padding="max_length",
    )

    offset_mapping = encoding["offset_mapping"]
    labels = [LABEL2ID["O"]] * len(offset_mapping)

    # Filter to clinical entities only
    clinical_entities = [
        e for e in entities
        if e["class"].replace(" ", "_") in CLINICAL_LABELS
        or e["class"] in CLINICAL_LABELS
    ]

    for entity in clinical_entities:
        start_char = entity["start"]
        end_char = entity["end"]
        entity_class = entity["class"].replace(" ", "_")

        if entity_class not in CLINICAL_LABELS:
            continue

        b_label = LABEL2ID.get(f"B-{entity_class}", LABEL2ID["O"])
        i_label = LABEL2ID.get(f"I-{entity_class}", LABEL2ID["O"])

        first = True
        for idx, (token_start, token_end) in enumerate(offset_mapping):
            if token_start == token_end == 0:
                continue  # special tokens
            if token_start >= start_char and token_end <= end_char:
                labels[idx] = b_label if first else i_label
                first = False

    return {
        "input_ids": encoding["input_ids"],
        "attention_mask": encoding["attention_mask"],
        "labels": labels,
    }


def build_dataset(
    model_name: str = "bert-base-uncased",
    max_samples: int = 3000,
    output_dir: str = "data/processed",
) -> DatasetDict:
    """
    Load raw data, convert to IOB format, split into train/val/test.
    """
    print("=" * 50)
    print("ClinicalMind — Dataset Construction")
    print("=" * 50)

    print("\nLoading knowledgator/biomed_NER...")
    raw = load_dataset("knowledgator/biomed_NER")
    print(f"  Raw documents: {len(raw['train'])}")

    print(f"  Using up to {max_samples} documents")
    print(f"  Clinical entity types: {CLINICAL_LABELS}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print(f"  Tokenizer: {model_name}")

    # Convert to IOB format
    print("\nConverting spans to IOB2 tokens...")
    processed = []
    entity_counts = Counter()

    for i, example in enumerate(raw["train"]):
        if i >= max_samples:
            break
        if i % 500 == 0:
            print(f"  Processing {i}/{min(max_samples, len(raw['train']))}...")

        result = spans_to_iob(
            example["text"],
            example["entities"],
            tokenizer,
            LABEL_LIST,
        )
        processed.append(result)

        for entity in example["entities"]:
            entity_counts[entity["class"]] += 1

    print(f"\nProcessed {len(processed)} documents")
    print("\nEntity type distribution (all types):")
    for entity_type, count in entity_counts.most_common(10):
        print(f"  {entity_type}: {count:,}")

    # Label distribution in processed data
    all_labels = [label for doc in processed for label in doc["labels"]]
    label_counts = Counter(all_labels)
    print("\nIOB label distribution (non-O):")
    for label_id, count in sorted(label_counts.items()):
        if label_id != LABEL2ID["O"] and count > 0:
            print(f"  {ID2LABEL[label_id]}: {count:,}")

    # Train/val/test split
    n = len(processed)
    train_end = int(n * 0.8)
    val_end = int(n * 0.9)

    dataset = DatasetDict({
        "train": Dataset.from_list(processed[:train_end]),
        "validation": Dataset.from_list(processed[train_end:val_end]),
        "test": Dataset.from_list(processed[val_end:]),
    })

    print(f"\nSplit sizes:")
    print(f"  Train: {len(dataset['train'])}")
    print(f"  Validation: {len(dataset['validation'])}")
    print(f"  Test: {len(dataset['test'])}")

    # Save
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(f"{output_dir}/ner_dataset")
    print(f"\nDataset saved to {output_dir}/ner_dataset")

    # Save label info
    with open(f"{output_dir}/label_info.json", "w") as f:
        json.dump({
            "label_list": LABEL_LIST,
            "label2id": LABEL2ID,
            "id2label": ID2LABEL,
            "num_labels": len(LABEL_LIST),
        }, f, indent=2)

    return dataset


if __name__ == "__main__":
    dataset = build_dataset(
        model_name="bert-base-uncased",
        max_samples=3000,
    )
    print("\nDataset construction complete!")
    print(f"Labels: {LABEL_LIST}")