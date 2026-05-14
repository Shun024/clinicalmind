"""
ClinicalMind inference pipeline.
Extracts clinical entities from free text using fine-tuned BioBERT.
"""

from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
from pathlib import Path


ENTITY_COLOURS = {
    "CHEMICALS": "#4488FF",
    "CLINICAL_DRUG": "#00CC88",
    "DISEASE_DISORDER": "#FF4444",
    "DIAGNOSTIC_PROCEDURE": "#FFB300",
    "THERAPEUTIC_PROCEDURE": "#AA44FF",
    "BODY_PART_ORGAN_OR_ORGAN_COMPONENT": "#FF8844",
    "SIGN_SYMPTOM": "#FF44AA",
}

EXAMPLE_TEXTS = [
    "The patient was administered metformin 500mg twice daily for type 2 diabetes mellitus. Blood glucose monitoring showed improvement after 4 weeks.",
    "MRI of the brain revealed a small lesion in the left temporal lobe consistent with glioblastoma multiforme. The patient underwent craniotomy and temozolomide chemotherapy.",
    "Chest X-ray demonstrated bilateral pneumonia. The patient was started on amoxicillin and azithromycin. Oxygen saturation improved from 88% to 96%.",
    "The patient presented with severe chest pain radiating to the left arm. ECG showed ST elevation consistent with acute myocardial infarction. Aspirin and heparin were administered immediately.",
]


def load_model(model_path: str = "data/processed/biomedbert_model"):
    """Load fine-tuned BioBERT NER model."""
    if not Path(model_path).exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. "
            "Run src/models/ner_trainer.py first."
        )

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForTokenClassification.from_pretrained(model_path)

    ner_pipeline = pipeline(
        "ner",
        model=model,
        tokenizer=tokenizer,
        aggregation_strategy="simple",
    )
    return ner_pipeline


def extract_entities(text: str, ner_pipeline) -> list:
    """Extract clinical entities from text."""
    results = ner_pipeline(text)
    entities = []
    for r in results:
        entity_type = r["entity_group"]
        if entity_type == "O":
            continue
        entities.append({
            "text": r["word"],
            "type": entity_type,
            "score": round(r["score"], 3),
            "start": r["start"],
            "end": r["end"],
            "colour": ENTITY_COLOURS.get(entity_type, "#888888"),
        })
    return entities


def run_demo():
    """Run inference on example clinical texts."""
    print("=" * 50)
    print("ClinicalMind — Inference Demo")
    print("=" * 50)

    print("\nLoading BioBERT NER model...")
    ner = load_model()
    print("Model loaded.")

    for i, text in enumerate(EXAMPLE_TEXTS, 1):
        print(f"\n[Example {i}]")
        print(f"Text: {text[:100]}...")
        entities = extract_entities(text, ner)
        if entities:
            print("Entities found:")
            for e in entities:
                print(f"  [{e['type']}] '{e['text']}' (confidence: {e['score']:.3f})")
        else:
            print("  No entities found.")


if __name__ == "__main__":
    run_demo()