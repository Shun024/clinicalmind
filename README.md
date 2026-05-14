# 🏥 ClinicalMind

Biomedical NLP pipeline for clinical entity extraction. Fine-tunes BioBERT on 2,400 PubMed abstracts to identify clinical drugs, chemicals, diagnostic procedures, and therapeutic procedures from free text — with an interactive Streamlit demo.

---

## Results

| Model | F1 | Precision | Recall |
|---|---|---|---|
| BERT (general domain) | 0.181 | 0.161 | 0.206 |
| **BioBERT (biomedical domain)** | **0.213** | **0.163** | **0.309** |
| Delta | **+0.033** | +0.002 | **+0.103** |

**Key finding:** Domain-adapted BioBERT improves recall by +10.3pp — it finds significantly more clinical entities than general BERT. Recall is the critical metric in clinical NLP: missing a drug or adverse event is more dangerous than a false positive.

---

## Architecture

```
knowledgator/biomed_NER (4,840 PubMed abstracts)
        │
        ▼
Span-to-IOB Conversion
├── Character spans → token-level IOB2 tags
├── BioBERT subword tokenization alignment
└── 7 clinical entity types filtered
        │
        ▼
        ├── BERT (general)     F1 0.181  ← baseline
        └── BioBERT (domain)   F1 0.213  ← +18% improvement
                │
                ▼
        Streamlit Demo
        ├── Inline text annotation with colour highlighting
        ├── Entity table with confidence scores
        └── Example clinical texts
```

---

## Entity Types

| Entity | Description | Example |
|---|---|---|
| CLINICAL_DRUG | Pharmaceutical substances | metformin, temozolomide |
| CHEMICALS | Chemical compounds | glucose, acetic acid |
| DISEASE_DISORDER | Medical conditions | glioblastoma, diabetes |
| DIAGNOSTIC_PROCEDURE | Clinical tests | MRI, ECG, blood test |
| THERAPEUTIC_PROCEDURE | Treatments | craniotomy, chemotherapy |
| BODY_PART | Anatomical structures | temporal lobe, brain |
| SIGN_SYMPTOM | Clinical findings | chest pain, ST elevation |

---

## Clinical Relevance

Clinical NLP is one of the NHS's highest-priority AI initiatives:
- **NHS** processes 1.2 billion clinical documents annually — 80% unstructured
- **Pharmacovigilance** requires detecting adverse drug events buried in clinical notes
- **ICD-10 coding** costs the NHS ~£600M/year in manual annotation effort
- **MHRA regulations** require explainable AI for clinical decision support

---

## Stack

BioBERT · HuggingFace Transformers · seqeval · Streamlit · MLflow · pandas

---

## Quickstart3D

```bash
git clone https://github.com/Shun024/clinicalmind.git
cd clinicalmind
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install "transformers==4.44.0"

# Build dataset
PYTHONPATH=. python -m src.data.loader

# Train models (warning: ~3hrs on CPU, use GPU/Colab for speed)
PYTHONPATH=. python -m src.models.ner_trainer

# Launch demo
PYTHONPATH=. streamlit run src/serving/app.py
```

---

## Why Domain Adaptation Matters

General BERT was pre-trained on Wikipedia and BooksCorpus — text that rarely contains terms like "temozolomide", "glioblastoma", or "craniotomy". BioBERT was pre-trained on 18 billion tokens from PubMed abstracts and full-text articles, giving it a vocabulary and semantic understanding tuned to biomedical language. The +10.3pp recall improvement directly demonstrates this advantage.

---

## Author

**Shun Le Yi Mon (Sheryl)** · Data Scientist · NLP & GenAI  
[LinkedIn](https://www.linkedin.com/in/shunleyimon724) · [GitHub](https://github.com/Shun024)