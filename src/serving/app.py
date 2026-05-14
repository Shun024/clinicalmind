"""
ClinicalMind — Streamlit Demo
Interactive clinical entity extraction from free text.
"""

import streamlit as st
from src.serving.inference import load_model, extract_entities, EXAMPLE_TEXTS, ENTITY_COLOURS

st.set_page_config(
    page_title="ClinicalMind",
    page_icon="🏥",
    layout="wide",
)

st.title("🏥 ClinicalMind")
st.caption(
    "Biomedical NLP · Clinical Entity Extraction · "
    "Fine-tuned BioBERT on 2,400 PubMed abstracts"
)
st.divider()

# Sidebar
with st.sidebar:
    st.header("About")
    st.markdown("""
    **ClinicalMind** extracts clinical entities from biomedical text using a fine-tuned BioBERT model.

    **Entity types detected:**
    """)
    for entity_type, colour in ENTITY_COLOURS.items():
        st.markdown(
            f'<span style="background-color:{colour};color:white;'
            f'padding:2px 8px;border-radius:4px;font-size:0.75rem;">'
            f'{entity_type.replace("_", " ")}</span>',
            unsafe_allow_html=True,
        )

    st.divider()
    st.header("Model Comparison")
    st.markdown("""
    | Model | F1 | Recall |
    |---|---|---|
    | BERT (general) | 0.181 | 0.206 |
    | **BioBERT** | **0.213** | **0.309** |

    BioBERT improves recall by **+10.3pp** — finds significantly more clinical entities than general BERT.
    """)

    st.divider()
    st.header("Example Texts")
    for i, text in enumerate(EXAMPLE_TEXTS):
        if st.button(f"Example {i+1}", use_container_width=True):
            st.session_state.input_text = text


# Load model
@st.cache_resource
def get_model():
    return load_model()


with st.spinner("Loading BioBERT model..."):
    ner = get_model()

# Input
text = st.text_area(
    "Enter clinical text",
    value=st.session_state.get("input_text", EXAMPLE_TEXTS[0]),
    height=150,
    placeholder="Enter clinical notes, discharge summaries, or PubMed abstracts...",
)

if st.button("Extract Entities", type="primary", use_container_width=True):
    if not text.strip():
        st.warning("Please enter some text.")
    else:
        with st.spinner("Extracting clinical entities..."):
            entities = extract_entities(text, ner)

        if not entities:
            st.info("No clinical entities detected in this text.")
        else:
            # Highlighted text
            st.subheader("Annotated Text")
            highlighted = text
            offset = 0
            for e in sorted(entities, key=lambda x: x["start"]):
                start = e["start"] + offset
                end = e["end"] + offset
                tag = (
                    f'<mark style="background-color:{e["colour"]}33;'
                    f'border-bottom:2px solid {e["colour"]};'
                    f'border-radius:3px;padding:1px 2px;">'
                    f'{highlighted[start:end]}'
                    f'<sup style="font-size:0.6rem;color:{e["colour"]};'
                    f'font-weight:bold;">{e["type"].split("_")[0]}</sup>'
                    f'</mark>'
                )
                highlighted = highlighted[:start] + tag + highlighted[end:]
                offset += len(tag) - (end - start)

            st.markdown(
                f'<div style="line-height:2.2;font-size:1rem;'
                f'padding:15px;border-radius:8px;'
                f'background:#1a1a2e;">{highlighted}</div>',
                unsafe_allow_html=True,
            )

            # Entity table
            st.subheader(f"Extracted Entities ({len(entities)} found)")
            import pandas as pd
            entity_df = pd.DataFrame([{
                "Entity": e["text"],
                "Type": e["type"].replace("_", " "),
                "Confidence": f"{e['score']:.3f}",
            } for e in entities])
            st.dataframe(entity_df, use_container_width=True, hide_index=True)

            # Summary stats
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Entities Found", len(entities))
            with col2:
                avg_conf = sum(e["score"] for e in entities) / len(entities)
                st.metric("Avg Confidence", f"{avg_conf:.3f}")
            with col3:
                types = set(e["type"] for e in entities)
                st.metric("Entity Types", len(types))