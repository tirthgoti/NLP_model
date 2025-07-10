import streamlit as st
from transformers import pipeline
from diffusers import StableDiffusionPipeline
import torch
import networkx as nx
import matplotlib.pyplot as plt

# Set Streamlit config
st.set_page_config(page_title="EcoNLP AI App", layout="centered")
st.title("🌍 EcoNLP: Environmental AI Assistant")

# Create 4 tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "🧠 Text Classifier",
    "🎨 Text-to-Image",
    "🗺️ NER + Graph",
    "✍️ Text Infilling"
])

# ---------------- TAB 1: Environmental Text Classifier ---------------- #
with tab1:
    st.header("🧠 Environmental Text Classifier")
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    labels = ["Forest", "Ocean", "Desert", "Urban", "Agriculture"]
    user_input = st.text_area("✍️ Enter your sentence:", height=100)

    if st.button("Classify", key="classify"):
        if not user_input.strip():
            st.warning("Please enter a sentence.")
        else:
            result = classifier(user_input, candidate_labels=labels)
            top_label = result['labels'][0]
            st.success(f"**Predicted Category:** {top_label}")
            st.subheader("🔎 Confidence Scores")
            for label, score in zip(result['labels'], result['scores']):
                st.write(f"**{label}:** {score:.3f}")

# ---------------- TAB 2: Text-to-Image Generator ---------------- #
with tab2:
    st.header("🎨 Text-to-Image Generator")
    prompt = st.text_input("📝 Describe a scene to generate:", placeholder="A misty forest with tall pine trees")

    if st.button("Generate Image", key="generate"):

        if not prompt.strip():
            st.warning("Please enter a prompt.")
        else:
            with st.spinner("Generating image..."):

                @st.cache_resource
                def load_image_model():
                    return StableDiffusionPipeline.from_pretrained(
                        "runwayml/stable-diffusion-v1-5",
                        torch_dtype=torch.float16,
                        revision="fp16",
                        use_auth_token=True  # replace with your Hugging Face token if needed
                    ).to("cuda" if torch.cuda.is_available() else "cpu")

                pipe = load_image_model()
                image = pipe(prompt).images[0]
                st.image(image, caption="🖼️ Generated Image", use_column_width=True)

# ---------------- TAB 3: Named Entity Recognition + Graph ---------------- #
with tab3:
    st.header("🗺️ Named Entity Recognition with Graph")
    ner_input = st.text_area("🔍 Enter a sentence for entity detection:")

    if st.button("Extract Entities", key="ner"):
        if not ner_input.strip():
            st.warning("Please enter some text.")
        else:
            with st.spinner("Extracting entities..."):
                ner_pipeline = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")
                entities = ner_pipeline(ner_input)

                if entities:
                    st.success("Entities detected:")
                    for ent in entities:
                        st.write(f"- **{ent['word']}** ({ent['entity_group']})")

                    st.subheader("🕸️ Entity Graph")
                    G = nx.Graph()
                    for ent in entities:
                        G.add_node(ent['word'], label=ent['entity_group'])

                    for i in range(len(entities) - 1):
                        G.add_edge(entities[i]['word'], entities[i + 1]['word'])

                    fig, ax = plt.subplots()
                    pos = nx.spring_layout(G)
                    nx.draw(G, pos, with_labels=True, node_color='skyblue',
                            edge_color='gray', node_size=2000, font_size=10, ax=ax)
                    st.pyplot(fig)
                else:
                    st.info("No entities found.")

# ---------------- TAB 4: Context-Aware Text Infilling ---------------- #
with tab4:
    st.header("✍️ Context-Aware Text Infilling (Masked Language Model)")
    st.markdown("""
    Type a sentence with one or more `[MASK]` tokens (e.g.,  
    **"Forests are [MASK] to biodiversity and help regulate the [MASK]."**)  
    and let the AI complete it.
    """)

    infill_input = st.text_area("📝 Enter masked sentence:", placeholder="Forests are [MASK] to biodiversity.")

    if st.button("Fill Mask", key="infilling"):
        if "[MASK]" not in infill_input.upper():
            st.warning("Please include at least one [MASK] token in your sentence.")
        else:
            with st.spinner("Filling in the blanks..."):

                @st.cache_resource
                def load_mask_filler():
                    return pipeline("fill-mask", model="bert-base-uncased")

                fill_mask = load_mask_filler()
                prepared_input = infill_input.replace("[mask]", "[MASK]").replace("[MASK]", fill_mask.tokenizer.mask_token)

                try:
                    results = fill_mask(prepared_input)
                    if isinstance(results, list):
                        st.subheader("🎯 Top Predictions:")
                        for i, r in enumerate(results[:5]):
                            filled = prepared_input.replace(fill_mask.tokenizer.mask_token, r['token_str'], 1)
                            st.write(f"{i+1}. {filled}  (Score: {r['score']:.3f})")
                    else:
                        st.write("No predictions returned.")
                except Exception as e:
                    st.error(f"Error: {e}")
