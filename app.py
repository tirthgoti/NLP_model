import gradio as gr
from transformers import pipeline
from diffusers import StableDiffusionPipeline
import torch
from PIL import Image
import matplotlib.pyplot as plt
import networkx as nx
from io import BytesIO

# Load Models
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
ner = pipeline("ner", grouped_entities=True)
fill_mask = pipeline("fill-mask", model="bert-base-uncased")

if torch.cuda.is_available():
    image_pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16,
        revision="fp16",
        use_auth_token=True  # OR use os.getenv("HF_TOKEN")
    ).to("cuda")
else:
    image_pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5"
    ).to("cpu")

# Functions
def classify_text(text):
    labels = ["Forest", "Ocean", "Desert", "Urban", "Agriculture"]
    result = classifier(text, candidate_labels=labels)
    return f"Predicted Label: {result['labels'][0]} (Score: {result['scores'][0]:.2f})"

def generate_image(prompt):
    image = image_pipe(prompt).images[0]
    return image

def extract_ner_graph(text):
    entities = ner(text)
    G = nx.Graph()

    for ent in entities:
        G.add_node(ent['word'], label=ent['entity_group'])

    for i in range(len(entities) - 1):
        G.add_edge(entities[i]['word'], entities[i + 1]['word'])

    fig, ax = plt.subplots()
    pos = nx.spring_layout(G)
    labels = nx.get_node_attributes(G, "label")
    node_colors = ["skyblue" if labels[n] == "LOC" else "lightgreen" for n in G.nodes]
    nx.draw(G, pos, with_labels=True, node_color=node_colors, edge_color='gray', node_size=2000, font_size=9, ax=ax)
    nx.draw_networkx_labels(G, pos, labels={n: f"{n}\\n({labels[n]})" for n in labels})
    
    buf = BytesIO()
    plt.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)

def fill_text(text):
    if "[MASK]" not in text.upper():
        return ["Please include at least one [MASK] token."]
    tokenized = text.replace("[mask]", "[MASK]").replace("[MASK]", fill_mask.tokenizer.mask_token)
    results = fill_mask(tokenized)
    return [r["sequence"] for r in results[:5]]

# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("# 🌍 EcoNLP: Environmental AI Assistant")

    with gr.Tab("🧠 Sentence Classification"):
        txt_input = gr.Textbox(label="Enter environmental sentence")
        txt_output = gr.Textbox(label="Classification Result")
        btn = gr.Button("Classify")
        btn.click(classify_text, txt_input, txt_output)

    with gr.Tab("🎨 Text-to-Image"):
        img_input = gr.Textbox(label="Describe an image")
        img_output = gr.Image(type="pil")
        btn_img = gr.Button("Generate Image")
        btn_img.click(generate_image, img_input, img_output)

    with gr.Tab("🗺️ NER + Graph"):
        ner_input = gr.Textbox(label="Sentence with named entities")
        ner_output = gr.Image(type="pil")
        ner_btn = gr.Button("Visualize NER")
        ner_btn.click(extract_ner_graph, ner_input, ner_output)

    with gr.Tab("✍️ Fill in the Blank"):
        mask_input = gr.Textbox(label="Sentence with [MASK]")
        mask_output = gr.Textbox(label="Top 5 Completions", lines=6)
        fill_btn = gr.Button("Fill Mask")
        fill_btn.click(fill_text, mask_input, mask_output)

demo.launch()
