import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import gradio as gr
import trimesh
import numpy as np
import tempfile
from threading import Thread
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from trimesh.exchange.gltf import export_glb

# Load the model and tokenizer only once
MODEL_PATH = "Zhengyi/LLaMA-Mesh"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, device_map="auto")

TERMINATORS = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

def apply_gradient_color(mesh_text: str) -> str:
    """
    Apply a gradient color to the mesh vertices based on the Y-axis and save as GLB.
    """
    with tempfile.NamedTemporaryFile(suffix=".obj", delete=False) as temp_file:
        temp_file.write(mesh_text.encode())
        temp_file_path = temp_file.name
    
    mesh = trimesh.load_mesh(temp_file_path, file_type='obj')
    y_values = mesh.vertices[:, 1]
    y_normalized = (y_values - y_values.min()) / (y_values.max() - y_values.min())

    colors = np.zeros((len(mesh.vertices), 4))
    colors[:, 0] = y_normalized
    colors[:, 2] = 1 - y_normalized
    colors[:, 3] = 1.0
    mesh.visual.vertex_colors = colors

    glb_path = temp_file_path.replace(".obj", ".glb")
    with open(glb_path, "wb") as f:
        f.write(export_glb(mesh))
    
    return glb_path

def chat_llama3_8b(message: str, history: list, temperature: float, max_new_tokens: int):
    """ Generate a streaming response using the llama3-8b model."""
    conversation = [
        {"role": "user", "content": user} if i % 2 == 0 else {"role": "assistant", "content": assistant}
        for i, (user, assistant) in enumerate(history)
    ]
    conversation.append({"role": "user", "content": message})

    input_ids = tokenizer.apply_chat_template(conversation, return_tensors="pt").to(model.device)
    streamer = TextIteratorStreamer(tokenizer, timeout=10.0, skip_prompt=True, skip_special_tokens=True)

    generate_kwargs = {
        "input_ids": input_ids,
        "streamer": streamer,
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0,
        "temperature": temperature,
        "eos_token_id": TERMINATORS,
    }
    
    Thread(target=model.generate, kwargs=generate_kwargs).start()
    outputs = []
    for text in streamer:
        outputs.append(text)
        yield "".join(outputs)

with gr.Blocks(fill_height=True) as demo:
    gr.Markdown("# LLaMA-Mesh: Generate and Visualize 3D Meshes")
    with gr.Row():
        chatbot = gr.Chatbot(height=450, label='Chat with LLaMA-Mesh')
        output_model = gr.Model3D(label="3D Mesh Visualization", interactive=False)
    
    mesh_input = gr.Textbox(label="3D Mesh Input", placeholder="Paste OBJ format here...", lines=5)
    visualize_button = gr.Button("Visualize 3D Mesh")
    
    visualize_button.click(fn=apply_gradient_color, inputs=[mesh_input], outputs=[output_model])
    
    gr.ChatInterface(
        fn=chat_llama3_8b,
        chatbot=chatbot,
        additional_inputs=[
            gr.Slider(minimum=0, maximum=1, step=0.1, value=0.95, label="Temperature"),
            gr.Slider(minimum=128, maximum=8192, step=1, value=4096, label="Max new tokens"),
        ],
        examples=[
            ["Create a 3D model of a table"],
            ["Create a low-poly 3D model of a tree"],
        ],
    )

demo.launch()
