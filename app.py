import os
# Disable parallelism for tokenizers to prevent possible issues with multi-threading
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import gradio as gr         # For building the interactive UI
import trimesh             # For loading and processing 3D mesh data
import numpy as np         # For numerical operations, especially with arrays
import tempfile            # For creating temporary files to work with mesh data
from threading import Thread  # To run model generation in a separate thread
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer  # For using the language model with streaming
from trimesh.exchange.gltf import export_glb  # For exporting the mesh to GLB format

# Define the model path and load both the tokenizer and model once to avoid reloading for every request.
MODEL_PATH = "Zhengyi/LLaMA-Mesh"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, device_map="auto")

# Define tokens that indicate the end of the generation (end-of-sequence tokens)
TERMINATORS = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

def apply_gradient_color(mesh_text: str) -> str:
    """
    Apply a gradient color to the mesh vertices based on the Y-axis and save the result as a GLB file.

    Parameters:
        mesh_text (str): A string containing the OBJ format mesh data.
    
    Returns:
        str: The file path to the generated GLB file.
    """
    # Create a temporary file to save the OBJ data, ensuring the file is not automatically deleted
    with tempfile.NamedTemporaryFile(suffix=".obj", delete=False) as temp_file:
        # Write the mesh text into the temporary file
        temp_file.write(mesh_text.encode())
        temp_file_path = temp_file.name  # Save the file path for later use
    
    # Load the mesh from the temporary OBJ file
    mesh = trimesh.load_mesh(temp_file_path, file_type='obj')
    
    # Extract the Y-axis values from the vertices (second column in the vertex array)
    y_values = mesh.vertices[:, 1]
    # Normalize the Y values between 0 and 1 to use as gradient parameters
    y_normalized = (y_values - y_values.min()) / (y_values.max() - y_values.min())

    # Create an array for colors with RGBA channels (initially all zeros)
    colors = np.zeros((len(mesh.vertices), 4))
    # Assign the normalized Y value to the red channel
    colors[:, 0] = y_normalized
    # Use the inverse of the normalized Y value for the blue channel
    colors[:, 2] = 1 - y_normalized
    # Set the alpha channel to 1 (fully opaque)
    colors[:, 3] = 1.0
    # Apply the generated colors to the mesh vertices
    mesh.visual.vertex_colors = colors

    # Define the output file path by replacing the .obj extension with .glb
    glb_path = temp_file_path.replace(".obj", ".glb")
    # Open the new file in binary write mode and export the mesh as GLB
    with open(glb_path, "wb") as f:
        f.write(export_glb(mesh))
    
    # Return the path to the GLB file for use in the UI
    return glb_path

def chat_llama3_8b(message: str, history: list, temperature: float, max_new_tokens: int):
    """
    Generate a streaming chat response using the LLaMA-Mesh model.

    Parameters:
        message (str): The latest message from the user.
        history (list): The conversation history, structured as a list of (user, assistant) pairs.
        temperature (float): Controls randomness in the model output.
        max_new_tokens (int): Maximum number of tokens to generate in the response.
    
    Yields:
        str: The incremental text output from the model.
    """
    # Convert the conversation history into a list of dictionaries with roles ("user" or "assistant")
    conversation = [
        {"role": "user", "content": user} if i % 2 == 0 else {"role": "assistant", "content": assistant}
        for i, (user, assistant) in enumerate(history)
    ]
    # Append the new user message to the conversation
    conversation.append({"role": "user", "content": message})

    # Format the conversation into input ids using the tokenizer's chat template
    input_ids = tokenizer.apply_chat_template(conversation, return_tensors="pt").to(model.device)
    # Set up a streamer to get the generated tokens in a streaming fashion
    streamer = TextIteratorStreamer(tokenizer, timeout=10.0, skip_prompt=True, skip_special_tokens=True)

    # Set up arguments for the model generation
    generate_kwargs = {
        "input_ids": input_ids,
        "streamer": streamer,
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0,  # Enable sampling if temperature is greater than zero
        "temperature": temperature,
        "eos_token_id": TERMINATORS,   # Use the terminator tokens to end generation
    }
    
    # Start the generation process in a separate thread to allow streaming
    Thread(target=model.generate, kwargs=generate_kwargs).start()
    outputs = []
    # Iterate over the generated text as it streams in
    for text in streamer:
        outputs.append(text)
        # Yield the current concatenated output to update the UI
        yield "".join(outputs)

# Build the Gradio user interface using Blocks for flexible layout
with gr.Blocks(fill_height=True) as demo:
    # Title and header for the UI
    gr.Markdown("# LLaMA-Mesh: Generate and Visualize 3D Meshes")
    
    # Create a row to display the chat interface and 3D model viewer side by side
    with gr.Row():
        # Chatbot component for interacting with the language model
        chatbot = gr.Chatbot(height=450, label='Chat with LLaMA-Mesh')
        # 3D model visualization component for showing the generated mesh
        output_model = gr.Model3D(label="3D Mesh Visualization", interactive=False)
    
    # Textbox for the user to input a mesh in OBJ format
    mesh_input = gr.Textbox(label="3D Mesh Input", placeholder="Paste OBJ format here...", lines=5)
    # Button to trigger visualization of the 3D mesh with the applied gradient color
    visualize_button = gr.Button("Visualize 3D Mesh")
    
    # When the visualize button is clicked, call the apply_gradient_color function
    # The input is the mesh text and the output is shown in the Model3D component
    visualize_button.click(fn=apply_gradient_color, inputs=[mesh_input], outputs=[output_model])
    
    # Set up the chat interface for interacting with the LLaMA-Mesh model
    gr.ChatInterface(
        fn=chat_llama3_8b,
        chatbot=chatbot,
        additional_inputs=[
            # Slider to adjust the temperature (controls randomness)
            gr.Slider(minimum=0, maximum=1, step=0.1, value=0.95, label="Temperature"),
            # Slider to set the maximum number of new tokens to generate
            gr.Slider(minimum=128, maximum=8192, step=1, value=4096, label="Max new tokens"),
        ],
        # Provide example prompts for the chat interface
        examples=[
            ["Create a 3D model of a table"],
            ["Create a low-poly 3D model of a tree"],
        ],
    )

# Launch the Gradio interface to start the web app
demo.launch()
