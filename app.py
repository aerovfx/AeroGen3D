import os
import gradio as gr
import numpy as np
import trimesh
import tempfile
import threading
from transformers import AutoModelForCausalLM, AutoTokenizer

# Tắt cảnh báo parallelism để tránh lỗi khi chạy tokenizer
os.environ["TOKENIZERS_PARALLELISM"] = "false"
HF_TOKEN = os.getenv("HF_TOKEN", None)

# Load model and tokenizer
model_path = "Zhengyi/LLaMA-Mesh"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype="auto")

def chat_llama3_8b(message: str, history: list, temperature: float, max_new_tokens: int) -> str:
    """ Sinh phản hồi từ mô hình ngôn ngữ LLaMA. """
    inputs = tokenizer.encode(message, return_tensors="pt").to(model.device)
    output = model.generate(
        inputs, max_new_tokens=max_new_tokens, temperature=temperature, pad_token_id=tokenizer.eos_token_id
    )
    return tokenizer.decode(output[0], skip_special_tokens=True)

def apply_gradient_color(mesh_text: str) -> str:
    """ Thêm màu gradient vào lưới 3D theo trục Y. """
    try:
        with tempfile.NamedTemporaryFile(suffix=".obj", delete=False) as temp_obj:
            temp_obj.write(mesh_text.encode("utf-8"))
            temp_obj.close()
            mesh = trimesh.load_mesh(temp_obj.name)
        
        # Kiểm tra mesh hợp lệ
        if mesh.vertices.shape[0] == 0:
            return "Mesh không hợp lệ. Kiểm tra lại dữ liệu đầu vào."
        
        y_min, y_max = mesh.vertices[:, 1].min(), mesh.vertices[:, 1].max()
        y_normalized = (mesh.vertices[:, 1] - y_min) / (y_max - y_min + 1e-6)
        colors = np.zeros((mesh.vertices.shape[0], 4))
        colors[:, 0] = np.clip(y_normalized, 0, 1)  # Kênh đỏ
        colors[:, 2] = np.clip(1 - y_normalized, 0, 1)  # Kênh xanh
        colors[:, 3] = 1  # Alpha = 1
        mesh.visual.vertex_colors = (colors * 255).astype(np.uint8)
        
        # Lưu mesh đã xử lý thành file GLB
        with tempfile.NamedTemporaryFile(suffix=".glb", delete=False) as temp_glb:
            mesh.export(temp_glb.name)
            return temp_glb.name
    except Exception as e:
        return f"Lỗi khi xử lý mesh: {str(e)}"

def visualize_mesh(mesh_text: str):
    """ Hiển thị lưới 3D từ văn bản OBJ. """
    try:
        with tempfile.NamedTemporaryFile(suffix=".obj", delete=False) as temp_obj:
            temp_obj.write(mesh_text.encode("utf-8"))
            temp_obj.close()
            return temp_obj.name
    except Exception as e:
        return f"Lỗi khi hiển thị mesh: {str(e)}"

# Gradio UI
def gradio_interface():
    with gr.Blocks(css=".container { width: 100%; height: 100vh; }") as demo:
        gr.Markdown("## 🎨 LLaMA 3D Mesh Generator 🎨")
        chatbot = gr.Chatbot(height=450, placeholder="Nhập mô tả của bạn...")
        
        with gr.Row():
            input_box = gr.Textbox(placeholder="Nhập dữ liệu OBJ hoặc văn bản mô tả 3D", lines=6)
            btn_generate = gr.Button("Tạo 3D Mesh")
        
        with gr.Row():
            temperature_slider = gr.Slider(0, 1, 0.7, label="Temperature")
            token_slider = gr.Slider(128, 4096, 2048, label="Max Tokens")
        
        output_mesh = gr.Model3D()
        
        btn_generate.click(fn=apply_gradient_color, inputs=[input_box], outputs=[output_mesh])
    
    return demo

demo = gradio_interface()
demo.launch()
