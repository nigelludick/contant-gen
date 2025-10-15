# contant-gen
import gradio as gr
from diffusers import StableDiffusionPipeline
import torch
import time
import os
from collections import deque

# ✅ Load lightweight Stable Diffusion model
pipe = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    torch_dtype=torch.float32
)

# ✅ Optimize for CPU (best for free plan)
pipe.enable_attention_slicing()
pipe.enable_sequential_cpu_offload()
pipe = pipe.to("cpu")

# 🎨 Style presets
STYLE_PRESETS = {
    "Realistic": "ultra realistic, detailed lighting, 8k, lifelike",
    "Cartoon": "cartoon style, colorful, outlined, 2D art",
    "Anime": "anime style, clean lines, vibrant colors, studio ghibli style",
    "Digital Art": "digital painting, fantasy, concept art, detailed brush strokes",
    "Sketch": "pencil sketch, black and white, hand-drawn"
}

# 🧠 Keep last 5 images in memory for the gallery
gallery_history = deque(maxlen=5)

# 🧩 Generate image with style, progress, and gallery
def generate_image(prompt, style, progress=gr.Progress(track_tqdm=True)):
    if not prompt:
        return None, "Please enter a description.", list(gallery_history)
    
    styled_prompt = f"{prompt}, {STYLE_PRESETS[style]}"

    # Simulate progress (optional visual)
    for i in range(3):
        time.sleep(0.5)
        progress(i / 3, desc=f"Generating in {style} style...")

    # Generate the image
    image = pipe(styled_prompt).images[0]

    # Save for download
    file_path = "generated_image.png"
    image.save(file_path)

    # Add to gallery history
    gallery_history.append(image)

    return image, file_path, list(gallery_history)

# 🖼️ Gradio interface
demo = gr.Interface(
    fn=generate_image,
    inputs=[
        gr.Textbox(label="Describe your image", placeholder="e.g. A futuristic robot in the desert"),
        gr.Dropdown(choices=list(STYLE_PRESETS.keys()), label="Choose a style", value="Realistic")
    ],
    outputs=[
        gr.Image(label="Generated Image"),
        gr.File(label="Download Image"),
        gr.Gallery(label="Your Recent Creations", columns=5, height="auto")
    ],
    title="AI Image Generator 🎨",
    description="Type a prompt and choose a style — Realistic, Cartoon, Anime, and more!",
    theme="soft"
)

# 🚀 Launch app
if __name__ == "__main__":
    demo.launch()
