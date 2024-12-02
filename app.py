import os
import torch
import time
import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM
from tld.diffusion import DiffusionTransformer
from tld.configs import LTDConfig, DenoiserConfig, DenoiserLoad
import numpy as np
from PIL import Image

# Image Generation Model Setup
denoiser_cfg = DenoiserConfig(
    image_size=32, 
    noise_embed_dims=256, 
    patch_size=2, 
    embed_dim=768, 
    dropout=0, 
    n_layers=12, 
    text_emb_size=768
)

denoiser_load = DenoiserLoad(**{
    'dtype': torch.float32, 
    'file_url': 'https://huggingface.co/apapiu/small_ldt/resolve/main/state_dict_378000.pth', 
    'local_filename': 'state_dict_378000.pth'
})

cfg = LTDConfig(denoiser_cfg=denoiser_cfg, denoiser_load=denoiser_load)
diffusion_transformer = DiffusionTransformer(cfg)

# Set PyTorch to use all available CPU cores
num_cores = os.cpu_count()
torch.set_num_threads(num_cores)
print(f"Using {num_cores} CPU cores.")

# Text Model Setup
model_name = 'mllmTeam/PhoneLM-1.5B-Instruct'
model = AutoModelForCausalLM.from_pretrained(model_name, device_map='cpu', trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_name)

def generate_text_response(question):
    start_time = time.time()
    prompt = [{"role": "user", "content": question}]
    input_text = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
    inp = tokenizer(input_text, return_tensors="pt")
    inp = {k: v.to('cpu') for k, v in inp.items()}
    out = model.generate(**inp, max_length=256, do_sample=True, temperature=0.7, top_p=0.7)
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    text = text.split("\n")[-1]
    end_time = time.time()
    elapsed_time = end_time - start_time
    return text

def generate_image(prompt, class_guidance=6, num_imgs=1, seed=11):
    start_time = time.time()
    try:
        # Generate the image
        out = diffusion_transformer.generate_image_from_text(
            prompt=prompt, 
            class_guidance=class_guidance, 
            num_imgs=num_imgs, 
            seed=seed
        )
        
        # Convert to PIL Image if it's not already
        if isinstance(out, torch.Tensor):
            out = out.squeeze().permute(1, 2, 0).numpy()
        
        # Ensure the image is in the right format for Gradio
        if isinstance(out, np.ndarray):
            # Normalize pixel values to 0-255 range
            out = ((out - out.min()) * (1/(out.max() - out.min()) * 255)).astype('uint8')
            out = Image.fromarray(out)
        
        end_time = time.time()
        print(f"Image generation time: {end_time - start_time:.2f} seconds")
        return out
    except Exception as e:
        print(f"Image generation error: {e}")
        return None

def chat_with_ai(message, history):
    max_history_length = 1  # Adjust as needed
    history = history[-max_history_length:]
    if message.startswith('@imagine'):
        # Extract prompt after '@imagine'
        image_prompt = message.split('@imagine', 1)[1].strip()
        image = generate_image(image_prompt)
        
        if image:
            return "", history, image
        else:
            return "", history + [[message, "Failed to generate image."]], None
    else:
        response = generate_text_response(message)
        return response, history + [[message, response]], None



# Create Gradio interface
with gr.Blocks(title="BlazeChat Image Generator") as demo:
    #################
    gr.Markdown("# ⚡Fast CPU-Powered Chat & Image Generation")
    gr.Markdown("Generate text and images using advanced AI models on CPU. Use `@imagine [prompt]` to create images or chat naturally.")
    gr.Markdown("https://github.com/SanshruthR/CPU_BlazeChat")
    ####################
    chatbot = gr.Chatbot()
    msg = gr.Textbox(label="Enter your message")
    ####submit button
    submit_button = gr.Button("Submit")
    ##########
    clear = gr.Button("Clear")
    img_output = gr.Image(label="Generated Image")

    msg.submit(chat_with_ai, [msg, chatbot], [msg, chatbot, img_output])

    ####################binding with submit
    submit_button.click(chat_with_ai, [msg, chatbot], [msg, chatbot, img_output])



    ###################
    clear.click(lambda: None, None, chatbot, queue=False)

# Launch the demo
demo.launch(debug=True,ssr_mode=False)