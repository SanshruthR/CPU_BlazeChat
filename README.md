# BlazeChat - Fast CPU-Powered Chat & Image Generation

![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-v1.12-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Gradio](https://img.shields.io/badge/Gradio-3.1.4-blueviolet?style=for-the-badge&logo=gradio&logoColor=white)
[![Deployed on Hugging Face](https://img.shields.io/badge/Deployed%20on-Hugging%20Face-yellow?style=for-the-badge&logo=huggingface&logoColor=white)](https://huggingface.co/spaces/Sanshruth/CPU_BlazeChat)

![image](https://github.com/user-attachments/assets/9f76c7f0-e46b-4f1f-a170-a53e4fe5a76d)


## Overview

**CPU BlazeChat** is a fast and efficient CPU-powered chat and image generation system. Using the `PhoneLM-1.5B-Instruct` model for text generation and a custom diffusion model for image creation, it allows users to interact via text or create images with natural language prompts. The system is designed to be lightweight, operating in near real-time with minimal computational overhead.

## Features

- **Text Generation**: Generates text responses using the `PhoneLM-1.5B-Instruct` model.
- **Image Generation**: Allows users to generate images with the `@imagine [prompt]` command, powered by a lightweight diffusion model.
- **CPU Optimization**: Runs efficiently on CPU using multi-threading to maximize processing power.
- **Gradio Interface**: An intuitive, interactive interface built with Gradio for real-time chat and image generation.
- **Real-time Performance**: Optimized for fast generation with a focus on low latency.

## Model Details

- **Text Generation Model**: `PhoneLM-1.5B-Instruct`  
  - **Performance**: 
    - HellaSwag: 66.9
    - WinoGrande: 63.0
    - PIQA: 77.3
    - SciQ: 88.8
    - BoolQ: 65.5
    - ARC Easy: 69.7
    - ARC Challenge: 39.9
    - **Average**: 67.31
- **Image Generation Model**: Custom diffusion model built using PyTorch with small model size (~100MM parameters) for fast generation.

## How It Works

1. **Text Interaction**: Users input messages in a chat interface. The `PhoneLM-1.5B-Instruct` model generates responses based on the input.
2. **Image Creation**: Users can enter `@imagine [prompt]` to generate an image using the diffusion model.
3. **Efficient CPU Utilization**: The application uses all available CPU cores to maximize processing speed, providing near real-time results.

## Usage

1. Clone the repository and install the dependencies:
    ```bash
    git clone https://github.com/SanshruthR/CPU_BlazeChat.git
    cd CPU_BlazeChat
    pip install -r requirements.txt
    ```

2. Run the script:
    ```bash
    python app.py
    ```

3. Access the Gradio interface to start chatting or generate images:
    - **Text Chat**: Ask questions and receive AI-generated text responses.
    - **Image Generation**: Use `@imagine [prompt]` to generate images based on textual descriptions.

### Deployment

This project is deployed on **Hugging Face Spaces**. You can interact with the app via the following link:

[Live Demo on Hugging Face](https://huggingface.co/spaces/Sanshruth/CPU_BlazeChat)

## License

This project is licensed under the MIT License.

## Additional Information

This repository is designed to be a lightweight diffusion model in PyTorch:
- **Speed**: Generates images in near real-time.
- **Size**: Uses a model with ~100MM parameters.
- **Quality**: Not SOTA, but provides a reasonably good result for its size.
- **Training**: Can be trained in under 50 hours on a single GPU (A100 or equivalent).
- **Code**: Simple, self-contained codebase (~400 lines) for both the model and training loop.
## Acknowledgements
```
@misc{yi2023mllm,
  title = {mllm: fast and lightweight multimodal LLM inference engine for mobile and edge devices},
  author = {Rongjie Yi and Xiang Li and Qichen Qiu and Zhenyan Lu and Hao Zhang and Daliang Xu and Liming Yang and Weikai Xie and Chenghua Wang and Mengwei Xu},
  year = {2023},
  publisher = {mllm Team},
  url = {https://github.com/UbiquitousLearning/mllm}
}
```





