import os
import json
import random
import numpy as np
import torch
import requests
from PIL import Image
import io
import base64
import math

# --- Constants ---
# Define API endpoints at the top for easy management.
# 根据阿里云官方文档，文生文和多模态模型使用不同的API端点。
# According to the official Aliyun documentation, text-generation and multimodal models use different endpoints.
QWEN_TEXT_API_URL = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"
QWEN_MULTIMODAL_API_URL = "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"

# Fallback for utility functions if running in a non-standard environment
try:
    from ..utils.api_utils import load_prompt_options, get_prompt_content
    from ..utils.env_manager import get_api_key
except ImportError:
    print("[QwenAPILLMNode] Warning: Could not import utility functions. Using fallback mechanisms.")
    def get_api_key(service_name="DASHSCOPE_API_KEY"):
        key = os.getenv(service_name)
        if not key:
            print(f"[QwenAPILLMNode] ERROR: API key for '{service_name}' not found in environment variables.")
        return key

def tensor_to_base64(tensor):
    """
    Converts a ComfyUI image tensor to a Base64 encoded string.
    The tensor is expected to be in the format (batch, height, width, channels)
    with float values in the range [0, 1].
    """
    if tensor is None:
        return None
    # Squeeze the batch dimension if it's 1
    if tensor.dim() == 4 and tensor.shape[0] == 1:
        tensor = tensor.squeeze(0)
    
    # Convert tensor to numpy array and scale to 0-255
    image_np = (tensor.cpu().numpy() * 255).astype(np.uint8)
    
    # Convert numpy array to PIL Image
    pil_image = Image.fromarray(image_np)
    
    # Save PIL Image to a byte buffer
    buffered = io.BytesIO()
    pil_image.save(buffered, format="JPEG") # Use JPEG for smaller size, PNG is also an option
    
    # Encode the byte buffer to Base64
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

class QwenAPILLMNode:
    """
    A ComfyUI node to interact with the Alibaba Cloud Qwen (DashScope) API.
    It supports both text-only and multimodal (vision-language) models by dynamically
    selecting the correct API endpoint.
    """
    # Define input types for the node
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "api_key": ("STRING", {"default": "", "multiline": False, "dynamicPrompts": False}),
                "model": (["qwen-vl-max", "qwen-vl-plus", "qwen-plus", "qwen-turbo", "qwen-max", "qwen-max-longcontext"], {"default": "qwen-vl-plus"}),
                "prompt": ("STRING", {"default": "Describe this image in detail.", "multiline": True, "dynamicPrompts": True}),
                "system_message": ("STRING", {"default": "You are a helpful assistant.", "multiline": True, "dynamicPrompts": False}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "temperature": ("FLOAT", {"default": 0.85, "min": 0.0, "max": 2.0, "step": 0.01}),
                "top_p": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.01}),
                "max_tokens": ("INT", {"default": 1500, "min": 1, "max": 8000, "step": 1}),
                "max_retries": ("INT", {"default": 1, "min": 1, "max": 5, "step": 1}),
            },
            "optional": {
                "image_input": ("IMAGE",),
            }
        }

    # Define return types and names for the node
    RETURN_TYPES = ("STRING", "STRING", "BOOLEAN")
    RETURN_NAMES = ("generated_text", "status_message", "is_success")
    FUNCTION = "execute_qwen_request"
    CATEGORY = "Lumi/LLM"

    def execute_qwen_request(self, api_key, model, prompt, system_message, seed, temperature, top_p, max_tokens, max_retries, image_input=None):
        # --- 1. Get API Key ---
        # Prioritize the key from the input field, then fall back to environment variables.
        final_api_key = api_key if api_key else get_api_key("DASHSCOPE_API_KEY")
        if not final_api_key:
            return ("", "API Key is missing. Please provide it in the node or set DASHSCOPE_API_KEY environment variable.", False)

        # --- 2. Determine API URL and Validate Inputs ---
        # The core fix: select the URL based on the model name.
        is_vl_model = 'vl' in model.lower()
        api_url = QWEN_MULTIMODAL_API_URL if is_vl_model else QWEN_TEXT_API_URL

        if is_vl_model and image_input is None:
            return ("", f"Model '{model}' is a vision-language model, but no image was provided.", False)
        if not is_vl_model and image_input is not None:
            return ("", f"Model '{model}' is a text-only model and cannot process images. Please use a 'vl' model.", False)

        # --- 3. Construct Headers and Parameters ---
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {final_api_key}"
        }
        
        params = {
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "seed": random.randint(1, 10000) if seed == 0 else seed,
        }

        # --- 4. Construct Payload (Messages) ---
        # This logic correctly builds the payload for either text-only or multimodal requests.
        messages = []
        if system_message:
            messages.append({"role": "system", "content": [{"text": system_message}]})

        user_content = []
        if image_input is not None:
            try:
                base64_image = tensor_to_base64(image_input)
                user_content.append({"image": base64_image})
            except Exception as e:
                error_message = f"Failed to convert image to Base64: {e}"
                print(f"[QwenAPILLMNode] {error_message}")
                return ("", error_message, False)
        
        user_content.append({"text": prompt})
        messages.append({"role": "user", "content": user_content})

        payload = {
            "model": model,
            "input": {"messages": messages},
            "parameters": params
        }
        
        # --- 5. Execute API Request with Retries ---
        for attempt in range(max_retries):
            print(f"[QwenAPILLMNode] Sending request to Qwen API ({api_url}) (Attempt {attempt + 1}/{max_retries})")
            try:
                with requests.post(api_url, headers=headers, json=payload, stream=False) as response:
                    response.raise_for_status()  # Raises HTTPError for bad responses (4xx or 5xx)
                    
                    response_data = response.json()
                    
                    # Check for errors in the response body
                    if "output" not in response_data or "choices" not in response_data["output"]:
                        error_msg = response_data.get("message", "Invalid response structure from API.")
                        raise Exception(error_msg)

                    generated_text = response_data["output"]["choices"][0]["message"]["content"][0]["text"]
                    status_message = f"Success (HTTP {response.status_code})"
                    print(f"[QwenAPILLMNode] {status_message}")
                    return (generated_text, status_message, True)

            except requests.exceptions.RequestException as e:
                # This catches network errors, timeouts, and HTTPError
                status_code = e.response.status_code if e.response is not None else "N/A"
                error_text = e.response.text if e.response is not None else str(e)
                error_message = f"Request failed with status {status_code}. Response: {error_text}"
                print(f"[QwenAPILLMNode] {error_message}")
                
                if attempt + 1 < max_retries:
                    print("[QwenAPILLMNode] Retrying...")
                    continue
                else:
                    return ("", error_message, False)
        
        # This should not be reached if max_retries >= 1
        return ("", "An unknown error occurred after all retries.", False)

# ComfyUI-specific mappings
NODE_CLASS_MAPPINGS = {
    "QwenAPILLMNode": QwenAPILLMNode
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "QwenAPILLMNode": "Qwen API (Lumi)"
}
