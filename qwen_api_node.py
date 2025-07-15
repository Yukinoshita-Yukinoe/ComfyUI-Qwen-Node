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
    Converts a ComfyUI image tensor to a Base64 encoded string with the
    required data URI scheme for the Qwen API.
    The tensor is expected to be in the format (batch, height, width, channels)
    with float values in the range [0, 1].
    """
    if tensor is None:
        return None
    if tensor.dim() == 4 and tensor.shape[0] == 1:
        tensor = tensor.squeeze(0)
    image_np = (tensor.cpu().numpy() * 255).astype(np.uint8)
    pil_image = Image.fromarray(image_np)
    buffered = io.BytesIO()
    pil_image.save(buffered, format="JPEG")
    base64_string = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return f"data:image/jpeg;base64,{base64_string}"

class QwenAPILLMNode:
    """
    A ComfyUI node to interact with the Alibaba Cloud Qwen (DashScope) API.
    It supports both text-only and multimodal (vision-language) models by dynamically
    selecting the correct API endpoint.
    """
    @classmethod
    def INPUT_TYPES(s):
        # Updated model list based on the latest Aliyun documentation.
        model_list = [
            "qwen-plus-latest",
            "qwen-vl-max", 
            "qwen-vl-plus", 
            "qwen-max", 
            "qwen-max-longcontext", 
            "qwen-plus", 
            "qwen-turbo", 
            "qwen-long", 
            "qwen-audio-turbo"
        ]
        return {
            "required": {
                "api_key": ("STRING", {"default": "", "multiline": False, "dynamicPrompts": False}),
                "model": (model_list, {"default": "qwen-plus-latest"}),
                "prompt": ("STRING", {"default": "Hello, Qwen!", "multiline": True, "dynamicPrompts": True}),
                "system_message": ("STRING", {"default": "You are a helpful assistant.", "multiline": True, "dynamicPrompts": False}),
                # --- Parameter order restored as per user request ---
                "temperature": ("FLOAT", {"default": 0.85, "min": 0.0, "max": 2.0, "step": 0.01}),
                "top_p": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.01}),
                "max_tokens": ("INT", {"default": 1500, "min": 1, "max": 8000, "step": 1}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "enable_search": ("BOOLEAN", {"default": False, "label_on": "enabled", "label_off": "disabled"}),
                "enable_thinking": ("BOOLEAN", {"default": False, "label_on": "enabled", "label_off": "disabled"}),
                "max_retries": ("INT", {"default": 1, "min": 1, "max": 5, "step": 1}),
            },
            "optional": {
                "image_input": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "BOOLEAN")
    RETURN_NAMES = ("generated_text", "status_message", "is_success")
    FUNCTION = "execute_qwen_request"
    CATEGORY = "Lumi/LLM"

    def execute_qwen_request(self, api_key, model, prompt, system_message, temperature, top_p, max_tokens, seed, enable_search, enable_thinking, max_retries, image_input=None):
        final_api_key = api_key if api_key else get_api_key("DASHSCOPE_API_KEY")
        if not final_api_key:
            return ("", "API Key is missing.", False)

        is_vl_model = 'vl' in model.lower()
        api_url = QWEN_MULTIMODAL_API_URL if is_vl_model else QWEN_TEXT_API_URL

        if is_vl_model and image_input is None:
            return ("", f"Model '{model}' is a vision-language model, but no image was provided.", False)
        if not is_vl_model and image_input is not None:
            return ("", f"Model '{model}' cannot process images. Please use a 'vl' model.", False)
        if enable_thinking and model != 'qwen-plus-latest':
            return ("", f"Enable Thinking (Code Interpreter) is only supported for 'qwen-plus-latest' model.", False)

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {final_api_key}"
        }
        
        params = {
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "seed": random.randint(1, 10000) if seed == 0 else seed,
            "enable_search": enable_search,
        }

        messages = []
        if system_message:
            messages.append({"role": "system", "content": [{"text": system_message}]})

        user_content = []
        if is_vl_model and image_input is not None:
            try:
                base64_image = tensor_to_base64(image_input)
                user_content.append({"image": base64_image})
            except Exception as e:
                return ("", f"Failed to convert image to Base64: {e}", False)
        
        user_content.append({"text": prompt})
        messages.append({"role": "user", "content": user_content})

        payload = {
            "model": model,
            "input": {"messages": messages},
            "parameters": params
        }
        
        if enable_thinking and model == 'qwen-plus-latest':
            payload['tools'] = [{"type": "code_interpreter"}]
            payload['parameters']['result_format'] = 'message'

        for attempt in range(max_retries):
            print(f"[QwenAPILLMNode] Sending request to Qwen API ({api_url}) (Attempt {attempt + 1}/{max_retries})")
            try:
                with requests.post(api_url, headers=headers, json=payload, stream=False) as response:
                    response.raise_for_status()
                    response_data = response.json()
                    
                    generated_text = ""
                    # --- FIX: Handle both 'message' and 'text' response formats ---
                    output = response_data.get("output", {})
                    
                    # Case 1: Handle 'message' format (for VL models and tools)
                    if "choices" in output and output["choices"]:
                        content_list = output["choices"][0].get("message", {}).get("content", [])
                        if isinstance(content_list, list):
                            for item in content_list:
                                if "text" in item:
                                    generated_text = item["text"]
                                    break
                        elif isinstance(content_list, str): # Fallback for simple string content
                             generated_text = content_list
                        
                        # Handle case where a tool was used but no text was returned
                        if not generated_text and output["choices"][0].get("finish_reason") == "tool_calls":
                            generated_text = "Model used a tool (e.g. Code Interpreter). Check logs for details."

                    # Case 2: Handle simple 'text' format (for standard text models)
                    elif "text" in output:
                        generated_text = output["text"]

                    # If no text could be extracted, raise an error
                    if not generated_text:
                        raise Exception("No valid text content found in the API response.")

                    status_message = f"Success (HTTP {response.status_code})"
                    print(f"[QwenAPILLMNode] {status_message}")
                    return (generated_text, status_message, True)

            except Exception as e:
                # Improved error logging
                error_message = f"!!! Exception during processing !!! {type(e).__name__}: {e}"
                try:
                    # Try to get more specific error from API response if possible
                    error_text = response.text
                    error_message = f"Request failed with status {response.status_code}. Response: {error_text}"
                except:
                    pass
                
                print(f"[QwenAPILLMNode] {error_message}")
                
                if attempt + 1 < max_retries:
                    print("[QwenAPILLMNode] Retrying...")
                    continue
                else:
                    return ("", error_message, False)
        
        return ("", "An unknown error occurred after all retries.", False)

NODE_CLASS_MAPPINGS = {"QwenAPILLMNode": QwenAPILLMNode}
NODE_DISPLAY_NAME_MAPPINGS = {"QwenAPILLMNode": "Qwen API (Lumi)"}
