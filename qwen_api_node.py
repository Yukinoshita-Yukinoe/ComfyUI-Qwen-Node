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
import codecs

# --- Constants ---
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
    Converts a single ComfyUI image tensor (1, H, W, C) to a Base64 encoded string
    with the required data URI scheme for the Qwen API.
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
    It supports both text-only and multimodal (vision-language) models, including
    multi-image inputs with different processing strategies.
    """
    @classmethod
    def INPUT_TYPES(s):
        # Model list restored and updated as per user request
        model_list = [
            "qwen-plus-latest",     # Restored
            "qwen3-vl-plus",        # Renamed as requested
            "qwen-vl-max", 
            "qwen3-max",            # Renamed as requested
            "qwen-max-longcontext", 
            "qwen-plus", 
            "qwen-turbo", 
            "qwen-long", 
            "qwen-audio-turbo"
        ]
        return {
            "required": {
                # --- Original parameter order preserved ---
                "api_key": ("STRING", {"default": "", "multiline": False, "dynamicPrompts": False}),
                "model": (model_list, {"default": "qwen-plus-latest"}),
                "prompt": ("STRING", {"default": "Hello, Qwen!", "multiline": True, "dynamicPrompts": True}),
                "system_message": ("STRING", {"default": "You are a helpful assistant.", "multiline": True, "dynamicPrompts": False}),
                "temperature": ("FLOAT", {"default": 0.85, "min": 0.0, "max": 2.0, "step": 0.01}),
                "top_p": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.01}),
                "max_tokens": ("INT", {"default": 1500, "min": 1, "max": 8000, "step": 1}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "enable_search": ("BOOLEAN", {"default": False, "label_on": "enabled", "label_off": "disabled"}),
                "enable_thinking": ("BOOLEAN", {"default": False, "label_on": "enabled", "label_off": "disabled"}), # Restored
                "max_retries": ("INT", {"default": 1, "min": 1, "max": 5, "step": 1}),
                # --- New parameters added at the end ---
                "multi_image_mode": (["Native Batch", "Sequential"], {"default": "Native Batch"}),
                "sequential_delimiter": ("STRING", {"default": "\\n\\n---\\n\\n", "multiline": True}),
            },
            "optional": {
                "image_input": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "BOOLEAN")
    RETURN_NAMES = ("generated_text", "status_message", "is_success")
    FUNCTION = "execute_qwen_request"
    CATEGORY = "Lumi/LLM"

    def _make_api_call(self, api_url, headers, payload, max_retries):
        """ Helper function to perform the API call with retries. """
        # This helper function remains unchanged and correctly handles various response formats.
        for attempt in range(max_retries):
            print(f"[QwenAPILLMNode] Sending request to Qwen API ({api_url}) (Attempt {attempt + 1}/{max_retries})")
            try:
                with requests.post(api_url, headers=headers, json=payload, stream=False, timeout=120) as response:
                    response.raise_for_status()
                    response_data = response.json()
                    
                    generated_text = ""
                    output = response_data.get("output", {})
                    
                    if "choices" in output and output["choices"]:
                        content_list = output["choices"][0].get("message", {}).get("content", [])
                        if isinstance(content_list, list):
                            for item in content_list:
                                if "text" in item:
                                    generated_text = item["text"]
                                    break
                        elif isinstance(content_list, str):
                            generated_text = content_list
                        
                        if not generated_text and output["choices"][0].get("finish_reason") == "tool_calls":
                            generated_text = "Model used a tool (e.g. Code Interpreter). Check logs for details."
                    elif "text" in output:
                        generated_text = output["text"]

                    if not generated_text:
                        if "message" in response_data:
                            raise Exception(f"API returned an error: {response_data['message']}")
                        raise Exception("No valid text content found in the API response.")

                    status_message = f"Success (HTTP {response.status_code})"
                    return (generated_text, status_message, True)

            except Exception as e:
                error_message = f"!!! Exception during processing !!! {type(e).__name__}: {e}"
                try:
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

    def execute_qwen_request(self, api_key, model, prompt, system_message, temperature, top_p, max_tokens, seed, enable_search, enable_thinking, max_retries, multi_image_mode, sequential_delimiter, image_input=None):
        final_api_key = api_key if api_key else get_api_key("DASHSCOPE_API_KEY")
        if not final_api_key:
            return ("", "API Key is missing.", False)

        is_vl_model = 'vl' in model.lower()
        api_url = QWEN_MULTIMODAL_API_URL if is_vl_model else QWEN_TEXT_API_URL

        # --- Input Validation ---
        if is_vl_model and image_input is None:
            return ("", f"Model '{model}' is a vision-language model, but no image was provided.", False)
        if not is_vl_model and image_input is not None:
            return ("", f"Model '{model}' cannot process images. Please use a 'vl' model.", False)
        # Restored validation for enable_thinking
        if enable_thinking and model != 'qwen-plus-latest':
            return ("", f"Enable Thinking (Code Interpreter) is only supported for 'qwen-plus-latest' model.", False)

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {final_api_key}"
        }
        
        base_params = {
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "seed": random.randint(1, 10000) if seed == 0 else seed,
            "enable_search": enable_search,
        }
        
        # --- Sequential Mode for Multi-Image ---
        if is_vl_model and image_input is not None and image_input.shape[0] > 1 and multi_image_mode == 'Sequential':
            all_results = []
            all_statuses = []
            overall_success = True
            
            processed_delimiter = codecs.decode(sequential_delimiter, 'unicode_escape')

            for i in range(image_input.shape[0]):
                print(f"[QwenAPILLMNode] Processing image {i+1}/{image_input.shape[0]} in Sequential mode.")
                single_image_tensor = image_input[i:i+1]
                
                messages = []
                if system_message:
                    messages.append({"role": "system", "content": [{"text": system_message}]})
                
                user_content = []
                try:
                    base64_image = tensor_to_base64(single_image_tensor)
                    user_content.append({"image": base64_image})
                except Exception as e:
                    error_msg = f"Failed to convert image {i+1} to Base64: {e}"
                    all_results.append(f"[ERROR: {error_msg}]")
                    all_statuses.append(error_msg)
                    overall_success = False
                    continue
                
                user_content.append({"text": prompt})
                messages.append({"role": "user", "content": user_content})

                payload = {
                    "model": model,
                    "input": {"messages": messages},
                    "parameters": base_params.copy()
                }
                # Note: 'enable_thinking' is not applicable in sequential mode as it's a model-specific feature
                # and this loop is for VL models. The initial check prevents this combination.

                text, status, success = self._make_api_call(api_url, headers, payload, max_retries)
                all_results.append(text)
                all_statuses.append(f"Image {i+1}: {status}")
                if not success:
                    overall_success = False

            final_text = processed_delimiter.join(all_results)
            final_status = "\n".join(all_statuses)
            return (final_text, final_status, overall_success)

        # --- Native Batch Mode (or single image) / Text-only Mode ---
        messages = []
        if system_message:
            messages.append({"role": "system", "content": [{"text": system_message}]})

        user_content = []
        if is_vl_model and image_input is not None:
            for i in range(image_input.shape[0]):
                try:
                    single_image_tensor = image_input[i:i+1]
                    base64_image = tensor_to_base64(single_image_tensor)
                    user_content.append({"image": base64_image})
                except Exception as e:
                    return ("", f"Failed to convert image {i+1} to Base64: {e}", False)
        
        user_content.append({"text": prompt})
        messages.append({"role": "user", "content": user_content})

        payload = {
            "model": model,
            "input": {"messages": messages},
            "parameters": base_params.copy()
        }
        
        # Restore logic for enable_thinking
        if enable_thinking and model == 'qwen-plus-latest':
            payload['tools'] = [{"type": "code_interpreter"}]
            payload['parameters']['result_format'] = 'message'

        return self._make_api_call(api_url, headers, payload, max_retries)

NODE_CLASS_MAPPINGS = {"QwenAPILLMNode": QwenAPILLMNode}
NODE_DISPLAY_NAME_MAPPINGS = {"QwenAPILLMNode": "Qwen API (Lumi)"}