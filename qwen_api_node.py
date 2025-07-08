import os
import json
import random
import numpy as np
import torch
import requests # Required for making HTTP requests
from PIL import Image
import io
import base64

# 尝试从父目录导入工具函数，如果失败则使用本地/占位符版本
# This attempts to import utility functions from a parent directory.
# If it fails (e.g., when run standalone or in a different structure),
# it will use placeholder/local versions. This is common in ComfyUI custom nodes.
try:
    from ..utils.api_utils import load_prompt_options, get_prompt_content # Assuming these might be useful later
    from ..utils.env_manager import ensure_env_file, get_api_key # For API key management if you prefer environment variables
except ImportError:
    print("[QwenAPILLMNode] Warning: Could not import utility functions from parent directory. Using fallback mechanisms if needed.")
    # Fallback: Define dummy functions if these are strictly needed by the template structure
    # but not essential for the core Qwen API call logic shown here.
    def load_prompt_options(files): return {}
    def get_prompt_content(options, preset): return ""
    def ensure_env_file(): pass
    def get_api_key(service_name="DASHSCOPE_API_KEY"): # Qwen uses DASHSCOPE_API_KEY
        return os.getenv(service_name)

# Helper function to convert ComfyUI image tensor to Base64
# 辅助函数：将 ComfyUI 图像张量转换为 Base64 编码字符串
def comfy_image_to_base64(image_tensor: torch.Tensor) -> str:
    """
    Converts a ComfyUI image tensor (batch, height, width, channels, float 0-1)
    to a Base64 encoded PNG string.
    将 ComfyUI 图像张量（批次、高度、宽度、通道，浮点 0-1）
    转换为 Base64 编码的 PNG 字符串。
    """
    if image_tensor.ndim == 4:
        # Assuming batch size is 1 for single image input from a node
        image_tensor = image_tensor[0]

    # Convert from 0-1 float to 0-255 uint8
    image_tensor = (image_tensor * 255).byte()

    # Determine image mode (RGB or RGBA)
    if image_tensor.shape[2] == 4:
        mode = "RGBA"
    else:
        mode = "RGB"

    # Convert to PIL Image
    image_pil = Image.fromarray(image_tensor.cpu().numpy(), mode)

    # Save to BytesIO and base64 encode
    buffered = io.BytesIO()
    image_pil.save(buffered, format="PNG") # PNG is a good default for base64
    return base64.b64encode(buffered.getvalue()).decode('utf-8')


class QwenAPILLMNode:
    # Qwen API documentation: https://help.aliyun.com/zh/model-studio/use-qwen-by-calling-api
    # Model list based on user request and DashScope documentation.
    # 通义千问模型列表 (请参考阿里云官方文档获取最新列表和模型能力)
    QWEN_MODELS = [
        # 通义千问 通用模型
        "qwen-turbo",
        "qwen-plus",
        "qwen-max",
        "qwen-max-longcontext", # 长文本
        "qwen-plus-latest",     # 通义千问Plus最新版
        "qwen-max-latest",      # 通义千问Max最新版
        # 通义千问 VL 系列 (视觉语言模型, 文本部分可调用)
        "qwen-vl-plus",
        "qwen-vl-max",
        # 更多模型请参考阿里云官方文档，并确认其在DashScope上的可用性
        # For more models, please refer to the official Alibaba Cloud documentation
        # and confirm their availability on DashScope.
    ]

    # API Endpoint
    # API 请求地址
    API_URL = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"multiline": False, "default": os.getenv("DASHSCOPE_API_KEY", ""), "tooltip": "Your Alibaba Cloud DashScope API Key. You can also set the DASHSCOPE_API_KEY environment variable.\n您的阿里云DashScope API密钥。您也可以设置 DASHSCOPE_API_KEY 环境变量。"}),
                "model": (cls.QWEN_MODELS, {"tooltip": "Select the Qwen model to use.\n选择要使用的通义千问模型。"}),
                "prompt": ("STRING", {"multiline": True, "default": "Hello, Qwen!", "tooltip": "The main prompt or question for the model.\n给模型的主要提示或问题。"}),
                "system_message": ("STRING", {"multiline": True, "default": "You are a helpful assistant.", "tooltip": "Optional system message to guide the model's behavior.\n可选的系统消息，用于指导模型的行为。"}),
                "temperature": ("FLOAT", {"default": 0.85, "min": 0.0, "max": 2.0, "step": 0.05, "tooltip": "Controls randomness. Lower for more deterministic, higher for more creative output.\n控制输出的随机性。较低的值使输出更具确定性，较高的值使输出更具创造性。"}),
                "top_p": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Nucleus sampling. Considers the smallest set of tokens whose cumulative probability exceeds top_p.\n核心采样。模型会考虑累积概率超过 top_p 的最小词汇集。"}),
                "max_tokens": ("INT", {"default": 1500, "min": 1, "max": 30000, "step": 1, "tooltip": "Maximum number of tokens to generate in the response. Note: qwen-max-longcontext supports up to 30k tokens, other models vary.\n响应中生成的最大令牌数。注意：qwen-max-longcontext 支持高达30k令牌, 其他模型有所不同。"}), # Increased max for longcontext
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**32 - 1, "tooltip": "Seed for random number generation for reproducibility. 0 for random.\n用于可复现性的随机数种子。0表示随机。"}),
                "enable_search": ("BOOLEAN", {"default": False, "tooltip": "Whether to enable internet search (only for models that support it, e.g., qwen-plus, qwen-turbo, qwen-max, qwen-plus-latest, qwen-max-latest). Check documentation.\n是否启用互联网搜索（仅适用于支持此功能的模型，例如 qwen-plus, qwen-turbo, qwen-max, qwen-plus-latest, qwen-max-latest）。请查阅文档。"}),
                "enable_thinking": ("BOOLEAN", {"default": False, "tooltip": "Whether to enable the model's thinking process (only for models that support it and with streaming enabled). Current non-streaming implementation does NOT support this parameter. Check documentation.\n是否启用模型的思考过程（仅适用于支持此功能的模型且开启流式传输时）。当前非流式实现不支持此参数。请查阅文档。"}),
                "max_retries": ("INT", {"default": 1, "min": 0, "max": 5, "step": 1, "tooltip": "Maximum number of retries in case of API request failure.\nAPI请求失败时的最大重试次数。"}),
            },
            "optional": {
                # Changed to IMAGE type for ComfyUI node connection
                "image_input": ("IMAGE", {"tooltip": "Optional image input for multimodal models (e.g., qwen-vl-plus). Connect an image node here.\n用于多模态模型（如 qwen-vl-plus）的可选图像输入。请连接一个图像节点。"}),
                # Kept as STRING, but clarified tooltip for URL requirement
                "video_input_url": ("STRING", {"multiline": False, "default": "", "tooltip": "Optional video URL input for multimodal models (e.g., qwen-vl-max). Must be a publicly accessible URL. Leave empty if not used.\n用于多模态模型（如 qwen-vl-max）的可选视频URL输入。必须是可公开访问的URL。不使用时请留空。"}),
            }
        }

    RETURN_TYPES = ("STRING", "BOOLEAN", "STRING")
    RETURN_NAMES = ("generated_text", "success", "error_message")
    FUNCTION = "execute_qwen_request"
    CATEGORY = "LLM/Qwen API" # You can change the category
    OUTPUT_NODE = True

    def execute_qwen_request(self, api_key, model, prompt, system_message, temperature, top_p, max_tokens, seed, enable_search, enable_thinking, max_retries, image_input=None, video_input_url=""):
        # If API key is not provided directly, try to get it from environment variable
        # 如果未直接提供 API 密钥，则尝试从环境变量中获取
        if not api_key:
            api_key = os.getenv("DASHSCOPE_API_KEY")

        if not api_key:
            return ("", False, "API Key is missing. Please provide it directly or set the DASHSCOPE_API_KEY environment variable.\nAPI密钥缺失。请直接提供或设置 DASHSCOPE_API_KEY 环境变量。")

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            # "X-DashScope-SSE": "enable" # For streaming, if implemented later
        }

        current_seed = seed
        if seed == 0:
            current_seed = random.randint(1, 10000)

        messages = []
        if system_message and system_message.strip():
            messages.append({"role": "system", "content": system_message})

        # Prepare user message content, supporting multimodal input
        user_content_parts = []
        if prompt and prompt.strip():
            user_content_parts.append({"text": prompt})

        is_multimodal_model = model in ["qwen-vl-plus", "qwen-vl-max"]

        # Handle image input
        if image_input is not None:
            if not is_multimodal_model:
                return ("", False, f"Image input provided, but '{model}' is not a multimodal model (e.g., qwen-vl-plus). Please select a VL model.\n提供了图像输入，但 '{model}' 不是多模态模型（例如 qwen-vl-plus）。请选择一个VL模型。")
            try:
                # Convert ComfyUI image tensor to Base64
                image_base64_data = comfy_image_to_base64(image_input)
                user_content_parts.append({"image": image_base64_data})
            except Exception as e:
                return ("", False, f"Failed to process image input: {str(e)}\n处理图像输入失败：{str(e)}")

        # Handle video URL input
        if video_input_url and video_input_url.strip():
            if not is_multimodal_model:
                return ("", False, f"Video URL input provided, but '{model}' is not a multimodal model (e.g., qwen-vl-max). Please select a VL model.\n提供了视频URL输入，但 '{model}' 不是多模态模型（例如 qwen-vl-max）。请选择一个VL模型。")
            user_content_parts.append({"video": {"url": video_input_url}})

        # If no prompt and no media, return error or default to empty prompt
        if not user_content_parts:
            return ("", False, "No prompt, image, or video input provided. Please provide at least one.\n未提供提示、图像或视频输入。请至少提供一个。")

        messages.append({"role": "user", "content": user_content_parts})

        payload = {
            "model": model,
            "input": {
                "messages": messages
            },
            "parameters": {
                "result_format": "message",
                "temperature": temperature,
                "top_p": top_p,
                "max_tokens": max_tokens,
                "seed": current_seed,
            }
        }

        # Add enable_search parameter if enabled and supported model
        if enable_search and model in ["qwen-plus", "qwen-turbo", "qwen-max", "qwen-plus-latest", "qwen-max-latest"]:
            payload["parameters"]["enable_search"] = True

        # Handle enable_thinking parameter (requires streaming, not implemented here)
        if enable_thinking:
             print("[QwenAPILLMNode] Warning: enable_thinking is enabled but the current implementation does not support streaming calls required by this parameter. The parameter will not be sent.")
             # We don't add payload["parameters"]["enable_thinking"] = True here

        # Debug Print: Print the payload before sending the request
        print("[QwenAPILLMNode] Request Payload:")
        print(json.dumps(payload, indent=2))

        generated_text = ""
        success_flag = False
        error_msg = ""

        for attempt in range(max_retries + 1):
            try:
                print(f"[QwenAPILLMNode] Sending request to Qwen API (Attempt {attempt + 1}/{max_retries + 1})")
                response = requests.post(self.API_URL, headers=headers, json=payload, timeout=120)
                response.raise_for_status()

                response_data = response.json()

                if response_data.get("output") and response_data["output"].get("choices"):
                    generated_text = response_data["output"]["choices"][0]["message"]["content"]
                    success_flag = True
                    error_msg = f"Success (Request ID: {response_data.get('request_id', 'N/A')})"
                    break
                elif response_data.get("output") and response_data["output"].get("text"): # Fallback for older/different API responses
                    generated_text = response_data["output"]["text"]
                    success_flag = True
                    error_msg = f"Success (Request ID: {response_data.get('request_id', 'N/A')})"
                    break
                else:
                    error_code = response_data.get("code", "Unknown Error Code")
                    error_message_detail = response_data.get("message", "No detailed error message from API.")
                    error_msg = f"API Error: Code {error_code} - {error_message_detail} (Request ID: {response_data.get('request_id', 'N/A')})"
                    print(f"[QwenAPILLMNode] Error: {error_msg}")
                    print(f"[QwenAPILLMNode] API Error Response Text: {response.text}")


            except requests.exceptions.RequestException as e:
                error_msg = f"Request failed: {str(e)}"
                print(f"[QwenAPILLMNode] RequestException: {error_msg}")
                if 'response' in locals() and response is not None:
                     print(f"[QwenAPILLMNode] Exception Response Text: {response.text}")
            except json.JSONDecodeError as e:
                error_msg = f"Failed to decode JSON response: {str(e)}. Response text: {response.text[:200]}"
                print(f"[QwenAPILLMNode] JSONDecodeError: {error_msg}")
                if 'response' in locals() and response is not None:
                     print(f"[QwenAPILLMNode] JSON Decode Error Response Text: {response.text}")
            except Exception as e:
                error_msg = f"An unexpected error occurred: {str(e)}"
                print(f"[QwenAPILLMNode] Unexpected error: {error_msg}")


            if attempt < max_retries:
                print(f"[QwenAPILLMNode] Retrying...")
                if torch.cuda.is_available(): # Check if CUDA is available before trying to empty cache
                    torch.cuda.empty_cache()
                # import time # Import time if you use time.sleep
                # time.sleep(random.uniform(1, 3))

        if not success_flag and not error_msg:
            error_msg = "Failed after all retries without specific error."

        return (generated_text, success_flag, error_msg)

# ComfyUI Node Mappings
# ComfyUI 节点映射
NODE_CLASS_MAPPINGS = {
    "QwenAPILLMNode": QwenAPILLMNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "QwenAPILLMNode": "Qwen API LLM Node (DashScope)"
}

if __name__ == "__main__":
    # This section is for testing the node logic outside of ComfyUI
    # 此部分用于在 ComfyUI 环境之外测试节点逻辑
    print("Testing QwenAPILLMNode...")

    api_key_env = os.getenv("DASHSCOPE_API_KEY")
    if not api_key_env:
        print("Error: DASHSCOPE_API_KEY environment variable not set. Please set it to test.")
    else:
        node = QwenAPILLMNode()

        # --- Test Case 1: Text-only request (qwen-turbo) ---
        print("\n--- Test Case 1: Text-only request (qwen-turbo) ---")
        text_output, success, status = node.execute_qwen_request(
            api_key=api_key_env,
            model="qwen-turbo",
            prompt="用中文写一首关于秋天的短诗。(Write a short poem about autumn in Chinese.)",
            system_message="你是一个乐于助人的AI助手，擅长写诗。(You are a helpful AI assistant skilled in writing poetry.)",
            temperature=0.7,
            top_p=0.8,
            max_tokens=100,
            seed=12347,
            enable_search=False,
            enable_thinking=False,
            max_retries=1,
            image_input=None, # No image input for text-only model
            video_input_url="" # No video input
        )
        print("\n--- Test Result 1 ---")
        print(f"Success: {success}")
        print(f"Status/Error: {status}")
        print(f"Generated Text:\n{text_output}")

        # --- Test Case 2: Image input with a VL model (qwen-vl-plus) ---
        print("\n--- Test Case 2: Image input (qwen-vl-plus) ---")
        # Create a dummy image tensor for testing purposes
        # In a real ComfyUI setup, this would come from an "IMAGE" input node
        dummy_image_tensor = torch.zeros((1, 64, 64, 3), dtype=torch.float32) # 1x64x64 RGB black image
        text_output, success, status = node.execute_qwen_request(
            api_key=api_key_env,
            model="qwen-vl-plus",
            prompt="描述这张图片。(Describe this image.)",
            system_message="",
            temperature=0.7,
            top_p=0.8,
            max_tokens=100,
            seed=0,
            enable_search=False,
            enable_thinking=False,
            max_retries=1,
            image_input=dummy_image_tensor, # Pass the dummy tensor
            video_input_url=""
        )
        print("\n--- Test Result 2 ---")
        print(f"Success: {success}")
        print(f"Status/Error: {status}")
        print(f"Generated Text:\n{text_output}")

        # --- Test Case 3: Video URL input with a VL model (qwen-vl-max) ---
        print("\n--- Test Case 3: Video URL input (qwen-vl-max) ---")
        # Replace with a real, publicly accessible video URL for actual testing
        dummy_video_url = "https://example.com/your_video.mp4"
        text_output, success, status = node.execute_qwen_request(
            api_key=api_key_env,
            model="qwen-vl-max",
            prompt="总结这个视频的内容。(Summarize the content of this video.)",
            system_message="",
            temperature=0.7,
            top_p=0.8,
            max_tokens=100,
            seed=0,
            enable_search=False,
            enable_thinking=False,
            max_retries=1,
            image_input=None,
            video_input_url=dummy_video_url # Pass the video URL string
        )
        print("\n--- Test Result 3 ---")
        print(f"Success: {success}")
        print(f"Status/Error: {status}")
        print(f"Generated Text:\n{text_output}")

        # --- Test Case 4: Error case - Image input with a non-VL model ---
        print("\n--- Test Case 4: Error - Image input with non-VL model (qwen-turbo) ---")
        text_output, success, status = node.execute_qwen_request(
            api_key=api_key_env,
            model="qwen-turbo", # Non-VL model
            prompt="描述这张图片。(Describe this image.)",
            system_message="",
            temperature=0.7,
            top_p=0.8,
            max_tokens=100,
            seed=0,
            enable_search=False,
            enable_thinking=False,
            max_retries=1,
            image_input=dummy_image_tensor, # Image input provided
            video_input_url=""
        )
        print("\n--- Test Result 4 ---")
        print(f"Success: {success}")
        print(f"Status/Error: {status}")
        print(f"Generated Text:\n{text_output}")
