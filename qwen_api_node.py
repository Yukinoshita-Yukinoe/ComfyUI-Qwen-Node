import base64
import codecs
import io
import os
import random
from typing import Any

import numpy as np
import requests
from PIL import Image

# --- Constants ---
# According to the official Aliyun documentation, text-generation and multimodal
# models use different DashScope endpoints.
QWEN_TEXT_API_URL = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"
QWEN_MULTIMODAL_API_URL = "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"

THINKING_DISABLED = "disabled"
THINKING_TOGGLE = "toggle"
THINKING_ONLY = "only"

QWEN_MODEL_SPECS = {
    "qwen-plus-latest": {
        "api_url": QWEN_TEXT_API_URL,
        "supports_image": False,
        "thinking_mode": THINKING_TOGGLE,
    },
    "qwen-plus": {
        "api_url": QWEN_TEXT_API_URL,
        "supports_image": False,
        "thinking_mode": THINKING_TOGGLE,
    },
    "qwen-flash": {
        "api_url": QWEN_TEXT_API_URL,
        "supports_image": False,
        "thinking_mode": THINKING_TOGGLE,
    },
    "qwen-turbo-latest": {
        "api_url": QWEN_TEXT_API_URL,
        "supports_image": False,
        "thinking_mode": THINKING_TOGGLE,
    },
    "qwen-turbo": {
        "api_url": QWEN_TEXT_API_URL,
        "supports_image": False,
        "thinking_mode": THINKING_TOGGLE,
    },
    "qwen3-max": {
        "api_url": QWEN_TEXT_API_URL,
        "supports_image": False,
        "thinking_mode": THINKING_TOGGLE,
    },
    "qwen3-max-preview": {
        "api_url": QWEN_TEXT_API_URL,
        "supports_image": False,
        "thinking_mode": THINKING_TOGGLE,
    },
    "qwen3.6-plus": {
        "api_url": QWEN_MULTIMODAL_API_URL,
        "supports_image": True,
        "thinking_mode": THINKING_TOGGLE,
    },
    "qwen3.5-plus": {
        "api_url": QWEN_MULTIMODAL_API_URL,
        "supports_image": True,
        "thinking_mode": THINKING_TOGGLE,
    },
    "qwen3.5-flash": {
        "api_url": QWEN_MULTIMODAL_API_URL,
        "supports_image": True,
        "thinking_mode": THINKING_TOGGLE,
    },
    "qwen3-coder-plus": {
        "api_url": QWEN_TEXT_API_URL,
        "supports_image": False,
        "thinking_mode": THINKING_DISABLED,
    },
    "qwen3-coder-flash": {
        "api_url": QWEN_TEXT_API_URL,
        "supports_image": False,
        "thinking_mode": THINKING_DISABLED,
    },
    "qwen-long-latest": {
        "api_url": QWEN_TEXT_API_URL,
        "supports_image": False,
        "thinking_mode": THINKING_DISABLED,
    },
    "qwen-long": {
        "api_url": QWEN_TEXT_API_URL,
        "supports_image": False,
        "thinking_mode": THINKING_DISABLED,
    },
    "qwen3-vl-plus": {
        "api_url": QWEN_MULTIMODAL_API_URL,
        "supports_image": True,
        "thinking_mode": THINKING_TOGGLE,
    },
    "qwen3-vl-flash": {
        "api_url": QWEN_MULTIMODAL_API_URL,
        "supports_image": True,
        "thinking_mode": THINKING_TOGGLE,
    },
    "qwen-vl-max-latest": {
        "api_url": QWEN_MULTIMODAL_API_URL,
        "supports_image": True,
        "thinking_mode": THINKING_DISABLED,
    },
    "qwen-vl-max": {
        "api_url": QWEN_MULTIMODAL_API_URL,
        "supports_image": True,
        "thinking_mode": THINKING_DISABLED,
    },
    "qwen-vl-plus-latest": {
        "api_url": QWEN_MULTIMODAL_API_URL,
        "supports_image": True,
        "thinking_mode": THINKING_DISABLED,
    },
    "qwen-vl-plus": {
        "api_url": QWEN_MULTIMODAL_API_URL,
        "supports_image": True,
        "thinking_mode": THINKING_DISABLED,
    },
    # Legacy entries kept for existing workflows.
    "qwen-max-longcontext": {
        "api_url": QWEN_TEXT_API_URL,
        "supports_image": False,
        "thinking_mode": THINKING_DISABLED,
    },
    "qwen-audio-turbo": {
        "api_url": QWEN_TEXT_API_URL,
        "supports_image": False,
        "thinking_mode": THINKING_DISABLED,
    },
}


def build_model_display_name(model_name: str, model_spec: dict[str, Any]) -> str:
    thinking_mode = model_spec["thinking_mode"]
    if thinking_mode == THINKING_ONLY:
        return f"{model_name} [T-only]"
    if thinking_mode == THINKING_TOGGLE:
        return f"{model_name} [T]"
    return model_name


QWEN_MODEL_DISPLAY_NAMES = tuple(
    build_model_display_name(model_name, model_spec)
    for model_name, model_spec in QWEN_MODEL_SPECS.items()
)
QWEN_DISPLAY_NAME_TO_MODEL = {
    display_name: model_name
    for model_name, display_name in zip(QWEN_MODEL_SPECS.keys(), QWEN_MODEL_DISPLAY_NAMES)
}
QWEN_MODEL_TO_DISPLAY_NAME = {
    model_name: display_name
    for display_name, model_name in QWEN_DISPLAY_NAME_TO_MODEL.items()
}


# Fallback for utility functions if running in a non-standard environment
try:
    from ..utils.env_manager import get_api_key
except ImportError:
    print("[QwenAPILLMNode] Warning: Could not import utility functions. Using fallback mechanisms.")

    def get_api_key(service_name: str = "DASHSCOPE_API_KEY") -> str | None:
        key = os.getenv(service_name)
        if not key:
            print(f"[QwenAPILLMNode] ERROR: API key for '{service_name}' not found in environment variables.")
        return key


def tensor_to_base64(tensor: Any) -> str | None:
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
    base64_string = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{base64_string}"


def get_model_spec(model_name: str) -> dict[str, Any]:
    spec = QWEN_MODEL_SPECS.get(model_name)
    if spec is not None:
        return spec

    normalized_model_name = model_name.lower()
    is_multimodal = any(
        keyword in normalized_model_name
        for keyword in ("qwen3.6-plus", "qwen3.5-", "qwen3-vl", "qwen-vl", "qvq")
    )
    thinking_mode = THINKING_DISABLED
    if "thinking" in normalized_model_name or normalized_model_name.startswith("qwq"):
        thinking_mode = THINKING_ONLY
    elif any(
        keyword in normalized_model_name
        for keyword in (
            "qwen3.6-plus",
            "qwen3.5-",
            "qwen3-vl",
            "qwen3-max",
            "qwen3-max-preview",
            "qwen3-max-2026-01-23",
            "qwen-plus",
            "qwen-flash",
            "qwen-turbo",
        )
    ):
        thinking_mode = THINKING_TOGGLE
    return {
        "api_url": QWEN_MULTIMODAL_API_URL if is_multimodal else QWEN_TEXT_API_URL,
        "supports_image": is_multimodal,
        "thinking_mode": thinking_mode,
    }


def resolve_model_name(selected_model_name: str) -> str:
    return QWEN_DISPLAY_NAME_TO_MODEL.get(selected_model_name, selected_model_name)


def extract_message_text(message: dict[str, Any]) -> str:
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        return content.get("text", "")
    if isinstance(content, list):
        text_segments = []
        for item in content:
            if isinstance(item, dict) and item.get("text"):
                text_segments.append(item["text"])
        return "\n".join(text_segments).strip()
    return ""


class QwenAPILLMNode:
    """
    A ComfyUI node to interact with the Alibaba Cloud Qwen (DashScope) API.
    It supports both text-only and multimodal models, including multi-image
    inputs with native batch and sequential processing strategies.
    """

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "api_key": ("STRING", {"default": "", "multiline": False, "dynamicPrompts": False}),
                "model": (QWEN_MODEL_DISPLAY_NAMES, {"default": QWEN_MODEL_TO_DISPLAY_NAME["qwen-plus-latest"]}),
                "prompt": ("STRING", {"default": "Hello, Qwen!", "multiline": True, "dynamicPrompts": True}),
                "system_message": ("STRING", {"default": "You are a helpful assistant.", "multiline": True, "dynamicPrompts": False}),
                "temperature": ("FLOAT", {"default": 0.85, "min": 0.0, "max": 2.0, "step": 0.01}),
                "top_p": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.01}),
                "max_tokens": ("INT", {"default": 1500, "min": 1, "max": 65536, "step": 1}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "enable_search": ("BOOLEAN", {"default": False, "label_on": "enabled", "label_off": "disabled"}),
                "enable_thinking": ("BOOLEAN", {"default": False, "label_on": "enabled", "label_off": "disabled"}),
                "max_retries": ("INT", {"default": 1, "min": 1, "max": 5, "step": 1}),
                "multi_image_mode": (["Native Batch", "Sequential"], {"default": "Native Batch"}),
                "sequential_delimiter": ("STRING", {"default": "\\n\\n---\\n\\n", "multiline": True}),
            },
            "optional": {
                "image_input": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "BOOLEAN")
    RETURN_NAMES = ("generated_text", "status_message", "is_success")
    FUNCTION = "execute_qwen_request"
    CATEGORY = "Lumi/LLM"

    def _build_messages(
        self,
        system_message: str,
        prompt: str,
        api_url: str,
        image_input: Any = None,
    ) -> list[dict[str, Any]]:
        uses_multimodal_api = api_url == QWEN_MULTIMODAL_API_URL
        messages = []

        if system_message:
            system_content = [{"text": system_message}] if uses_multimodal_api else system_message
            messages.append({"role": "system", "content": system_content})

        if uses_multimodal_api:
            user_content = []
            if image_input is not None:
                for image_index in range(image_input.shape[0]):
                    single_image_tensor = image_input[image_index:image_index + 1]
                    base64_image = tensor_to_base64(single_image_tensor)
                    user_content.append({"image": base64_image})
            user_content.append({"text": prompt})
            messages.append({"role": "user", "content": user_content})
        else:
            messages.append({"role": "user", "content": prompt})

        return messages

    def _make_api_call(
        self,
        api_url: str,
        headers: dict[str, str],
        payload: dict[str, Any],
        max_retries: int,
    ) -> tuple[str, str, bool]:
        """Helper function to perform the API call with retries."""
        for attempt in range(max_retries):
            response = None
            print(f"[QwenAPILLMNode] Sending request to Qwen API ({api_url}) (Attempt {attempt + 1}/{max_retries})")
            try:
                with requests.post(api_url, headers=headers, json=payload, stream=False, timeout=120) as response:
                    response.raise_for_status()
                    response_data = response.json()

                    output = response_data.get("output", {})
                    generated_text = ""

                    if "choices" in output and output["choices"]:
                        message = output["choices"][0].get("message", {})
                        generated_text = extract_message_text(message)
                    elif "text" in output:
                        generated_text = output["text"]

                    if not generated_text:
                        if "message" in response_data:
                            raise Exception(f"API returned an error: {response_data['message']}")
                        raise Exception("No valid text content found in the API response.")

                    status_message = f"Success (HTTP {response.status_code})"
                    return (generated_text, status_message, True)

            except Exception as error:
                error_message = f"!!! Exception during processing !!! {type(error).__name__}: {error}"
                if response is not None:
                    try:
                        error_message = f"Request failed with status {response.status_code}. Response: {response.text}"
                    except Exception:
                        pass

                print(f"[QwenAPILLMNode] {error_message}")

                if attempt + 1 < max_retries:
                    print("[QwenAPILLMNode] Retrying...")
                    continue
                return ("", error_message, False)

        return ("", "An unknown error occurred after all retries.", False)

    def execute_qwen_request(
        self,
        api_key: str,
        model: str,
        prompt: str,
        system_message: str,
        temperature: float,
        top_p: float,
        max_tokens: int,
        seed: int,
        enable_search: bool,
        enable_thinking: bool,
        max_retries: int,
        multi_image_mode: str,
        sequential_delimiter: str,
        image_input: Any = None,
    ) -> tuple[str, str, bool]:
        final_api_key = api_key if api_key else get_api_key("DASHSCOPE_API_KEY")
        if not final_api_key:
            return ("", "API Key is missing.", False)

        resolved_model = resolve_model_name(model)
        model_spec = get_model_spec(resolved_model)
        api_url = model_spec["api_url"]
        supports_image = model_spec["supports_image"]
        thinking_mode = model_spec["thinking_mode"]

        if image_input is not None and not supports_image:
            return ("", f"Model '{resolved_model}' cannot process image input.", False)
        if enable_thinking and thinking_mode == THINKING_DISABLED:
            return ("", f"Enable Thinking is not supported for model '{resolved_model}'.", False)

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {final_api_key}",
        }

        base_params = {
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "seed": random.randint(1, 10000) if seed == 0 else seed,
            "enable_search": enable_search,
            "result_format": "message",
        }
        if thinking_mode == THINKING_TOGGLE:
            base_params["enable_thinking"] = enable_thinking

        if supports_image and image_input is not None and image_input.shape[0] > 1 and multi_image_mode == "Sequential":
            all_results = []
            all_statuses = []
            overall_success = True
            processed_delimiter = codecs.decode(sequential_delimiter, "unicode_escape")

            for image_index in range(image_input.shape[0]):
                print(
                    f"[QwenAPILLMNode] Processing image {image_index + 1}/{image_input.shape[0]} in Sequential mode."
                )
                single_image_tensor = image_input[image_index:image_index + 1]

                try:
                    messages = self._build_messages(
                        system_message,
                        prompt,
                        api_url,
                        image_input=single_image_tensor,
                    )
                except Exception as error:
                    error_message = f"Failed to convert image {image_index + 1} to Base64: {error}"
                    all_results.append(f"[ERROR: {error_message}]")
                    all_statuses.append(error_message)
                    overall_success = False
                    continue

                payload = {
                    "model": resolved_model,
                    "input": {"messages": messages},
                    "parameters": base_params.copy(),
                }

                text, status, success = self._make_api_call(api_url, headers, payload, max_retries)
                all_results.append(text)
                all_statuses.append(f"Image {image_index + 1}: {status}")
                if not success:
                    overall_success = False

            final_text = processed_delimiter.join(all_results)
            final_status = "\n".join(all_statuses)
            return (final_text, final_status, overall_success)

        try:
            messages = self._build_messages(system_message, prompt, api_url, image_input=image_input)
        except Exception as error:
            return ("", f"Failed to convert image to Base64: {error}", False)

        payload = {
            "model": resolved_model,
            "input": {"messages": messages},
            "parameters": base_params.copy(),
        }
        return self._make_api_call(api_url, headers, payload, max_retries)

NODE_CLASS_MAPPINGS = {"QwenAPILLMNode": QwenAPILLMNode}
NODE_DISPLAY_NAME_MAPPINGS = {"QwenAPILLMNode": "Qwen API (Lumi)"}
