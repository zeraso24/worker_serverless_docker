"""
Qwen Image Edit Inpaint Node
A dedicated node for Qwen Image Edit inpainting functionality.
Independent version - no external dependencies required.
"""

import os
import io
import tempfile
import configparser

import torch
import numpy as np
import requests
from PIL import Image
from fal_client.client import SyncClient


# ==================== FAL API UTILITIES ====================

class FalConfig:
    """Singleton class to handle FAL configuration and client setup."""

    _instance = None
    _client = None
    _key = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(FalConfig, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self, api_key_override=None):
        """Initialize configuration and API key."""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        config_path = os.path.join(parent_dir, "config.ini")

        config = configparser.ConfigParser()
        config.read(config_path)

        try:
            # Priority: 1. API key from node parameter, 2. Environment variable, 3. config.ini
            if api_key_override and api_key_override.strip() and api_key_override != "<your_fal_api_key_here>":
                self._key = api_key_override.strip()
                print("FAL_KEY found in node parameter")
            elif os.environ.get("FAL_KEY") is not None:
                print("FAL_KEY found in environment variables")
                self._key = os.environ["FAL_KEY"]
            else:
                print("FAL_KEY not found in environment variables")
                self._key = config["API"]["FAL_KEY"]
                print("FAL_KEY found in config.ini")
                os.environ["FAL_KEY"] = self._key
                print("FAL_KEY set in environment variables")

            # Check if FAL key is the default placeholder
            if self._key == "<your_fal_api_key_here>" or not self._key or self._key.strip() == "":
                print("WARNING: You are using the default FAL API key placeholder or empty key!")
                print("Please set your actual FAL API key in either:")
                print("1. The node parameter 'api_key' (recommended)")
                print("2. The config.ini file under [API] section")
                print("3. Or as an environment variable named FAL_KEY")
                print("Get your API key from: https://fal.ai/dashboard/keys")
        except KeyError:
            print("Error: FAL_KEY not found in config.ini or environment variables")

    def get_client(self, api_key_override=None):
        """Get or create the FAL client."""
        # If API key override is provided, use it; otherwise use the stored key
        key_to_use = api_key_override if api_key_override and api_key_override.strip() else self._key
        
        if not key_to_use or key_to_use == "<your_fal_api_key_here>":
            raise ValueError("FAL API key is not set. Please provide a valid API key.")
        
        # Create a new client if key is different, or reuse existing client
        if self._client is None or (api_key_override and api_key_override != self._key):
            self._client = SyncClient(key=key_to_use)
        return self._client

    def get_key(self):
        """Get the FAL API key."""
        return self._key


class ImageUtils:
    """Utility functions for image processing."""

    @staticmethod
    def tensor_to_pil(image):
        """Convert image tensor to PIL Image."""
        try:
            # Convert the image tensor to a numpy array
            if isinstance(image, torch.Tensor):
                image_np = image.cpu().numpy()
            else:
                image_np = np.array(image)

            # Ensure the image is in the correct format (H, W, C)
            if image_np.ndim == 4:
                image_np = image_np.squeeze(0)  # Remove batch dimension if present
            if image_np.ndim == 2:
                image_np = np.stack([image_np] * 3, axis=-1)  # Convert grayscale to RGB
            elif image_np.shape[0] == 3:
                image_np = np.transpose(
                    image_np, (1, 2, 0)
                )  # Change from (C, H, W) to (H, W, C)

            # Normalize the image data to 0-255 range
            if image_np.dtype == np.float32 or image_np.dtype == np.float64:
                image_np = (image_np * 255).astype(np.uint8)

            # Convert to PIL Image
            return Image.fromarray(image_np)
        except Exception as e:
            print(f"Error converting tensor to PIL: {str(e)}")
            return None

    @staticmethod
    def upload_image(image, api_key=None):
        """Upload image tensor to FAL and return URL."""
        try:
            pil_image = ImageUtils.tensor_to_pil(image)
            if not pil_image:
                return None

            # Save the image to a temporary file
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
                pil_image.save(temp_file, format="PNG")
                temp_file_path = temp_file.name

            # Upload the temporary file
            fal_config = FalConfig()
            client = fal_config.get_client(api_key_override=api_key)
            image_url = client.upload_file(temp_file_path)
            return image_url
        except Exception as e:
            print(f"Error uploading image: {str(e)}")
            return None
        finally:
            # Clean up the temporary file
            if "temp_file_path" in locals():
                os.unlink(temp_file_path)
                
    @staticmethod
    def upload_file(file_path, api_key=None):
        """Upload a file to FAL and return URL."""
        try:
            fal_config = FalConfig()
            client = fal_config.get_client(api_key_override=api_key)
            file_url = client.upload_file(file_path)
            return file_url
        except Exception as e:
            print(f"Error uploading file: {str(e)}")
            return None
        
    @staticmethod
    def mask_to_image(mask):
        """Convert mask tensor to image tensor."""
        result = (
            mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1]))
            .movedim(1, -1)
            .expand(-1, -1, -1, 3)
        )
        return result


class ResultProcessor:
    """Utility functions for processing API results."""

    @staticmethod
    def process_image_result(result):
        """Process image generation result and return tensor."""
        try:
            images = []
            for img_info in result["images"]:
                img_url = img_info["url"]
                img_response = requests.get(img_url)
                img = Image.open(io.BytesIO(img_response.content))
                img_array = np.array(img).astype(np.float32) / 255.0
                images.append(img_array)

            # Stack the images along a new first dimension
            stacked_images = np.stack(images, axis=0)

            # Convert to PyTorch tensor
            img_tensor = torch.from_numpy(stacked_images)

            return (img_tensor,)
        except Exception as e:
            print(f"Error processing image result: {str(e)}")
            return ResultProcessor.create_blank_image()

    @staticmethod
    def process_single_image_result(result):
        """Process single image result and return tensor."""
        try:
            img_url = result["image"]["url"]
            img_response = requests.get(img_url)
            img = Image.open(io.BytesIO(img_response.content))
            img_array = np.array(img).astype(np.float32) / 255.0

            # Stack the images along a new first dimension
            stacked_images = np.stack([img_array], axis=0)

            # Convert to PyTorch tensor
            img_tensor = torch.from_numpy(stacked_images)
            return (img_tensor,)
        except Exception as e:
            print(f"Error processing single image result: {str(e)}")
            return ResultProcessor.create_blank_image()

    @staticmethod
    def create_blank_image():
        """Create a blank black image tensor."""
        blank_img = Image.new("RGB", (512, 512), color="black")
        img_array = np.array(blank_img).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_array)[None,]
        return (img_tensor,)


class ApiHandler:
    """Utility functions for API interactions."""

    @staticmethod
    def submit_and_get_result(endpoint, arguments, api_key=None):
        """Submit job to FAL API and get result."""
        try:
            fal_config = FalConfig()
            client = fal_config.get_client(api_key_override=api_key)
            handler = client.submit(endpoint, arguments=arguments)
            return handler.get()
        except Exception as e:
            print(f"Error submitting to {endpoint}: {str(e)}")
            raise e

    @staticmethod
    def handle_video_generation_error(model_name, error):
        """Handle video generation errors consistently."""
        print(f"Error generating video with {model_name}: {str(error)}")
        return ("Error: Unable to generate video.",)

    @staticmethod
    def handle_image_generation_error(model_name, error):
        """Handle image generation errors consistently."""
        print(f"Error generating image with {model_name}: {str(error)}")
        return ResultProcessor.create_blank_image()

    @staticmethod
    def handle_text_generation_error(model_name, error):
        """Handle text generation errors consistently."""
        print(f"Error generating text with {model_name}: {str(error)}")
        return ("Error: Unable to generate text.",)


# ==================== QWEN IMAGE EDIT INPAINT NODE ====================


class QwenImageEditInpaint:
    """
    Qwen Image Edit Inpaint Node
    Performs inpainting using Qwen Image Edit model.
    Requires: image, mask, and prompt.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Image to be inpainted"}),
                "mask": ("MASK", {"tooltip": "Mask indicating areas to inpaint"}),
                "prompt": ("STRING", {"default": "", "multiline": True, "tooltip": "Text prompt describing the desired inpainting"}),
            },
            "optional": {
                "api_key": ("STRING", {"default": "", "tooltip": "FAL API key (optional, overrides config.ini and environment variable)"}),
                "acceleration": (["none", "regular"], {"default": "regular", "tooltip": "Acceleration level for processing"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "inpaint"
    CATEGORY = "plexard_FAL"
    DESCRIPTION = "Qwen Image Edit Inpaint - Inpainting with prompt and mask"

    def inpaint(
        self,
        image,
        mask,
        prompt,
        api_key="",
        acceleration="regular",
    ):
        """
        Perform inpainting using Qwen Image Edit model.
        
        Args:
            image: Input image tensor (required)
            mask: Mask tensor indicating areas to inpaint (required)
            prompt: Text prompt describing the desired inpainting (required)
            api_key: FAL API key (optional, overrides config.ini and environment variable)
            acceleration: Acceleration level - "none" or "regular" (default: "regular")
            
        Returns:
            Tuple containing the inpainted image tensor
        """
        model_name = "Qwen Image Edit Inpaint"
        
        # Validate and sanitize input parameters
        try:
            # Validate acceleration
            valid_accelerations = ["none", "regular"]
            if acceleration not in valid_accelerations:
                acceleration = "regular"
        except Exception as e:
            print(f"Warning: Error validating parameters: {str(e)}, using defaults")
            acceleration = "regular"
        
        # Get API key for uploads (priority: node parameter > environment > config.ini)
        api_key_for_upload = api_key if api_key and api_key.strip() and api_key != "<your_fal_api_key_here>" else None
        
        # Upload input image
        image_url = ImageUtils.upload_image(image, api_key=api_key_for_upload)
        if not image_url:
            print(f"Error: Failed to upload image for {model_name}")
            return ResultProcessor.create_blank_image()
        
        # Process mask - convert MASK to IMAGE if needed
        try:
            if isinstance(mask, torch.Tensor):
                # Check if it's a mask tensor (2D or 3D without color channels)
                if mask.dim() == 2 or (mask.dim() == 3 and mask.shape[0] == 1):
                    mask_image = ImageUtils.mask_to_image(mask)
                    mask_url = ImageUtils.upload_image(mask_image, api_key=api_key_for_upload)
                else:
                    # If it's already an IMAGE tensor (4D with batch dimension)
                    mask_url = ImageUtils.upload_image(mask, api_key=api_key_for_upload)
            else:
                # If it's already an IMAGE, upload directly
                mask_url = ImageUtils.upload_image(mask, api_key=api_key_for_upload)
        except Exception as e:
            print(f"Error processing mask: {str(e)}")
            # Fallback: try uploading directly
            mask_url = ImageUtils.upload_image(mask, api_key=api_key_for_upload)
        
        if not mask_url:
            print(f"Error: Failed to upload mask for {model_name}")
            return ResultProcessor.create_blank_image()

        # Build API arguments according to fal.ai Qwen Image Edit API schema
        arguments = {
            "prompt": prompt,
            "image_url": image_url,
            "mask_url": mask_url,
        }
        
        # Add acceleration only if it's not "none" (some APIs may not accept "none")
        if acceleration and acceleration != "none":
            arguments["acceleration"] = acceleration

        # Submit request to FAL API
        try:
            result = ApiHandler.submit_and_get_result(
                "fal-ai/qwen-image-edit/inpaint", arguments, api_key=api_key_for_upload
            )
            return ResultProcessor.process_image_result(result)
        except Exception as e:
            print(f"Error in {model_name}: {str(e)}")
            return ApiHandler.handle_image_generation_error(model_name, e)


# Node mapping for ComfyUI
NODE_CLASS_MAPPINGS = {
    "QwenImageEditInpaint": QwenImageEditInpaint,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "QwenImageEditInpaint": "Qwen Image Edit Inpaint",
}

