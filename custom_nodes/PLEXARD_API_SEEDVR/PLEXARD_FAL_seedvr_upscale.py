"""
SeedVR2 Image Upscale Node
A dedicated node for SeedVR2 image upscaling functionality.
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


# ==================== SEEDVR2 IMAGE UPSCALE NODE ====================


class SeedVR2Upscale:
    """
    SeedVR2 Image Upscale Node
    Performs image upscaling using SeedVR2 model.
    Requires: image.
    Supports factor-based or target resolution-based upscaling.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Image to be upscaled"}),
            },
            "optional": {
                "api_key": ("STRING", {"default": "", "tooltip": "FAL API key (optional, overrides config.ini and environment variable)"}),
                "upscale_mode": (["factor", "target"], {"default": "factor", "tooltip": "The mode to use for the upscale. 'factor' uses upscale_factor, 'target' uses target_resolution"}),
                "upscale_factor": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 8.0, "step": 0.1, "tooltip": "Upscaling factor to be used when upscale_mode is 'factor'"}),
                "target_resolution": (["720p", "1080p", "1440p", "2160p"], {"default": "1080p", "tooltip": "The target resolution to upscale to when upscale_mode is 'target'"}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 2147483647, "tooltip": "Random seed used for the generation process (-1 = random)"}),
                "noise_scale": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "The noise scale to use for the generation process"}),
                "output_format": (["png", "jpg", "webp"], {"default": "jpg", "tooltip": "The format of the output image"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "upscale"
    CATEGORY = "plexard_FAL"
    DESCRIPTION = "SeedVR2 Image Upscale - Upscale images using SeedVR2 model"

    def upscale(
        self,
        image,
        api_key="",
        upscale_mode="factor",
        upscale_factor=2.0,
        target_resolution="1080p",
        seed=-1,
        noise_scale=0.1,
        output_format="jpg",
    ):
        """
        Perform image upscaling using SeedVR2 model.
        
        Args:
            image: Input image tensor (required)
            api_key: FAL API key (optional, overrides config.ini and environment variable)
            upscale_mode: Mode for upscaling - "factor" or "target" (default: "factor")
            upscale_factor: Upscaling factor when mode is "factor" (default: 2.0)
            target_resolution: Target resolution when mode is "target" (default: "1080p")
            seed: Random seed for generation (-1 = random, default: -1)
            noise_scale: Noise scale for generation process (default: 0.1)
            output_format: Output image format - "png", "jpg", or "webp" (default: "jpg")
            
        Returns:
            Tuple containing the upscaled image tensor
        """
        model_name = "SeedVR2 Image Upscale"
        
        # Validate and sanitize input parameters
        try:
            # Validate upscale_mode
            valid_modes = ["factor", "target"]
            if upscale_mode not in valid_modes:
                upscale_mode = "factor"
            
            # Validate upscale_factor
            if upscale_factor is None or not isinstance(upscale_factor, (int, float)):
                upscale_factor = 2.0
            upscale_factor = float(max(1.0, min(8.0, upscale_factor)))
            
            # Validate target_resolution
            valid_resolutions = ["720p", "1080p", "1440p", "2160p"]
            if target_resolution not in valid_resolutions:
                target_resolution = "1080p"
            
            # Validate noise_scale
            if noise_scale is None or not isinstance(noise_scale, (int, float)):
                noise_scale = 0.1
            noise_scale = float(max(0.0, min(1.0, noise_scale)))
            
            # Validate output_format
            valid_formats = ["png", "jpg", "webp"]
            if output_format not in valid_formats:
                output_format = "jpg"
            
            # Validate seed
            if seed == -1:
                seed = None  # API will use random seed
            elif seed is not None:
                seed = int(max(-1, min(2147483647, seed)))
        except Exception as e:
            print(f"Warning: Error validating parameters: {str(e)}, using defaults")
            upscale_mode = "factor"
            upscale_factor = 2.0
            target_resolution = "1080p"
            noise_scale = 0.1
            output_format = "jpg"
            seed = None
        
        # Get API key for uploads (priority: node parameter > environment > config.ini)
        api_key_for_upload = api_key if api_key and api_key.strip() and api_key != "<your_fal_api_key_here>" else None
        
        # Upload input image
        image_url = ImageUtils.upload_image(image, api_key=api_key_for_upload)
        if not image_url:
            print(f"Error: Failed to upload image for {model_name}")
            return ResultProcessor.create_blank_image()

        # Build API arguments according to fal.ai SeedVR2 Upscale API schema
        arguments = {
            "image_url": image_url,
            "upscale_mode": upscale_mode,
            "upscale_factor": upscale_factor,
            "target_resolution": target_resolution,
            "noise_scale": noise_scale,
            "output_format": output_format,
        }
        
        # Add seed only if provided (not -1)
        if seed is not None:
            arguments["seed"] = seed

        # Submit request to FAL API
        try:
            result = ApiHandler.submit_and_get_result(
                "fal-ai/seedvr/upscale/image", arguments, api_key=api_key_for_upload
            )
            return ResultProcessor.process_single_image_result(result)
        except Exception as e:
            print(f"Error in {model_name}: {str(e)}")
            return ApiHandler.handle_image_generation_error(model_name, e)


# Node mapping for ComfyUI
NODE_CLASS_MAPPINGS = {
    "SeedVR2Upscale": SeedVR2Upscale,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SeedVR2Upscale": "SeedVR2 Image Upscale",
}

