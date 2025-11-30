import torch
from PIL import Image
import numpy as np

class SkinModifierNode:
    """
    ComfyUI node for skin texture modifiers that can be toggled on/off
    Compatible with SeedDream prompt system
    """
    
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pores_level": (["off", "low", "medium", "high"], {"default": "off"}),
                "facial_hair_level": (["off", "low", "medium", "high"], {"default": "off"}),
                "freckles_level": (["off", "low", "medium", "high"], {"default": "off"}),
                "wrinkles_level": (["off", "low", "medium", "high"], {"default": "off"}),
                "skin_tone_level": (["off", "low", "medium", "high"], {"default": "off"}),
                "blemishes_level": (["off", "low", "medium", "high"], {"default": "off"}),
                "redness_level": (["off", "low", "medium", "high"], {"default": "off"}),
                "roughness_level": (["off", "low", "medium", "high"], {"default": "off"}),
            },
            "optional": {
                "base_prompt": ("STRING", {"default": "", "multiline": True}),
                "pores_custom": ("STRING", {"default": "", "multiline": True}),
                "facial_hair_custom": ("STRING", {"default": "", "multiline": True}),
                "freckles_custom": ("STRING", {"default": "", "multiline": True}),
                "wrinkles_custom": ("STRING", {"default": "", "multiline": True}),
                "skin_tone_custom": ("STRING", {"default": "", "multiline": True}),
                "blemishes_custom": ("STRING", {"default": "", "multiline": True}),
                "redness_custom": ("STRING", {"default": "", "multiline": True}),
                "roughness_custom": ("STRING", {"default": "", "multiline": True}),
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("modified_prompt", "generic_prompt", "added_prompts")
    FUNCTION = "modify_prompt"
    CATEGORY = "Prompt Modifiers"
    
    def modify_prompt(self, pores_level, facial_hair_level, freckles_level, 
                     wrinkles_level, skin_tone_level, blemishes_level, 
                     redness_level, roughness_level, base_prompt="",
                     pores_custom="", facial_hair_custom="", freckles_custom="",
                     wrinkles_custom="", skin_tone_custom="", blemishes_custom="",
                     redness_custom="", roughness_custom=""):
        
        # Define the prompt modifiers in English
        modifiers = {
            "pores": {
                "low": "smooth skin texture, minimal visible pores",
                "medium": "natural skin texture, slightly visible pores", 
                "high": "detailed skin texture, prominent visible pores"
            },
            "facial_hair": {
                "low": "clean-shaven, smooth skin",
                "medium": "light facial hair, subtle stubble",
                "high": "visible facial hair, natural hair texture"
            },
            "freckles": {
                "low": "clear complexion, minimal freckles",
                "medium": "natural freckles, scattered across face",
                "high": "prominent freckles, abundant across face"
            },
            "wrinkles": {
                "low": "youthful skin, fine lines",
                "medium": "marked wrinkles expresions lines visible",
                "high": "deep wrinkles and lines visible"
            },
            "skin_tone": {
                "low": "even skin tone, uniform complexion",
                "medium": "natural skin tone, slight variations",
                "high": "diverse skin tone, natural variations"
            },
            "blemishes": {
                "low": " minimal imperfections",
                "medium": "some imperfections on skin",
                "high": "visible imperfections"
            },
            "redness": {
                "low": "skin with visible redness",
                "medium": "slight soft redness ",
                "high": "flushed skin, visible redness"
            },
            "roughness": {
                "low": "soft skin texture, smooth feel",
                "medium": "natural skin texture, moderate roughness",
                "high": "rough skin texture, weathered appearance"
            }
        }
        
        # Collect active modifiers with custom prompts
        active_modifiers = []
        added_prompts = []
        
        # Helper function to get the right prompt (custom or default)
        def get_prompt(category, level, custom_text):
            if custom_text.strip():
                return custom_text.strip()
            else:
                return modifiers[category][level]
        
        # Check each category and add to prompt if not "off"
        if pores_level != "off":
            prompt_text = get_prompt("pores", pores_level, pores_custom)
            full_prompt = f"important to add: {prompt_text}"
            active_modifiers.append(full_prompt)
            added_prompts.append(f"pores ({pores_level}): {full_prompt}")
            
        if facial_hair_level != "off":
            prompt_text = get_prompt("facial_hair", facial_hair_level, facial_hair_custom)
            full_prompt = f"important to add: {prompt_text}"
            active_modifiers.append(full_prompt)
            added_prompts.append(f"facial_hair ({facial_hair_level}): {full_prompt}")
            
        if freckles_level != "off":
            prompt_text = get_prompt("freckles", freckles_level, freckles_custom)
            full_prompt = f"important to add: {prompt_text}"
            active_modifiers.append(full_prompt)
            added_prompts.append(f"freckles ({freckles_level}): {full_prompt}")
            
        if wrinkles_level != "off":
            prompt_text = get_prompt("wrinkles", wrinkles_level, wrinkles_custom)
            full_prompt = f"important to add: {prompt_text}"
            active_modifiers.append(full_prompt)
            added_prompts.append(f"wrinkles ({wrinkles_level}): {full_prompt}")
            
        if skin_tone_level != "off":
            prompt_text = get_prompt("skin_tone", skin_tone_level, skin_tone_custom)
            full_prompt = f"important to add: {prompt_text}"
            active_modifiers.append(full_prompt)
            added_prompts.append(f"skin_tone ({skin_tone_level}): {full_prompt}")
            
        if blemishes_level != "off":
            prompt_text = get_prompt("blemishes", blemishes_level, blemishes_custom)
            full_prompt = f"important to add: {prompt_text}"
            active_modifiers.append(full_prompt)
            added_prompts.append(f"blemishes ({blemishes_level}): {full_prompt}")
            
        if redness_level != "off":
            prompt_text = get_prompt("redness", redness_level, redness_custom)
            full_prompt = f"important to add: {prompt_text}"
            active_modifiers.append(full_prompt)
            added_prompts.append(f"redness ({redness_level}): {full_prompt}")
            
        if roughness_level != "off":
            prompt_text = get_prompt("roughness", roughness_level, roughness_custom)
            full_prompt = f"important to add: {prompt_text}"
            active_modifiers.append(full_prompt)
            added_prompts.append(f"roughness ({roughness_level}): {full_prompt}")
        
        # Prepare outputs
        generic_prompt = base_prompt or "No base prompt provided"
        added_prompts_text = "\n".join(added_prompts) if added_prompts else "No modifiers selected yet"
        
        # Combine base prompt with modifiers
        if base_prompt and active_modifiers:
            modified_prompt = f"{base_prompt}, {', '.join(active_modifiers)}"
        elif active_modifiers:
            modified_prompt = ', '.join(active_modifiers)
        else:
            modified_prompt = base_prompt
            
        return (modified_prompt, generic_prompt, added_prompts_text)

# Node mapping for ComfyUI
NODE_CLASS_MAPPINGS = {
    "SkinModifierNode": SkinModifierNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SkinModifierNode": "Skin Texture Modifier"
}
