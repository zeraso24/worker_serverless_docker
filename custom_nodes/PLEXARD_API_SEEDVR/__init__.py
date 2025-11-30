import importlib
import traceback

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

try:
    imported_module = importlib.import_module(".PLEXARD_FAL_nano_banana_edit", __name__)
    NODE_CLASS_MAPPINGS = {**NODE_CLASS_MAPPINGS, **imported_module.NODE_CLASS_MAPPINGS}
    NODE_DISPLAY_NAME_MAPPINGS = {
        **NODE_DISPLAY_NAME_MAPPINGS,
        **imported_module.NODE_DISPLAY_NAME_MAPPINGS,
    }
    print(f"✓ PLEXARD_FAL_nano_banana_edit: Loaded {len(imported_module.NODE_CLASS_MAPPINGS)} node(s)")
except Exception as e:
    print(f"✗ PLEXARD_FAL_nano_banana_edit: Failed to load nodes")
    print(f"  Error: {str(e)}")
    traceback.print_exc()

try:
    imported_module = importlib.import_module(".PLEXARD_FAL_qwen_image_edit_inpaint", __name__)
    NODE_CLASS_MAPPINGS = {**NODE_CLASS_MAPPINGS, **imported_module.NODE_CLASS_MAPPINGS}
    NODE_DISPLAY_NAME_MAPPINGS = {
        **NODE_DISPLAY_NAME_MAPPINGS,
        **imported_module.NODE_DISPLAY_NAME_MAPPINGS,
    }
    print(f"✓ PLEXARD_FAL_qwen_image_edit_inpaint: Loaded {len(imported_module.NODE_CLASS_MAPPINGS)} node(s)")
except Exception as e:
    print(f"✗ PLEXARD_FAL_qwen_image_edit_inpaint: Failed to load nodes")
    print(f"  Error: {str(e)}")
    traceback.print_exc()

try:
    imported_module = importlib.import_module(".PLEXARD_FAL_seedvr_upscale", __name__)
    NODE_CLASS_MAPPINGS = {**NODE_CLASS_MAPPINGS, **imported_module.NODE_CLASS_MAPPINGS}
    NODE_DISPLAY_NAME_MAPPINGS = {
        **NODE_DISPLAY_NAME_MAPPINGS,
        **imported_module.NODE_DISPLAY_NAME_MAPPINGS,
    }
    print(f"✓ PLEXARD_FAL_seedvr_upscale: Loaded {len(imported_module.NODE_CLASS_MAPPINGS)} node(s)")
except Exception as e:
    print(f"✗ PLEXARD_FAL_seedvr_upscale: Failed to load nodes")
    print(f"  Error: {str(e)}")
    traceback.print_exc()

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
