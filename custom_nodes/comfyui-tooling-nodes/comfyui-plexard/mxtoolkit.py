import torch
import numpy as np
from PIL import Image
import math

class PlexardNode:
    """
    Nodo Plexard que redimensiona una imagen manteniendo aspect ratio y asegurando
    que las dimensiones sean múltiplos de un valor específico.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "target_size": ("INT", {
                    "default": 4096,
                    "min": 64,
                    "max": 8192,
                    "step": 64
                }),
                "multiple": ("INT", {
                    "default": 64,
                    "min": 1,
                    "max": 256,
                    "step": 1
                }),
            },
            "optional": {
                "max_width": ("INT", {
                    "default": 4096,
                    "min": 64,
                    "max": 8192,
                    "step": 64
                }),
                "max_height": ("INT", {
                    "default": 4096,
                    "min": 64,
                    "max": 8192,
                    "step": 64
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "INT", "INT", "FLOAT")
    RETURN_NAMES = ("image", "width", "height", "scale_factor")
    FUNCTION = "resize_image"
    CATEGORY = "image/processing"
    
    def resize_image(self, image, target_size, multiple, max_width=4096, max_height=4096):
        """
        Redimensiona la imagen manteniendo aspect ratio y asegurando múltiplos.
        """
        # Obtener dimensiones originales
        batch_size, height, width, channels = image.shape
        original_width = width
        original_height = height
        
        # Calcular aspect ratio
        aspect_ratio = original_width / original_height
        
        # Calcular dimensiones objetivo manteniendo aspect ratio
        if original_width >= original_height:
            # Imagen más ancha que alta
            target_width = min(target_size, max_width)
            target_height = int(target_width / aspect_ratio)
        else:
            # Imagen más alta que ancha
            target_height = min(target_size, max_height)
            target_width = int(target_height * aspect_ratio)
        
        # Ajustar a múltiplos
        final_width = self._adjust_to_multiple(target_width, multiple, max_width)
        final_height = self._adjust_to_multiple(target_height, multiple, max_height)
        
        # Recalcular para mantener aspect ratio lo mejor posible
        final_width, final_height = self._optimize_dimensions(
            final_width, final_height, aspect_ratio, multiple, max_width, max_height
        )
        
        # Calcular factor de escala
        scale_factor = min(final_width / original_width, final_height / original_height)
        
        # Redimensionar la imagen
        resized_image = torch.nn.functional.interpolate(
            image.permute(0, 3, 1, 2),  # Cambiar a (batch, channels, height, width)
            size=(final_height, final_width),
            mode='bilinear',
            align_corners=False
        ).permute(0, 2, 3, 1)  # Volver a (batch, height, width, channels)
        
        return (resized_image, final_width, final_height, scale_factor)
    
    def _adjust_to_multiple(self, value, multiple, max_value):
        """
        Ajusta un valor para que sea múltiplo del número especificado,
        sin exceder el valor máximo.
        """
        # Encontrar el múltiplo más cercano hacia abajo
        adjusted = (value // multiple) * multiple
        
        # Si el valor ajustado es 0, usar el múltiplo mínimo
        if adjusted == 0:
            adjusted = multiple
        
        # Asegurar que no exceda el máximo
        return min(adjusted, max_value)
    
    def _optimize_dimensions(self, width, height, aspect_ratio, multiple, max_width, max_height):
        """
        Optimiza las dimensiones para mantener el aspect ratio lo mejor posible
        mientras respeta los múltiplos y límites máximos.
        """
        # Intentar ajustar el ancho primero
        optimal_width = self._adjust_to_multiple(width, multiple, max_width)
        optimal_height = int(optimal_width / aspect_ratio)
        optimal_height = self._adjust_to_multiple(optimal_height, multiple, max_height)
        
        # Recalcular el ancho basado en la altura ajustada
        optimal_width = int(optimal_height * aspect_ratio)
        optimal_width = self._adjust_to_multiple(optimal_width, multiple, max_width)
        
        # Verificar que no excedamos los límites
        if optimal_width > max_width:
            optimal_width = self._adjust_to_multiple(max_width, multiple, max_width)
            optimal_height = int(optimal_width / aspect_ratio)
            optimal_height = self._adjust_to_multiple(optimal_height, multiple, max_height)
        
        if optimal_height > max_height:
            optimal_height = self._adjust_to_multiple(max_height, multiple, max_height)
            optimal_width = int(optimal_height * aspect_ratio)
            optimal_width = self._adjust_to_multiple(optimal_width, multiple, max_width)
        
        return optimal_width, optimal_height

# Mapeo de nodos
NODE_CLASS_MAPPINGS = {
    "PlexardNode": PlexardNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PlexardNode": "Plexard Node",
}
