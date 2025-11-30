# Skin Texture Modifier Node for ComfyUI

Este nodo permite modificar prompts de manera granular para controlar diferentes aspectos de la textura de la piel en generación de imágenes.

## Características

- **8 categorías de modificadores**: Poros, Vello facial, Pecas, Arrugas, Tono de piel, Manchas, Enrojecimiento, y Aspereza
- **4 niveles por categoría**: Off, Bajo, Medio, Alto
- **Compatibilidad con SeedDream**: Se integra perfectamente con el sistema de prompts de SeedDream
- **Estado inicial**: Todos los modificadores desactivados por defecto
- **Prompts en inglés**: Todas las descripciones están traducidas al inglés

## Uso

1. Conecta el nodo a tu workflow de ComfyUI
2. Selecciona el nivel deseado para cada categoría de textura de piel
3. Opcionalmente, proporciona un prompt base
4. El nodo generará un prompt modificado que puedes usar con SeedDream

## Categorías Disponibles

### Poros (Pores)
- **Bajo**: "smooth skin texture, minimal visible pores"
- **Medio**: "natural skin texture, slightly visible pores"
- **Alto**: "detailed skin texture, prominent visible pores"

### Vello Facial (Facial Hair)
- **Bajo**: "clean-shaven, smooth skin"
- **Medio**: "light facial hair, subtle stubble"
- **Alto**: "visible facial hair, natural hair texture"

### Pecas (Freckles)
- **Bajo**: "clear complexion, minimal freckles"
- **Medio**: "natural freckles, scattered across face"
- **Alto**: "prominent freckles, abundant across face"

### Arrugas (Wrinkles)
- **Bajo**: "youthful skin, fine lines"
- **Medio**: "mature skin, moderate wrinkles"
- **Alto**: "aged skin, deep wrinkles and lines"

### Tono de Piel (Skin Tone)
- **Bajo**: "even skin tone, uniform complexion"
- **Medio**: "natural skin tone, slight variations"
- **Alto**: "diverse skin tone, natural variations"

### Manchas (Blemishes)
- **Bajo**: "clear skin, minimal imperfections"
- **Medio**: "natural skin, some blemishes"
- **Alto**: "realistic skin, visible imperfections"

### Enrojecimiento (Redness)
- **Bajo**: "neutral skin tone, no redness"
- **Medio**: "natural skin tone, slight redness"
- **Alto**: "flushed skin, visible redness"

### Aspereza (Roughness)
- **Bajo**: "soft skin texture, smooth feel"
- **Medio**: "natural skin texture, moderate roughness"
- **Alto**: "rough skin texture, weathered appearance"

## Instalación

1. Copia la carpeta `comfy-plexard_switches` a tu directorio de custom nodes de ComfyUI
2. Reinicia ComfyUI
3. El nodo aparecerá en la categoría "Prompt Modifiers"
