# PLEXARD FAL Flux Stitch

Nodo personalizado de ComfyUI para inpainting avanzado usando **FLUX.1 Kontext LoRA**.

## 🎯 Características

- ✅ **Completamente independiente** - No requiere otros custom nodes
- 🎨 Inpainting con imágenes de referencia
- 🔧 Soporte para hasta 2 LoRAs simultáneos
- ⚙️ Control completo de parámetros (steps, guidance, strength, etc.)
- 🔐 Safety checker integrado

## 📦 Instalación

### Requisitos

Instala las dependencias necesarias:

```bash
pip install -r requirements.txt
```

O manualmente:
```bash
pip install fal-client torch Pillow requests numpy
```

### Configuración de API Key

1. **Opción A:** Edita el archivo `config.ini`:
   ```ini
   [API]
   FAL_KEY = tu_api_key_aqui
   ```

2. **Opción B:** Configura una variable de entorno:
   ```bash
   # Windows PowerShell
   $env:FAL_KEY = "tu_api_key_aqui"
   
   # Windows CMD
   set FAL_KEY=tu_api_key_aqui
   
   # Linux/Mac
   export FAL_KEY="tu_api_key_aqui"
   ```

🔑 Obtén tu API key en: https://fal.ai/dashboard/keys

## 🚀 Uso

### Inputs Requeridos

- **image**: Imagen base a modificar
- **mask**: Máscara indicando áreas a inpaint (blanco = inpaint, negro = mantener)
- **reference_image**: Imagen de referencia para guiar el inpainting
- **prompt**: Descripción del resultado deseado

### Inputs Opcionales

| Parámetro | Tipo | Default | Rango | Descripción |
|-----------|------|---------|-------|-------------|
| `num_inference_steps` | INT | 30 | 1-100 | Pasos de inferencia |
| `guidance_scale` | FLOAT | 2.5 | 0.0-20.0 | Escala de CFG |
| `strength` | FLOAT | 0.88 | 0.0-1.0 | Fuerza del inpainting |
| `num_images` | INT | 1 | 1-4 | Número de imágenes a generar |
| `seed` | INT | -1 | -1 a 2³² | Seed para reproducibilidad (-1 = aleatorio) |
| `enable_safety_checker` | BOOL | True | - | Activar safety checker |
| `output_format` | CHOICE | png | png/jpeg | Formato de salida |
| `acceleration` | CHOICE | none | none/regular/high | Nivel de aceleración |
| `sync_mode` | BOOL | False | - | Modo síncrono |
| `lora_path_1` | STRING | "" | - | Ruta o URL del primer LoRA |
| `lora_scale_1` | FLOAT | 1.0 | 0.0-2.0 | Escala del primer LoRA |
| `lora_path_2` | STRING | "" | - | Ruta o URL del segundo LoRA |
| `lora_scale_2` | FLOAT | 1.0 | 0.0-2.0 | Escala del segundo LoRA |

### Output

- **image**: Imagen resultante del inpainting

## 💡 Ejemplos de Uso

### Ejemplo Básico
```
1. Carga una imagen
2. Crea una máscara de las áreas a modificar
3. Proporciona una imagen de referencia
4. Escribe un prompt descriptivo
5. Ajusta parámetros según necesites
6. Ejecuta el nodo
```

### Con LoRAs
Puedes usar hasta 2 LoRAs para personalizar el estilo:
- **lora_path_1**: URL o ruta del primer LoRA
- **lora_scale_1**: Intensidad del efecto (0.0 = sin efecto, 2.0 = máximo)

## 🔧 Troubleshooting

### El nodo no aparece en ComfyUI

1. Verifica que las dependencias estén instaladas:
   ```bash
   pip list | findstr "fal-client torch Pillow"
   ```

2. Revisa la consola de ComfyUI al iniciar para ver errores

3. Asegúrate de que el archivo `config.ini` existe y tiene tu API key

### Error de API Key

```
Error: FAL_KEY not found in config.ini or environment variables
```

**Solución:** Configura tu API key en `config.ini` o como variable de entorno.

### Error de conexión

Si falla la conexión a FAL API, verifica:
- Tu conexión a Internet
- Que tu API key sea válida
- Que tengas créditos disponibles en tu cuenta de FAL

## 📝 Notas

- El nodo sube temporalmente las imágenes a FAL para procesamiento
- Las imágenes temporales se eliminan automáticamente después de usarse
- Si `seed = -1`, cada ejecución producirá resultados diferentes
- El safety checker puede rechazar contenido inapropiado

## 📄 Licencia

Este nodo es independiente y no requiere atribución a otros proyectos.

## 🔗 Enlaces

- FAL AI Dashboard: https://fal.ai/dashboard
- FAL API Keys: https://fal.ai/dashboard/keys
- Documentación de FLUX: https://fal.ai/models/fal-ai/flux-kontext-lora

---

**Creado para ComfyUI**

