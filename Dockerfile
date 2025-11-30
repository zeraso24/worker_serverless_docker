# start from a clean base image (replace <version> with the desired release)
FROM runpod/worker-comfyui:5.1.0-base

# Copy local custom nodes to ComfyUI custom_nodes directory
COPY custom_nodes/ /comfyui/custom_nodes/
