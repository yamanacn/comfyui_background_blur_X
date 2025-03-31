"""
ComfyUI深度感知背景虚化插件
自动识别图像中的前景和背景，对背景应用虚化效果，同时保持前景清晰
"""

from .depth_background_blur_node import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS'] 