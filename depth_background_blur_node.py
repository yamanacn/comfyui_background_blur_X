import torch
import numpy as np
import cv2
from PIL import Image

"""
深度感知背景虚化插件
自动识别图像中的前景和背景，对背景应用虚化效果，同时保持前景清晰
可选输入用户掩码来保护特定区域不被虚化
支持多种专业摄影模糊效果
"""

class DepthBackgroundBlur:
    """
    深度感知背景虚化节点
    支持多种摄影级模糊效果
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "depth_map": ("IMAGE",),
                "blur_strength": ("FLOAT", {
                    "default": 5.0,
                    "min": 0.0,
                    "max": 100.0,
                    "step": 0.5,
                }),
                "blur_style": (["高斯模糊", "散景效果", "动态模糊", "方框模糊", 
                              "方向性模糊", "双重曝光", "棱镜模糊"], {
                    "default": "散景效果"
                }),
                "foreground_threshold": ("FLOAT", {
                    "default": 0.3,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                }),
                "transition_smoothness": ("FLOAT", {
                    "default": 0.2,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                }),
                "mask_strength": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                }),
                "effect_angle": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 360.0,
                    "step": 5.0,
                }),
                "effect_center_x": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                }),
                "effect_center_y": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                }),
                "effect_gradient": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                }),
                "effect_intensity": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 3.0,
                    "step": 0.1,
                }),
            },
            "optional": {
                "mask": ("MASK",),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_background_blur"
    CATEGORY = "image/processing"
    
    def apply_background_blur(self, image, depth_map, blur_strength, blur_style, 
                            foreground_threshold, transition_smoothness, 
                            mask_strength, effect_angle=0.0, effect_center_x=0.5, 
                            effect_center_y=0.5, effect_gradient=0.5, 
                            effect_intensity=1.0, mask=None):
        """
        应用深度感知背景虚化效果
        
        参数:
            image: 输入图像
            depth_map: 深度图
            blur_strength: 背景模糊强度
            blur_style: 虚化方式
            foreground_threshold: 前景识别阈值
            transition_smoothness: 过渡平滑度
            mask_strength: 掩码强度
            effect_angle: 效果角度 (用于方向性模糊等)
            effect_center_x: 效果中心X坐标 (0-1, 用于径向模糊等)
            effect_center_y: 效果中心Y坐标 (0-1, 用于径向模糊等)
            effect_gradient: 效果渐变强度 (用于渐变模糊等)
            effect_intensity: 效果整体强度
            mask: 可选的保护掩码
            
        返回:
            处理后的图像
        """
        # 将输入图像和深度图从ComfyUI格式转换为OpenCV格式
        processed_image = self._process_comfy_image(image)
        processed_depth = self._process_comfy_depth(depth_map)
        
        # 输出输入图像和深度图的形状信息
        print(f"输入图像形状: {processed_image.shape}")
        print(f"深度图形状: {processed_depth.shape}")
        
        # 处理掩码（如果提供）
        final_mask = self._prepare_mask(processed_image, processed_depth, 
                                       foreground_threshold, transition_smoothness, 
                                       mask, mask_strength)
        
        # 准备额外的效果参数
        effect_params = {
            'effect_angle': effect_angle,
            'effect_center_x': effect_center_x,
            'effect_center_y': effect_center_y,
            'effect_gradient': effect_gradient,
            'effect_intensity': effect_intensity,
            'image_height': processed_image.shape[0],
            'image_width': processed_image.shape[1]
        }
                
        # 应用背景虚化
        blurred_image = self._apply_blur(processed_image, blur_style, blur_strength, **effect_params)
        
        # 合成最终图像
        result = self._blend_images(processed_image, blurred_image, final_mask)
        
        # 转换回ComfyUI格式并返回
        return (self._to_comfy_image(result),)
    
    def _process_comfy_image(self, comfy_image):
        """将ComfyUI图像格式转换为OpenCV格式"""
        # ComfyUI图像格式通常是torch.Tensor，形状为(B, H, W, C)，值范围0-1
        # 转换为numpy数组，形状为(H, W, C)，值范围0-255
        if isinstance(comfy_image, torch.Tensor):
            # 获取第一帧（如果是批处理的话）
            if len(comfy_image.shape) == 4:
                image_np = comfy_image[0].cpu().numpy()
            else:
                image_np = comfy_image.cpu().numpy()
            
            # 确保值范围在0-1之间
            if image_np.max() <= 1.0:
                image_np = (image_np * 255).astype(np.uint8)
            else:
                image_np = image_np.astype(np.uint8)
            
            # 如果是RGB格式，转换为BGR（OpenCV使用BGR）
            if image_np.shape[2] == 3:
                return cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            return image_np
        else:
            # 如果已经是numpy数组，直接返回
            return comfy_image
    
    def _process_comfy_depth(self, comfy_depth):
        """处理深度图，确保其为单通道并与图像大小匹配"""
        depth_np = self._process_comfy_image(comfy_depth)
        
        # 如果是BGR或RGB格式，转换为灰度图
        if len(depth_np.shape) == 3 and depth_np.shape[2] > 1:
            depth_np = cv2.cvtColor(depth_np, cv2.COLOR_BGR2GRAY)
        
        # 确保深度图为单通道
        if len(depth_np.shape) > 2:
            depth_np = depth_np[:, :, 0]
        
        # 归一化深度图到0-255范围
        if depth_np.max() > 0:
            depth_np = ((depth_np - depth_np.min()) / (depth_np.max() - depth_np.min()) * 255).astype(np.uint8)
        
        return depth_np
    
    def _prepare_mask(self, image, depth_map, threshold, smoothness, user_mask=None, mask_strength=1.0):
        """准备最终的处理掩码，结合深度图自动生成的掩码和用户提供的掩码"""
        # 基于深度图创建自动前景掩码
        auto_mask = self._create_depth_mask(depth_map, threshold)
        
        # 应用平滑过渡
        if smoothness > 0:
            # 计算模糊半径，确保至少为1
            blur_radius = max(1, int(smoothness * 30))
            # 应用高斯模糊使边缘平滑
            auto_mask = cv2.GaussianBlur(auto_mask, (blur_radius * 2 + 1, blur_radius * 2 + 1), 0)
        
        # 如果提供了用户掩码，处理并与自动掩码结合
        if user_mask is not None:
            processed_user_mask = self._process_user_mask(user_mask, image.shape[:2])
            
            # 根据mask_strength结合掩码
            if mask_strength >= 1.0:
                # 用户掩码完全覆盖自动掩码
                final_mask = processed_user_mask
            else:
                # 根据强度混合两个掩码
                final_mask = cv2.addWeighted(
                    auto_mask, 1 - mask_strength,
                    processed_user_mask, mask_strength,
                    0
                )
        else:
            final_mask = auto_mask
        
        # 确保掩码形状与输入图像匹配，并且是3通道（如果输入图像是3通道）
        if len(image.shape) == 3 and image.shape[2] == 3:
            if len(final_mask.shape) == 2:
                # 扩展为3通道掩码
                final_mask = np.repeat(final_mask[:, :, np.newaxis], 3, axis=2)
        
        return final_mask
    
    def _create_depth_mask(self, depth_map, threshold):
        """根据深度图创建前景/背景分离掩码"""
        # 归一化深度图
        if depth_map.max() > 0:  # 避免除以零
            normalized_depth = depth_map / 255.0
        else:
            normalized_depth = np.zeros_like(depth_map, dtype=np.float32)
        
        # 创建二值掩码，值大于阈值的被视为前景
        # 深度图中，较亮的区域（高值）通常代表较近的物体
        foreground_mask = (normalized_depth > threshold).astype(np.uint8) * 255
        
        return foreground_mask
    
    def _process_user_mask(self, user_mask, target_shape):
        """处理用户提供的掩码，确保其格式和大小正确"""
        # 处理ComfyUI掩码格式
        if isinstance(user_mask, torch.Tensor):
            # ComfyUI的掩码通常是单通道的
            mask_np = user_mask.cpu().numpy()
            
            # 如果是批处理的，取第一帧
            if len(mask_np.shape) == 3:
                mask_np = mask_np[0]
            
            # 确保值范围为0-255
            if mask_np.max() <= 1.0:
                mask_np = (mask_np * 255).astype(np.uint8)
        else:
            mask_np = user_mask
        
        # 调整大小以匹配目标形状
        if mask_np.shape[:2] != target_shape:
            mask_np = cv2.resize(mask_np, (target_shape[1], target_shape[0]), 
                              interpolation=cv2.INTER_LINEAR)
        
        return mask_np
    
    def _apply_blur(self, image, blur_style, blur_strength, **kwargs):
        """应用选定样式的模糊效果"""
        # 确保模糊强度至少为1（对于kernel size）
        k_size = max(1, int(blur_strength))
        
        # 分块处理非常大的图像
        if image.shape[1] > 1024:  # 如果图像宽度大于1024像素
            return self._apply_blur_chunked(image, blur_style, blur_strength, **kwargs)
        
        # 根据选择的模糊样式应用不同的模糊效果
        if blur_style == "高斯模糊":
            # 高斯模糊：简单平滑的模糊效果
            # kernel size需要是奇数
            k_size = k_size * 2 + 1
            return cv2.GaussianBlur(image, (k_size, k_size), 0)
            
        elif blur_style == "散景效果":
            # 散景效果：模拟相机光圈散景效果
            return self._apply_bokeh_effect(image, blur_strength)
            
        elif blur_style == "动态模糊":
            # 动态模糊：模拟运动模糊
            return self._apply_motion_blur(image, blur_strength, **kwargs)
            
        elif blur_style == "方框模糊":
            # 方框模糊：像素化的模糊效果
            k_size = k_size * 2 + 1
            return cv2.boxFilter(image, -1, (k_size, k_size))
            
        elif blur_style == "方向性模糊":
            # 方向性模糊：沿特定方向的模糊效果
            return self._apply_directional_blur(image, blur_strength, **kwargs)
            
        elif blur_style == "双重曝光":
            # 双重曝光：将图像与其模糊版本叠加
            return self._apply_double_exposure_blur(image, blur_strength, **kwargs)
            
        elif blur_style == "棱镜模糊":
            # 棱镜模糊：色彩通道分离并位移
            return self._apply_prism_blur(image, blur_strength, **kwargs)
            
        # 默认返回高斯模糊
        k_size = k_size * 2 + 1
        return cv2.GaussianBlur(image, (k_size, k_size), 0)
    
    def _apply_blur_chunked(self, image, blur_style, blur_strength, **kwargs):
        """分块处理大图像的模糊效果"""
        height, width = image.shape[:2]
        channels = 1 if len(image.shape) == 2 else image.shape[2]
        
        # 创建与原图像大小相同的输出图像
        result = np.zeros_like(image)
        
        # 定义块大小和重叠区域
        chunk_width = 512
        overlap = 32  # 重叠区域大小，用于避免块之间的边界效应
        
        for x in range(0, width, chunk_width - overlap * 2):
            # 计算当前块的起始和结束位置
            start_x = max(0, x - overlap if x > 0 else 0)
            end_x = min(width, x + chunk_width + overlap if x > 0 else chunk_width)
            
            # 提取当前块
            chunk = image[:, start_x:end_x].copy()
            
            # 对当前块应用模糊效果（递归调用，传递所有参数）
            blurred_chunk = self._apply_blur(chunk, blur_style, blur_strength, **kwargs)
            
            # 将模糊处理后的块放回结果图像
            # 如果是第一个块，直接复制
            if x == 0:
                result[:, start_x:end_x] = blurred_chunk
            else:
                # 如果不是第一个块，需要处理重叠区域
                # 计算需要混合的区域
                blend_start = start_x
                blend_end = min(start_x + overlap * 2, width)
                
                # 创建渐变权重掩码用于平滑过渡
                blend_mask = np.linspace(0, 1, blend_end - blend_start)
                
                # 根据通道数调整混合掩码
                if channels > 1:
                    blend_mask = np.repeat(blend_mask[np.newaxis, :, np.newaxis], channels, axis=2)
                else:
                    blend_mask = blend_mask[np.newaxis, :]
                
                # 混合重叠区域
                result[:, blend_start:blend_end] = (
                    result[:, blend_start:blend_end] * (1 - blend_mask) +
                    blurred_chunk[:, :blend_end-blend_start] * blend_mask
                )
                
                # 复制剩余的非重叠区域
                result[:, blend_end:end_x] = blurred_chunk[:, blend_end-blend_start:end_x-blend_start]
        
        return result
    
    def _apply_bokeh_effect(self, image, blur_strength):
        """应用散景效果模糊"""
        # 创建圆形卷积核来模拟散景效果
        k_size = max(3, int(blur_strength) * 2 + 1)
        kernel = np.zeros((k_size, k_size), np.uint8)
        center = k_size // 2
        radius = center
        cv2.circle(kernel, (center, center), radius, 1, -1)
        
        # 归一化内核
        kernel = kernel.astype(np.float32) / kernel.sum()
        
        # 应用滤波器
        try:
            # 尝试使用滤波器
            result = cv2.filter2D(image, -1, kernel)
            return result
        except Exception as e:
            print(f"应用散景效果时出错: {e}")
            # 如果失败，回退到高斯模糊
            return cv2.GaussianBlur(image, (k_size, k_size), 0)
    
    def _apply_motion_blur(self, image, blur_strength, **kwargs):
        """应用动态模糊效果，模拟运动"""
        # 获取效果参数
        angle = kwargs.get('effect_angle', 0)  # 运动方向
        intensity = kwargs.get('effect_intensity', 1.0)  # 效果强度
        
        # 创建运动模糊内核
        k_size = max(3, int(blur_strength * intensity)) * 2 + 1
        
        try:
            # 创建一个空的卷积核
            kernel = np.zeros((k_size, k_size))
            
            # 根据指定角度设置核中的值
            angle_rad = np.radians(angle)
            
            # 核的中心点
            center = k_size // 2
            
            # 在角度方向上设置一条线
            for i in range(k_size):
                # 计算沿角度方向的坐标
                x = int(center + (i - center) * np.cos(angle_rad))
                y = int(center + (i - center) * np.sin(angle_rad))
                
                # 确保坐标在核范围内
                if 0 <= x < k_size and 0 <= y < k_size:
                    kernel[y, x] = 1
            
            # 确保核不是空的
            if np.sum(kernel) == 0:
                kernel[center, center] = 1
            
            # 归一化核
            kernel = kernel / np.sum(kernel)
            
            # 应用卷积
            return cv2.filter2D(image, -1, kernel)
            
        except Exception as e:
            print(f"应用动态模糊时出错: {e}")
            
            # 如果上述方法失败，尝试传统的水平和垂直模糊组合
            try:
                # 水平方向的运动模糊
                kernel_h = np.zeros((k_size, k_size))
                kernel_h[k_size // 2, :] = 1.0 / k_size
                
                # 应用水平运动模糊
                blurred_h = cv2.filter2D(image, -1, kernel_h)
                
                # 垂直方向的运动模糊
                kernel_v = np.zeros((k_size, k_size))
                kernel_v[:, k_size // 2] = 1.0 / k_size
                blurred_v = cv2.filter2D(image, -1, kernel_v)
                
                # 结合水平和垂直模糊效果
                return cv2.addWeighted(blurred_h, 0.5, blurred_v, 0.5, 0)
            except:
                # 如果还是失败，回退到高斯模糊
                return cv2.GaussianBlur(image, (k_size, k_size), 0)
    
    def _apply_directional_blur(self, image, blur_strength, **kwargs):
        """应用方向性模糊，沿特定角度方向模糊"""
        # 获取效果参数
        angle = kwargs.get('effect_angle', 0)  # 模糊方向的角度
        intensity = kwargs.get('effect_intensity', 1.0)  # 效果强度
        
        # 调整模糊强度
        k_size = max(3, int(blur_strength * intensity))
        
        try:
            # 基于角度创建方向性核
            # 角度为0时表示水平方向，90表示垂直方向
            
            # 如果核大小太小，使用至少3x3的核
            k_size = max(3, k_size)
            
            # 创建一个空的卷积核
            kernel = np.zeros((k_size, k_size), dtype=np.float32)
            
            # 计算角度的弧度
            angle_rad = np.radians(angle)
            
            # 找到核的中心点
            center = k_size // 2
            
            # 在核的中心画一条线
            for i in range(k_size):
                # 映射到-center到center的范围
                offset = i - center
                
                # 计算在给定角度下的坐标
                x = center + int(offset * np.cos(angle_rad))
                y = center + int(offset * np.sin(angle_rad))
                
                # 确保坐标在有效范围内
                if 0 <= x < k_size and 0 <= y < k_size:
                    kernel[y, x] = 1.0
            
            # 如果核中没有非零元素（可能由于角度和核大小的组合），使用中心点
            if np.sum(kernel) == 0:
                kernel[center, center] = 1.0
            
            # 归一化核以保持图像亮度
            kernel = kernel / np.sum(kernel)
            
            # 应用自定义卷积
            return cv2.filter2D(image, -1, kernel)
        
        except Exception as e:
            print(f"应用方向性模糊时出错: {e}")
            # 如果失败，回退到简单的高斯模糊
            k_size = max(3, int(blur_strength)) * 2 + 1
            return cv2.GaussianBlur(image, (k_size, k_size), 0)
    
    def _apply_double_exposure_blur(self, image, blur_strength, **kwargs):
        """应用双重曝光效果，将图像与其模糊版本叠加"""
        # 获取效果参数
        intensity = kwargs.get('effect_intensity', 1.0)  # 混合强度
        blend_mode = int(kwargs.get('effect_angle', 0)) % 4  # 使用角度参数选择混合模式
        
        try:
            # 创建高斯模糊版本
            k_size = max(3, int(blur_strength * intensity)) * 2 + 1
            blurred = cv2.GaussianBlur(image, (k_size, k_size), 0)
            
            # 根据选择的混合模式应用不同的图层混合算法
            if blend_mode == 0:  # 正片叠底（常规混合）
                # alpha混合：result = alpha*img1 + (1-alpha)*img2
                alpha = 0.6  # 控制原始图像的比例
                result = cv2.addWeighted(image, alpha, blurred, 1-alpha, 0)
                
            elif blend_mode == 1:  # 屏幕混合
                # 屏幕混合：result = 1 - (1-img1) * (1-img2)
                # 先将图像归一化到0-1
                img_norm = image.astype(np.float32) / 255.0
                blur_norm = blurred.astype(np.float32) / 255.0
                
                # 应用屏幕混合公式
                result_norm = 1.0 - (1.0 - img_norm) * (1.0 - blur_norm)
                
                # 控制混合强度
                alpha = min(1.0, intensity * 0.7)
                result_norm = img_norm * (1-alpha) + result_norm * alpha
                
                # 转换回0-255范围
                result = (result_norm * 255).astype(np.uint8)
                
            elif blend_mode == 2:  # 叠加混合
                # 叠加混合：对于暗区域，使用正片叠底；对于亮区域，使用屏幕混合
                img_norm = image.astype(np.float32) / 255.0
                blur_norm = blurred.astype(np.float32) / 255.0
                
                # 创建掩码分离暗区域和亮区域
                mask = img_norm < 0.5
                
                # 对暗区域应用正片叠底
                multiply = img_norm * blur_norm
                
                # 对亮区域应用屏幕混合
                screen = 1.0 - (1.0 - img_norm) * (1.0 - blur_norm)
                
                # 组合结果
                result_norm = np.where(mask, 2 * multiply, 1 - 2 * (1 - img_norm) * (1 - blur_norm))
                
                # 控制混合强度
                alpha = min(1.0, intensity * 0.7)
                result_norm = img_norm * (1-alpha) + result_norm * alpha
                
                # 转换回0-255范围
                result = (result_norm * 255).astype(np.uint8)
                
            else:  # 柔光混合
                # 柔光混合：类似于叠加，但更柔和
                img_norm = image.astype(np.float32) / 255.0
                blur_norm = blurred.astype(np.float32) / 255.0
                
                # 创建掩码分离暗区域和亮区域
                mask = blur_norm < 0.5
                
                # 应用柔光混合公式
                result_norm = np.where(
                    mask, 
                    2 * img_norm * blur_norm + img_norm**2 * (1 - 2 * blur_norm),
                    2 * img_norm * (1 - blur_norm) + np.sqrt(img_norm) * (2 * blur_norm - 1)
                )
                
                # 控制混合强度
                alpha = min(1.0, intensity * 0.7)
                result_norm = img_norm * (1-alpha) + result_norm * alpha
                
                # 转换回0-255范围，确保在有效范围内
                result = np.clip(result_norm * 255, 0, 255).astype(np.uint8)
            
            return result
        
        except Exception as e:
            print(f"应用双重曝光模糊时出错: {e}")
            # 如果失败，回退到简单的混合
            return cv2.addWeighted(image, 0.6, blurred, 0.4, 0)
    
    def _apply_prism_blur(self, image, blur_strength, **kwargs):
        """应用棱镜模糊，分离RGB通道并添加位移"""
        # 检查图像是否为彩色
        if len(image.shape) < 3 or image.shape[2] < 3:
            # 如果不是彩色图像，返回原图
            print("棱镜模糊需要彩色图像")
            return image
        
        # 获取效果参数
        intensity = kwargs.get('effect_intensity', 1.0)  # 棱镜效果强度
        angle = kwargs.get('effect_angle', 0)  # 通道分离方向
        
        try:
            # 分离RGB通道
            b_channel, g_channel, r_channel = cv2.split(image)
            
            # 计算位移量，基于模糊强度和强度参数
            displacement = max(1, int(blur_strength * intensity * 0.5))
            
            # 根据角度确定位移方向
            angle_rad = np.radians(angle)
            dx = int(displacement * np.cos(angle_rad))
            dy = int(displacement * np.sin(angle_rad))
            
            # 创建平移矩阵
            r_translation = np.float32([[1, 0, dx], [0, 1, dy]])
            b_translation = np.float32([[1, 0, -dx], [0, 1, -dy]])
            
            # 图像尺寸
            height, width = image.shape[:2]
            
            # 平移红色和蓝色通道
            r_shifted = cv2.warpAffine(r_channel, r_translation, (width, height), borderMode=cv2.BORDER_REFLECT)
            b_shifted = cv2.warpAffine(b_channel, b_translation, (width, height), borderMode=cv2.BORDER_REFLECT)
            
            # 组合通道
            result = cv2.merge([b_shifted, g_channel, r_shifted])
            
            # 添加轻微的高斯模糊以柔化边缘
            k_size = max(3, int(blur_strength * 0.3)) * 2 + 1
            result = cv2.GaussianBlur(result, (k_size, k_size), 0)
            
            return result
        
        except Exception as e:
            print(f"应用棱镜模糊时出错: {e}")
            # 如果失败，回退到原始图像
            return image
    
    def _blend_images(self, original, blurred, mask):
        """根据掩码混合原始图像和模糊图像"""
        # 确保掩码和图像具有相同的形状
        if original.shape != blurred.shape:
            print(f"警告: 原始图像形状 {original.shape} 与模糊图像形状 {blurred.shape} 不匹配")
            return original
        
        if mask.shape[:2] != original.shape[:2]:
            print(f"警告: 掩码形状 {mask.shape} 与图像形状 {original.shape} 不匹配")
            # 调整掩码大小
            mask = cv2.resize(mask, (original.shape[1], original.shape[0]))
        
        try:
            # 归一化掩码以便与图像混合
            if mask.max() > 1:
                normalized_mask = mask.astype(np.float32) / 255.0
            else:
                normalized_mask = mask
            
            # 混合图像：前景（mask=1）保持原样，背景（mask=0）模糊
            result = original * normalized_mask + blurred * (1 - normalized_mask)
            
            # 确保结果在有效范围内
            return np.clip(result, 0, 255).astype(np.uint8)
        except Exception as e:
            print(f"混合图像时出错: {e}")
            # 如果混合失败，返回原始图像
            return original
    
    def _to_comfy_image(self, cv_image):
        """将OpenCV图像格式转换回ComfyUI格式"""
        # 如果是BGR格式，转换为RGB
        if len(cv_image.shape) == 3 and cv_image.shape[2] == 3:
            rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        else:
            rgb_image = cv_image
        
        # 转换为torch张量
        tensor_image = torch.from_numpy(rgb_image).float() / 255.0
        
        # 添加批次维度
        if len(tensor_image.shape) == 3:
            tensor_image = tensor_image.unsqueeze(0)
        
        return tensor_image


# 注册节点
NODE_CLASS_MAPPINGS = {
    "DepthBackgroundBlur": DepthBackgroundBlur
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DepthBackgroundBlur": "深度感知背景虚化"
} 