"""
YOLO ONNX 模型推理测试脚本
支持检测（detect）和分割（segment）模型
"""

import cv2
import numpy as np
import onnxruntime as ort
from pathlib import Path


class YOLOv11ONNXInference:
    """YOLO ONNX 模型推理类（支持检测和分割）"""
    
    def __init__(self, model_path: str, conf_threshold: float = 0.25, iou_threshold: float = 0.45):
        """
        初始化 YOLO ONNX 推理器
        
        Args:
            model_path: ONNX 模型文件路径
            conf_threshold: 置信度阈值，默认 0.25
            iou_threshold: NMS IoU 阈值，默认 0.45
        """
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        
        # 加载 ONNX 模型
        self.session = ort.InferenceSession(
            model_path,
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        
        # 获取模型输入信息
        self.input_name = self.session.get_inputs()[0].name
        input_shape = self.session.get_inputs()[0].shape
        self.input_height = input_shape[2]  # 通常是 640
        self.input_width = input_shape[3]   # 通常是 640
        
        # 检查模型输出，判断是检测模型还是分割模型
        self.output_names = [output.name for output in self.session.get_outputs()]
        self.num_outputs = len(self.output_names)
        
        # 分割模型通常有2个输出：检测结果和原型掩码
        self.is_segmentation = self.num_outputs == 2
        
        # 获取输出信息
        if self.is_segmentation:
            # 分割模型：第一个输出是检测结果，第二个是原型掩码
            self.output_name = self.output_names[0]
            self.proto_name = self.output_names[1]
            proto_shape = self.session.get_outputs()[1].shape
            self.mask_dim = proto_shape[1]  # 掩码维度，通常是32
            print(f"✓ 检测到分割模型（Segmentation Model）")
            print(f"  输出1（检测）: {self.output_name}")
            print(f"  输出2（原型掩码）: {self.proto_name}, 形状: {proto_shape}")
        else:
            # 检测模型：只有一个输出
            self.output_name = self.output_names[0]
            print(f"✓ 检测到检测模型（Detection Model）")
            print(f"  输出: {self.output_name}")
        
        print(f"模型加载成功: {model_path}")
        print(f"输入尺寸: {self.input_width}x{self.input_height}")
        print(f"置信度阈值: {self.conf_threshold}, IoU阈值: {self.iou_threshold}")
    
    def letterbox(self, img: np.ndarray, new_shape: tuple = (640, 640)) -> tuple:
        """
        保持宽高比缩放图片，并添加填充
        
        Args:
            img: 输入图片 (H, W, C)
            new_shape: 目标尺寸 (height, width)
            
        Returns:
            img: 缩放并填充后的图片
            pad: 填充值 (top, left)
            gain: 缩放比例
        """
        shape = img.shape[:2]  # 当前尺寸 [height, width]
        
        # 计算缩放比例
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        
        # 计算新尺寸
        new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))  # (width, height)
        
        # 计算填充
        dw = (new_shape[1] - new_unpad[0]) / 2
        dh = (new_shape[0] - new_unpad[1]) / 2
        
        # 缩放图片
        if shape[::-1] != new_unpad:
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        
        # 添加填充
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        
        return img, (top, left), r
    
    def preprocess(self, image_path: str) -> tuple:
        """
        预处理图片
        
        Args:
            image_path: 图片路径
            
        Returns:
            image_data: 预处理后的图片数据 (1, 3, H, W)
            pad: 填充值 (top, left)
            gain: 缩放比例
            original_img: 原始图片
        """
        # 读取图片
        original_img = cv2.imread(image_path)
        if original_img is None:
            raise ValueError(f"无法读取图片: {image_path}")
        
        # BGR 转 RGB
        img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        
        # Letterbox 缩放
        img, pad, gain = self.letterbox(img, (self.input_height, self.input_width))
        
        # 归一化到 [0, 1]
        image_data = img.astype(np.float32) / 255.0
        
        # 转换为 (C, H, W) 格式
        image_data = np.transpose(image_data, (2, 0, 1))
        
        # 添加 batch 维度: (1, C, H, W)
        image_data = np.expand_dims(image_data, axis=0)
        
        return image_data, pad, gain, original_img
    
    def process_mask(self, protos: np.ndarray, masks_in: np.ndarray, bboxes: np.ndarray, 
                     shape: tuple, pad: tuple, gain: float) -> np.ndarray:
        """
        处理分割掩码：将原型掩码和掩码系数结合生成最终掩码
        
        Args:
            protos: 原型掩码 (1, mask_dim, mask_h, mask_w) 或 (mask_dim, mask_h, mask_w)
            masks_in: 掩码系数 (N, mask_dim)，N是检测数量
            bboxes: 边界框 (N, 4)，格式为 [x1, y1, x2, y2]
            shape: 原始图片尺寸 (height, width)
            pad: 填充值 (top, left)
            gain: 缩放比例
            
        Returns:
            masks: 二值掩码 (N, height, width)
        """
        # 处理原型掩码维度
        if len(protos.shape) == 4:
            protos = protos[0]  # 移除batch维度: (mask_dim, mask_h, mask_w)
        
        c, mh, mw = protos.shape  # mask_dim, mask_h, mask_w
        
        # 矩阵乘法：masks_in @ protos.view(c, -1)
        # masks_in: (N, c), protos_flat: (c, mh*mw)
        protos_flat = protos.reshape(c, -1).astype(np.float32)  # (c, mh*mw)
        masks = masks_in.astype(np.float32) @ protos_flat  # (N, mh*mw)
        masks = masks.reshape(-1, mh, mw)  # (N, mh, mw)
        
        # 应用sigmoid激活函数
        masks = 1 / (1 + np.exp(-np.clip(masks, -250, 250)))  # sigmoid，防止溢出
        
        # 处理每个掩码
        masks_resized = []
        img_h, img_w = shape
        
        for i, mask in enumerate(masks):
            # 计算去除填充后的尺寸
            top, left = int(pad[0]), int(pad[1])
            bottom = self.input_height - int(pad[0])
            right = self.input_width - int(pad[1])
            
            # 先缩放到输入尺寸（考虑填充）
            mask_input = cv2.resize(mask, (self.input_width, self.input_height), interpolation=cv2.INTER_LINEAR)
            
            # 去除填充
            mask_unpadded = mask_input[top:bottom, left:right]
            
            # 缩放到原图尺寸
            mask_orig = cv2.resize(mask_unpadded, (img_w, img_h), interpolation=cv2.INTER_LINEAR)
            
            # 裁剪到边界框内（YOLO的做法）
            x1, y1, x2, y2 = bboxes[i]
            x1, y1, x2, y2 = max(0, int(x1)), max(0, int(y1)), min(img_w, int(x2)), min(img_h, int(y2))
            
            # 创建最终掩码，只在边界框区域内保留
            mask_final = np.zeros((img_h, img_w), dtype=np.float32)
            if x2 > x1 and y2 > y1:
                # 裁剪掩码到边界框
                mask_cropped = mask_orig[y1:y2, x1:x2]
                mask_final[y1:y2, x1:x2] = mask_cropped
            
            masks_resized.append(mask_final)
        
        masks = np.array(masks_resized)
        
        # 转换为二值掩码
        masks = (masks > 0.5).astype(np.uint8)
        
        return masks
    
    def postprocess(self, outputs: list, pad: tuple, gain: float, 
                    original_img: np.ndarray) -> tuple:
        """
        后处理：解析模型输出，应用 NMS，处理分割掩码（如果存在）
        
        Args:
            outputs: 模型输出列表
            pad: 填充值 (top, left)
            gain: 缩放比例
            original_img: 原始图片
            
        Returns:
            detections: 检测结果列表，每个元素为 [x1, y1, x2, y2, conf, class_id]
            masks: 分割掩码 (N, H, W)，如果是检测模型则为 None
        """
        # 获取检测输出
        pred_output = outputs[0]
        
        # 转置并压缩维度: (num_detections, 4+num_classes+mask_dim)
        pred_output = np.transpose(np.squeeze(pred_output, axis=0))
        
        # 存储检测框、分数、类别和掩码系数
        boxes = []
        scores = []
        class_ids = []
        mask_coeffs = [] if self.is_segmentation else None
        
        # 原始图片尺寸
        img_h, img_w = original_img.shape[:2]
        
        # 判断输出格式（需要根据实际模型确定）
        # 对于分割模型，通常输出格式为: [x, y, w, h, class_scores..., mask_coeffs...]
        # 我们需要推断类别数量
        if self.is_segmentation:
            # 尝试推断类别数量：假设掩码维度是32（常见值）
            # 输出维度 = 4 (bbox) + num_classes + mask_dim
            # 我们可以通过检查输出维度来推断
            total_dim = pred_output.shape[1]
            # 常见的掩码维度是32，类别数可能是80（COCO）或其他
            # 这里我们尝试常见的组合
            possible_mask_dims = [32, 64, 128]
            num_classes = None
            for mask_dim_guess in possible_mask_dims:
                if (total_dim - 4 - mask_dim_guess) > 0:
                    num_classes = total_dim - 4 - mask_dim_guess
                    break
            if num_classes is None:
                # 如果无法推断，假设掩码维度是32
                num_classes = total_dim - 4 - self.mask_dim
        else:
            num_classes = pred_output.shape[1] - 4
        
        # 解析每个检测结果
        for i in range(pred_output.shape[0]):
            # 提取边界框坐标 (中心点格式: x, y, w, h)
            x_center, y_center, w, h = pred_output[i][0:4]
            
            # 提取类别分数
            classes_scores = pred_output[i][4:4+num_classes]
            
            # 提取掩码系数（如果是分割模型）
            if self.is_segmentation:
                mask_coeff = pred_output[i][4+num_classes:]
            
            # 找到最大分数和对应类别
            max_score = np.max(classes_scores)
            class_id = np.argmax(classes_scores)
            
            # 过滤低置信度检测
            if max_score < self.conf_threshold:
                continue
            
            # 去除填充并缩放回原图尺寸
            x_center = (x_center - pad[1]) / gain
            y_center = (y_center - pad[0]) / gain
            w = w / gain
            h = h / gain
            
            # 转换为左上角坐标格式
            x1 = int(x_center - w / 2)
            y1 = int(y_center - h / 2)
            x2 = int(x_center + w / 2)
            y2 = int(y_center + h / 2)
            
            # 限制在图片范围内
            x1 = max(0, min(x1, img_w))
            y1 = max(0, min(y1, img_h))
            x2 = max(0, min(x2, img_w))
            y2 = max(0, min(y2, img_h))
            
            boxes.append([x1, y1, x2 - x1, y2 - y1])  # [x, y, w, h] 格式用于 NMS
            scores.append(float(max_score))
            class_ids.append(int(class_id))
            if self.is_segmentation:
                mask_coeffs.append(mask_coeff)
        
        # 应用 NMS
        detections = []
        masks = None
        
        if len(boxes) > 0:
            indices = cv2.dnn.NMSBoxes(boxes, scores, self.conf_threshold, self.iou_threshold)
            
            # 处理 NMS 结果
            if len(indices) > 0:
                # indices 可能是 numpy 数组或嵌套数组
                if isinstance(indices, np.ndarray):
                    indices = indices.flatten()
                
                # 提取保留的检测结果
                kept_boxes = []
                kept_mask_coeffs = []
                
                for idx in indices:
                    x, y, w, h = boxes[idx]
                    detections.append([
                        x, y, x + w, y + h,  # x1, y1, x2, y2
                        scores[idx],
                        class_ids[idx]
                    ])
                    kept_boxes.append([x, y, x + w, y + h])
                    if self.is_segmentation:
                        kept_mask_coeffs.append(mask_coeffs[idx])
                
                # 处理分割掩码
                if self.is_segmentation and len(kept_mask_coeffs) > 0:
                    protos = outputs[1]  # 原型掩码
                    protos = np.squeeze(protos, axis=0)  # 移除batch维度
                    masks = self.process_mask(
                        protos, 
                        np.array(kept_mask_coeffs),
                        np.array(kept_boxes),
                        (img_h, img_w),
                        pad,
                        gain
                    )
        
        return detections, masks
    
    def draw_detections(self, img: np.ndarray, detections: list, masks: np.ndarray = None) -> np.ndarray:
        """
        在图片上绘制检测结果和分割掩码（如果存在）
        
        Args:
            img: 输入图片
            detections: 检测结果列表
            masks: 分割掩码 (N, H, W)，可选
            
        Returns:
            img: 绘制了检测框和掩码的图片
        """
        # 生成颜色调色板
        np.random.seed(42)  # 固定随机种子，确保颜色一致
        colors = np.random.randint(0, 255, size=(100, 3), dtype=np.uint8)
        
        # 如果存在掩码，先绘制掩码
        if masks is not None and len(masks) > 0:
            for i, mask in enumerate(masks):
                if i >= len(detections):
                    break
                
                class_id = detections[i][5]
                color = colors[class_id % len(colors)]
                
                # 创建彩色掩码
                colored_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
                colored_mask[mask > 0] = color
                
                # 将掩码叠加到原图上（半透明）
                img = cv2.addWeighted(img, 1.0, colored_mask, 0.4, 0)
        
        # 绘制边界框和标签
        for i, det in enumerate(detections):
            x1, y1, x2, y2, conf, class_id = det
            
            # 选择颜色
            color = tuple(map(int, colors[class_id % len(colors)]))
            
            # 绘制边界框
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            
            # 准备标签文本
            label = f"Class {class_id}: {conf:.2f}"
            
            # 计算文本尺寸
            (label_width, label_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            
            # 绘制标签背景
            cv2.rectangle(
                img,
                (int(x1), int(y1) - label_height - baseline - 5),
                (int(x1) + label_width, int(y1)),
                color,
                -1
            )
            
            # 绘制标签文本
            cv2.putText(
                img,
                label,
                (int(x1), int(y1) - baseline - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )
        
        return img
    
    def infer(self, image_path: str, save_path: str = None) -> tuple:
        """
        执行推理
        
        Args:
            image_path: 输入图片路径
            save_path: 保存结果图片的路径（可选）
            
        Returns:
            detections: 检测结果列表
            result_img: 绘制了检测框和掩码的图片
            masks: 分割掩码（如果是分割模型），否则为 None
        """
        # 预处理
        image_data, pad, gain, original_img = self.preprocess(image_path)
        
        # 推理
        if self.is_segmentation:
            outputs = self.session.run([self.output_name, self.proto_name], {self.input_name: image_data})
        else:
            outputs = self.session.run([self.output_name], {self.input_name: image_data})
        
        # 后处理（NMS + 掩码处理）
        detections, masks = self.postprocess(outputs, pad, gain, original_img)
        
        # 绘制检测结果和掩码
        result_img = original_img.copy()
        result_img = self.draw_detections(result_img, detections, masks)
        
        # 保存结果
        if save_path:
            cv2.imwrite(save_path, result_img)
            print(f"结果已保存到: {save_path}")
        
        return detections, result_img, masks


def main():
    """主函数"""
    # 模型路径
    model_path = "ultralytics/runs/segment/carparts_yolov8_s/weights/best.onnx"
    
    # 图片路径
    image_path = "/data1/zhn/test2/2.jpg"
    
    # 检查文件是否存在
    if not Path(model_path).exists():
        print(f"错误: 模型文件不存在: {model_path}")
        return
    
    if not Path(image_path).exists():
        print(f"错误: 图片文件不存在: {image_path}")
        return
    
    # 创建推理器
    print("=" * 60)
    print("YOLOv11 ONNX 推理测试")
    print("=" * 60)
    
    inferencer = YOLOv11ONNXInference(
        model_path=model_path,
        conf_threshold=0.25,  # 置信度阈值
        iou_threshold=0.45    # NMS IoU 阈值
    )
    
    # 执行推理
    print(f"\n正在推理图片: {image_path}")
    detections, result_img, masks = inferencer.infer(
        image_path=image_path,
        save_path="ultralytics/trainPlugin/result_onnx.jpg"  # 保存结果
    )
    
    # 打印检测结果
    task_type = "分割" if inferencer.is_segmentation else "检测"
    print(f"\n{task_type}到 {len(detections)} 个目标:")
    for i, det in enumerate(detections, 1):
        x1, y1, x2, y2, conf, class_id = det
        mask_info = f", 掩码面积={np.sum(masks[i-1])}像素" if masks is not None and i <= len(masks) else ""
        print(f"  目标 {i}: 类别={class_id}, 置信度={conf:.3f}, "
              f"位置=({int(x1)}, {int(y1)}, {int(x2)}, {int(y2)}){mask_info}")
    
    if masks is not None:
        print(f"\n分割掩码信息: {len(masks)} 个掩码，尺寸 {masks[0].shape}")
    
    print("\n推理完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()

