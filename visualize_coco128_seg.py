import cv2
import numpy as np
import os
from pathlib import Path
import random

def plot_one_box(x, img, color=None, label=None, line_thickness=3):
    """画一个边界框"""
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

def plot_segments(img, segments, color=None):
    """画分割多边形"""
    color = color or [random.randint(0, 255) for _ in range(3)]
    segments = segments.astype(np.int32)
    cv2.fillPoly(img, [segments], color=color)
    
def visualize_labels(img_path, label_path):
    """可视化单张图片的标注"""
    # 读取图片
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"无法读取图片: {img_path}")
        return
    
    # 创建一个用于显示分割的overlay
    overlay = img.copy()
    
    # 读取标签文件
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            lines = f.readlines()
            
        # 处理每个标注
        for line in lines:
            data = line.strip().split()
            cls = int(data[0])  # 类别ID
            segments = np.array([float(x) for x in data[1:]]).reshape(-1, 2)
            
            # 将相对坐标转换为绝对坐标
            segments[:, 0] *= img.shape[1]
            segments[:, 1] *= img.shape[0]
            
            # 获取边界框
            x_min, y_min = segments.min(0)
            x_max, y_max = segments.max(0)
            box = [x_min, y_min, x_max, y_max]
            
            # 随机生成颜色
            color = [random.randint(0, 255) for _ in range(3)]
            
            # 绘制分割区域
            plot_segments(overlay, segments, color)
            
            # 绘制边界框和类别标签
            plot_one_box(box, img, color, f'Class {cls}', 2)
    
    # 合并原图和分割overlay
    alpha = 0.4
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    
    return img

def main():
    # 设置数据集路径
    dataset_path = Path('datasets/coco128-seg')
    images_path = dataset_path / 'images/train2017'
    labels_path = dataset_path / 'labels/train2017'
    
    # 创建输出目录
    output_dir = Path('visualization_results')
    output_dir.mkdir(exist_ok=True)
    
    # 处理所有图片
    for img_path in images_path.glob('*.jpg'):
        # 获取对应的标签文件路径
        label_path = labels_path / f"{img_path.stem}.txt"
        
        # 可视化图片和标注
        result = visualize_labels(img_path, label_path)
        
        if result is not None:
            # 保存结果
            output_path = output_dir / f"vis_{img_path.name}"
            cv2.imwrite(str(output_path), result)
            print(f"已保存可视化结果到: {output_path}")

if __name__ == '__main__':
    main() 