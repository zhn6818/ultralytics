from pathlib import Path

from ultralytics import YOLO

# ==================== 配置路径 ====================
base_yaml = "./ultralytics/cfg/models/11/yolo11-seg.yaml"
# 设置为 True 表示从检查点继续训练，False 表示从预训练权重开始训练
resume = False  # 修改此变量来切换训练模式

if resume:
    # 从检查点继续训练
    pretrained_weights = "runs/segment/yolo_seg_dataset3/weights/best.pt"
else:
    # 从预训练权重开始训练
    pretrained_weights = "./ultralytics/yolo11l-seg.pt"

# ==================== 继续训练（使用策略1的改进配置） ====================
# 加载预训练模型
model = YOLO(pretrained_weights)  # 使用预训练权重开始训练

# 使用策略1的改进配置进行训练
results = model.train(
    data="./ultralytics/cfg/datasets/yolo_seg_dataset.yaml",
    epochs=50,  # 增加总epoch数
    imgsz=1024,   # 使用标准尺寸
    device="mps",
    name="yolo_seg_dataset",  # 指定实验名称，结果保存在 runs/segment/yolo_seg_dataset/
    batch=2,     # batch size
    resume=resume,  # 继续训练时设置为True

    # 模型设置
    freeze=0,    # 不冻结任何层，让整个网络适应你的数据

    # 模型保存设置
    save=True,        # 保存训练checkpoint（会自动保存best.pt和last.pt）
    save_period=10,   # 每10个epoch保存一次checkpoint

    # 损失函数权重
    box=7.5,      # 边界框损失权重
    cls=0.5,      # 分类损失权重
    dfl=1.5,      # Distribution Focal Loss权重

    # ==================== 策略1的改进：学习率设置 ====================
    lr0=0.001,     # 初始学习率（继续训练时可适当降低，如0.0001-0.0005）
    lrf=0.01,      # 最终学习率因子（最终学习率 = lr0 * lrf = 0.001 * 0.01 = 0.00001）
    cos_lr=True,   # 启用余弦学习率调度器，有助于精细调优和跳出局部最优

    # ==================== 策略1的改进：优化器设置 ====================
    optimizer='AdamW',  # 使用AdamW优化器，通常比SGD更稳定，适合fine-tuning
    momentum=0.937,     # SGD的momentum（AdamW不使用，但保留以防切换）
    weight_decay=0.0005,  # 权重衰减（L2正则化）

    # ==================== 策略1的改进：早停策略 ====================
    patience=50,  # 如果50个epoch没有提升就停止训练

        # 禁用大部分数据增强，但保留最小缩放以避免处理错误
        hsv_h=0.0,        # HSV色调增强
        hsv_s=0.0,        # HSV饱和度增强
        hsv_v=0.0,        # HSV亮度增强
        degrees=0.0,      # 旋转角度
        translate=0.0,    # 平移
        scale=0.1,        # 保留最小缩放（0.0可能导致处理错误）
        shear=0.0,        # 剪切
        perspective=0.0,   # 透视变换
        flipud=0.0,       # 垂直翻转
        fliplr=0.0,       # 水平翻转
        bgr=0.0,          # BGR通道交换
        mosaic=0.0,       # Mosaic增强
        mixup=0.0,        # MixUp增强
        cutmix=0.0,       # CutMix增强
        copy_paste=0.0,   # 分割copy-paste增强
        erasing=0.0,      # 随机擦除
        auto_augment=None,  # 禁用自动增强策略
    )