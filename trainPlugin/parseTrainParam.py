"""
训练参数解析模块

该模块负责解析所有与模型、数据增强和训练相关的命令行参数，
保持 YoloTrain.py 的简洁性。

作者：SteelAiLab
日期：2025
"""

import argparse
from typing import Dict, Any


def add_model_args(parser: argparse.ArgumentParser,
                    DEFAULT_MODEL_FAMILY: str,
                    DEFAULT_MODEL_SIZE: str,
                    DEFAULT_TRAIN_MODE: str,
                    DEFAULT_TASK_NAME: str,
                    SUPPORTED_MODEL_SIZES: list,
                    SUPPORTED_TRAIN_MODES: list) -> None:
    """
    添加所有模型相关的命令行参数到 ArgumentParser

    Args:
        parser: argparse.ArgumentParser 实例
        DEFAULT_MODEL_FAMILY: 默认模型系列
        DEFAULT_MODEL_SIZE: 默认模型大小
        DEFAULT_TRAIN_MODE: 默认训练模式
        DEFAULT_TASK_NAME: 默认任务名称
        SUPPORTED_MODEL_SIZES: 支持的模型大小列表
        SUPPORTED_TRAIN_MODES: 支持的训练模式列表
    """
    # ===== 模型配置组 =====
    model_group = parser.add_argument_group('模型配置', '模型系列、大小、训练模式等参数')

    model_group.add_argument(
        "--model-family",
        type=str,
        default=DEFAULT_MODEL_FAMILY,
        help="模型系列(默认: {})。支持变体: yolo11/yolov11, yolov5/yolo5, yolov8/yolo8".format(DEFAULT_MODEL_FAMILY),
    )
    model_group.add_argument(
        "--model-size",
        type=str,
        default=DEFAULT_MODEL_SIZE,
        choices=SUPPORTED_MODEL_SIZES,
        help="模型大小(默认: {})".format(DEFAULT_MODEL_SIZE),
    )
    model_group.add_argument(
        "--train-mode",
        type=str,
        default=DEFAULT_TRAIN_MODE,
        choices=SUPPORTED_TRAIN_MODES,
        help="训练模式(默认: {})".format(DEFAULT_TRAIN_MODE),
    )
    model_group.add_argument(
        "--task-name",
        type=str,
        default=DEFAULT_TASK_NAME,
        help="训练任务名称(默认格式: model_family_model_size_detect)，使用固定任务名，每次训练会覆盖之前的结果",
    )
    model_group.add_argument(
        "--seg",
        action="store_true",
        help="训练 YOLO 实例分割模型(instance segmentation)。如果指定，将使用分割模型(如 yolo11n-seg.pt)",
    )

    # ===== 训练配置组 =====
    train_group = parser.add_argument_group('训练配置', '批次大小、轮数、设备等训练参数')

    train_group.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="训练轮数(epochs，默认: 10)",
    )
    train_group.add_argument(
        "--batch",
        type=int,
        default=4,
        help="批次大小(batch size，默认: 4)",
    )
    train_group.add_argument(
        "--imgsz",
        type=int,
        default=1024,
        help="输入图像尺寸(image size，默认: 1024)",
    )
    train_group.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "mps", "cpu"],
        help="训练设备：auto(自动选择，默认)、cuda、mps或cpu",
    )
    train_group.add_argument(
        "--runs-dir",
        type=str,
        default=None,
        help="自定义训练输出根目录(默认使用项目内 ultralytics/trainPlugin/runs)。会在其下创建 detect/instance 子目录。",
    )

    # ===== 数据配置组 =====
    data_group = parser.add_argument_group('数据配置', '数据集路径、优化器等参数')

    data_group.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="数据集根目录路径(包含 classes.txt、images/、labels/ 子目录)。如果指定，将自动生成配置文件",
    )
    data_group.add_argument(
        "--optimizer",
        type=str,
        default=None,
        choices=["SGD", "Adam", "AdamW", "RMSprop"],
        help="优化器类型(如果不指定，使用配置文件中的默认值：AdamW)",
    )
    data_group.add_argument(
        "--lr0",
        type=float,
        default=None,
        help="初始学习率(如果不指定，使用配置文件中的默认值：0.001)",
    )


def add_augmentation_args(parser: argparse.ArgumentParser) -> None:
    """
    添加所有数据增强相关的命令行参数到 ArgumentParser

    Args:
        parser: argparse.ArgumentParser 实例
    """
    # ===== 基础几何增强组 =====
    basic_geo_group = parser.add_argument_group('基础几何增强', '基础几何变换增强参数')

    basic_geo_group.add_argument(
        "--scale",
        type=float,
        default=None,
        help="图像缩放幅度范围(0-1)，例如：0.5 表示缩放±50%(默认：0.1)"
    )
    basic_geo_group.add_argument(
        "--translate",
        type=float,
        default=None,
        help="图像平移幅度比例(0-1)，例如：0.1 表示平移±10%(默认：0.0)"
    )
    basic_geo_group.add_argument(
        "--fliplr",
        type=float,
        default=None,
        help="水平翻转概率(0-1)，例如：0.5 表示50%概率翻转(默认：0.0)"
    )
    basic_geo_group.add_argument(
        "--flipud",
        type=float,
        default=None,
        help="垂直翻转概率(0-1)，例如：0.5 表示50%概率翻转(默认：0.0)"
    )
    basic_geo_group.add_argument(
        "--degrees",
        type=float,
        default=None,
        help="图像旋转角度范围(度)，例如：10 表示旋转±10度(默认：0.0)"
    )

    # ===== 高级几何增强组 =====
    advanced_geo_group = parser.add_argument_group('高级几何增强', '高级几何变换增强参数')

    advanced_geo_group.add_argument(
        "--shear",
        type=float,
        default=None,
        help="图像剪切角度范围(度)，例如：5 表示剪切±5度(默认：0.0)"
    )
    advanced_geo_group.add_argument(
        "--perspective",
        type=float,
        default=None,
        help="透视变换强度(0-0.001)，数值越大透视效果越强(默认：0.0)"
    )

    # ===== 颜色空间增强组 =====
    color_group = parser.add_argument_group('颜色空间增强', 'HSV和颜色空间增强参数')

    color_group.add_argument(
        "--hsv_h",
        type=float,
        default=None,
        help="HSV色调增强幅度(0-1)，例如：0.015 表示色调变化±1.5%(默认：0.0)"
    )
    color_group.add_argument(
        "--hsv_s",
        type=float,
        default=None,
        help="HSV饱和度增强幅度(0-1)，例如：0.7 表示饱和度变化±70%(默认：0.0)"
    )
    color_group.add_argument(
        "--hsv_v",
        type=float,
        default=None,
        help="HSV亮度增强幅度(0-1)，例如：0.4 表示亮度变化±40%(默认：0.0)"
    )
    color_group.add_argument(
        "--bgr",
        type=float,
        default=None,
        help="RGB转BGR通道概率(0-1)，0表示不转换(默认：0.0)"
    )

    # ===== 混合增强组 =====
    mix_group = parser.add_argument_group('混合增强', 'Mosaic、MixUp等混合增强参数')

    mix_group.add_argument(
        "--mosaic",
        type=float,
        default=None,
        help="Mosaic数据增强概率(0-1)，0表示不使用(默认：0.0)"
    )
    mix_group.add_argument(
        "--mixup",
        type=float,
        default=None,
        help="MixUp数据增强概率(0-1)，0表示不使用(默认：0.0)"
    )
    mix_group.add_argument(
        "--cutmix",
        type=float,
        default=None,
        help="CutMix数据增强概率(0-1)，0表示不使用(默认：0.0)"
    )
    mix_group.add_argument(
        "--close_mosaic",
        type=int,
        default=None,
        help="在最后N个epoch关闭mosaic增强，0表示不关闭(默认：0)"
    )

    # ===== 特殊增强组 =====
    special_group = parser.add_argument_group('特殊增强', 'Copy-Paste、随机擦除等特殊增强参数')

    special_group.add_argument(
        "--copy_paste",
        type=float,
        default=None,
        help="Copy-Paste增强概率(0-1)，主要用于分割任务(默认：0.0)"
    )
    special_group.add_argument(
        "--erasing",
        type=float,
        default=None,
        help="随机擦除增强概率(0-1)，0表示不进行随机擦除(默认：0.0)"
    )
    special_group.add_argument(
        "--auto_augment",
        type=str,
        default=None,
        choices=[None, "randaugment", "augmix", "autoaugment"],
        help="自动增强策略：randaugment、augmix、autoaugment 或 None(默认：None)"
    )

    # ===== 训练参数组 =====
    train_group = parser.add_argument_group('训练参数', '多尺度、Dropout等训练相关参数')

    train_group.add_argument(
        "--multi_scale",
        action="store_true",
        help="启用多尺度训练，随机改变输入图像大小"
    )
    train_group.add_argument(
        "--crop_fraction",
        type=float,
        default=None,
        help="数据裁剪使用的比例(0-1)，1.0表示不裁剪(默认：1.0)"
    )
    train_group.add_argument(
        "--dropout",
        type=float,
        default=None,
        help="分类头的dropout率(0-1)，防止过拟合，0表示不使用(默认：0.0)"
    )
    train_group.add_argument(
        "--workers",
        type=int,
        default=None,
        help="数据加载器的工作线程数，0表示主线程加载(默认：0)"
    )


def update_model_kwargs(train_kwargs: Dict[str, Any], args) -> None:
    """
    根据命令行参数更新模型相关参数字典

    Args:
        train_kwargs: 训练参数字典
        args: 解析后的命令行参数
    """
    # 更新训练基本参数
    if args.epochs is not None:
        train_kwargs["epochs"] = args.epochs

    if args.batch is not None:
        train_kwargs["batch"] = args.batch

    if args.imgsz is not None:
        train_kwargs["imgsz"] = args.imgsz

    # 更新优化器参数
    if args.optimizer is not None:
        train_kwargs["optimizer"] = args.optimizer
        print(f"  使用优化器: {args.optimizer}")

    if args.lr0 is not None:
        train_kwargs["lr0"] = args.lr0
        print(f"  初始学习率: {args.lr0}")


def update_train_kwargs(train_kwargs: Dict[str, Any], args) -> None:
    """
    根据命令行参数更新数据增强和训练参数字典

    Args:
        train_kwargs: 训练参数字典
        args: 解析后的命令行参数
    """
    # ===== 基础几何增强 =====
    if args.scale is not None:
        train_kwargs["scale"] = args.scale
        print(f"  图像缩放幅度: ±{args.scale * 100:.1f}%")

    if args.translate is not None:
        train_kwargs["translate"] = args.translate
        print(f"  图像平移幅度: ±{args.translate * 100:.1f}%")

    if args.fliplr is not None:
        train_kwargs["fliplr"] = args.fliplr
        print(f"  水平翻转概率: {args.fliplr}")

    if args.flipud is not None:
        train_kwargs["flipud"] = args.flipud
        print(f"  垂直翻转概率: {args.flipud}")

    if args.degrees is not None:
        train_kwargs["degrees"] = args.degrees
        print(f"  旋转角度范围: ±{args.degrees}度")

    # ===== 高级几何增强 =====
    if args.shear is not None:
        train_kwargs["shear"] = args.shear
        print(f"  剪切角度范围: ±{args.shear}度")

    if args.perspective is not None:
        train_kwargs["perspective"] = args.perspective
        print(f"  透视变换强度: {args.perspective}")

    # ===== 颜色空间增强 =====
    if args.hsv_h is not None:
        train_kwargs["hsv_h"] = args.hsv_h
        print(f"  HSV色调增强: ±{args.hsv_h}")

    if args.hsv_s is not None:
        train_kwargs["hsv_s"] = args.hsv_s
        print(f"  HSV饱和度增强: ±{args.hsv_s}")

    if args.hsv_v is not None:
        train_kwargs["hsv_v"] = args.hsv_v
        print(f"  HSV亮度增强: ±{args.hsv_v}")

    if args.bgr is not None:
        train_kwargs["bgr"] = args.bgr
        print(f"  RGB转BGR概率: {args.bgr}")

    # ===== 混合增强 =====
    if args.mosaic is not None:
        train_kwargs["mosaic"] = args.mosaic
        print(f"  Mosaic增强概率: {args.mosaic}")

    if args.mixup is not None:
        train_kwargs["mixup"] = args.mixup
        print(f"  MixUp增强概率: {args.mixup}")

    if args.cutmix is not None:
        train_kwargs["cutmix"] = args.cutmix
        print(f"  CutMix增强概率: {args.cutmix}")

    if args.close_mosaic is not None:
        train_kwargs["close_mosaic"] = args.close_mosaic
        if args.close_mosaic > 0:
            print(f"  最后{args.close_mosaic}个epoch关闭mosaic增强")

    # ===== 特殊增强 =====
    if args.copy_paste is not None:
        train_kwargs["copy_paste"] = args.copy_paste
        print(f"  Copy-Paste增强概率: {args.copy_paste}")

    if args.erasing is not None:
        train_kwargs["erasing"] = args.erasing
        print(f"  随机擦除概率: {args.erasing}")

    if args.auto_augment is not None:
        train_kwargs["auto_augment"] = args.auto_augment
        print(f"  自动增强策略: {args.auto_augment if args.auto_augment else 'None'}")

    # ===== 训练参数 =====
    if args.multi_scale:
        train_kwargs["multi_scale"] = True
        print("  多尺度训练: 已启用")

    if args.crop_fraction is not None:
        train_kwargs["crop_fraction"] = args.crop_fraction
        print(f"  裁剪比例: {args.crop_fraction}")

    if args.dropout is not None:
        train_kwargs["dropout"] = args.dropout
        print(f"  Dropout率: {args.dropout}")

    if args.workers is not None:
        train_kwargs["workers"] = args.workers
        print(f"  数据加载线程数: {args.workers}")