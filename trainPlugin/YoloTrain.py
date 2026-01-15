from copy import deepcopy
import argparse
import sys
import os
from pathlib import Path
from typing import Dict, Any, Callable, Optional, Tuple

# 添加项目根目录到 sys.path，确保能找到 ultralytics 模块
# YoloTrain.py 位于: ultralytics/trainPlugin/YoloTrain.py
# 项目根目录: SteelAiLab/
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from ultralytics import YOLO

# 导入参数解析模块
from parseTrainParam import (
    add_model_args,
    add_augmentation_args,
    update_model_kwargs,
    update_train_kwargs
)

# 从本地配置文件导入训练相关参数
from config import (
    get_model_config,
    get_task_full_name,
    generate_dataset_yaml,
    get_device_for_training,
    get_runs_root,
    TRAIN_KWARGS,
    TRAIN_DATA,
    SUPPORTED_MODEL_FAMILIES,
    SUPPORTED_MODEL_SIZES,
    SUPPORTED_TRAIN_MODES,
    DEFAULT_MODEL_FAMILY,
    DEFAULT_MODEL_SIZE,
    DEFAULT_TRAIN_MODE,
    DEFAULT_TASK_NAME,
)


def normalize_model_family(user_input: str) -> str:
    """
    将用户输入的模型系列变体标准化为代码中使用的标准格式。
    
    支持的映射：
    - yolo11 或 yolov11 → yolo11
    - yolov5 或 yolo5 → yolov5
    - yolov8 或 yolo8 → yolov8
    
    Args:
        user_input: 用户输入的模型系列名称（不区分大小写）
    
    Returns:
        标准化后的模型系列名称
    
    Raises:
        ValueError: 如果输入无法映射到支持的模型系列
    """
    user_input_lower = user_input.lower().strip()
    
    # 定义变体到标准格式的映射
    # 注意：yolo11 的标准格式是 "yolo11"（不带v），而 yolov5 和 yolov8 的标准格式带v
    variant_mapping = {
        # yolo11 的变体（标准格式是 yolo11）
        "yolo11": "yolo11",
        "yolov11": "yolo11",
        # yolov5 的变体（标准格式是 yolov5）
        "yolov5": "yolov5",
        "yolo5": "yolov5",
        # yolov8 的变体（标准格式是 yolov8）
        "yolov8": "yolov8",
        "yolo8": "yolov8",
    }
    
    # 首先尝试直接匹配映射表
    normalized = variant_mapping.get(user_input_lower)
    
    # 如果直接匹配失败，检查是否已经是标准格式
    if normalized is None:
        if user_input_lower in SUPPORTED_MODEL_FAMILIES:
            normalized = user_input_lower
        else:
            # 尝试通过移除 'v' 字符进行模糊匹配
            # 例如：yolov11 → yolo11, yolo5 → yolov5
            input_without_v = user_input_lower.replace("v", "")
            
            # 检查移除v后的输入是否能匹配到某个变体
            for variant, standard in variant_mapping.items():
                variant_without_v = variant.replace("v", "")
                if variant_without_v == input_without_v:
                    normalized = standard
                    break
            
            # 如果仍然无法匹配，报错
            if normalized is None:
                raise ValueError(
                    f"不支持的模型系列: '{user_input}'。"
                    f"支持的变体: yolo11/yolov11, yolov5/yolo5, yolov8/yolo8。"
                    f"标准格式: {', '.join(SUPPORTED_MODEL_FAMILIES)}"
                )
    
    # 验证标准化后的值在支持的列表中（双重检查，确保安全）
    if normalized not in SUPPORTED_MODEL_FAMILIES:
        raise ValueError(
            f"标准化后的模型系列 '{normalized}' 不在支持的列表中: {SUPPORTED_MODEL_FAMILIES}"
        )
    
    return normalized


def parse_args() -> argparse.Namespace:
    """解析命令行参数，选择要训练的模型系列、大小和训练模式。"""
    parser = argparse.ArgumentParser(
        description="快速训练 YOLO 模型（支持 yolov5/yolov8/yolo11 的 n/s/m/l/x 变体，支持检测和分割任务，支持三种训练模式）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 1. 使用官方预训练权重训练 yolo11s（默认模式）
  python YoloTrain.py
  python YoloTrain.py --train-mode pretrained
  
  # 2. 从零开始训练 yolo11n（不使用预训练权重）
  python YoloTrain.py --model-size n --train-mode scratch
  
  # 3. 继续之前中断的训练（从 checkpoint 恢复）
  python YoloTrain.py --train-mode resume
  
  # 4. 训练不同的模型系列和大小
  python YoloTrain.py --model-family yolov8 --model-size m --train-mode pretrained
  python YoloTrain.py --model-family yolov5 --model-size l --train-mode scratch
  
  # 5. 使用自定义任务名称（推荐，方便管理）
  python YoloTrain.py --task-name my_steel_detector --train-mode pretrained
  python YoloTrain.py --task-name exp_001 --model-size m --train-mode scratch
  
  # 6. 使用自定义数据集（自动生成配置文件）
  python YoloTrain.py --data-root /data1/zhn/africa --task-name africa
  python YoloTrain.py --data-root /data1/dataset/steel --task-name steel --model-size m
  
  # 7. 自定义训练参数（epochs、batch、imgsz）
  python YoloTrain.py --epochs 100 --batch 8 --imgsz 640
  python YoloTrain.py --model-size m --epochs 200 --batch 16 --imgsz 1280
  python YoloTrain.py --task-name my_exp --epochs 50 --batch 4 --imgsz 512
  
  # 8. 训练实例分割模型（使用 --seg 选项）
  python YoloTrain.py --seg
  python YoloTrain.py --seg --model-size m --epochs 100
  python YoloTrain.py --seg --model-family yolov8 --model-size l --train-mode pretrained
  python YoloTrain.py --seg --data-root /data1/dataset/seg --task-name my_seg

  # 9. 自定义优化器参数
  python YoloTrain.py --optimizer SGD --lr0 0.01
  python YoloTrain.py --optimizer Adam --lr0 0.0001
  python YoloTrain.py --optimizer AdamW --lr0 0.001
  python YoloTrain.py --model-size m --optimizer Adam --lr0 0.0005 --epochs 100

  # 10. 自定义数据增强参数
  # 基础几何增强
  python YoloTrain.py --fliplr 0.5 --degrees 10 --scale 0.5
  python YoloTrain.py --flipud 0.1 --translate 0.1 --shear 5

  # 颜色空间增强
  python YoloTrain.py --hsv_h 0.015 --hsv_s 0.7 --hsv_v 0.4

  # 混合增强（YOLO特色）
  python YoloTrain.py --mosaic 1.0 --mixup 0.1 --close_mosaic 10
  python YoloTrain.py --cutmix 0.2 --copy_paste 0.3

  # 训练参数
  python YoloTrain.py --multi_scale --dropout 0.1 --workers 8
  python YoloTrain.py --crop_fraction 0.8 --auto_augment randaugment

任务类型说明:
  - 默认: 目标检测任务（detect），使用检测模型（如 yolo11s.pt）
  - --seg: 实例分割任务（instance），使用分割模型（如 yolo11s-seg.pt）

训练模式说明:
  - scratch: 从 YAML 配置文件构建模型，从零开始训练（不使用预训练权重）
  - pretrained: 使用官方预训练权重进行迁移学习（默认，推荐）
  - resume: 从之前保存的 checkpoint（last.pt）继续训练

数据集说明:
  - 默认使用 coco8 数据集进行训练
  - 使用 --data-root 指定自定义数据集目录，将自动生成配置文件
  - 支持两种数据集目录结构（自动检测）：
    
    结构1（按 images/labels 分组）:
    data_root/
    ├── classes.txt       # 类别文件，每行一个类别名称
    ├── images/
    │   ├── train/       # 训练集图片
    │   ├── val/         # 验证集图片
    │   └── test/        # 测试集图片（可选）
    └── labels/
        ├── train/       # 训练集标注（YOLO格式）
        ├── val/         # 验证集标注
        └── test/        # 测试集标注（可选）
    
    结构2（按 train/val/test 分组）:
    data_root/
    ├── classes.txt       # 类别文件，每行一个类别名称
    ├── train/
    │   ├── images/      # 训练集图片
    │   └── labels/      # 训练集标注（YOLO格式）
    ├── val/
    │   ├── images/      # 验证集图片
    │   └── labels/      # 验证集标注
    └── test/            # 测试集（可选）
        ├── images/      # 测试集图片
        └── labels/      # 测试集标注

重要说明:
  - 训练结果会保存到固定的文件夹 runs/detect/任务名称/
  - 每次训练会覆盖该文件夹中的内容（exist_ok=True）
  - 不会创建递增文件夹（如 train、train2、train3）
  - 建议使用 --task-name 参数为不同实验指定不同的任务名
        """
    )

    # 添加所有模型相关的参数
    add_model_args(
        parser=parser,
        DEFAULT_MODEL_FAMILY=DEFAULT_MODEL_FAMILY,
        DEFAULT_MODEL_SIZE=DEFAULT_MODEL_SIZE,
        DEFAULT_TRAIN_MODE=DEFAULT_TRAIN_MODE,
        DEFAULT_TASK_NAME=DEFAULT_TASK_NAME,
        SUPPORTED_MODEL_SIZES=SUPPORTED_MODEL_SIZES,
        SUPPORTED_TRAIN_MODES=SUPPORTED_TRAIN_MODES
    )

    # 添加所有数据增强相关的参数
    add_augmentation_args(parser)

    return parser.parse_args()


def _args_to_data_augment_dict(args) -> Dict[str, Any]:
    """
    从 args 对象提取数据增强参数并转换为字典
    
    Args:
        args: argparse.Namespace 对象
    
    Returns:
        数据增强参数字典
    """
    data_augment = {}
    
    # 基础几何增强
    if hasattr(args, 'scale') and args.scale is not None:
        data_augment['scale'] = args.scale
    if hasattr(args, 'translate') and args.translate is not None:
        data_augment['translate'] = args.translate
    if hasattr(args, 'fliplr') and args.fliplr is not None:
        data_augment['fliplr'] = args.fliplr
    if hasattr(args, 'flipud') and args.flipud is not None:
        data_augment['flipud'] = args.flipud
    if hasattr(args, 'degrees') and args.degrees is not None:
        data_augment['degrees'] = args.degrees
    
    # 高级几何增强
    if hasattr(args, 'shear') and args.shear is not None:
        data_augment['shear'] = args.shear
    if hasattr(args, 'perspective') and args.perspective is not None:
        data_augment['perspective'] = args.perspective
    
    # 颜色空间增强
    if hasattr(args, 'hsv_h') and args.hsv_h is not None:
        data_augment['hsv_h'] = args.hsv_h
    if hasattr(args, 'hsv_s') and args.hsv_s is not None:
        data_augment['hsv_s'] = args.hsv_s
    if hasattr(args, 'hsv_v') and args.hsv_v is not None:
        data_augment['hsv_v'] = args.hsv_v
    if hasattr(args, 'bgr') and args.bgr is not None:
        data_augment['bgr'] = args.bgr
    
    # 混合增强
    if hasattr(args, 'mosaic') and args.mosaic is not None:
        data_augment['mosaic'] = args.mosaic
    if hasattr(args, 'mixup') and args.mixup is not None:
        data_augment['mixup'] = args.mixup
    if hasattr(args, 'cutmix') and args.cutmix is not None:
        data_augment['cutmix'] = args.cutmix
    if hasattr(args, 'close_mosaic') and args.close_mosaic is not None:
        data_augment['close_mosaic'] = args.close_mosaic
    
    # 特殊增强
    if hasattr(args, 'copy_paste') and args.copy_paste is not None:
        data_augment['copy_paste'] = args.copy_paste
    if hasattr(args, 'erasing') and args.erasing is not None:
        data_augment['erasing'] = args.erasing
    if hasattr(args, 'auto_augment') and args.auto_augment is not None:
        data_augment['auto_augment'] = args.auto_augment
    
    # 训练参数
    if hasattr(args, 'multi_scale') and args.multi_scale:
        data_augment['multi_scale'] = True
    if hasattr(args, 'crop_fraction') and args.crop_fraction is not None:
        data_augment['crop_fraction'] = args.crop_fraction
    if hasattr(args, 'dropout') and args.dropout is not None:
        data_augment['dropout'] = args.dropout
    if hasattr(args, 'workers') and args.workers is not None:
        data_augment['workers'] = args.workers
    
    return data_augment if data_augment else None


def _apply_data_augment_kwargs(train_kwargs: Dict[str, Any], data_augment: Optional[Dict[str, Any]]) -> None:
    """
    将数据增强参数字典应用到训练参数中
    
    Args:
        train_kwargs: 训练参数字典（会被修改）
        data_augment: 数据增强参数字典
    """
    if not data_augment:
        return
    
    # 基础几何增强
    for key in ['scale', 'translate', 'fliplr', 'flipud', 'degrees']:
        if key in data_augment and data_augment[key] is not None:
            train_kwargs[key] = data_augment[key]
    
    # 高级几何增强
    for key in ['shear', 'perspective']:
        if key in data_augment and data_augment[key] is not None:
            train_kwargs[key] = data_augment[key]
    
    # 颜色空间增强
    for key in ['hsv_h', 'hsv_s', 'hsv_v', 'bgr']:
        if key in data_augment and data_augment[key] is not None:
            train_kwargs[key] = data_augment[key]
    
    # 混合增强参数
    for key in ['mosaic', 'mixup', 'cutmix', 'close_mosaic']:
        if key in data_augment and data_augment[key] is not None:
            train_kwargs[key] = data_augment[key]
    
    # 特殊增强
    for key in ['copy_paste', 'erasing', 'auto_augment']:
        if key in data_augment and data_augment[key] is not None:
            train_kwargs[key] = data_augment[key]
    
    # 训练参数
    for key in ['multi_scale', 'crop_fraction', 'dropout', 'workers']:
        if key in data_augment and data_augment[key] is not None:
            train_kwargs[key] = data_augment[key]


def train_yolo_model(
    model_family: str,
    model_size: str,
    task_type: str = "detect",
    task_prefix: Optional[str] = None,
    data_root: Optional[str] = None,
    output_path: Optional[str] = None,
    runs_dir: Optional[str] = None,
    train_mode: str = "pretrained",
    pretrained_model_path: Optional[str] = None,
    epochs: int = 10,
    batch_size: int = 4,
    imgsz: int = 640,
    device: str = "auto",
    lr0: Optional[float] = None,
    data_augment: Optional[Dict[str, Any]] = None,
    callbacks: Optional[Dict[str, Callable]] = None,
    verbose: bool = True,
    **kwargs
) -> Tuple[Any, str, str]:
    """
    核心YOLO训练函数，提取了训练的核心逻辑，供多个调用方复用
    
    Args:
        model_family: 模型系列 (yolo11/yolov8/yolov5)，应该是已标准化的值
        model_size: 模型大小 (n/s/m/l/x)
        task_type: 任务类型 (detect/instance)
        task_prefix: 任务名称前缀，如果为None则使用DEFAULT_TASK_NAME
        data_root: 数据集根目录路径
        output_path: 输出路径（优先使用，如果指定则忽略runs_dir）
        runs_dir: runs目录（如果output_path未指定时使用）
        train_mode: 训练模式 (scratch/pretrained/resume)
        pretrained_model_path: 预训练模型路径（可选）
        epochs: 训练轮数
        batch_size: 批次大小
        imgsz: 图像尺寸
        device: 训练设备（auto/cuda/mps/cpu）
        lr0: 初始学习率（可选）
        data_augment: 数据增强参数字典（可选）
        callbacks: 回调函数字典，格式如 {"on_train_batch_end": callback_func, ...}（可选）
        verbose: 是否打印详细信息
        **kwargs: 其他训练参数（会直接传递给train_kwargs）
    
    Returns:
        Tuple[train_results, task_name, runs_root]: 训练结果、任务名称、运行根目录
    
    Raises:
        ValueError: 如果数据集路径不存在或生成数据集配置失败
        SystemExit: 如果模型配置获取失败（仅在verbose模式下打印错误信息）
    """
    # 确定任务名称前缀
    if task_prefix is None:
        task_prefix = DEFAULT_TASK_NAME
    
    # 标准化任务类型
    if task_type.lower() in ["instance", "instance_segmentation", "segment"]:
        task_type = "instance"
    else:
        task_type = "detect"
    
    # 生成完整的任务名称
    task_name = get_task_full_name(task_prefix, model_family, model_size, task_type)
    
    # 确定输出根目录（优先使用output_path）
    if output_path:
        runs_root = output_path
        os.makedirs(runs_root, exist_ok=True)
    else:
        runs_root = get_runs_root(task_type, runs_dir)
    
    # 处理数据集配置
    dataset_config_file = None
    if data_root:
        if not os.path.exists(data_root):
            error_msg = f"数据集路径不存在: {data_root}"
            if verbose:
                print(f"错误: {error_msg}")
            raise ValueError(error_msg)
        
        try:
            if verbose:
                print("\n" + "=" * 80)
                print("自动生成数据集配置文件")
                print("=" * 80)
            dataset_config_file = generate_dataset_yaml(
                data_root=data_root,
                task_name=task_prefix,
                model_family=model_family,
                model_size=model_size,
                task_type=task_type,
                runs_dir=runs_dir
            )
            if verbose:
                print("=" * 80 + "\n")
        except Exception as e:
            error_msg = f"生成数据集配置文件失败: {e}"
            if verbose:
                print(f"✗ {error_msg}")
            raise ValueError(error_msg) from e
    
    # 获取模型配置
    model_path, is_yaml = get_model_config(
        model_family,
        model_size,
        train_mode,
        task_prefix,
        task_type,
        runs_dir
    )
    
    # 如果指定了预训练模型路径，使用指定的路径
    if pretrained_model_path and os.path.exists(pretrained_model_path):
        model_path = pretrained_model_path
    
    # 打印训练信息（如果verbose）
    if verbose:
        print("=" * 80)
        print(f"训练配置:")
        print(f"  模型系列: {model_family}")
        print(f"  模型大小: {model_size}")
        print(f"  任务类型: {task_type} ({'实例分割' if task_type == 'instance' else '目标检测'})")
        print(f"  训练模式: {train_mode}")
        print(f"  任务前缀: {task_prefix}")
        print(f"  完整任务名: {task_name}")
        if data_root:
            print(f"  数据集根目录: {data_root}")
            if dataset_config_file:
                print(f"  数据集配置: {dataset_config_file}")
        else:
            print(f"  数据集配置: {TRAIN_DATA} (默认)")
        print(f"  覆盖模式: 已启用（exist_ok=True，将覆盖已存在的训练结果）")
        print(f"  训练轮数: {epochs}")
        print(f"  批次大小: {batch_size}")
        print(f"  图像尺寸: {imgsz}")
        
        # 获取训练设备
        device_str = get_device_for_training(device) if device == "auto" else device
        print(f"  训练设备: {device_str} ({device}参数)")
        
        if train_mode == "scratch":
            print(f"  模型配置: {model_path} (从 YAML 配置文件构建，从零开始训练)")
        elif train_mode == "pretrained":
            print(f"  预训练权重: {model_path} (使用官方预训练权重)")
        else:  # resume
            if "last.pt" in model_path:
                print(f"  恢复训练: {model_path} (从 checkpoint 继续训练)")
            else:
                print(f"  预训练权重: {model_path} (checkpoint 不存在，已自动回退到预训练模式)")
        print("=" * 80)
    
    # 加载模型（可以是权重文件或 YAML 配置文件）
    model = YOLO(model_path)
    
    # 注册回调函数（如果有）
    if callbacks:
        for event_name, callback_func in callbacks.items():
            model.add_callback(event_name, callback_func)
    
    # 准备训练参数
    train_kwargs = deepcopy(TRAIN_KWARGS)
    train_kwargs["name"] = task_name
    
    # 如果生成了自定义数据集配置文件，使用它；否则使用默认配置
    if dataset_config_file:
        train_kwargs["data"] = dataset_config_file
    # 如果没有指定 data_root，train_kwargs["data"] 已经在 TRAIN_KWARGS 中设置
    
    # 设置基本训练参数
    train_kwargs["epochs"] = epochs
    train_kwargs["batch"] = batch_size
    train_kwargs["imgsz"] = imgsz
    
    # 获取训练设备
    if device == "auto":
        train_device = get_device_for_training(device)
    else:
        train_device = device
    train_kwargs["device"] = train_device
    
    # 设置学习率（如果指定）
    if lr0 is not None and lr0 > 0:
        train_kwargs["lr0"] = lr0
    
    # 应用数据增强参数
    if data_augment:
        _apply_data_augment_kwargs(train_kwargs, data_augment)
    
    # 应用额外的kwargs参数
    train_kwargs.update(kwargs)
    
    # 判断是否真正执行 resume：只有当 model_path 是 checkpoint 文件时才设置 resume=True
    is_actually_resuming = train_mode == "resume" and "last.pt" in str(model_path)
    train_kwargs["resume"] = is_actually_resuming
    
    # 设置输出目录
    train_kwargs["project"] = runs_root
    
    # 开始训练
    if verbose:
        print(f"\n开始训练，结果将保存到: {runs_root}/{task_name}/")
        print(f"注意: 如果该文件夹已存在，将覆盖之前的训练结果\n")
    
    train_results = model.train(**train_kwargs)
    
    if verbose:
        # 打印训练完成信息
        print("=" * 80)
        print(f"训练完成！")
        print(f"结果保存在: {runs_root}/{task_name}/")
        print(f"  - 最佳权重: {runs_root}/{task_name}/weights/best.pt")
        print(f"  - 最后权重: {runs_root}/{task_name}/weights/last.pt")
        print(f"  - 训练曲线: {runs_root}/{task_name}/results.png")
        print(f"  - 训练参数: {runs_root}/{task_name}/args.yaml")
        if dataset_config_file and data_root:
            config_filename = Path(dataset_config_file).name
            print(f"  - 数据集配置: {runs_root}/{task_name}/{config_filename}")
        print("=" * 80)
    
    return train_results, task_name, runs_root


def main() -> None:
    """
    训练入口函数，支持三种训练模式：
    1. scratch: 从零开始训练（不使用预训练权重）
    2. pretrained: 使用官方预训练权重训练（默认）
    3. resume: 从 checkpoint 继续训练
    
    支持自定义任务名称，每次训练使用固定文件夹（覆盖模式）
    """
    args = parse_args()
    
    # 标准化模型系列名称（支持变体输入）
    try:
        model_family = normalize_model_family(args.model_family)
    except ValueError as e:
        print(f"错误: {e}")
        sys.exit(1)
    
    model_size = args.model_size.lower()
    train_mode = args.train_mode.lower()
    
    # 确定任务类型：如果指定了 --seg，使用 instance，否则使用 detect
    task_type = "instance" if args.seg else "detect"
    
    # 确定任务名称前缀：优先使用用户指定的，否则使用默认值
    task_prefix = args.task_name if args.task_name else DEFAULT_TASK_NAME
    
    # 从 args 提取数据增强参数
    data_augment = _args_to_data_augment_dict(args)
    
    # 提取优化器参数（如果指定）
    lr0 = args.lr0 if hasattr(args, 'lr0') and args.lr0 is not None else None
    optimizer = args.optimizer if hasattr(args, 'optimizer') and args.optimizer is not None else None
    
    # 准备额外的 kwargs（用于 optimizer 等参数）
    extra_kwargs = {}
    if optimizer:
        extra_kwargs['optimizer'] = optimizer
    
    # 调用核心训练函数
    try:
        train_results, task_name, runs_root = train_yolo_model(
            model_family=model_family,
            model_size=model_size,
            task_type=task_type,
            task_prefix=task_prefix,
            data_root=args.data_root if hasattr(args, 'data_root') and args.data_root else None,
            output_path=None,  # main() 使用默认路径
            runs_dir=args.runs_dir if hasattr(args, 'runs_dir') and args.runs_dir else None,
            train_mode=train_mode,
            pretrained_model_path=None,  # main() 使用默认模型配置
            epochs=args.epochs,
            batch_size=args.batch,
            imgsz=args.imgsz,
            device=args.device,
            lr0=lr0,
            data_augment=data_augment,
            callbacks=None,  # main() 不使用回调
            verbose=True,
            **extra_kwargs
        )
    except ValueError as e:
        print(f"错误: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()