"""
训练配置文件：集中管理 YOLO 训练脚本中用到的所有可调参数

支持三种训练模式：
1. scratch: 从 YAML 配置文件构建模型，从零开始训练（不使用预训练权重）
2. pretrained: 使用官方预训练权重进行迁移学习
3. resume: 从之前的 checkpoint 继续训练
"""

from pathlib import Path
from typing import Dict, Any, Literal, List
import yaml
import re
import torch

# ==================== 路径相关配置 ====================
# YAML 配置文件基础路径
YAML_BASE_DIR: str = "./ultralytics/ultralytics/cfg/models"

# 训练数据配置
TRAIN_DATA: str = "./ultralytics/ultralytics/cfg/datasets/coco8.yaml"  # 默认数据集配置
DATASETS_CONFIG_DIR: str = "./ultralytics/ultralytics/cfg/datasets"  # 数据集配置文件存放目录

# 项目内统一的 runs 根目录（用于强制训练输出位置，避免引用全局 Ultralytics 设置）
PROJECT_RUNS_DIR: Path = Path(__file__).resolve().parent.parent / "runs"


def get_runs_root(task_type: str = "detect", base_dir: str | Path | None = None) -> str:
    """
    获取训练输出的根目录（区分 detect/instance），允许通过 base_dir 覆盖默认路径。
    
    Args:
        task_type: 任务类型，"detect" 或 "instance"
        base_dir: 自定义 runs 根目录（可选），为空则使用 PROJECT_RUNS_DIR
    
    Returns:
        对应任务类型的 runs 根目录字符串路径
    """
    task_type = "instance" if task_type.lower() == "instance" else "detect"
    root_base = Path(base_dir).expanduser().resolve() if base_dir else PROJECT_RUNS_DIR
    runs_root = root_base / task_type 
    runs_root.mkdir(parents=True, exist_ok=True)
    return str(runs_root)

# ==================== 模型配置 ====================
# 支持的训练模式
TrainMode = Literal["scratch", "pretrained", "resume"]
SUPPORTED_TRAIN_MODES: list[str] = ["scratch", "pretrained", "resume"]

# 支持的模型系列
SUPPORTED_MODEL_FAMILIES: list[str] = ["yolov5", "yolov8", "yolo11"]

# 支持的模型大小
SUPPORTED_MODEL_SIZES: list[str] = ["n", "s", "m", "l", "x"]

# 默认配置
DEFAULT_MODEL_FAMILY: str = "yolo11"
DEFAULT_MODEL_SIZE: str = "s"
DEFAULT_TRAIN_MODE: str = "resume"
DEFAULT_TASK_NAME: str = "detect"  # 默认任务名前缀，完整格式为 {task_name}_{model_family}_{model_size}
DEFAULT_TASK_TYPE: str = "detect"  # 默认任务类型：detect（检测）或 instance（实例分割）


def get_task_full_name(task_name: str, model_family: str, model_size: str, task_type: str = "detect") -> str:
    """
    生成完整的任务名称。
    
    Args:
        task_name: 任务名称前缀，例如 "detect", "instance" 等
        model_family: 模型系列，例如 "yolo11", "yolov8", "yolov5"
        model_size: 模型大小，例如 "n", "s", "m", "l", "x"
        task_type: 任务类型，例如 "detect" 或 "instance"
    
    Returns:
        完整的任务名称，格式为 {task_name}_{model_family}_{model_size}
        例如: "detect_yolo11_s", "instance_yolov8_m"
    """
    return f"{task_name}_{model_family}_{model_size}"


def read_classes_from_file(classes_file: str) -> List[str]:
    """
    从 classes.txt 文件中读取类别列表。
    
    支持多种格式：
    - "class_name" - 只有类别名称
    - "1:class_name" - 数字ID后跟冒号
    - "1,class_name" - 数字ID后跟逗号
    - "1;class_name" - 数字ID后跟分号
    - "1 class_name" - 数字ID后跟空格
    - 混合格式如 "1: class_name", "1, class_name" 等
    
    Args:
        classes_file: classes.txt 文件路径
    
    Returns:
        类别名称列表（按文件中的顺序）
    
    Raises:
        FileNotFoundError: 如果 classes.txt 文件不存在
        ValueError: 如果文件为空或格式错误
    """
    classes_path = Path(classes_file)
    
    if not classes_path.exists():
        raise FileNotFoundError(f"类别文件不存在: {classes_file}")
    
    classes = []
    with open(classes_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:  # 跳过空行
                continue
            
            # 尝试解析带ID的格式，支持逗号、分号、冒号和空格作为分隔符
            # 例如: "1:class_name", "1, class_name", "1; class_name", "1 class_name"
            # 正则解释:
            # ^\s*(\d+) : 匹配行首可能的空格和数字ID
            # [\s:;,]+  : 匹配一个或多个分隔符（空格、冒号、分号、逗号）
            # (.+)$     : 匹配剩余的所有内容作为类别名称
            match = re.match(r'^\s*(\d+)[\s:;,]+(.+)$', line)
            if match:
                class_name = match.group(2).strip()
                if class_name:
                    classes.append(class_name)
                    continue
            
            # 格式3: 直接是类别名称（没有ID）
            classes.append(line)
    
    if not classes:
        raise ValueError(f"类别文件为空或无法解析: {classes_file}")
    
    return classes


def generate_dataset_yaml(
    data_root: str,
    task_name: str,
    model_family: str,
    model_size: str,
    output_dir: str = None,
    task_type: str = "detect",
    runs_dir: str | Path | None = None
) -> str:
    """
    根据数据集根目录自动生成 YOLO 格式的数据集配置文件。
    
    自动遍历数据集根目录下所有包含图片的文件夹，根据路径名称中包含 train/val/test 进行分类。
    支持多路径训练集配置（train 可以是列表格式）。
    
    支持的图片格式：bmp, dng, jpeg, jpg, mpo, png, tif, tiff, webp, pfm, heic
    
    智能处理规则：
    - 遍历所有包含图片的目录，根据路径名称分类（包含 train/val/test）
    - train 可以是多个路径（列表格式），例如：["images/train", "images/train_extra"]
    - val 如果不存在，自动使用 train 的第一个路径；如果存在多个，使用第一个
    - test 如果不存在，不会在配置文件中添加 test 字段
    
    示例目录结构：
    data_root/
    ├── classes.txt
    ├── images/
    │   ├── train/          # 会被识别为训练集
    │   ├── train_extra/    # 会被识别为训练集（多路径）
    │   ├── val/            # 会被识别为验证集
    │   └── test/           # 会被识别为测试集
    └── labels/
        ├── train/
        ├── train_extra/
        ├── val/
        └── test/
    
    Args:
        data_root: 数据集根目录路径，例如 "/data1/zhn/carparts"
        task_name: 任务名称前缀
        model_family: 模型系列
        model_size: 模型大小
        output_dir: 配置文件输出目录，默认为训练输出目录 runs/detect/{task_name}_{model_family}_{model_size}/
        task_type: 任务类型，"detect"（检测）或 "instance"（实例分割）
    
    Returns:
        生成的 YAML 配置文件路径
    
    Raises:
        FileNotFoundError: 如果数据集根目录或 classes.txt 不存在
        ValueError: 如果没有找到任何包含图片的训练集目录
    """
    # 支持的图片格式
    IMG_FORMATS = {"bmp", "dng", "jpeg", "jpg", "mpo", "png", "tif", "tiff", "webp", "pfm", "heic"}
    
    data_root_path = Path(data_root).resolve()
    
    # 验证数据集根目录
    if not data_root_path.exists():
        raise FileNotFoundError(f"数据集根目录不存在: {data_root}")
    
    classes_file = data_root_path / "classes.txt"
    if not classes_file.exists():
        raise FileNotFoundError(f"类别文件不存在: {classes_file}")
    
    def contains_images(dir_path: Path) -> bool:
        """检查目录是否包含图片文件"""
        if not dir_path.is_dir():
            return False
        try:
            for file in dir_path.iterdir():
                if file.is_file():
                    suffix = file.suffix.lower().lstrip('.')
                    if suffix in IMG_FORMATS:
                        return True
        except PermissionError:
            return False
        return False
    
    def get_relative_path(full_path: Path) -> str:
        """获取相对于 data_root_path 的路径"""
        try:
            return str(full_path.relative_to(data_root_path))
        except ValueError:
            # 如果不在同一路径下，返回绝对路径（虽然不应该发生）
            return str(full_path)
    
    # 遍历所有目录，找到包含图片的目录
    train_paths = []  # 训练集路径列表
    val_paths = []    # 验证集路径列表
    test_paths = []   # 测试集路径列表
    
    # 递归遍历所有目录
    for dir_path in data_root_path.rglob('*'):
        if not dir_path.is_dir():
            continue
        
        # 检查目录是否包含图片
        if not contains_images(dir_path):
            continue
        
        # 获取相对路径
        rel_path = get_relative_path(dir_path)
        rel_path_lower = rel_path.lower()
        
        # 根据路径名称分类（不区分大小写）
        # 检查路径中是否包含 train/val/test 关键词
        if 'train' in rel_path_lower:
            train_paths.append(rel_path)
        elif 'val' in rel_path_lower or 'valid' in rel_path_lower:
            val_paths.append(rel_path)
        elif 'test' in rel_path_lower:
            test_paths.append(rel_path)
    
    # 去重并排序
    train_paths = sorted(list(set(train_paths)))
    val_paths = sorted(list(set(val_paths)))
    test_paths = sorted(list(set(test_paths)))
    
    # 验证：至少需要一个训练集路径
    if not train_paths:
        raise ValueError(
            f"未找到任何包含图片的训练集目录。请确保数据集根目录下存在包含 train 关键词的目录，"
            f"且该目录中包含图片文件。\n"
            f"数据集根目录: {data_root_path}"
        )
    
    # 处理训练集路径
    # 如果只有一个训练路径，直接使用字符串；如果有多个，使用列表
    if len(train_paths) == 1:
        train_config = train_paths[0]
    else:
        train_config = train_paths  # 列表格式
    
    # 处理验证集路径
    use_train_as_val = False
    if val_paths:
        # 如果存在验证集路径，使用第一个
        val_config = val_paths[0]
        if len(val_paths) > 1:
            print(f"⚠ 找到多个验证集路径，使用第一个: {val_config}")
            print(f"  其他验证集路径: {', '.join(val_paths[1:])}")
    else:
        # 如果不存在验证集，使用训练集的第一个路径
        if isinstance(train_config, list):
            val_config = train_config[0]
        else:
            val_config = train_config
        use_train_as_val = True
        print(f"⚠ 未找到验证集目录，将使用训练集路径作为验证集: {val_config}")
    
    # 处理测试集路径（可选）
    test_config = None
    if test_paths:
        # 如果存在测试集路径，使用第一个
        test_config = test_paths[0]
        if len(test_paths) > 1:
            print(f"⚠ 找到多个测试集路径，使用第一个: {test_config}")
            print(f"  其他测试集路径: {', '.join(test_paths[1:])}")
    
    # 读取类别
    classes = read_classes_from_file(str(classes_file))
    
    # 生成配置文件名
    config_filename = f"{task_name}_{model_family}_{model_size}_data.yaml"
    
    # 确定输出目录：默认放在训练输出目录中，和模型权重在一起
    if output_dir is None:
        full_task_name = get_task_full_name(task_name, model_family, model_size, task_type)
        runs_root = get_runs_root(task_type, runs_dir)
        output_dir = f"{runs_root}/{full_task_name}"
    
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    
    config_file_path = output_dir_path / config_filename
    
    # 构建 YAML 配置内容
    yaml_config = {
        'path': str(data_root_path),  # 数据集根目录（绝对路径）
        'train': train_config,         # 训练集图片路径（相对于 path），可以是字符串或列表
        'val': val_config,             # 验证集图片路径（相对于 path）
        'names': {i: name for i, name in enumerate(classes)},  # 类别字典
    }
    
    # 只有当 test 路径存在时才添加到配置中
    if test_config is not None:
        yaml_config['test'] = test_config
    
    # 写入 YAML 文件
    with open(config_file_path, 'w', encoding='utf-8') as f:
        yaml.dump(yaml_config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    
    # 打印详细信息
    print(f"✓ 成功生成数据集配置文件")
    print(f"  配置文件: {config_file_path}")
    print(f"  数据集路径: {data_root_path}")
    print(f"  训练集路径数量: {len(train_paths)}")
    if isinstance(train_config, list):
        print(f"  训练集路径: {train_config}")
    else:
        print(f"  训练集路径: {train_config}")
    print(f"  验证集路径: {val_config}{' (使用训练集)' if use_train_as_val else ''}")
    if test_config:
        print(f"  测试集路径: {test_config}")
    print(f"  类别数量: {len(classes)}")
    print(f"  类别列表: {', '.join(classes[:5])}{'...' if len(classes) > 5 else ''}")
    print(f"  说明: 配置文件将与训练权重保存在同一目录")
    
    return str(config_file_path)


def get_yaml_path(model_family: str, model_size: str, task_type: str = "detect") -> str:
    """
    根据模型系列、大小和任务类型生成 YAML 配置文件路径（包含 scale 信息）。
    
    注意：虽然实际的 YAML 文件名是 yolo11.yaml，但我们返回 yolo11n.yaml 这样的格式，
    Ultralytics 会自动识别 scale 后缀（n/s/m/l/x）并应用对应的缩放参数。

    Args:
        model_family: 模型系列，例如 "yolo11", "yolov8", "yolov5"
        model_size: 模型大小，例如 "n", "s", "m", "l", "x"
        task_type: 任务类型，"detect"（检测）或 "instance"（实例分割）
    
    Returns:
        YAML 配置文件路径（包含 scale），例如 "./ultralytics/ultralytics/cfg/models/11/yolo11n.yaml"
        或 "./ultralytics/ultralytics/cfg/models/11/yolo11n-seg.yaml"
    """
    model_family = model_family.lower()
    model_size = model_size.lower()
    task_type = task_type.lower()
    
    # 确定后缀：如果是实例分割任务，添加 -seg（ultralytics框架要求）
    suffix = "-seg" if task_type == "instance" else ""
    
    # 对于 yolo11，返回 ./ultralytics/ultralytics/cfg/models/11/yolo11n.yaml 或 yolo11n-seg.yaml
    if model_family == "yolo11":
        return f"{YAML_BASE_DIR}/11/yolo11{model_size}{suffix}.yaml"
    # 对于 yolov8，返回 ./ultralytics/ultralytics/cfg/models/v8/yolov8n.yaml 或 yolov8n-seg.yaml
    elif model_family == "yolov8":
        return f"{YAML_BASE_DIR}/v8/yolov8{model_size}{suffix}.yaml"
    # 对于 yolov5，返回 ./ultralytics/ultralytics/cfg/models/v5/yolov5n.yaml 或 yolov5n-seg.yaml
    elif model_family == "yolov5":
        return f"{YAML_BASE_DIR}/v5/yolov5{model_size}{suffix}.yaml"
    else:
        raise ValueError(f"不支持的模型系列: {model_family}，支持的系列: {SUPPORTED_MODEL_FAMILIES}")


def get_official_weight_name(model_family: str, model_size: str, task_type: str = "detect") -> str:
    """
    根据模型系列、大小和任务类型生成官方权重名称。

    Args:
        model_family: 模型系列，例如 "yolo11", "yolov8", "yolov5"
        model_size: 模型大小，例如 "n", "s", "m", "l", "x"
        task_type: 任务类型，"detect"（检测）或 "instance"（实例分割）
    
    Returns:
        官方权重名称，例如 "yolo11s.pt", "yolov8n-seg.pt"
    """
    model_family = model_family.lower()
    model_size = model_size.lower()
    task_type = task_type.lower()
    
    # 确定后缀：如果是实例分割任务，添加 -seg（ultralytics框架要求）
    suffix = "-seg" if task_type == "instance" else ""
    
    # 对于 yolo11，格式为 yolo11n.pt 或 yolo11n-seg.pt
    if model_family == "yolo11":
        return f"yolo11{model_size}{suffix}.pt"
    # 对于 yolov8，格式为 yolov8n.pt 或 yolov8n-seg.pt
    elif model_family == "yolov8":
        return f"yolov8{model_size}{suffix}.pt"
    # 对于 yolov5，格式为 yolov5s.pt 或 yolov5s-seg.pt
    elif model_family == "yolov5":
        return f"yolov5{model_size}{suffix}.pt"
    else:
        raise ValueError(f"不支持的模型系列: {model_family}，支持的系列: {SUPPORTED_MODEL_FAMILIES}")


def get_model_config(
    model_family: str, 
    model_size: str, 
    train_mode: str = "pretrained",
    task_name: str = None,
    task_type: str = "detect",
    runs_dir: str | Path | None = None
) -> tuple[str, bool]:
    """
    根据模型系列、大小、训练模式和任务类型自动选择模型配置。
    
    Args:
        model_family: 模型系列，例如 "yolo11", "yolov8", "yolov5"
        model_size: 模型大小，例如 "n", "s", "m", "l", "x"
        train_mode: 训练模式，可选值：
            - "scratch": 从 YAML 配置文件从头训练（不使用预训练权重）
            - "pretrained": 使用官方预训练权重进行训练
            - "resume": 从之前的 checkpoint 继续训练（如果 checkpoint 不存在，自动回退到 pretrained）
        task_name: 任务名称前缀（用于 resume 模式定位 checkpoint），如果为 None 则使用 DEFAULT_TASK_NAME
        task_type: 任务类型，"detect"（检测）或 "instance"（实例分割）
    
    Returns:
        (model_path, is_yaml): 
            - model_path: 模型路径（权重文件路径或 YAML 文件路径）
            - is_yaml: True 表示是 YAML 配置文件（从零训练），False 表示是权重文件
    
    Raises:
        ValueError: 如果训练模式不支持
    
    Note:
        在 resume 模式下，如果找不到 checkpoint 文件，会自动回退到 pretrained 模式，避免训练失败。
    """
    model_family = model_family.lower()
    model_size = model_size.lower()
    train_mode = train_mode.lower()
    task_type = task_type.lower()
    
    # 验证训练模式
    if train_mode not in SUPPORTED_TRAIN_MODES:
        raise ValueError(
            f"不支持的训练模式: {train_mode}，支持的模式: {', '.join(SUPPORTED_TRAIN_MODES)}"
        )
    
    # 根据任务类型选择输出目录（支持自定义 runs 根目录）
    runs_root = get_runs_root(task_type, runs_dir)
    
    # 模式1: 从 checkpoint 继续训练
    if train_mode == "resume":
        # 确定任务名称：如果未指定则使用默认前缀
        if task_name is None:
            task_name = DEFAULT_TASK_NAME
        # 生成完整的任务名称：{task_name}_{model_family}_{model_size}
        full_task_name = get_task_full_name(task_name, model_family, model_size, task_type)
        model_dir = f"{runs_root}/{full_task_name}"
        checkpoint_path = f"{model_dir}/weights/last.pt"
        
        # 检查 checkpoint 文件是否存在
        checkpoint_file = Path(checkpoint_path)
        if checkpoint_file.exists():
            print(f"✓ 找到 checkpoint 文件: {checkpoint_path}")
            return checkpoint_path, False
        else:
            # checkpoint 不存在，自动回退到预训练模式
            print(f"⚠ checkpoint 文件不存在: {checkpoint_path}")
            print(f"⚠ 自动回退到 pretrained 模式，使用官方预训练权重")
            weight_name = get_official_weight_name(model_family, model_size, task_type)
            return weight_name, False

    # 模式2: 使用官方预训练权重
    elif train_mode == "pretrained":
        weight_name = get_official_weight_name(model_family, model_size, task_type)
        return weight_name, False

    # 模式3: 从 YAML 配置文件构建模型（从零开始训练）
    else:  # train_mode == "scratch"
        yaml_path = get_yaml_path(model_family, model_size, task_type)
        return yaml_path, True


def get_device_for_training(device: str = "auto") -> str:
    """获取训练设备，支持自动选择，优先检查MPS支持"""
    if device == "auto":
        # 优先检查MPS支持
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"
        elif torch.cuda.is_available():
            return "cuda"
        else:
            return "cpu"
    else:
        # 验证指定设备是否可用
        if device == "cuda":
            # 先检查MPS是否可用，如果可用则优先使用MPS
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                print("警告: 检测到MPS支持，自动切换到MPS")
                return "mps"
            elif not torch.cuda.is_available():
                print("警告: CUDA不可用且MPS不可用，自动切换到CPU")
                return "cpu"
            else:
                return "cuda"
        elif device == "mps" and (not hasattr(torch.backends, 'mps') or not torch.backends.mps.is_available()):
            print("警告: MPS不可用，自动切换到CPU")
            return "cpu"
        return device

# ==================== 训练超参数配置 ====================
# 这里的所有键会直接传给 model.train(**TRAIN_KWARGS)
TRAIN_KWARGS: Dict[str, Any] = {
    # 数据与基本训练设置
    "data": TRAIN_DATA,
    "epochs": 10,        # 增加总 epoch 数
    "imgsz": 1024,       # 输入图像尺寸
    "device": "auto",    # 使用自动设备选择
    # 指定训练输出目录根路径，防止使用全局Ultralytics runs_dir
    "project": str(PROJECT_RUNS_DIR),
    # 实验名称（可以在调用处按模型类型覆盖，如 f"{model_type}_detect"）
    "name": "yolo_detect",
    "batch": 4,
    # resume 参数由训练模式决定，不在这里设置
    "exist_ok": True,    # 允许覆盖已存在的训练文件夹，不创建递增文件夹

    # 模型设置
    "freeze": 0,         # 不冻结任何层

    # 模型保存设置
    "save": True,        # 保存 checkpoint（best.pt & last.pt）
    "save_period": 10,   # 每 10 个 epoch 保存一次

    # 损失函数权重
    "box": 7.5,
    "cls": 0.5,
    "dfl": 1.5,

    # 学习率设置
    "lr0": 0.001,        # 初始学习率
    "lrf": 0.01,         # 最终学习率因子
    "cos_lr": True,      # 余弦学习率调度

    # 优化器设置
    "optimizer": "AdamW",
    "momentum": 0.937,
    "weight_decay": 0.0005,

    # 早停策略
    "patience": 50,

    # 数据增强（基本关闭，仅保留轻微缩放）
    # 基础几何变换（最常用）
    "scale": 0.1,              # 图像缩放幅度范围，轻微缩放有助于提升模型鲁棒性
    "translate": 0.0,          # 图像平移幅度比例，0表示不进行平移
    "fliplr": 0.0,             # 水平翻转概率，0表示不进行水平翻转
    "flipud": 0.0,             # 垂直翻转概率，0表示不进行垂直翻转
    "degrees": 0.0,            # 图像旋转角度范围，0表示不进行旋转

    # 高级几何变换（较常用）
    "shear": 0.0,              # 图像剪切角度范围，0表示不进行剪切变换
    "perspective": 0.0,        # 透视变换强度，0表示不进行透视变换

    # 颜色空间变换（常用）
    "hsv_h": 0.0,              # HSV色调增强幅度，0表示不进行色调调整
    "hsv_s": 0.0,              # HSV饱和度增强幅度，0表示不进行饱和度调整
    "hsv_v": 0.0,              # HSV亮度增强幅度，0表示不进行亮度调整
    "bgr": 0.0,                # RGB转BGR通道概率，0表示不进行通道转换

    # 混合增强（YOLO特色，训练前期常用）
    "mosaic": 0.0,             # Mosaic数据增强概率，0表示不使用Mosaic增强
    "mixup": 0.0,              # MixUp数据增强概率，0表示不使用MixUp增强
    "cutmix": 0.0,             # CutMix数据增强概率，0表示不使用CutMix增强
    "close_mosaic": 0,         # 在最后N个epoch关闭mosaic增强，0表示不关闭

    # 特殊增强（较少使用）
    "copy_paste": 0.0,         # Copy-Paste增强概率，0表示不使用Copy-Paste增强
    "erasing": 0.0,            # 随机擦除增强概率，0表示不进行随机擦除
    "auto_augment": None,      # 自动增强策略，None表示不使用自动增强

    # 多尺度和其他训练参数
    "multi_scale": False,      # 多尺度训练，随机改变输入图像大小，提高模型对不同尺度目标的泛化能力
    "crop_fraction": 1.0,      # 数据裁剪使用的比例，1.0表示不裁剪，使用完整图像

    # 模型和训练相关
    "dropout": 0.0,            # 分类头的dropout率，防止过拟合，0表示不使用dropout
    "workers": 0,              # 数据加载器的工作线程数，0表示主线程加载，影响数据加载速度
}


