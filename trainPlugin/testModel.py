"""训练模型测试脚本 - 单张图片测试"""

from pathlib import Path
import sys
# 添加项目根目录到 sys.path，确保能找到 ultralytics 模块
# YoloTrain.py 位于: ultralytics/trainPlugin/YoloTrain.py
# 项目根目录: SteelAiLab/
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from ultralytics import YOLO

# 加载模型
model = YOLO("ultralytics/runs/detect/afica_yolo11_s/weights/best.pt")

# 遍历验证集图片
image_dir = Path("/data1/zhn/test/")
image_paths = sorted(
    [
        p
        for p in image_dir.glob("*")
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    ]
)

if not image_paths:
    raise FileNotFoundError(f"未在 {image_dir} 下找到可用图片")

for image_path in image_paths:
    print(f"\n开始处理: {image_path.name}")
    results = model.predict(
        source=str(image_path), imgsz=1024, conf=0.25, device="cpu", save=True
    )

    # 显示结果
    result = results[0]
    if result.boxes is not None and len(result.boxes):
        print(f"检测到 {len(result.boxes)} 个对象:")
        for box in result.boxes:
            cls_name = model.names[int(box.cls[0])]
            conf = float(box.conf[0])
            print(f"  {cls_name}: {conf:.2%}")
    else:
        print("未检测到对象")

