from ultralytics import YOLO

# Load a model
model = YOLO("/Users/zhanghaining/git/ultralytics/ultralytics/cfg/models/11/yolo11-seg.yaml")  # build a new model from YAML
model = YOLO("/Users/zhanghaining/git/ultralytics/yolo11n-seg.pt")  # load a pretrained model (recommended for training)
# model = YOLO("yolo11n-seg.yaml").load("yolo11n.pt")  # build from YAML and transfer weights

# Train the model
results = model.train(data="/Users/zhanghaining/git/ultralytics/ultralytics/cfg/datasets/coco128-seg.yaml", epochs=100, imgsz=640, device="mps")