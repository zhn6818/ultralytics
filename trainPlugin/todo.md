# TODO：训练参数增强任务列表

## 已完成任务 ✅
### 1. 添加优化器和学习率参数支持
- --optimizer: 优化器类型 (SGD, Adam, AdamW, RMSprop)
- --lr0: 初始学习率
- [x] 修改 YoloTrain.py 的 parse_args() 函数
- [x] 修改 YoloTrain.py 的 main() 函数
- [x] 更新帮助文档，添加使用示例

## 新任务 📝
### 2. 添加缺失的训练参数（7个）

#### 需要添加的7个参数
1. [x] multi_scale: False  # 多尺度训练，随机改变输入图像大小
2. [x] close_mosaic: 0     # 在最后N个epoch关闭mosaic增强，0表示不关闭
3. [x] mask_ratio: 1.0     # 分割任务中mask的下采样比例
4. [x] overlap_mask: True  # 训练期间是否合并实例掩码
5. [x] crop_fraction: 1.0  # 数据裁剪使用的比例
6. [x] dropout: 0.0        # 分类头的dropout率，防止过拟合
7. [x] workers: 0          # 数据加载器的工作线程数，0表示主线程加载

#### 实施步骤
1. [x] 在 config.py 的 TRAIN_KWARGS 中添加这7个新参数
2. [x] 为每个参数添加详细的中文注释说明其作用

## 使用示例

### 优化器参数（已完成）
```bash
# 使用默认优化器参数
python YoloTrain.py

# 使用 SGD 优化器
python YoloTrain.py --optimizer SGD --lr0 0.01

# 使用 Adam 优化器，自定义学习率
python YoloTrain.py --optimizer Adam --lr0 0.0001

# 使用 AdamW 优化器，自定义学习率
python YoloTrain.py --optimizer AdamW --lr0 0.001
```

### 新增训练参数（已完成）
已在 config.py 中添加5个参数：
```python
# 新增训练参数
"multi_scale": False,      # 多尺度训练，随机改变输入图像大小
"close_mosaic": 0,         # 最后N个epoch关闭mosaic增强
"crop_fraction": 1.0,      # 数据裁剪使用的比例
"dropout": 0.0,            # 分类头的dropout率，防止过拟合
"workers": 0,              # 数据加载器的工作线程数
```

### 3. 创建 parseTrainParam.py 管理参数解析（已完成）

#### 支持的22个数据增强命令行参数
1. [x] --scale: 图像缩放幅度
2. [x] --translate: 图像平移幅度
3. [x] --fliplr: 水平翻转概率
4. [x] --flipud: 垂直翻转概率
5. [x] --degrees: 旋转角度范围
6. [x] --shear: 剪切角度
7. [x] --perspective: 透视变换强度
8. [x] --hsv_h: HSV色调增强
9. [x] --hsv_s: HSV饱和度增强
10. [x] --hsv_v: HSV亮度增强
11. [x] --bgr: RGB转BGR概率
12. [x] --mosaic: Mosaic增强概率
13. [x] --mixup: MixUp增强概率
14. [x] --cutmix: CutMix增强概率
15. [x] --close_mosaic: 关闭mosaic的epoch数
16. [x] --copy_paste: Copy-Paste增强概率
17. [x] --erasing: 随机擦除概率
18. [x] --auto_augment: 自动增强策略
19. [x] --multi_scale: 多尺度训练
20. [x] --crop_fraction: 裁剪比例
21. [x] --dropout: Dropout率
22. [x] --workers: 数据加载线程数

#### 完成的功能
1. [x] 创建 parseTrainParam.py 文件
2. [x] 实现参数分组展示（6个参数组）
3. [x] 创建 add_augmentation_args() 函数
4. [x] 创建 update_train_kwargs() 函数
5. [x] 在 YoloTrain.py 中导入并使用新模块
6. [x] 更新帮助文档，添加数据增强参数使用示例

### 数据增强参数使用示例
```bash
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

# 组合使用
python YoloTrain.py \
  --optimizer SGD --lr0 0.01 \
  --fliplr 0.5 --degrees 10 \
  --mosaic 1.0 \
  --multi_scale \
  --workers 8
```