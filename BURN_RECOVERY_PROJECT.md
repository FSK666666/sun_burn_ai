# 灼烧恢复网络项目复现说明

本文档记录当前项目的数据、合成灼烧标签、网络结构、训练流程、可视化和推理方法。目标是让别人拿到同样的数据目录和代码后，可以完整复现实验。

## 1. 项目目标

本项目面向红外图像中的太阳灼烧/残留亮斑恢复任务。

训练样本由连续红外帧组成。每个样本包含 5 帧干净红外图像，并人为合成一个固定空间位置的灼烧痕迹。网络输入是加入灼烧后的 5 帧图像，输出两个结果：

- `P`：灼烧概率图，表示每个像素是否属于灼烧区域。
- `C`：灼烧强度/偏置图，表示需要从输入帧中减掉多少亮度。

恢复时使用：

```python
restored = future_frame - (P > threshold) * C
```

## 2. 当前代码文件

核心文件如下：

```text
C:\Users\17874\Documents\python
├── preview_burn_on_training_groups.py   # 合成灼烧标签、预览、自动写入
├── burn_recovery_net.py                 # BurnRecoveryNet 网络结构
├── train_burn_recovery.py               # 训练脚本
├── 思路.txt                             # 网络设计思路
└── BURN_RECOVERY_PROJECT.md             # 本文档
```

其中 `preview_burn_on_training_groups_multiprocess.py`、`preview_burn_on_training_groups_multithread.py` 是历史/实验版本，当前主要使用 `preview_burn_on_training_groups.py`。

## 3. 数据目录

当前数据根目录：

```text
C:\Users\17874\Documents\python\datasets
```

灼烧恢复训练数据位于：

```text
C:\Users\17874\Documents\python\datasets\burn_recovery
├── train       # 训练集，46304 个样本
├── val         # 验证集，2880 个样本
├── test        # KAIST 测试集，44500 个样本
├── test_flir   # FLIR 外部测试集，3309 个样本
└── manifests   # 数据划分清单
```

数据划分规则记录在：

```text
C:\Users\17874\Documents\python\datasets\burn_recovery\manifests\summary.txt
```

当前 summary 内容要点：

```text
rule: overlapping windows, frames t,t+10,t+20,t+30,t+40, stride 1 within each original sequence
KAIST train=set00-set04, val=set05, test=set06-set11
FLIR video_thermal_test is external test_flir only
train: 46304 samples
val: 2880 samples
test: 44500 samples
test_flir: 3309 samples
file placement: hardlinks=484965 copies=0
```

也就是说，每个样本不是 5 个相邻帧，而是在原始连续序列中取：

```text
t, t+10, t+20, t+30, t+40
```

并且用 stride 1 产生重叠样本，提升样本数量。

## 4. 单个样本结构

每个样本文件夹形如：

```text
sample_000001
├── 000.jpg
├── 001.jpg
├── 002.jpg
├── 003.jpg
├── 004.jpg
├── synthetic_burn_trace.npy
├── synthetic_burn_trace.png
├── synthetic_burn_probability.png
└── synthetic_burn_metadata.txt
```

含义：

- `000.jpg` 到 `004.jpg`：5 帧干净红外图像。
- `synthetic_burn_trace.npy`：float32 灼烧强度图，训练时作为 `C_target`。
- `synthetic_burn_trace.png`：灼烧强度可视化图。
- `synthetic_burn_probability.png`：灼烧区域二值图，灼烧强度大于 0 的位置为 255，其余为 0。
- `synthetic_burn_metadata.txt`：合成参数、seed、最大强度等元信息。

训练脚本最依赖的是：

```text
000.jpg ... 004.jpg
synthetic_burn_trace.npy
```

## 5. Python 环境

推荐使用 Python 3.12。当前环境路径示例：

```text
C:\Users\17874\AppData\Local\Programs\Python\Python312\python.exe
```

安装基础依赖：

```powershell
python -m pip install numpy opencv-python matplotlib tensorboard
```

如果使用 NVIDIA GPU，需要安装 CUDA 版 PyTorch，而不是普通 PyPI 镜像里的 CPU 版：

```powershell
python -m pip uninstall torch torchvision torchaudio -y
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

如果 `cu128` 不适配，可尝试：

```powershell
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

检查 GPU 是否可用：

```powershell
python -c "import torch; print(torch.cuda.is_available()); print(torch.version.cuda); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```

期望输出包含：

```text
True
NVIDIA ...
```

## 6. 合成灼烧标签

合成脚本：

```text
preview_burn_on_training_groups.py
```

关键配置在文件顶部：

```python
TRAIN_ROOT = r"C:\Users\17874\Documents\python\datasets\burn_recovery\train"
AUTO_WRITE_BURN_ARTIFACTS = True
AUTO_WRITE_MAX_GROUPS = None
AUTO_WRITE_PRINT_EVERY = 100
AUTO_WRITE_WORKERS = min(4, max(1, (os.cpu_count() or 2) - 1))
```

如果 `AUTO_WRITE_BURN_ARTIFACTS=True`，运行脚本会直接为 `TRAIN_ROOT` 下所有样本写入灼烧标签，不弹出预览窗口：

```powershell
python preview_burn_on_training_groups.py
```

如果希望人工预览每个样本，需要改为：

```python
AUTO_WRITE_BURN_ARTIFACTS = False
```

预览模式下有两个按钮：

- `Reset burn`：当前样本不变，重新生成灼烧痕迹。
- `Save burn maps`：保存当前灼烧标签到样本目录，然后进入下一个样本。

### 6.1 当前灼烧合成参数

主要参数：

```python
BURN_BASE_SIZE_SCALE = 1.5
BURN_SIZE_MULTIPLIER = 1.0
BURN_BLUR_WITH_SIZE = False
BURN_DISC_RADIUS_MULTIPLIER = 0.5
BURN_DISC_OVERLAP_RATIO = 0.8
BURN_HORIZONTAL_BIAS = True
BURN_MAX_STRIPE_ANGLE_DEG = 12
BURN_DISC_ROUNDNESS_JITTER = 0.12
BURN_DWELL_INTENSITY_VARIATION = 0.35
BLUR_SIGMA = 0.8
BURN_CLUSTER_EDGE_BLUR_MULTIPLIER = 0.25
BURN_STRIPE_EDGE_BLUR_MULTIPLIER = 1.0
BURN_STRIPE_DISC_EDGE_SOFTNESS = 0.32
MASK_THRESHOLD = 8.0
BURN_ACTIVE_THRESHOLD = 0.5
TEMPORAL_SCALE_RANGE = (0.92, 1.08)
```

生成逻辑包含三类灼烧形态：

- `cluster`：点状/团簇灼烧。
- `broken_chain`：断续条带灼烧。
- `polyline`：短折线条带灼烧。

`ALLOW_MULTI_PATTERN=True` 时，一个样本可能组合多个形态。

### 6.2 合成强度与一致性

同一个样本的 5 帧使用相同空间位置和形状的 `burn_map`。训练读取时会对 5 帧乘一个小范围随机时间强度因子：

```python
TEMPORAL_SCALE_RANGE = (0.92, 1.08)
```

所以空间痕迹一致，但每帧强度可以略有变化。

`BURN_ACTIVE_THRESHOLD` 用于截断低强度尾巴，避免引入高斯长尾导致整张图都被改变。只有灼烧区域像素会改变。

## 7. 网络结构

网络文件：

```text
burn_recovery_net.py
```

模型：

```python
BurnRecoveryNet(in_frames=5, base_channels=32)
```

输入：

```text
x: [B, 5, H, W]
```

输出：

```text
P_logits: [B, 1, H, W]，灼烧概率 logits，训练时用于 Focal/BCEWithLogits 类损失
C: [B, 1, H, W]，灼烧有效偏置强度图，经过 sigmoid，范围 0~1
```

结构概要：

1. `temporal_fuse`
   - 1x1 卷积将 5 帧融合到 `base_channels`。
   - 后接 depthwise separable conv。

2. 编码器
   - `enc1`: 下采样到 1/2 尺寸，通道变为 `2 * base_channels`。
   - `enc2`: 下采样到 1/4 尺寸，通道变为 `4 * base_channels`。

3. bottleneck
   - 多尺度轻量块 `LiteMultiScaleBlock`。
   - depthwise separable conv。

4. 解码器
   - `dec1`: 双线性上采样，与 `enc1` skip feature 拼接。
   - `dec2`: 双线性上采样，与 `temporal_fuse` skip feature 拼接。

5. 双输出头
   - `prob_head`: 输出 `P`。
   - `correction_head`: 输出 `C`。

核心模块：

- `DepthwiseSeparableConv`: depthwise 3x3 + pointwise 1x1。
- `LiteMultiScaleBlock`: 本地卷积分支 + dilation=2 上下文分支 + 1x1 融合。

## 8. 训练脚本

训练脚本：

```text
train_burn_recovery.py
```

运行：

```powershell
python train_burn_recovery.py
```

当前训练配置位于 `TrainConfig`：

```python
train_root = Path(r"C:\Users\17874\Documents\python\datasets\burn_recovery\train")
val_root = Path(r"C:\Users\17874\Documents\python\datasets\burn_recovery\val")
output_dir = Path(r"C:\Users\17874\Documents\python\checkpoints\burn_recovery")

image_size = None
batch_size = 6
num_workers = 0
epochs = 30
lr = 1e-3
weight_decay = 1e-4
use_amp = True

base_channels = 32
p_loss_weight = 1.0
c_active_loss_weight = 1.0
c_bg_loss_weight = 0.10
c_global_loss_weight = 0.20
gradient_loss_weight = 0.10
dice_loss_weight = 0.50
focal_alpha = 0.25
focal_gamma = 2.0
mask_threshold = 8.0 / 255.0
background_change_threshold = 1.0 / 255.0
saturation_percentile = 1.0

temporal_scale_range = (0.92, 1.08)
train_no_burn_probability = 0.15
val_no_burn_probability = 0.15
generate_missing_burn = True
val_generate_missing_burn = False
validate_samples_on_init = False
max_train_samples = None
max_val_samples = 500
val_subset_seed = 20260705
save_every_epoch = True
use_tensorboard = True
vis_every_epoch = True
vis_num_samples = 4
progress_print_every = 50
use_scheduler = True
grad_clip_norm = 1.0
```

注意：8GB 显存下，`image_size=None` 且 `batch_size=6` 可能 OOM。更稳的推荐配置是：

```python
image_size = (256, 320)
batch_size = 4
use_amp = True
```

如果希望全分辨率训练：

```python
image_size = None
batch_size = 1
use_amp = True
```

### 8.1 训练时数据如何构造

每个样本读取：

```text
000.jpg ... 004.jpg
synthetic_burn_trace.npy
```

脚本将图像归一化到 `[0, 1]`：

```python
clean = frame / 255.0
correction = synthetic_burn_trace / 255.0
mask = correction >= (8.0 / 255.0)
```

输入网络前，会把灼烧加到干净图上。为了避免 8 bit 图像饱和，同时保持标签代表同一个固定探测器灼烧状态，当前使用统一缩放，而不是逐像素裁剪标签：

```python
max_burn = correction[None, :, :] * temporal_scale
headroom = 1.0 - clean
ratio = headroom[max_burn > 1e-6] / max_burn[max_burn > 1e-6]
global_scale = min(1.0, percentile(ratio, 1.0))

scaled = max_burn * global_scale
burned = clip(clean + scaled, 0, 1)
c_target = correction * global_scale
mask = c_target >= (8.0 / 255.0)
```

这样 `P_target` 和 `C_target` 描述的是同一个最终灼烧状态，避免出现 `mask=1` 但 `correction≈0` 的矛盾监督。

其中训练集 temporal scale 在 `(0.92, 1.08)` 随机变化，验证集固定为 `(1.0, 1.0)`。

为了让网络学会“没有灼烧时不要修复”，训练和验证中还会加入一定比例的无灼烧负样本：

```python
train_no_burn_probability = 0.15
val_no_burn_probability = 0.15
```

当某个样本被选为无灼烧样本时：

```python
correction = 0
mask = 0
burned = clean
```

训练集的无灼烧样本是随机出现的，用于增强鲁棒性；验证集的无灼烧样本由固定 seed 决定，保证每个 epoch 的验证结果可比较。

### 8.2 损失函数

总损失：

```text
loss = FocalWithLogits(P_logits, mask)
     + 0.5 * Dice(sigmoid(P_logits), mask)
     + 1.0 * L1(C_active, C_target_active)
     + 0.10 * L1(C_background, 0)
     + 0.20 * L1(C, C_target)
     + 0.10 * GradientL1(C, C_target)
```

对应代码：

```python
loss_focal = focal_loss_with_logits(prob_logits, p_target)
loss_dice = dice_loss(torch.sigmoid(prob_logits), p_target)
l1_active = abs(correction - c_target)[active].mean()
l1_bg = abs(correction[background]).mean()
l1_global = abs(correction - c_target).mean()
loss_gradient = gradient_l1(correction, c_target)
```

含义：

- `FocalWithLogits + Dice`：缓解灼烧区域小、背景像素多导致的类别不平衡。
- `active L1`：重点监督灼烧区域内的有效偏置强度。
- `background L1`：约束无灼烧背景不要被错误修复。
- `global L1`：保持整幅图的偏置估计稳定。
- `GradientL1`：让预测偏置图的边缘形状更接近标签。

### 8.3 验证集抽样

当前每个 epoch 不跑完整验证集，而是从 `val` 中固定随机抽 500 个样本：

```python
max_val_samples = 500
val_subset_seed = 20260705
```

这 500 个样本在训练开始时确定，每个 epoch 都相同，因此验证 loss 可比较，同时节省时间。

测试集 `test` 和 `test_flir` 不在训练循环中每 epoch 运行。建议训练完成后用 `best.pt` 单独完整测试。

验证集默认要求使用预生成标签：

```python
val_generate_missing_burn = False
```

这样不同 epoch 的验证集标签固定，`val_loss` 和 `best.pt` 选择可复现。

训练完成后可运行独立评估脚本：

```powershell
python test_burn_recovery.py --checkpoint C:\Users\17874\Documents\python\checkpoints\burn_recovery\best.pt
```

测试脚本会优先读取 checkpoint 中保存的 `config`，恢复：

```text
base_channels
image_size
mask_threshold
saturation_percentile
batch_size
```

如果命令行显式传入 `--batch-size` 或 `--image-height/--image-width`，则以命令行为准。

默认会依次评估：

```text
datasets\burn_recovery\val
datasets\burn_recovery\test
datasets\burn_recovery\test_flir
```

默认不临时生成缺失的灼烧标签，以保证测试可复现。如果确实需要临时生成，可加：

```powershell
python test_burn_recovery.py --generate-missing-burn
```

指标采用全数据集累计 TP/FP/FN 和误差总量后统一计算，不再简单平均每个 batch 的 Dice/IoU。

### 8.4 训练进度

训练开始会打印：

```text
device: cuda
train samples: 46304
val samples: 500 / 2880
output dir: ...
image_size: ...
batch_size: ...
amp: True
```

每 50 个 batch 打印一次当前 epoch 内进度：

```text
train batch 50/7718 loss=...
train batch 100/7718 loss=...
val batch 50/84 loss=...
```

每个 epoch 结束打印：

```text
epoch 001/30 time=...s train_loss=... val_loss=... val_dice=... val_iou=... val_active_mae=... val_bg_mae=...
```

### 8.5 输出文件

训练输出目录：

```text
C:\Users\17874\Documents\python\checkpoints\burn_recovery
```

包含：

```text
epoch_001.pt
epoch_002.pt
...
best.pt
runs\
vis\
```

说明：

- `epoch_XXX.pt`：每个 epoch 的模型权重。
- `best.pt`：验证 loss 最低的模型。
- `runs`：TensorBoard 日志。
- `vis`：验证样本可视化结果。

## 9. TensorBoard 可视化

启动：

```powershell
tensorboard --logdir C:\Users\17874\Documents\python\checkpoints\burn_recovery\runs
```

浏览器打开：

```text
http://localhost:6006
```

可以查看：

- `train/loss`
- `train/loss_prob`
- `train/focal`
- `train/dice_loss`
- `train/l1_active`
- `train/l1_bg`
- `train/l1_global`
- `train/gradient`
- `train/dice`
- `train/iou`
- `train/active_mae`
- `train/bg_mae`
- `val/loss`
- `val/loss_prob`
- `val/focal`
- `val/dice_loss`
- `val/l1_active`
- `val/l1_bg`
- `val/l1_global`
- `val/gradient`
- `val/dice`
- `val/iou`
- `val/active_mae`
- `val/bg_mae`
- `lr`

## 10. 图像可视化

每个 epoch 会保存若干验证样本可视化：

```text
C:\Users\17874\Documents\python\checkpoints\burn_recovery\vis\epoch_001
```

每张可视化图包含：

```text
input burned
target clean
restored direct
restored gated
pred P
target mask
pred C
target C
```

含义：

- `input burned`：加入合成灼烧后的输入帧。
- `target clean`：原始干净帧。
- `restored direct`：直接使用 `future - C` 的恢复结果，用于检查偏置头。
- `restored gated`：使用 `future - (P > 0.5) * C` 的恢复结果，用于检查概率门控。
- `pred P`：网络预测的灼烧概率。
- `target mask`：真实灼烧区域。
- `pred C`：网络预测的灼烧强度。
- `target C`：真实灼烧强度。

## 11. 推理方法

目前项目中提供了网络恢复函数：

```python
from burn_recovery_net import BurnRecoveryNet, restore_future_frame
```

基本流程：

```python
import cv2
import numpy as np
import torch

from burn_recovery_net import BurnRecoveryNet, restore_future_frame

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ckpt = torch.load(
    r"C:\Users\17874\Documents\python\checkpoints\burn_recovery\best.pt",
    map_location=device,
)

model = BurnRecoveryNet(in_frames=5, base_channels=32).to(device)
model.load_state_dict(ckpt["model"])
model.eval()

frame_paths = [
    r"path\to\000.jpg",
    r"path\to\001.jpg",
    r"path\to\002.jpg",
    r"path\to\003.jpg",
    r"path\to\004.jpg",
]

frames = []
for path in frame_paths:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = img.astype(np.float32) / 255.0
    frames.append(img)

x = torch.from_numpy(np.stack(frames, axis=0)).unsqueeze(0).to(device)

with torch.no_grad():
    prob_logits, correction = model(x)
    prob = torch.sigmoid(prob_logits)
    restored = restore_future_frame(x[:, -1:, :, :], prob, correction, threshold=0.5)

out = restored[0, 0].cpu().numpy()
out_u8 = np.clip(out * 255.0, 0, 255).astype(np.uint8)
cv2.imwrite("restored.png", out_u8)
```

如果训练时使用了 `image_size=(256, 320)`，推理时也建议先 resize 到同样尺寸，再视需要 resize 回原图尺寸。若训练时 `image_size=None`，可直接输入原始尺寸。

## 12. 复现实验步骤

从当前项目状态复现训练：

1. 确认数据目录存在：

```powershell
Test-Path C:\Users\17874\Documents\python\datasets\burn_recovery\train
Test-Path C:\Users\17874\Documents\python\datasets\burn_recovery\val
```

2. 检查样本结构：

```powershell
Get-ChildItem C:\Users\17874\Documents\python\datasets\burn_recovery\train\sample_000001
```

应看到：

```text
000.jpg
001.jpg
002.jpg
003.jpg
004.jpg
synthetic_burn_trace.npy
```

3. 如未生成灼烧标签，运行：

```powershell
python preview_burn_on_training_groups.py
```

注意先确认：

```python
TRAIN_ROOT = r"C:\Users\17874\Documents\python\datasets\burn_recovery\train"
AUTO_WRITE_BURN_ARTIFACTS = True
```

如果要给验证集或测试集也生成标签，需要修改 `TRAIN_ROOT` 指向对应目录后再运行。

4. 推荐先使用 8GB 显存安全配置：

```python
image_size = (256, 320)
batch_size = 4
use_amp = True
```

5. 开始训练：

```powershell
python train_burn_recovery.py
```

6. 查看进度：

```text
train batch ...
val batch ...
epoch ...
```

7. 查看 TensorBoard：

```powershell
tensorboard --logdir C:\Users\17874\Documents\python\checkpoints\burn_recovery\runs
```

8. 查看可视化图片：

```text
C:\Users\17874\Documents\python\checkpoints\burn_recovery\vis
```

9. 训练结束后使用：

```text
C:\Users\17874\Documents\python\checkpoints\burn_recovery\best.pt
```

进行推理或测试。

## 13. 常见问题

### 13.1 为什么 CUDA 可用但仍然 OOM？

8GB 显卡不能按任务管理器里的“专用 GPU 内存 + 共享 GPU 内存”相加来设置 batch。训练主要依赖专用显存。共享 GPU 内存来自系统内存，速度慢且容易卡顿。

如果 OOM，按顺序调整：

```python
batch_size = 4
```

再不行：

```python
batch_size = 2
```

全分辨率训练建议：

```python
image_size = None
batch_size = 1
```

低分辨率训练建议：

```python
image_size = (256, 320)
batch_size = 4
```

### 13.2 为什么把图像缩小也能训练？

因为训练脚本会同时缩放输入图像和灼烧标签：

```python
img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
burn = cv2.resize(burn, (w, h), interpolation=cv2.INTER_LINEAR)
```

所以网络看到的输入和标签尺寸一致，可以训练。

但缩小训练与全分辨率训练不完全一致。缩小会让细小痕迹和边缘更平滑。建议流程：

1. 用 `(256, 320)` 快速验证网络有效。
2. 再用 `image_size=None, batch_size=1` 做全分辨率微调。

### 13.3 训练时是否每个 epoch 跑测试集？

不是。当前每个 epoch 只跑验证集 `val` 的固定随机 500 个样本。`test` 和 `test_flir` 应在训练完成后单独用 `best.pt` 评估。

### 13.4 合成灼烧数据提速前后是否一致？

同一个样本、同一个 seed 下，优化前后的 `burn_map` 一致，最大差值为 0。

但如果重新批量生成且重新抽 seed，则新生成的文件不会与旧文件逐样本完全一致。要完全复现旧标签，应读取旧 `synthetic_burn_metadata.txt` 中的 `seed=...` 并复用。

### 13.5 当前 batch 进度怎么看？

脚本每 50 个 batch 打印：

```text
train batch 50/N loss=...
```

调整频率：

```python
progress_print_every = 10
```

关闭：

```python
progress_print_every = 0
```

## 14. 建议的正式实验流程

推荐按三个阶段做：

1. 快速调试阶段
   - `image_size=(256, 320)`
   - `batch_size=4`
   - `max_train_samples=500` 可选
   - 确认 loss 能下降、可视化结果合理。

2. 正式训练阶段
   - `max_train_samples=None`
   - `max_val_samples=500`
   - 根据显存选择 `image_size` 和 `batch_size`。

3. 全分辨率微调阶段
   - `image_size=None`
   - `batch_size=1`
   - 从前一阶段最优模型继续训练若干 epoch。

当前脚本尚未实现“从 checkpoint 继续训练”的命令行接口。如需断点续训，需要在 `main()` 中加载 `best.pt` 或指定 checkpoint 后再进入训练循环。
