# Sun Burn Raw 16bit 使用说明

这个目录是灼烧恢复项目的 raw 域版本，用于 `uint16` `.bin` / `.raw` 文件的迁移训练、合成灼烧标签生成和推理。

支持的典型分辨率：

```text
512x640
1024x1280
```

网络内部使用 `0~1` 浮点张量训练，输入输出文件仍保持 16bit raw 域。

## 归一化原则

本项目默认使用固定 bit 位范围归一化：

```python
x = raw_uint16 / raw_max
```

默认：

```text
raw_max = 65535
```

也就是说：

```text
raw 65535 -> 1.0
raw 32768 -> 0.5
raw 655   -> 0.01
```

不要对每张图做 `min-max` 归一化。原因是本任务要学习灼烧带来的绝对偏置量：

```text
burned = clean + burn_bias
target correction = burn_bias
```

如果每张图单独 `min-max`，不同帧、不同样本的 raw 灰度物理含义会被改变，模型学到的纠正量不再对应真实 raw 偏置。

如果真实数据只有 14bit 或 12bit 有效位，也不要逐图 `min-max`，而是统一改：

```text
14bit: raw_max = 16383
12bit: raw_max = 4095
```

当前代码默认是 16bit：

```text
raw_max = 65535
```

## 关于盲元

如果使用固定 bit 位范围归一化，训练过程不需要因为归一化而额外关心盲元。盲元不会像逐图 `min-max` 那样把整张图动态范围拉歪。

但这不代表盲元完全无影响。如果 raw 数据里存在大量固定坏点、死点、热噪点，它们仍可能影响网络学习。建议先按当前流程训练；如果后面发现模型把固定坏点误判为灼烧，再在 `raw_io.py` 中增加坏点表、坏点插值或固定 pattern 预处理。

## 代码文件

```text
burn_recovery_net.py              网络结构
raw_io.py                         uint16 bin/raw 读写、尺寸识别、归一化
build_raw_training_groups.py      从连续 raw 序列切 5 帧训练样本
synthetic_burn_raw.py             生成 raw 域合成灼烧标签
train_burn_recovery_raw.py        raw 域训练主脚本
train_small_raw.py                小规模 raw 试训配置
infer_burn_recovery_raw.py        raw 域推理脚本
README.md                         本说明
```

## 数据目录格式

你的原始 raw 数据可以是下面这种结构：

```text
raw_root/
  512/
    scene_001/
      frame_000001.bin
      frame_000002.bin
      frame_000003.bin
      device.txt
      weather.txt
      ...
    scene_002/
      ...
  1024/
    scene_101/
      frame_000001.bin
      frame_000002.bin
      frame_000003.bin
      device.txt
      weather.txt
      ...
```

说明：

```text
512  目录默认对应 512x640
1024 目录默认对应 1024x1280
场景目录内的 .txt 文件会被忽略
.bin/.raw 文件末尾允许带信息行，读取图像时只读前 H*W*2 字节，尾部信息行忽略
```

训练脚本最终使用的是切分后的 5 帧样本目录。每个样本文件夹包含 5 帧 raw：

```text
sample_000001/
  000.bin
  001.bin
  002.bin
  003.bin
  004.bin
```

生成灼烧标签后，会额外得到：

```text
sample_000001/
  synthetic_burn_trace_u16.npy
  synthetic_burn_trace_u16.bin
  synthetic_burn_mask_u16.bin
  synthetic_burn_trace_preview.png
  synthetic_burn_metadata_raw.txt
```

推荐数据目录：

```text
C:\Users\17874\Documents\python\datasets_raw\burn_recovery\train
C:\Users\17874\Documents\python\datasets_raw\burn_recovery\val
C:\Users\17874\Documents\python\datasets_raw\burn_recovery\test
```

代码识别 raw 尺寸的顺序：

```text
1. 如果路径中包含 512 / 512x640 / 640x512，则按 512x640
2. 如果路径中包含 1024 / 1024x1280 / 1280x1024，则按 1024x1280
3. 否则根据文件大小判断，并允许末尾存在信息行
```

如果仍无法判断，运行时用：

```text
--raw-shape HxW
```

例如：

```text
--raw-shape 512x640
--raw-shape 1024x1280
```

## 从连续 raw 序列切训练样本

如果原始数据是连续帧目录：

```text
raw_root/
  512/
    scene_001/
      frame_000000.bin
      frame_000001.bin
      frame_000002.bin
      device.txt
      weather.txt
      ...
  1024/
    scene_101/
      frame_000000.bin
      frame_000001.bin
      frame_000002.bin
      device.txt
      weather.txt
      ...
```

可以递归查找所有场景序列，并切成 5 帧训练样本。只切 512 数据：

```powershell
cd C:\Users\17874\Documents\python\sun_burn_raw

python build_raw_training_groups.py `
  --source-root C:\raw_root\512 `
  --output-root C:\Users\17874\Documents\python\datasets_raw\burn_recovery\train `
  --shape 512x640 `
  --frame-gap 10 `
  --stride 1
```

只切 1024 数据：

```powershell
python build_raw_training_groups.py `
  --source-root C:\raw_root\1024 `
  --output-root C:\Users\17874\Documents\python\datasets_raw\burn_recovery_1024\train `
  --shape 1024x1280 `
  --frame-gap 10 `
  --stride 1
```

也可以直接从原始根目录递归查找 `512` 和 `1024` 下所有场景：

```powershell
python build_raw_training_groups.py `
  --source-root C:\raw_root `
  --output-root C:\Users\17874\Documents\python\datasets_raw\burn_recovery_mixed\train `
  --frame-gap 10 `
  --stride 1
```

注意：如果一个训练目录里混合了 `512x640` 和 `1024x1280`，训练时需要满足下面任一条件：

```text
batch_size = 1
或使用 --image-size 把所有样本缩放到统一尺寸
或分开训练 512 模型和 1024 模型
```

每个样本使用：

```text
t, t+10, t+20, t+30, t+40
```

如果不想复制大文件，可以尝试硬链接：

```powershell
python build_raw_training_groups.py `
  --source-root C:\raw_root\512 `
  --output-root C:\Users\17874\Documents\python\datasets_raw\burn_recovery\train `
  --shape 512x640 `
  --frame-gap 10 `
  --stride 1 `
  --copy-mode hardlink
```

## 生成 raw 域灼烧标签

训练集：

```powershell
python synthetic_burn_raw.py `
  --root C:\Users\17874\Documents\python\datasets_raw\burn_recovery\train `
  --shape 512x640
```

验证集建议也提前生成，保证验证固定：

```powershell
python synthetic_burn_raw.py `
  --root C:\Users\17874\Documents\python\datasets_raw\burn_recovery\val `
  --shape 512x640
```

1024x1280：

```powershell
python synthetic_burn_raw.py `
  --root C:\Users\17874\Documents\python\datasets_raw\burn_recovery\train `
  --shape 1024x1280
```

常用参数：

```text
--min-peak / --max-peak       灼烧最大偏置强度，单位是 raw 灰度
--active-threshold            小于该 raw 灰度的尾巴直接截断
--size-multiplier             灼烧尺寸倍率
--stripe-probability          条状灼烧概率
--endian little|big           raw 字节序，默认 little
--seed                        随机种子
```

示例：

```powershell
python synthetic_burn_raw.py `
  --root C:\Users\17874\Documents\python\datasets_raw\burn_recovery\train `
  --shape 512x640 `
  --min-peak 1800 `
  --max-peak 9000 `
  --active-threshold 64 `
  --size-multiplier 1.0
```

## 小规模迁移训练

先建议小规模验证流程：

```powershell
python train_burn_recovery_raw.py `
  --train-root C:\Users\17874\Documents\python\datasets_raw\burn_recovery\train `
  --val-root C:\Users\17874\Documents\python\datasets_raw\burn_recovery\val `
  --raw-shape 512x640 `
  --batch-size 2 `
  --epochs 10 `
  --max-train-samples 1000 `
  --max-val-samples 300
```

如果要从 PNG 项目训练好的权重做迁移训练，用 `--pretrained`：

```powershell
python train_burn_recovery_raw.py `
  --train-root C:\Users\17874\Documents\python\datasets_raw\burn_recovery\train `
  --val-root C:\Users\17874\Documents\python\datasets_raw\burn_recovery\val `
  --raw-shape 512x640 `
  --batch-size 2 `
  --epochs 10 `
  --max-train-samples 1000 `
  --max-val-samples 300 `
  --pretrained C:\Users\17874\Documents\python\checkpoints\burn_recovery\best.pt
```

`--pretrained` 只加载模型权重，适合 PNG 到 raw 的迁移训练。

`--resume` 用于 raw 训练中断后继续训练，会恢复模型、优化器、scheduler 等状态。

## 推荐训练路线

当前推荐按三阶段走：

```text
1. 先用 --image-size 512x640 跑通 raw 迁移训练
2. 再用 1024x1280、batch_size=1 做少量原尺寸微调
3. 最后使用 1024x1280 原尺寸推理
```

这样做的原因：

```text
512x640 训练显存压力小，适合先验证 raw 数据读取、标签、loss 和迁移权重是否正常
1024x1280 原尺寸微调可以让模型适应真实高分辨率下的灼烧尺寸、边缘宽度和纹理尺度
最终 1024 原尺寸推理时，训练和推理分辨率一致，效果最稳
```

第一阶段：1024 数据缩放到 512x640 做迁移训练。

```powershell
python train_burn_recovery_raw.py `
  --train-root C:\Users\17874\Documents\python\datasets_raw\burn_recovery_1024\train `
  --val-root C:\Users\17874\Documents\python\datasets_raw\burn_recovery_1024\val `
  --raw-shape 1024x1280 `
  --image-size 512x640 `
  --batch-size 2 `
  --epochs 10 `
  --max-train-samples 1000 `
  --max-val-samples 300 `
  --pretrained C:\Users\17874\Documents\python\checkpoints\burn_recovery\best.pt
```

第二阶段：用第一阶段的 best checkpoint 做 1024x1280 原尺寸微调。

```powershell
python train_burn_recovery_raw.py `
  --train-root C:\Users\17874\Documents\python\datasets_raw\burn_recovery_1024\train `
  --val-root C:\Users\17874\Documents\python\datasets_raw\burn_recovery_1024\val `
  --raw-shape 1024x1280 `
  --batch-size 1 `
  --epochs 5 `
  --max-train-samples 1000 `
  --max-val-samples 300 `
  --pretrained C:\Users\17874\Documents\python\sun_burn_raw\checkpoints\burn_recovery_raw\best.pt `
  --output-dir C:\Users\17874\Documents\python\sun_burn_raw\checkpoints\burn_recovery_raw_1024_finetune
```

第三阶段：1024x1280 原尺寸推理。

```powershell
python infer_burn_recovery_raw.py `
  --checkpoint C:\Users\17874\Documents\python\sun_burn_raw\checkpoints\burn_recovery_raw_1024_finetune\best.pt `
  --frames scene_001\000.bin scene_001\001.bin scene_001\002.bin scene_001\003.bin scene_001\004.bin `
  --raw-shape 1024x1280 `
  --output-dir C:\Users\17874\Documents\python\sun_burn_raw\infer_out_1024
```

## 1024x1280 训练建议

`1024x1280` 的显存占用大约是 `512x640` 的 4 倍。建议先用：

```text
batch_size = 1
```

例如：

```powershell
python train_burn_recovery_raw.py `
  --train-root C:\Users\17874\Documents\python\datasets_raw\burn_recovery\train `
  --val-root C:\Users\17874\Documents\python\datasets_raw\burn_recovery\val `
  --raw-shape 1024x1280 `
  --batch-size 1 `
  --epochs 10 `
  --max-train-samples 1000 `
  --max-val-samples 300
```

如果显存不够，可以先缩放训练：

```powershell
python train_burn_recovery_raw.py `
  --train-root C:\Users\17874\Documents\python\datasets_raw\burn_recovery\train `
  --val-root C:\Users\17874\Documents\python\datasets_raw\burn_recovery\val `
  --raw-shape 1024x1280 `
  --image-size 512x640 `
  --batch-size 2 `
  --epochs 10
```

但正式效果最一致的方式仍然是原尺寸训练、原尺寸推理。

## TensorBoard

训练后日志在：

```text
输出目录\runs
```

例如：

```powershell
tensorboard --logdir C:\Users\17874\Documents\python\sun_burn_raw\checkpoints\burn_recovery_raw\runs
```

浏览器打开：

```text
http://localhost:6006
```

## 推理

输入 5 帧 raw：

```powershell
python infer_burn_recovery_raw.py `
  --checkpoint C:\Users\17874\Documents\python\sun_burn_raw\checkpoints\burn_recovery_raw\best.pt `
  --frames sample_000001\000.bin sample_000001\001.bin sample_000001\002.bin sample_000001\003.bin sample_000001\004.bin `
  --raw-shape 512x640 `
  --output-dir C:\Users\17874\Documents\python\sun_burn_raw\infer_out
```

输出文件：

```text
restored_last_u16.bin          恢复后的最后一帧
pred_correction_u16.bin        预测纠正量
pred_mask_u16.bin              预测灼烧区域，灼烧=65535，背景=0
pred_probability.npy           概率图，float32
pred_correction_u16.npy        预测纠正量，uint16
input_last_preview.png         输入最后一帧 8bit 预览
restored_last_preview.png      恢复结果 8bit 预览
pred_probability_preview.png   概率图 8bit 预览
pred_correction_preview.png    纠正量 8bit 预览
```

默认推理使用概率门控：

```text
restored = last_frame - (P > threshold) * correction
```

默认：

```text
threshold = 0.5
```

如果想直接减去 correction，不使用门控：

```powershell
python infer_burn_recovery_raw.py `
  --checkpoint C:\Users\17874\Documents\python\sun_burn_raw\checkpoints\burn_recovery_raw\best.pt `
  --frames sample_000001\000.bin sample_000001\001.bin sample_000001\002.bin sample_000001\003.bin sample_000001\004.bin `
  --raw-shape 512x640 `
  --output-dir C:\Users\17874\Documents\python\sun_burn_raw\infer_out `
  --no-gate
```

## 参数量和算力

当前默认网络：

```text
BurnRecoveryNet(in_frames=5, base_channels=32)
```

参数量：

```text
参数量：149,602
可训练参数：149,602
约 0.15M 参数
FP32 参数大小：约 0.57 MB
FP16 参数大小：约 0.29 MB
```

不同输入尺寸下的卷积计算量：

```text
256x320:
  1.85 GMACs
  约 3.71 GFLOPs

512x640:
  7.42 GMACs
  约 14.84 GFLOPs

1024x1280:
  29.68 GMACs
  约 59.36 GFLOPs
```

这里采用的口径：

```text
1 MAC = 1 次乘法 + 1 次加法
GFLOPs = 2 * GMACs
```

如果希望把 `512x640` 输入压缩到约 `2 GMACs`，最直接的方案是把基础通道数改小：

```text
base_channels=32: 7.42 GMACs, 149,602 params
base_channels=24: 4.32 GMACs,  87,242 params
base_channels=20: 3.08 GMACs,  62,302 params
base_channels=16: 2.05 GMACs,  41,522 params
base_channels=12: 1.23 GMACs,  24,902 params
base_channels=8 : 0.61 GMACs,  12,442 params
```

推荐的轻量化起点：

```text
BurnRecoveryNet(base_channels=16)
```

这版结构不变，训练和推理代码改动最小，适合后续做 NPU 部署或知识蒸馏。

## 训练目标

输入：

```text
5 帧 burned raw，归一化为 [5, H, W]
```

输出：

```text
P: 灼烧概率图
C: 灼烧纠正量图
```

标签：

```text
P_target = C_target > mask_threshold
C_target = effective correction / raw_max
```

损失：

```text
focal(P, P_target)
+ dice(P, P_target)
+ active L1(C, C_target)
+ background L1(C, 0)
+ global L1(C, C_target)
+ gradient L1(C, C_target)
```

当前默认权重：

```text
p=1.0
active=1.0
bg=1.0
global=0.5
grad=0.1
dice=0.5
```

## 当前实现假设

- raw 文件是单通道 `uint16`。
- 默认 little-endian。
- 默认 `raw_max=65535`。
- 原始目录可包含 `512` 和 `1024` 两类分辨率目录。
- 场景目录内 `.txt` 采集说明文件会被忽略。
- `.bin/.raw` 文件末尾允许有信息行，读取时只取前 `H*W*2` 字节作为图像数据。
- 样本仍是 5 帧输入。
- 网络预测最后一帧的灼烧区域和纠正量。
- 训练时输入、标签统一除以固定 `raw_max`。
- 不做逐图 `min-max` 归一化。

如果后续确认 raw 有固定黑电平、有效位宽不是 16bit、或存在严重坏点图案，再在 `raw_io.py` 中增加对应预处理参数。
