import numpy as np
import os
import re
import matplotlib.pyplot as plt
import cv2
from scipy.signal import convolve2d
from compare_gray_videos_slider import compare_gray_videos_slider
from load_dc import DCPipeline

# =========================
# Matplotlib
# =========================
plt.rcParams['font.sans-serif'] = ['AR PL UMing CN']
plt.rcParams['axes.unicode_minus'] = False


# =========================
# Inpaint（保持你原逻辑）
# =========================
def inpaint(img, mask, win_r=5):
    result = img.copy()
    for row in range(win_r, img.shape[0] - win_r):
        for col in range(win_r, img.shape[1] - win_r):
            if mask[row, col] == 0:
                continue
            win_img = result[row - win_r:row + win_r + 1,
                             col - win_r:col + win_r + 1]
            win_mask = mask[row - win_r:row + win_r + 1,
                            col - win_r:col + win_r + 1]
            result[row, col] = win_img[~win_mask.astype(bool)].mean()
    return img - result


# =========================
# 🚀 工程级重构核心模块
# =========================

class RobustBackground:
    """替代 img_min，避免天空固化"""
    def __init__(self, shape, alpha=0.03):
        self.bg = np.zeros(shape, np.float32)
        self.alpha = alpha
        self.init = False

    def update(self, frame):
        frame = frame.astype(np.float32)

        if not self.init:
            self.bg = frame.copy()
            self.init = True
            return self.bg

        # EMA background
        self.bg = (1 - self.alpha) * self.bg + self.alpha * frame
        return self.bg


def sky_normalization(frame, bg):
    """消除天空非均匀影响"""
    res = frame - bg

    sigma = cv2.GaussianBlur(res**2, (11, 11), 0) ** 0.5
    z = res / (sigma + 1e-6)

    return res, z


def burn_score(frame, z):
    """结构 + 统计联合评分"""
    lap = cv2.Laplacian(frame, cv2.CV_32F, ksize=3)
    edge = np.abs(lap)

    gx = cv2.Sobel(frame, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(frame, cv2.CV_32F, 0, 1)
    texture = np.sqrt(gx**2 + gy**2)

    edge_n = edge / (np.mean(edge) + 1e-6)
    tex_n = texture / (np.mean(texture) + 1e-6)

    score = np.abs(z) * 0.6 + edge_n * 0.3 + tex_n * 0.1
    return score


class TemporalMask:
    """时序稳定（解决天空误检波动）"""
    def __init__(self, shape, beta=0.85):
        self.acc = np.zeros(shape, np.float32)
        self.beta = beta

    def update(self, mask):
        self.acc = self.beta * self.acc + (1 - self.beta) * mask
        return self.acc > 0.4


# =========================
# 参数
# =========================
PLOT = True
SAVE = False

h, w = 512, 640

# =========================
# 你的 pipeline（保持不变）
# =========================
lib_path_md = "/data/付帅康/sun_burn/new_par/parameters_for_tnr+nle+sl+md/build/modules_script/libtest500_export.so"
para_path_md = "/data/付帅康/sun_burn/new_par/parameters_for_tnr+nle+sl+md"

pipe_md = DCPipeline(rows=h, cols=w,
                     lib_path=lib_path_md,
                     para_path=para_path_md)

pipe_md.set_bypass_all_except([
    "rs500_tnr",
    "rs500_sl",
    "rs500_nle2",
    "rs500_md",
])


# =========================
# 数据路径
# =========================
file_name = "/data/付帅康/sun_burn/bins/1/"
names = os.listdir(file_name)
names.sort(key=lambda x: int(re.findall(r'\d+', x)[0]) if re.findall(r'\d+', x) else -1)


# =========================
# 初始化模块（关键）
# =========================
bg_model = RobustBackground((h, w), alpha=0.03)
temporal = TemporalMask((h, w), beta=0.85)


# =========================
# 主循环
# =========================
for idx, name in enumerate(names[2500:]):

    print(f"{idx}/{len(names)}")

    img_uint16_file = os.path.join(file_name, name)
    img_uint16 = np.fromfile(img_uint16_file, dtype=np.uint16)
    frame = np.reshape(img_uint16, (h, w)).astype(np.float32)

    md_out_img = pipe_md.run(frame)

    # =========================
    # 1. 背景建模（替代 img_min）
    # =========================
    bg = bg_model.update(frame)

    # =========================
    # 2. sky normalization
    # =========================
    res, z = sky_normalization(frame, bg)

    # =========================
    # 3. burn score
    # =========================
    score = burn_score(frame, z)

    # =========================
    # 4. 初始 mask（替代 midfreq + threshold 100）
    # =========================
    mask0 = (score > 3.0).astype(np.uint8)

    # =========================
    # 5. 时序稳定
    # =========================
    burn_mask = temporal.update(mask0)

    # =========================
    # 6. 修复
    # =========================
    b_t = inpaint(res, burn_mask.astype(np.uint8))

    # =========================
    # 可视化
    # =========================
    if PLOT and idx % 5 == 0:
        plt.clf()

        plt.subplot(2, 3, 1)
        plt.title("frame")
        plt.imshow(frame, cmap='gray')

        plt.subplot(2, 3, 2)
        plt.title("background")
        plt.imshow(bg, cmap='gray')

        plt.subplot(2, 3, 3)
        plt.title("residual")
        plt.imshow(res, cmap='gray')

        plt.subplot(2, 3, 4)
        plt.title("score")
        plt.imshow(score, cmap='jet')
        plt.colorbar()

        plt.subplot(2, 3, 5)
        plt.title("burn mask")
        plt.imshow(burn_mask, cmap='gray')

        plt.subplot(2, 3, 6)
        plt.title("corrected")
        plt.imshow(frame - b_t, cmap='gray')

        plt.tight_layout()
        plt.show()

        input("press enter next frame...")


    # =========================
    # 保存（可选）
    # =========================
    if SAVE:
        out_dir = file_name + "_out_v3"
        os.makedirs(out_dir, exist_ok=True)

        img_out = np.clip(frame - b_t, 0, 65535).astype(np.uint16)
        img_out.tofile(os.path.join(out_dir, name))