import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import matplotlib.pyplot as plt

output =  torch.load("./output/generated_output_1756.pt")

# === Ground Truth ===
gt_output = torch.load("./output/ground_truth_1756.pt")

# 將預測拆成三幀（每幀 3 channels）
pred = output[0]  # [9, H, W]

# 將 24 張圖片拆成 3 天，每天 8 張圖
def split_by_day(tensor_24):  # tensor_24: [24, H, W]
    return torch.stack([
        tensor_24[0:8],    # 第一天的 8 張
        tensor_24[8:16],   # 第二天的 8 張
        tensor_24[16:24]   # 第三天的 8 張
    ])  # return: [3, 8, H, W]

pred_days = split_by_day(pred)       # shape: [3, 8, H, W]
gt_days = split_by_day(gt_output)    # shape: [3, 8, H, W]

# === 畫圖 ===
fig, axs = plt.subplots(2, 8, figsize=(24, 6))  # 2 rows: prediction, ground truth


day_idx = 0  # 第幾天（0: Day1, 1: Day2, 2: Day3）

fig, axs = plt.subplots(3, 8, figsize=(24, 9))  # 3 rows, 8 columns

for day_idx in range(3):  # 三天
    for t_idx in range(8):  # 每天 8 個時段
        im = axs[day_idx, t_idx].imshow(gt_days[day_idx, t_idx].cpu(), cmap='hot')
        axs[day_idx, t_idx].set_title(f"Pred D{day_idx+1} T{t_idx+1}")
        axs[day_idx, t_idx].axis('off')

fig.colorbar(im, ax=axs, orientation='vertical', fraction=0.02, pad=0.04)
date_str = "2015-07-15_3days"
# plt.savefig(f"./output/pred_vs_gt_3days({date_str}).png")

plt.show()