import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# === 1. 載入與處理資料 ===
output = torch.load("./output/generated_output_2520_1.pt")
gt_output = torch.load("./output/ground_truth_2520_1.pt")

if len(gt_output.shape) == 5:
    gt_output = gt_output.squeeze(2)

print("output shape:", output.shape)
print("gt_output shape:", gt_output.shape)

# 轉為攝氏溫度
pred = output[0] - 273.15
gt = gt_output[0] - 273.15

def average_by_day(tensor_24):
    return torch.stack([
        tensor_24[0:8].mean(dim=0),
        tensor_24[8:16].mean(dim=0),
        tensor_24[16:24].mean(dim=0)
    ])

pred_days = average_by_day(pred)
gt_days = average_by_day(gt)

# === 2. 設定經緯度網格 ===
H, W = pred_days.shape[1], pred_days.shape[2] 

# 設定範圍
lon_min, lon_max = 20, 120
lat_min, lat_max = 28, 65

lon = np.linspace(lon_min, lon_max, W)
lat = np.linspace(lat_min, lat_max, H)
lon_grid, lat_grid = np.meshgrid(lon, lat)

# === [修改點 1] 設定色階範圍 ===
# vmin 強制設為 25 (低於此不顯示)，vmax 取資料最大值
vmin = 25
vmax = max(pred_days.max().item(), gt_days.max().item())

# === 3. 繪圖 ===
fig, axs = plt.subplots(2, 3, figsize=(18, 8),
                        subplot_kw={'projection': ccrs.PlateCarree()})

rows_data = [pred_days, gt_days]
row_labels = ["Prediction", "Ground Truth"]

for row in range(2):
    for col in range(3):
        ax = axs[row, col]
        
        # === [修改點 2] 資料處理：將低於 25 度的值設為 NaN ===
        data = rows_data[row][col].cpu().numpy().copy() # 務必使用 copy，避免改到原始資料
        data[data < 25] = np.nan  # NaN 在 pcolormesh 中會變成透明/白色
        
        # === [修改點 3] 繪圖：使用 hot_r 並帶入 vmin ===
        im = ax.pcolormesh(lon_grid, lat_grid, data,
                           cmap='hot_r',  # _r 代表顏色反轉
                           vmin=vmin, vmax=vmax,
                           transform=ccrs.PlateCarree(),
                           shading='auto') 
        
        # 加入地圖特徵
        ax.add_feature(cfeature.COASTLINE, linewidth=1.2, color='black')
        ax.add_feature(cfeature.BORDERS, linewidth=0.8, linestyle=':')
        
        ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
        ax.set_title(f"{row_labels[row]} - Day {col+1}", fontsize=12, fontweight='bold')

        # 經緯度網格
        gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
        gl.top_labels = False
        gl.right_labels = False
        gl.xlabel_style = {'size': 9}
        gl.ylabel_style = {'size': 9}

# === 4. 共用 Colorbar ===
plt.tight_layout()
plt.subplots_adjust(bottom=0.15) 

cbar_ax = fig.add_axes([0.15, 0.05, 0.7, 0.03]) 
cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
cbar.set_label("Temperature (°C) (>25°C)", fontsize=14)
cbar.ax.tick_params(labelsize=12)

# === 5. 顯示 ===
plt.show()