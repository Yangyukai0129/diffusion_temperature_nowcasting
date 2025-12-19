import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np

# === 載入輸出與 GT ===
output = torch.load("./output/generated_output_2520.pt")
gt_output = torch.load("./output/ground_truth_2520.pt")
gt_output = gt_output.squeeze(2)

print("output shape:", output.shape)
print("gt_output shape:", gt_output.shape)

# === 取出第一個樣本 ===
pred = output[0] - 273.15
gt = gt_output[0] - 273.15

# === 平均前 8 張（模擬第 1 天平均）===
def average_by_day(tensor_24):
    return torch.stack([tensor_24[0:8].mean(dim=0)])  # 只平均前 8 張

pred_days = average_by_day(pred)
gt_days = average_by_day(gt)

# === 經緯度範圍 ===
lon_min, lon_max = 20, 120
lat_min, lat_max = 28, 65

lons = np.linspace(lon_min, lon_max, pred_days.shape[-1])
lats = np.linspace(lat_min, lat_max, pred_days.shape[-2])
lon_grid, lat_grid = np.meshgrid(lons, lats)

# === 固定色階範圍 ===
vmin, vmax = 15, 45

# === 熱浪格點 (範例) ===
heatwave_points = [(np.float64(57.625), np.float64(92.375)), 
                   (np.float64(52.625), np.float64(37.375)), 
                   (np.float64(52.625), np.float64(42.375)), 
                   (np.float64(47.625), np.float64(32.375)), 
                   (np.float64(47.625), np.float64(37.375)), 
                   (np.float64(47.625), np.float64(42.375)), 
                   (np.float64(47.625), np.float64(102.375)), 
                   (np.float64(42.625), np.float64(22.375)), 
                   (np.float64(42.625), np.float64(27.375)), 
                   (np.float64(42.625), np.float64(32.375)), 
                   (np.float64(42.625), np.float64(37.375)), 
                   (np.float64(42.625), np.float64(92.375)), 
                   (np.float64(42.625), np.float64(97.375)), 
                   (np.float64(42.625), np.float64(102.375)), 
                   (np.float64(42.625), np.float64(107.375)), 
                   (np.float64(37.625), np.float64(22.375)), 
                   (np.float64(37.625), np.float64(27.375)), 
                   (np.float64(37.625), np.float64(87.375)), 
                   (np.float64(37.625), np.float64(92.375)), 
                   (np.float64(37.625), np.float64(97.375)), 
                   (np.float64(37.625), np.float64(102.375)), 
                   (np.float64(37.625), np.float64(112.375)), 
                   (np.float64(37.625), np.float64(117.375)), 
                   (np.float64(32.625), np.float64(22.375))]

heatwave_lats = [pt[0] for pt in heatwave_points]
heatwave_lons = [pt[1] for pt in heatwave_points]

# === 畫圖 (橫向排列) ===
fig, axs = plt.subplots(1, 2, figsize=(12, 5),
                        subplot_kw={'projection': ccrs.PlateCarree()})

titles = ["Predicted Day 1", "Ground Truth Day 1"]
data_list = [pred_days[0].cpu(), gt_days[0].cpu()]

for i, ax in enumerate(axs):
    im = ax.pcolormesh(lon_grid, lat_grid, data_list[i],
                       cmap='hot', vmin=vmin, vmax=vmax,
                       transform=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.add_feature(cfeature.LAND, facecolor='lightgray')
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
    ax.set_title(titles[i], fontsize=12)

    # 經緯度網格
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 9}
    gl.ylabel_style = {'size': 9}
    gl.xlocator = plt.MultipleLocator(10)
    gl.ylocator = plt.MultipleLocator(10)

    # 標示熱浪格點
    ax.scatter(heatwave_lons, heatwave_lats, color='blue', s=30, marker='o', edgecolor='black', transform=ccrs.PlateCarree(), label='Heatwave')
    ax.legend(loc='upper right', fontsize=10)
# === Colorbar (水平放在下面) ===
cbar_width = 0.3
cbar_left = (1 - cbar_width) / 2
cbar_ax = fig.add_axes([cbar_left, 0.12, cbar_width, 0.02])
fig.colorbar(im, cax=cbar_ax, orientation='horizontal', label="Temperature (°C)")

plt.tight_layout(rect=[0, 0.1, 1, 1])
plt.show()
