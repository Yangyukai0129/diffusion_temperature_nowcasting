import torch
import torch.nn as nn
from tqdm import tqdm
import pandas as pd
import numpy as np
import os

from unet import UNet, ddim_inference
from data_utils import LazyWeatherDataset, prepare_file_list, compute_mean_std, denormalize
from torch.utils.data import DataLoader

# === 主要推理流程 ===
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("✅ 使用裝置:", device)

    # === 載入模型 ===
    checkpoint = torch.load("./saved_models/diffusion_model_allregion.pth", map_location=device)
    model = UNet(in_channels=24, out_channels=24, cond_channels=72, time_dim=32).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    beta = checkpoint["beta"].to(device)
    model.eval()
    print("✅ 模型載入完成")

    # === 準備資料 ===
    train_files, test_files = prepare_file_list("data/npy_files_3")
    cond_mean, cond_std, target_mean, target_std = compute_mean_std(train_files)

    # 建立 Dataset（Lazy load）
    test_dataset = LazyWeatherDataset(test_files, cond_mean, cond_std, target_mean, target_std)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

    # === 把測試集一次讀進來（含時間）===
    cond_test_list, target_test_list, time_test_list = [], [], []
    for cond, target, valid_time in test_loader:
        cond_test_list.append(cond)
        target_test_list.append(target)
        time_test_list.append(valid_time)

    cond_test = torch.cat(cond_test_list, dim=0).to(device)
    target_test = torch.cat(target_test_list, dim=0)  # 先保留 CPU
    time_test = np.concatenate(time_test_list, axis=0)  # numpy.datetime64
    print(f"✅ 測試集載入完成，共 {len(cond_test)} 筆")

    target_test = target_test.squeeze(2)  # 在 getitem 或 collate_fn 裡

    # 篩選6~8月的時間點索引 (要先轉成 pd.Timestamp)
    time_test_dt = pd.to_datetime(time_test, unit='s')
    print(time_test_dt[:20])
    print("月份分布：")
    print(pd.Series(time_test_dt.month).value_counts())
    summer_mask = (time_test_dt.month >= 6) & (time_test_dt.month <= 8)
    indices = np.where(summer_mask)[0]
    print(f"符合夏季(6~8月)的時間筆數: {len(indices)}")

    # 篩選 cond_test, target_test, time_test
    cond_test = cond_test[indices]
    target_test = target_test[indices]
    time_test = time_test[indices]
    time_test_dt = time_test_dt[indices]

    # === 推理（過程直接搬回 CPU，節省 GPU RAM）===
    print("🚀 開始生成...")
    generated_list = []
    with torch.no_grad():
        for i in tqdm(range(len(cond_test)), desc="Generating Samples"):
            gen = ddim_inference(model, cond_test[i:i+1], beta, device=device, eta=0.0, num_steps=15)
            generated_list.append(gen.cpu())  # ✅ 馬上搬回 CPU
    generated = torch.cat(generated_list, dim=0)  # CPU

    # === 反標準化（全程 CPU）===
    generated_denorm = denormalize(generated, target_mean, target_std)
    target_denorm = denormalize(target_test, target_mean, target_std)

    # import xarray as xr
    # import numpy as np

    # # 讀取原始資料，取得 lat/lon
    # obs_ds = xr.open_dataset("download_nc/5/merged_5deg_t.nc")
    # lats = obs_ds.latitude.values  # 原始 lat
    # lons = obs_ds.longitude.values  # 原始 lon


    # # 假設 generated_denorm 是 torch tensor 或 numpy array，shape = (N, 24, H, W)
    # # 只取第一天 8 個時段
    # first_day_data = generated_denorm[:, 0:8, :, :]  # shape = (N, 8, H, W)

    # # 時間序列 (假設你有對應的 N 筆時間，這裡舉例用日期陣列)
    # # 取推理對應的時間（你的 selected_times）
    # times = pd.to_datetime(time_test, unit='s')

    # # 建立 xarray Dataset
    # ds = xr.Dataset(
    #     {
    #         "temperature": (["time", "forecast_hour", "lat", "lon"], first_day_data.numpy()),
    #     },
    #     coords={
    #         "time": times,
    #         "forecast_hour": np.arange(1, 9),  # 1~8時段
    #         "lat": lats,
    #         "lon": lons,
    #     },
    # )

    # # 儲存成 nc
    # ds.to_netcdf("generated_first_day_3region.nc")


    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import os
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    # 把 tensor 轉成 numpy
    gen_np = generated_denorm.numpy()  # shape: (N, 24, H, W)
    tgt_np = target_denorm.numpy()

    # 取第一天的前 8 個時段
    forecast_hours = 8
    gen_day1 = gen_np[:, :forecast_hours, :, :]
    tgt_day1 = tgt_np[:, :forecast_hours, :, :]

    # 攤平成一維，方便做 boxplot
    # 每個時段做一個 box
    gen_data = [gen_day1[:, i, :, :].reshape(-1) for i in range(forecast_hours)]
    tgt_data = [tgt_day1[:, i, :, :].reshape(-1) for i in range(forecast_hours)]

    # 儲存成 npy
    np.save("gen_data.npy", gen_data)
    np.save("tgt_data.npy", tgt_data)

    # 畫圖
    plt.figure(figsize=(12,6))
    positions_gen = np.arange(1, forecast_hours*2, 2)     # 預測的 box 位置
    positions_tgt = np.arange(2, forecast_hours*2+1, 2)   # 真值的 box 位置

    plt.boxplot(gen_data, positions=positions_gen, widths=0.6, patch_artist=True,
                boxprops=dict(facecolor="lightblue"), medianprops=dict(color="blue"), tick_labels=[f"{i+1}" for i in range(forecast_hours)])
    plt.boxplot(tgt_data, positions=positions_tgt, widths=0.6, patch_artist=True,
                boxprops=dict(facecolor="lightcoral"), medianprops=dict(color="red"))

    plt.xticks(np.arange(1.5, forecast_hours*2, 2), [f"H{i+1}" for i in range(forecast_hours)])
    plt.xlabel("Forecast Hour")
    plt.ylabel("Temperature (K)")
    plt.title("Temperature Distribution: Predicted (blue) vs Target (red) - First 8 Hours")
    plt.grid(True, axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()
