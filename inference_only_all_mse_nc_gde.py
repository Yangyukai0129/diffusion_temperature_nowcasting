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
    checkpoint = torch.load("./saved_models/diffusion_model_community.pth", map_location=device)
    model = UNet(in_channels=24, out_channels=24, cond_channels=72, time_dim=32).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    beta = checkpoint["beta"].to(device)
    model.eval()
    print("✅ 模型載入完成")

    # === 準備資料 ===
    train_files, test_files = prepare_file_list("data/npy_community")
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

    # === 推理（過程直接搬回 CPU，節省 GPU RAM）===
    print("🚀 開始生成...")
    N = 5  # ensemble 次數
    all_generated = []
    all_times = []  # ✅ 收集 valid_time

    with torch.no_grad():
        for cond_batch, target_batch, valid_time in tqdm(test_loader, desc="Testing Batches"):

            cond_subset = cond_batch.to(device)
            target_subset = target_batch  # 還在標準化狀態

            if cond_subset.size(0) == 0:
                continue

            # 建立 batch_sum
            batch_sum = torch.zeros_like(target_subset, dtype=torch.float32)

            # 如果 shape 是 [B, 24, 1, H, W] -> squeeze 掉第 2 維
            if batch_sum.dim() == 5 and batch_sum.size(2) == 1:
                batch_sum = batch_sum.squeeze(2)

            for _ in range(N):
                gen = ddim_inference(model, cond_subset, beta, device=device, eta=0.0, num_steps=15)
                gen_cpu = gen.cpu()
                gen_denorm = denormalize(gen_cpu, target_mean, target_std)  # 預測結果反標準化
                batch_sum += gen_denorm
            
            # 平均後的預測結果
            batch_avg = batch_sum / N

            # target 也反標準化
            target_denorm = denormalize(target_subset, target_mean, target_std)

            all_generated.append(batch_avg.cpu())
            all_times.extend(valid_time.cpu().numpy())  # ✅ 收集時間

    # 合併所有 batch
    generated_final = torch.cat(all_generated, dim=0)  # (N, 24, H, W)
    times_final = pd.to_datetime(all_times, unit='s')  # ✅ 轉成 pandas datetime

    # === 取第一天 8 時段 ===
    generated_first_day = generated_final[:, 0:8, :, :].numpy().astype(np.float32)  # (N, 8, H, W)

    import xarray as xr

    # 從真實資料抓經緯度
    obs_ds = xr.open_dataset("download_nc/combined_community.nc")
    lats = obs_ds.latitude.values
    lons = obs_ds.longitude.values

    # 建立 xarray Dataset
    ds = xr.Dataset(
        {
            "temperature": (["time", "forecast_hour", "lat", "lon"], generated_first_day),
        },
        coords={
            "time": times_final,
            "forecast_hour": np.arange(1, 9),
            "lat": lats,
            "lon": lons,
        },
    )

    # 儲存成 NetCDF
    ds.to_netcdf("generate_data/generated_first_day_community_gde.nc")
    print("✅ NetCDF 檔案已儲存：generated_first_day_gde.nc")


    # import os
    # os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # 避免 OMP #15 錯誤

    # import matplotlib.pyplot as plt
    # import numpy as np

    # forecast_hours = 8  # 第一天天氣的時段數

    # # --- reshape & flatten --- 
    # # generated_first_day: (N, 8, H, W)
    # gen_current = generated_first_day.reshape(generated_first_day.shape[0], forecast_hours, -1)
    # gen_external = np.load("gen_data.npy", allow_pickle=True)
    # tgt_external = np.load("tgt_data.npy", allow_pickle=True)

    # gen_external = gen_external.reshape(gen_external.shape[0], forecast_hours, -1)
    # tgt_external = tgt_external.reshape(tgt_external.shape[0], forecast_hours, -1)

    # # 準備 boxplot 資料
    # gen_current_boxes = [gen_current[:, i, :].flatten() for i in range(forecast_hours)]
    # gen_external_boxes = [gen_external[:, i, :].flatten() for i in range(forecast_hours)]
    # tgt_external_boxes = [tgt_external[:, i, :].flatten() for i in range(forecast_hours)]

    # plt.figure(figsize=(12,6))

    # # 每個 forecast hour 3 個 box，間距用 3 的倍數
    # positions_current = np.arange(1, forecast_hours*3, 3)
    # positions_gen_ext = np.arange(2, forecast_hours*3+1, 3)
    # positions_tgt_ext = np.arange(3, forecast_hours*3+2, 3)

    # bp1 = plt.boxplot(gen_current_boxes, positions=positions_current, widths=0.6, patch_artist=True,
    #                 boxprops=dict(facecolor="lightblue", alpha=0.7), medianprops=dict(color="blue"))
    # bp2 = plt.boxplot(gen_external_boxes, positions=positions_gen_ext, widths=0.6, patch_artist=True,
    #                 boxprops=dict(facecolor="lightgreen", alpha=0.7), medianprops=dict(color="green"))
    # bp3 = plt.boxplot(tgt_external_boxes, positions=positions_tgt_ext, widths=0.6, patch_artist=True,
    #                 boxprops=dict(facecolor="lightcoral", alpha=0.7), medianprops=dict(color="red"))

    # # x 軸標籤放在每個 forecast hour 的中間
    # plt.xticks(np.arange(2, forecast_hours*3, 3), [f"H{i+1}" for i in range(forecast_hours)])
    # plt.xlabel("Forecast Hour")
    # plt.ylabel("Temperature (K)")
    # plt.title("Temperature Distribution per Forecast Hour")

    # plt.legend([bp1["boxes"][0], bp2["boxes"][0], bp3["boxes"][0]],
    #         ["Diffusion(GDE)", "Diffusion", "Origin"],
    #         loc="upper right")

    # plt.grid(True, axis="y", linestyle="--", alpha=0.5)
    # plt.tight_layout()
    # plt.show()
