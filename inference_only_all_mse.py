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
    checkpoint = torch.load("./saved_models/diffusion_model_3region.pth", map_location=device)
    model = UNet(in_channels=24, out_channels=24, cond_channels=72, time_dim=32).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    beta = checkpoint["beta"].to(device)
    model.eval()
    print("✅ 模型載入完成")

    # === 準備資料 ===
    train_files, test_files = prepare_file_list("data/npy_files_3region")
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

    # === 從 CSV 讀取 valid_time 列表 ===
    df = pd.read_csv("./data/valid_time(20).csv")
    target_dates = np.array([np.datetime64(t).astype("datetime64[D]") for t in df["valid_time"].tolist()])

    # 把測試集的時間轉成日期
    time_test = time_test.astype("datetime64[s]")  # 先轉秒
    time_test_dates = time_test.astype("datetime64[D]")

    # 找出符合日期的 index
    indices = np.isin(time_test_dates, target_dates).nonzero()[0]
    print(f"✅ 找到 {len(indices)} 筆符合條件的資料")

    # 取子集
    cond_subset = cond_test[indices]
    target_subset = target_test[indices]  # CPU



    target_test = target_test.squeeze(2)  # 在 getitem 或 collate_fn 裡




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

    # === 計算三天的 MSE（搬回 GPU 計算）===
    def split_days(tensor):
        return tensor[:, 0:8], tensor[:, 8:16], tensor[:, 16:24]

    gen_day1, gen_day2, gen_day3 = split_days(generated_denorm.to(device))
    gt_day1, gt_day2, gt_day3 = split_days(target_denorm.to(device))

    gt_day1_cpu = gt_day1.cpu()
    std_day1 = torch.std(gt_day1_cpu.view(-1)).item()
    print(f"Day 1 目標資料標準差: {std_day1:.4f} K")

    mse_fn = nn.MSELoss(reduction='mean')
    mse_day1 = mse_fn(gen_day1, gt_day1).item()
    mse_day2 = mse_fn(gen_day2, gt_day2).item()
    mse_day3 = mse_fn(gen_day3, gt_day3).item()
    mse_all = mse_fn(generated_denorm.to(device), target_denorm.to(device)).item()

    print(f"MSE (Day 1): {mse_day1:.6f}")
    print(f"MSE (Day 2): {mse_day2:.6f}")
    print(f"MSE (Day 3): {mse_day3:.6f}")
    print(f"MSE (All 3 Days): {mse_all:.6f}")

    rmse_day1 = np.sqrt(mse_day1).item()
    rmse_day2 = np.sqrt(mse_day2).item()
    rmse_day3 = np.sqrt(mse_day3).item()
    rmse_all  = np.sqrt(mse_all).item()

    print(f"RMSE (Day 1): {rmse_day1:.6f}")
    print(f"RMSE (Day 2): {rmse_day2:.6f}")
    print(f"RMSE (Day 3): {rmse_day3:.6f}")
    print(f"RMSE (All 3 Days): {rmse_all:.6f}")
