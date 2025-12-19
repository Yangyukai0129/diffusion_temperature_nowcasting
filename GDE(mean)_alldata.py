import torch
import torch.nn as nn
from tqdm import tqdm
import pandas as pd
import numpy as np

from unet import UNet, ddim_inference
from data_utils import LazyWeatherDataset, prepare_file_list, compute_mean_std, denormalize
from torch.utils.data import DataLoader

# === 主要推理流程 ===
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("✅ 使用裝置:", device)

    # === 載入模型 ===
    checkpoint = torch.load("./saved_models/diffusion_model_community_attention.pth", map_location=device)
    model = UNet(in_channels=24, out_channels=24, cond_channels=72, time_dim=32).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    beta = checkpoint["beta"].to(device)
    model.eval()
    print("✅ 模型載入完成")

    # === 準備資料 ===
    train_files, test_files = prepare_file_list("data/npy_community")
    cond_mean, cond_std, target_mean, target_std = compute_mean_std(train_files)

    test_dataset = LazyWeatherDataset(test_files, cond_mean, cond_std, target_mean, target_std)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=4, pin_memory=True)


    mse_fn = nn.MSELoss(reduction='mean')

    N = 15  # ensemble 次數

    all_generated = []
    all_targets = []

    with torch.no_grad():
        for cond_batch, target_batch, valid_time in tqdm(test_loader, desc="Testing Batches"):

            cond_subset = cond_batch.to(device)
            target_subset = target_batch  # 還在標準化狀態

            if cond_subset.size(0) == 0:
                continue

            # 累加 N 次推理結果
            batch_sum = torch.zeros_like(target_subset, dtype=torch.float32)

            # 這裡先檢查 shape，如果是 [B, 24, 1, H, W] 就 squeeze 掉
            if batch_sum.dim() == 5 and batch_sum.size(2) == 1:
                batch_sum = batch_sum.squeeze(2)

            for _ in range(N):
                gen = ddim_inference(model, cond_subset, beta, device=device, eta=0.0, num_steps=15)
                gen_cpu = gen.cpu()
                gen_denorm = denormalize(gen_cpu, target_mean, target_std)  # ✅ 預測結果反標準化
                batch_sum += gen_denorm
                
            # 平均後的預測結果
            batch_avg = batch_sum / N

            # target 也反標準化
            target_denorm = denormalize(target_subset, target_mean, target_std)

            all_generated.append(batch_avg)
            all_targets.append(target_denorm)

    # 合併所有 batch
    generated_final = torch.cat(all_generated, dim=0)
    target_final = torch.cat(all_targets, dim=0)

    if target_final.dim() == 5 and target_final.size(2) == 1:
        target_final = target_final.squeeze(2)

    # === 計算三天 MSE ===
    def split_days(tensor):
        return tensor[:, 0:8], tensor[:, 8:16], tensor[:, 16:24]

    gen_day1, gen_day2, gen_day3 = split_days(generated_final.to(device))
    gt_day1, gt_day2, gt_day3 = split_days(target_final.to(device))

    mse_day1 = mse_fn(gen_day1, gt_day1).item()
    mse_day2 = mse_fn(gen_day2, gt_day2).item()
    mse_day3 = mse_fn(gen_day3, gt_day3).item()
    mse_all = mse_fn(generated_final.to(device), target_final.to(device)).item()

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
