import torch
import torch.nn as nn
from tqdm import tqdm
import pandas as pd
import numpy as np

from unet import UNet, ddim_inference
from data_utils import LazyWeatherDataset, prepare_file_list, compute_mean_std, denormalize
from torch.utils.data import DataLoader
from skimage.metrics import structural_similarity as ssim

config = {'data_dir': 'data/8day_1day', 
                   'cond_steps': 64, 
                   'target_steps': 8, 
                   'in_channels': 8, 
                   'out_channels': 8, 
                   'cond_channels': 64, 
                   'time_dim': 32, 
                   'depth': 4, 
                   'base_channels': 128, 
                   'channel_mults': [1, 2, 2, 8]
                   }

# === 主要推理流程 ===
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("✅ 使用裝置:", device)

    # === 載入模型 ===
    checkpoint = torch.load("./saved_models/diffusion_model_8day1_1219.pth", map_location=device, weights_only=False)
    # model = UNet(config).to(device)
    model = UNet(in_channels=8, out_channels=8, cond_channels=64, time_dim=32).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    beta = checkpoint["beta"].to(device)
    model.eval()
    print("✅ 模型載入完成")

    # === 準備資料 ===
    train_files, test_files = prepare_file_list("data/8day_1day")
    cond_mean, cond_std, target_mean, target_std = compute_mean_std(train_files)

    test_dataset = LazyWeatherDataset(test_files, cond_mean, cond_std, target_mean, target_std)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=4, pin_memory=True)

    # === 從 CSV 讀取 valid_time 列表 ===
    df = pd.read_csv("./data/valid_time(20).csv")
    target_dates = np.array([np.datetime64(t).astype("datetime64[D]") for t in df["valid_time"].tolist()])

    mse_fn = nn.MSELoss(reduction='mean')

    N = 15  # ensemble 次數

    all_generated_avg = [] # 儲存平均後的預測
    all_generated_std = [] # 儲存預測的標準差
    all_targets = []

    with torch.no_grad():
        for cond_batch, target_batch, valid_time in tqdm(test_loader, desc="Testing Batches"):
            # 篩選符合日期的樣本
            valid_time_np = valid_time.numpy().astype("datetime64[s]").astype("datetime64[D]")
            mask = np.isin(valid_time_np, target_dates)
            if not mask.any():
                continue

            cond_subset = cond_batch[mask].to(device)
            target_subset = target_batch[mask]  # 還在標準化狀態

            if cond_subset.size(0) == 0:
                continue

            # 初始化累加變數
            batch_sum = torch.zeros_like(target_subset, dtype=torch.float32)
            batch_sum_sq = torch.zeros_like(target_subset, dtype=torch.float32) # 用於計算標準差

            # 這裡先檢查 shape，如果是 [B, 24, 1, H, W] 就 squeeze 掉
            if batch_sum.dim() == 5 and batch_sum.size(2) == 1:
                batch_sum = batch_sum.squeeze(2)
                batch_sum_sq = batch_sum_sq.squeeze(2)
            
            # 儲存 N 次推理的結果，用於後續計算標準差
            ensemble_results = []

            for _ in range(N):
                gen = ddim_inference(model, cond_subset, beta, device=device, eta=0.0, num_steps=15)
                gen_cpu = gen.cpu()
                gen_denorm = denormalize(gen_cpu, target_mean, target_std)  # ✅ 預測結果反標準化
                ensemble_results.append(gen_denorm)
                
            # 將 N 次結果堆疊起來，計算平均和標準差
            stacked_results = torch.stack(ensemble_results, dim=0) # shape: [N, B, C, H, W]
            batch_avg = torch.mean(stacked_results, dim=0) # shape: [B, C, H, W]
            batch_std = torch.std(stacked_results, dim=0)  # shape: [B, C, H, W]

            # target 也反標準化
            target_denorm = denormalize(target_subset, target_mean, target_std)

            all_generated_avg.append(batch_avg)
            all_generated_std.append(batch_std) # 儲存標準差
            all_targets.append(target_denorm)

    def compute_ssim_batch(pred, gt):
        pred_np = pred.detach().cpu().numpy().astype(np.float32)
        gt_np = gt.detach().cpu().numpy().astype(np.float32)
        ssim_scores = []
        for i in range(pred_np.shape[0]):  # 每筆樣本
            ch_scores = []
            for c in range(pred_np.shape[1]):
                ch_pred = pred_np[i, c]
                ch_gt = gt_np[i, c]
                # 確保 data_range 不為零或負數
                data_range_val = max(ch_gt.max() - ch_gt.min(), 1e-5)
                s = ssim(
                    ch_pred,
                    ch_gt,
                    data_range=data_range_val
                )
                ch_scores.append(s)
            ssim_scores.append(np.mean(ch_scores))
        return float(np.mean(ssim_scores))

    # 合併所有 batch
    generated_final_avg = torch.cat(all_generated_avg, dim=0)
    generated_final_std = torch.cat(all_generated_std, dim=0) # 合併標準差
    target_final = torch.cat(all_targets, dim=0)

    if target_final.dim() == 5 and target_final.size(2) == 1:
        target_final = target_final.squeeze(2)

    print("generated_final_avg shape:", generated_final_avg.shape)
    print("generated_final_std shape:", generated_final_std.shape)
    print("target_final shape:", target_final.shape)

    # === 計算三天 MSE ===
    def split_days(tensor_avg, tensor_std, tensor_gt):
        return tensor_avg[:, 0:8], tensor_std[:, 0:8], tensor_gt[:, 0:8]

    gen_day1_avg, gen_day1_std, gt_day1 = split_days(
        generated_final_avg.to(device),
        generated_final_std.to(device),
        target_final.to(device)
    )

    mse_day1 = mse_fn(gen_day1_avg, gt_day1).item()
    rmse_day1 = mse_day1 ** 0.5
    ssim_day1 = compute_ssim_batch(gen_day1_avg, gt_day1)

    # 計算預測標準差的平均值，作為不確定性的量度
    mean_pred_std_day1 = torch.mean(gen_day1_std).item()

    print(f"MSE (Day 1): {mse_day1:.6f}")
    print(f"RMSE (Day 1): {rmse_day1:.6f}")
    print(f"RMSE (Day 1) with Prediction Std Dev: {rmse_day1:.6f} ± {mean_pred_std_day1:.6f}")
    print(f"SSIM (Day 1): {ssim_day1:.6f}")

    # 可以進一步視覺化預測的標準差
    # 例如，針對某個預測樣本，繪製其預測值和標準差的熱力圖
    # (此處僅為文字說明，實際繪圖需要 matplotlib 或 seaborn)
    print("\n💡 提示：你可以進一步視覺化 'gen_day1_std' 來觀察預測的不確定性分佈。")