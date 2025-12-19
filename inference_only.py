import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os

from unet import UNet, ddim_inference
from data_utils import prepare_file_list, compute_mean_std, denormalize, LazyWeatherDataset
from torch.utils.data import DataLoader

config = {'data_dir': 'data/3day_1day', 
                   'cond_steps': 24, 
                   'target_steps': 8, 
                   'in_channels': 8, 
                   'out_channels': 8, 
                   'cond_channels': 24, 
                   'time_dim': 32, 
                   'depth': 2, 
                   'base_channels': 64, 
                   'channel_mults': [1, 4]
                   }

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("✅ 使用裝置:", device)

    # 1. 載入模型與 beta
    checkpoint = torch.load("./saved_models/diffusion_model_8day3.pth", map_location=device, weights_only=False)
    # model = UNet(config).to(device)
    model = UNet(in_channels=24, out_channels=24, cond_channels=64, time_dim=32).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    beta = checkpoint["beta"].to(device)
    model.eval()
    print("✅ 模型載入完成")

    # 2. 準備資料（完整載入）
    train_files, test_files = prepare_file_list("data/8day_3day")
    cond_mean, cond_std, target_mean, target_std = compute_mean_std(train_files)

    test_dataset = LazyWeatherDataset(test_files, cond_mean, cond_std, target_mean, target_std)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

    cond_list, target_list, time_list = [], [], []
    for cond, target, valid_time in test_loader:
        cond_list.append(cond)
        target_list.append(target)
        time_list.append(valid_time)

    cond_test = torch.cat(cond_list, dim=0)  # 全部 concat 後
    target_test = torch.cat(target_list, dim=0)  # CPU 也可視需求調整




    target_subset = target_test.squeeze(2)  # 在 getitem 或 collate_fn 裡




    time_test = np.concatenate(time_list, axis=0)  # numpy.datetime64
    print(f"✅ 測試集載入完成，共 {len(cond_test)} 筆")

    # 3. 指定你要預測的 target 起始日期
    target_date = np.datetime64("2024-07-18", "s")  # 範例目標日期，視情況調整

    # 4. 找對應 cond 時間的索引（因 cond 時間比 target 早 72 個時間單位）
    offset = np.timedelta64(64 * 3 * 3600, 's')  # 72 個 3 小時 -> 秒
    cond_dates = time_test  # cond 對應的時間戳，通常是 numpy.datetime64 格式

    # 假設 cond_dates 是秒為單位的 unix timestamp
    cond_dates_dt = cond_dates.astype('datetime64[s]')  # 或 'datetime64[ms]' 根據實際單位調整

    # 確認轉換後類型
    print("cond_dates_dt dtype:", cond_dates_dt.dtype)
    print("target_date dtype:", type(target_date))

    # 找 cond_dates 裡的哪個元素加 offset 等於 target_date
    idx_candidates = np.where(cond_dates_dt + offset == target_date)[0]
    if len(idx_candidates) == 0:
        raise ValueError("找不到對應的 cond 時間")
    idx = idx_candidates[0]

    print(f"選擇的 cond 時間：{cond_dates_dt[idx]}")
    print(f"對應的 target 時間起始：{target_date}")

    # 5. 取出指定條件，做單筆推理
    cond_sample = cond_test[idx].unsqueeze(0).to(device)

    print("🚀 開始推理...")
    with torch.no_grad():
        generated = ddim_inference(model, cond_sample, beta, device=device, eta=0.0, num_steps=15)

    # 6. 反標準化（CPU）
    generated_denorm = denormalize(generated.cpu(), target_mean, target_std)
    target_denorm = denormalize(target_test[idx].unsqueeze(0), target_mean, target_std)  # 同樣轉成 batch=1

    # 7. 儲存結果
    os.makedirs("./output", exist_ok=True)
    torch.save(generated_denorm, f"./output/generated_output_{idx}_1.pt")
    torch.save(target_denorm, f"./output/ground_truth_{idx}_1.pt")
    print(f"✅ 已儲存生成結果與 Ground Truth，索引：{idx}")
