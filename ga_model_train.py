# train_model_final.py (一個新的或修改過的訓練腳本)
import torch
import torch.nn as nn
from unet_ga import UNet # 使用我們可配置的 UNet
import torch.optim as optim
from data_utils import LazyWeatherDataset, prepare_file_list, compute_mean_std
from torch.utils.data import DataLoader
from unet_ga import train # 沿用您原有的 train 函數

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用裝置：{device}")

    # =========================================================
    # <<< 1. 手動定義從 GA 找到的最佳 Config >>>
    # =========================================================
    best_config = {'data_dir': 'data/8day_1day', 
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
    print("使用 GA 找到的最佳設定進行完整訓練:", best_config)

    # =========================================================
    # <<< 2. 使用 Config 中的 data_dir 來準備資料 >>>
    # =========================================================
    data_dir = best_config['data_dir']
    train_files, test_files = prepare_file_list(data_dir)
    cond_mean, cond_std, target_mean, target_std = compute_mean_std(train_files)
    train_dataset = LazyWeatherDataset(train_files, cond_mean, cond_std, target_mean, target_std)
    # 使用完整的 DataLoader，而不是 Subset
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)

    # =========================================================
    # <<< 3. 使用 Config 建立模型 >>>
    # =========================================================
    model = UNet(best_config).to(device)

    criterion = nn.L1Loss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4) # 可以從一個較高的學習率開始
    train_loss_history = []
    
    # =========================================================
    # <<< 4. 進行長時間的完整訓練 >>>
    # =========================================================
    train_loss_history, rmse_history, ssim_history, beta, alpha, alpha_cumprod = train(
        model, train_loader, num_epochs=50, device=device, # 訓練更多輪次
        optimizer=optimizer, criterion=criterion,
        train_loss_history=train_loss_history, # 傳遞 history 列表
        use_checkpoint=True
    )
    
    # 儲存最終模型
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": best_config, # 將設定也一併儲存
        "train_loss_history": train_loss_history,
        "rmse_history": rmse_history,
        "ssim_history": ssim_history,
        "beta": beta, # 新增
        "alpha": alpha, # 新增
        "alpha_cumprod": alpha_cumprod, # 新增
        "optimizer_state_dict": optimizer.state_dict(), # 新增
        # ... 其他需要儲存的資訊 ...
    }, "./saved_models/ga_best_model_unet_bit1.pth")
    print("GA 找到的最佳模型已訓練並儲存完成！")

if __name__ == "__main__":
    main()