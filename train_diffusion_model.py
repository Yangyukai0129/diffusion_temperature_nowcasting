import torch
import torch.nn as nn
from unet import UNet, train
import torch.optim as optim
from data_utils import LazyWeatherDataset, prepare_file_list, compute_mean_std
from torch.utils.data import DataLoader


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用裝置：{device}")

    # 建立模型
    model = UNet(in_channels=8, out_channels=8, cond_channels=64, time_dim=32).to(device)

    # 1. 取得檔案清單
    train_files, test_files = prepare_file_list("data/8day_1day")

    # 2. 計算 normalization 統計數值
    cond_mean, cond_std, target_mean, target_std = compute_mean_std(train_files)

    # 3. 建立 Dataset
    train_dataset = LazyWeatherDataset(train_files, cond_mean, cond_std, target_mean, target_std)

    # 4. DataLoader（開多 worker）
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)

    criterion = nn.L1Loss()
    train_loss_history = []

    # 第一階段訓練
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    train_loss_history, rmse_history, ssim_history, beta, alpha, alpha_cumprod = train(
        model, train_loader, 40, device,
        optimizer, criterion,
        train_loss_history,
        use_checkpoint=True  # 開啟梯度檢查點
    )

    # 釋放資源
    torch.save({
        "model_state_dict": model.state_dict(),
        "beta": beta,
        "alpha": alpha,
        "alpha_cumprod": alpha_cumprod,
        "optimizer_state_dict": optimizer.state_dict(),
        "train_loss_history": train_loss_history,
        "rmse_history": rmse_history,
        "ssim_history": ssim_history,
    }, "./saved_models/temp_stage4.pth")
    torch.cuda.empty_cache()

    # 第二階段 fine-tune
    ckpt = torch.load("./saved_models/temp_stage4.pth", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    beta, alpha, alpha_cumprod = ckpt["beta"].to(device), ckpt["alpha"].to(device), ckpt["alpha_cumprod"].to(device)

    optimizer = optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-6)
    # optimizer.load_state_dict(ckpt["optimizer_state_dict"])  # ← 若想延續訓練動量則加上

    train_loss_history2, rmse_history2, ssim_history2, _, _, _ = train(
        model, train_loader, 10, device, optimizer, criterion, [],
        use_checkpoint=True, beta=beta, alpha=alpha, alpha_cumprod=alpha_cumprod
    )

    # 合併歷史
    train_loss_history += train_loss_history2
    rmse_history += rmse_history2
    ssim_history += ssim_history2

    # 最終儲存
    torch.save({
        "model_state_dict": model.state_dict(),
        "beta": beta,
        "alpha": alpha,
        "alpha_cumprod": alpha_cumprod,
        "optimizer_state_dict": optimizer.state_dict(),
        "train_loss_history": train_loss_history,
        "rmse_history": rmse_history,
        "ssim_history": ssim_history,
    }, "./saved_models/diffusion_model_8day1_1219.pth")
    print("模型已儲存完成")

if __name__ == "__main__":
    main()