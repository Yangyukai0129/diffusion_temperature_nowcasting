import torch

# 1. 檢查版本 (應該會顯示帶有 dev 或 nightly 字樣的版本)
print(f"Torch Version: {torch.__version__}")

# 2. 檢查 CUDA 是否可用 (這是最關鍵的，應該要回傳 True)
print(f"CUDA Available: {torch.cuda.is_available()}")

# 3. 檢查目前抓到的 GPU 名稱
if torch.cuda.is_available():
    print(f"GPU Device: {torch.cuda.get_device_name(0)}")