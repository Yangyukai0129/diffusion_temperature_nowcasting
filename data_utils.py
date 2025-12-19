# data_utils.py
import torch
from torch.utils.data import Dataset
import numpy as np
import os

class LazyWeatherDataset(Dataset):
    def __init__(self, file_list, cond_mean=None, cond_std=None, target_mean=None, target_std=None):
        """
        file_list: [(cond_path, target_path), ...]
        mean/std: 事先算好的統計數值 (float 或 tensor)
        """
        self.file_list = file_list
        self.cond_mean = cond_mean
        self.cond_std = cond_std
        self.target_mean = target_mean
        self.target_std = target_std
        self.eps = 1e-6

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        cond_path, target_path, time_path = self.file_list[idx]
        cond = np.load(cond_path, mmap_mode="r")  # 不會一次載入整個檔案
        target = np.load(target_path, mmap_mode="r")
        valid_time = np.load(time_path)  # datetime64

        cond = torch.from_numpy(cond.copy()).float()
        target = torch.from_numpy(target.copy()).float()

        # 即時標準化
        if self.cond_mean is not None:
            cond = (cond - self.cond_mean) / (self.cond_std + self.eps)
        if self.target_mean is not None:
            target = (target - self.target_mean) / (self.target_std + self.eps)

        # 轉成 int timestamp（秒）
        time_int = int(valid_time.astype('datetime64[s]').astype(int))

        return cond, target, time_int


# def prepare_file_list(data_dir, split_ratio=0.8):
#     """把資料夾內 cond / target 檔案配對，切成 train/test"""
#     cond_files = sorted([os.path.join(data_dir, "cond", f) for f in os.listdir(os.path.join(data_dir, "cond"))])
#     target_files = sorted([os.path.join(data_dir, "target", f) for f in os.listdir(os.path.join(data_dir, "target"))])
#     time_files = sorted([os.path.join(data_dir, "time", f) for f in os.listdir(os.path.join(data_dir, "time"))])

#     N = len(cond_files)
#     split_index = int(N * split_ratio)

#     train_files = list(zip(cond_files[:split_index], target_files[:split_index], time_files[:split_index]))
#     test_files = list(zip(cond_files[split_index:], target_files[split_index:], time_files[split_index:]))

#     return train_files, test_files

import os
import numpy as np
import datetime

def prepare_file_list(data_dir, split_date="20200101", date_format="%Y%m%d"):
    cond_files = sorted([os.path.join(data_dir, "cond", f) for f in os.listdir(os.path.join(data_dir, "cond"))])
    target_files = sorted([os.path.join(data_dir, "target", f) for f in os.listdir(os.path.join(data_dir, "target"))])
    time_files = sorted([os.path.join(data_dir, "time", f) for f in os.listdir(os.path.join(data_dir, "time"))])

    split_dt = datetime.datetime.strptime(split_date, date_format)

    train_files, test_files = [], []
    all_times = []

    for c, t, tm in zip(cond_files, target_files, time_files):
        ts = np.load(tm)
        if isinstance(ts, np.ndarray):
            ts = ts.item()

        # robust 轉換：無論 ts 正負都能轉
        dt = np.datetime64(ts, 'ns').astype('M8[ms]').astype(datetime.datetime)
        all_times.append(dt)

        if dt <= split_dt:
            train_files.append((c, t, tm))
        else:
            test_files.append((c, t, tm))

    # Debug print
    print(f"[DEBUG] 最早時間: {min(all_times)}")
    print(f"[DEBUG] 最晚時間: {max(all_times)}")
    print(f"[DEBUG] 切分日期: {split_dt}")
    print(f"[DEBUG] Train 檔案數: {len(train_files)}")
    print(f"[DEBUG] Test 檔案數: {len(test_files)}")

    return train_files, test_files



def compute_mean_std(file_list, sample_size=1000):
    """隨機取部分樣本計算 mean/std（避免讀全部）"""
    idxs = np.random.choice(len(file_list), size=min(sample_size, len(file_list)), replace=False)
    cond_vals = []
    target_vals = []

    for idx in idxs:
        cond_path, target_path, time_path = file_list[idx]
        cond = np.load(cond_path, mmap_mode="r")
        target = np.load(target_path, mmap_mode="r")
        cond_vals.append(cond)
        target_vals.append(target)

    cond_vals = np.concatenate([x.flatten() for x in cond_vals])
    target_vals = np.concatenate([x.flatten() for x in target_vals])

    return cond_vals.mean(), cond_vals.std(), target_vals.mean(), target_vals.std()

def denormalize(tensor, mean, std, eps=1e-6):
    """
    把標準化後的 tensor 還原
    tensor: torch.Tensor (標準化過的)
    mean, std: float 或 torch.Tensor
    """
    if isinstance(mean, torch.Tensor):
        mean = mean.to(tensor.device)
    if isinstance(std, torch.Tensor):
        std = std.to(tensor.device)

    return tensor * (std + eps) + mean

