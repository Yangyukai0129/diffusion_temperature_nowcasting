import os
import numpy as np
import torch
import xarray as xr
from tqdm import tqdm

# =========================================================
# Step A: 讀取多個 NetCDF 並對齊時間
# =========================================================
def load_and_merge_datasets(file_var_pairs):
    """
    讀取多個 NetCDF 檔案，取共同時間，對齊後合併成 Dataset
    避免 xr.merge() 卡死，直接手動組合變數
    """
    datasets = []
    time_arrays = []

    # 讀檔 & 收集時間
    for file_path, var_list in file_var_pairs:
        ds = xr.open_dataset(file_path, chunks={"valid_time": 1000})
        datasets.append(ds[var_list])
        time_arrays.append(np.array(ds['valid_time'].values, dtype='datetime64[ns]'))

    # 取共同時間
    common_times = time_arrays[0]
    for arr in time_arrays[1:]:
        common_times = np.intersect1d(common_times, arr)
    common_times = np.sort(common_times)

    print(f"✅ 共同時間點數量: {len(common_times)}")

    # 對齊時間 & 載入
    for i in tqdm(range(len(datasets)), desc="對齊時間並載入記憶體"):
        datasets[i] = datasets[i].sel(valid_time=common_times).load()
        datasets[i] = datasets[i].reset_coords(drop=True)
        datasets[i].attrs = {}

    # 合併變數
    merged_ds = xr.Dataset()
    for i, (_, var_list) in enumerate(file_var_pairs):
        for var in var_list:
            merged_ds[var] = datasets[i][var]

    return merged_ds


# =========================================================
# Step B: Lazy 輸出成 .npy
# =========================================================
def lazy_export_time_windows(ds, cond_vars, target_var, cond_days, target_days,
                             months_filter=None, out_dir="data/npy_files"):
    """Lazy load 每個 time window，直接存成 .npy，避免吃滿記憶體"""

    total_steps = len(ds["valid_time"])
    max_start = total_steps - cond_days - target_days + 1
    valid_idx = np.arange(max_start)

    # 篩選月份
    if months_filter is not None:
        valid_times = ds["valid_time"].values
        months = valid_times.astype('datetime64[M]').astype(int) % 12 + 1
        print("資料月份分布:", np.unique(months, return_counts=True))
        mask = np.isin(months, months_filter)
        print(f"月份 mask 數量: {np.sum(mask)} / {len(mask)}")
        valid_idx = valid_idx[mask[valid_idx]]
        print(f"篩選月份後可用索引: {len(valid_idx)}")

    # 建資料夾
    os.makedirs(f"{out_dir}/cond", exist_ok=True)
    os.makedirs(f"{out_dir}/target", exist_ok=True)
    os.makedirs(f"{out_dir}/time", exist_ok=True)

    save_count = 0
    for save_i, i in enumerate(tqdm(valid_idx, desc="輸出 .npy 中")):
        # cond
        cond_list = []
        for var in cond_vars:
            da = ds[var]
            if "pressure_level" in da.dims:
                da = da.isel(pressure_level=0)
            arr = da.isel(valid_time=slice(i, i + cond_days)).values
            cond_list.append(arr)
        cond = np.stack(cond_list, axis=1)  # (T, C, H, W)

        # target
        da_tgt = ds[target_var]
        if "pressure_level" in da_tgt.dims:
            da_tgt = da_tgt.isel(pressure_level=0)
        target = da_tgt.isel(
            valid_time=slice(i + cond_days, i + cond_days + target_days)
        ).values
        target = np.expand_dims(target, axis=1)  # (T, 1, H, W)

        # NaN 過濾
        # if np.isnan(cond).any() or np.isnan(target).any():
        if np.isnan(cond).all():
            continue

        # 儲存
        np.save(f"{out_dir}/cond/{save_count:06d}.npy", cond.astype(np.float32))
        np.save(f"{out_dir}/target/{save_count:06d}.npy", target.astype(np.float32))
        np.save(f"{out_dir}/time/{save_count:06d}.npy", ds["valid_time"].values[i + cond_days])
        save_count += 1

    print(f"✅ 輸出完成，共 {save_count} 筆")


# =========================================================
# Step C: Lazy Dataset 讀 .npy 訓練
# =========================================================
class LazyWeatherDataset(torch.utils.data.Dataset):
    def __init__(self, cond_dir, target_dir, time_dir):
        self.cond_dir = cond_dir
        self.target_dir = target_dir
        self.time_dir = time_dir
        self.indices = sorted([f.split(".")[0] for f in os.listdir(cond_dir)])

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        cond = np.load(os.path.join(self.cond_dir, f"{self.indices[idx]}.npy"))
        target = np.load(os.path.join(self.target_dir, f"{self.indices[idx]}.npy"))
        time = np.load(os.path.join(self.time_dir, f"{self.indices[idx]}.npy"), allow_pickle=True)

        cond = torch.from_numpy(cond).float()
        target = torch.from_numpy(target).float()

        return cond, target, time


# =========================================================
# Step D: 主程式
# =========================================================
if __name__ == "__main__":
    file_var_pairs = [
        ("download_nc/CVPR.nc", ["t"]),
    ]

    cond_steps = 64
    target_steps = 24

    # 讀資料
    ds = load_and_merge_datasets(file_var_pairs)

    # Lazy 輸出
    lazy_export_time_windows(
        ds,
        cond_vars=["t"],
        target_var="t",
        cond_days=cond_steps,
        target_days=target_steps,
        months_filter=[6, 7, 8],
        out_dir="data/8day_3day"
    )
