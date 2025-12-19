import xarray as xr
import glob
from dask.diagnostics import ProgressBar

# 1. 讀取所有檔案路徑（請依實際情況修改）
file_paths = sorted(glob.glob("./download_nc/v_component_of_wind/*.nc"))  # 假設資料在 data 資料夾內

# 2. 合併所有檔案
# 開啟多檔案並啟用分塊
ds_all = xr.open_mfdataset(
    file_paths, 
    combine="by_coords", 
    parallel=True,        # 啟用 Dask 平行讀取
    chunks={"time": 100}  # 根據資料切分，調整這個值可測試效能
)

# 3. 過濾時間範圍：只保留 1980~2024 年
ds_filtered = ds_all.sel(valid_time=slice("1965-06-01", "2024-08-31"))

# 4. 儲存為新檔案（可選）
with ProgressBar():
    ds_filtered.to_netcdf("./download_nc/merged_0.05v.nc", compute=True)

# import xarray as xr
# import glob

# # 1. 讀取所有檔案路徑（請依實際情況修改）
# file_paths = sorted(glob.glob("./download_nc/*.nc"))  # 假設資料在 data 資料夾內

# # 2. 合併所有檔案
# ds_all = xr.open_mfdataset(file_paths, combine='by_coords')

# # 4. 儲存為新檔案（可選）
# ds_all.to_netcdf("./download_nc/merged_1965_2024.nc")