import xarray as xr
import numpy as np
from dask.diagnostics import ProgressBar

u = xr.open_dataset("./download_nc/0.05/merged_0.05u.nc", chunks={"time": 100})["u"]
v = xr.open_dataset("./download_nc/0.05/merged_0.05v.nc", chunks={"time": 100})["v"]

# 計算 wind speed
wind_speed = np.sqrt(u**2 + v**2)
wind_speed = wind_speed.compute()  # 這裡才會真的計算（分塊計算，減少記憶體壓力
wind_speed.name = "wind_speed"

# 計算 wind direction（0°~360°）
# wind_direction_deg = (np.degrees(np.arctan2(v, u)) + 360) % 360
# wind_direction_deg.name = "wind_direction"

# 合併為一個 Dataset
with ProgressBar():
    ds_wind = xr.Dataset({
        "wind_speed": wind_speed,
        # "wind_direction": wind_direction_deg
    })

# 輸出成 NetCDF
with ProgressBar():
    ds_wind.to_netcdf("./download_nc/0.05/merged_0.05wind.nc")