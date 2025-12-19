import xarray as xr
import pandas as pd

# 1. 載入數據
t = xr.open_dataset('./download_nc/0.05/merged_0.05.nc')
t = t.sel(pressure_level=1000)
temp = t['t']
temp = temp.assign_coords(valid_time=pd.to_datetime(temp.valid_time.values))

# 2. 粗化到 5°(用 mean)
temp_coarse = temp.coarsen(latitude=20, longitude=20, boundary='trim').mean()

# 3. 儲存
encoding = {'t': {'dtype': 'float', 'zlib': True, 'complevel': 5}}
temp_coarse.to_netcdf('./download_nc/merged_5deg_t.nc', encoding=encoding)