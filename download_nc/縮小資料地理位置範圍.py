import xarray as xr


ds = xr.open_dataset('download_nc/5/merged_5deg_t.nc')


# 選取子區域
# subset = ds.sel(
#     latitude=slice(60, 36.25),        # 用 36.25 而不是 36
#     longitude=slice(-12, 11.75)       # 用 11.75 而不是 12
# )
subset = ds.sel(latitude=slice(65, 28), longitude=slice(20, 120), valid_time=slice('1995-01-01', '2024-12-31'))  # 時間範圍


# 檢查結果
print(subset)

subset.to_netcdf("download_nc/CVPR.nc")

# import xarray as xr
# import numpy as np

# # 讀取原始資料
# ds = xr.open_dataset('download_nc/5/merged_5deg_t.nc')

# # 三個區域的經緯度範圍
# regions_bounds = {
#     "Central China": (90, 120.25, 25, 35.25),
#     "Northeastern Europe": (35, 60.25, 55, 70.25)
# }

# # 找出整體經緯度範圍 (cover 所有區域)
# lat_min = min([b[0] for b in regions_bounds.values()])
# lat_max = max([b[1] for b in regions_bounds.values()])
# lon_min = min([b[2] for b in regions_bounds.values()])
# lon_max = max([b[3] for b in regions_bounds.values()])

# # latitude 可能是降序
# if ds.latitude.values[0] > ds.latitude.values[-1]:
#     lat_slice = slice(lat_max, lat_min)
# else:
#     lat_slice = slice(lat_min, lat_max)
# lon_slice = slice(lon_min, lon_max)

# # 建一個空的 dataset，shape: (time, lat, lon)
# subset = ds.sel(latitude=lat_slice, longitude=lon_slice)
# t_combined = subset['t'].copy(deep=True)  # 用於建立 DataArray
# t_combined[:] = nan  # 先填滿 NaN

# # 把每個區域的資料填進對應位置
# for region_name, (r_lat_min, r_lat_max, r_lon_min, r_lon_max) in regions_bounds.items():
#     # latitude slice
#     if ds.latitude.values[0] > ds.latitude.values[-1]:
#         lat_s = slice(r_lat_max, r_lat_min)
#     else:
#         lat_s = slice(r_lat_min, r_lat_max)
#     lon_s = slice(r_lon_min, r_lon_max)
    
#     region_data = ds['t'].sel(latitude=lat_s, longitude=lon_s)
#     t_combined.loc[dict(latitude=region_data.latitude, longitude=region_data.longitude)] = region_data

# # 儲存成 NetCDF
# t_combined.to_netcdf("download_nc/combined_central_china_north_europe.nc")
# print("✅ 已儲存合併後的三個區域 NetCDF")
