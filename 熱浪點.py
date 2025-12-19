import xarray as xr
import pandas as pd
import numpy as np

# === 1. 讀取資料 ===
t = xr.open_dataset('download_nc/CVPR.nc', chunks={'valid_time': 1000})

# 將 valid_time 轉為 datetime
t = t.assign_coords(valid_time=pd.to_datetime(t.valid_time.values))

# 計算每日平均氣溫
temp_daily = t.resample(valid_time='1D').mean()
temp_daily = temp_daily.chunk({'valid_time': -1})

# 選擇研究時間範圍
temp_daily = temp_daily.sel(valid_time=slice("1965-06-01", "2024-08-31"))

# === 2. 計算每個格點的 95 百分位數門檻 ===
thresholds = temp_daily.quantile(0.95, dim='valid_time')
thr_t = thresholds['t']
thr_t_computed = thr_t.compute()
print("門檻最小值:", thr_t_computed.min().item())
print("門檻最大值:", thr_t_computed.max().item())

# === 3. 判斷熱浪事件 (大於門檻為 1，否則 0) ===
heatwave_events = (temp_daily > thresholds).astype('int8')

# 計算總熱浪事件數量
total_events = heatwave_events['t'].sum().compute().item()
print(f"總熱浪事件數量（所有時間 + 所有格點）: {total_events}")

# === 5. 定義函數查詢特定日期熱浪格點 ===
def get_heatwave_points(date_str, heatwave_da):
    """
    輸入:
        date_str: 'YYYY-MM-DD'
        heatwave_da: heatwave events DataArray, shape [time, lat, lon], 0/1
    輸出:
        heatwave_points: list of (lat, lon) 出現熱浪的格點
    """
    events_on_date = heatwave_da.sel(valid_time=date_str)['t']
    heatwave_mask = events_on_date == 1
    latitudes = events_on_date.latitude.values
    longitudes = events_on_date.longitude.values
    heatwave_lats, heatwave_lons = np.where(heatwave_mask)
    heatwave_points = [(latitudes[i], longitudes[j]) for i, j in zip(heatwave_lats, heatwave_lons)]
    return heatwave_points

# === 6. 範例查詢 ===
date = "2024-07-18"
points = get_heatwave_points(date, heatwave_events)
print(f"{date} 熱浪格點總數: {len(points)}")
print("部分熱浪格點經緯度:", points)