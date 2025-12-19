import xarray as xr
import pandas as pd

# 讀取你生成的資料
ds = xr.open_dataset("generate_data/generated_first_day_community_gde.nc")

# 1. 把每天的 8 個 forecast_hour 合併成每日最大值
ds_daily = ds.resample(time="1D").max(dim="time")  # 先確保 time 是 datetime64[ns]
ds_daily = ds_daily.max(dim="forecast_hour")       # 移除 forecast_hour 維度
ds_daily = ds_daily.sel(time=slice("2015-06-01", "2024-08-28"))

# 2. 計算 95 百分位數門檻（每個格點獨立計算）
thresholds = ds_daily.quantile(0.95, dim="time")
thr_t = thresholds['temperature']
thr_t_computed = thr_t.compute()  # 轉成已計算的 DataArray
print("Min:", thr_t_computed.min().item())
print("Max:", thr_t_computed.max().item())

# 3. 判斷是否為熱浪事件（1=是熱浪，0=不是）
heatwave_events = (ds_daily > thresholds).astype("int8")

# 4. 計算總熱浪事件數量（所有時間 + 所有網格點加總）
total_events = heatwave_events["temperature"].sum().compute().item()
print(f"總熱浪事件數量（所有時間 + 所有網格點）: {total_events}")

# 5. 儲存檔案（可選）
encoding = {"temperature": {"dtype": "int8", "zlib": True, "complevel": 5}}
heatwave_events.to_netcdf("heatwave_event_generate/heatwave_events_generated_first_day_community_gde.nc", encoding=encoding)
