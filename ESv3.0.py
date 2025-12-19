import xarray as xr
import numpy as np
from scipy import stats
import pickle
import pandas as pd

# 1. 讀取數據
heatwave_events = xr.open_dataset('heatwave_event_generate/heatwave_events_generated_first_day_community_gde.nc')['temperature']
tau_max = 10
n_time = heatwave_events.shape[0]
events_flat = heatwave_events.stack(grid=['lat', 'lon']).transpose('time', 'grid')
#n_grid = heatwave_events.shape[1] * heatwave_events.shape[2]
n_grid = events_flat.sizes['grid']

# 2. 提取日期和經緯度
dates = heatwave_events['time'].values
grid_coords = events_flat['grid'].coords
grid_latitudes = grid_coords['lat'].values  # 形狀為 (n_grid,)
grid_longitudes = grid_coords['lon'].values  # 形狀為 (n_grid,)

# 3. 定義函數
def compute_event_intervals(events):
    event_times = np.where(events == 1)[0]
    if len(event_times) < 2:
        return event_times, np.array([float('inf')])
    intervals = np.diff(event_times)
    return event_times, intervals

def compute_es_and_transactions(events_i, events_j, tau_max, i, j, sync_events):
    # 找出事件發生的時間索引
    t_i = np.where(events_i == 1)[0]
    t_j = np.where(events_j == 1)[0]
    n_i, n_j = len(t_i), len(t_j)

    if n_i == 0 or n_j == 0:
        return 0.0, 0

    # 間隔
    intervals_i = np.diff(t_i) if n_i > 1 else np.array([], dtype=np.int32)
    intervals_j = np.diff(t_j) if n_j > 1 else np.array([], dtype=np.int32)

    # 計算每個事件前後間隔
    t_i_prev = np.r_[np.inf, intervals_i]
    t_i_next = np.r_[intervals_i, np.inf]
    t_j_prev = np.r_[np.inf, intervals_j]
    t_j_next = np.r_[intervals_j, np.inf]

    # tau_ij 矩陣
    tau_i = np.minimum(t_i_prev, t_i_next)
    tau_j = np.minimum(t_j_prev, t_j_next)
    Tau = 0.5 * np.minimum(tau_i[:, None], tau_j[None, :])

    # 時間差矩陣
    T_diff = np.abs(t_i[:, None] - t_j[None, :])

    # 找符合條件的同步事件
    mask = (T_diff < Tau) & (T_diff <= tau_max)
    es_ij = np.count_nonzero(mask)

    # 更新 sync_events
    if es_ij > 0:
        ti_sync, tj_sync = np.where(mask)
        for a_idx, b_idx in zip(ti_sync, tj_sync):
            t1 = t_i[a_idx]
            t2 = t_j[b_idx]
            if t1 not in sync_events:
                sync_events[t1] = set()
            if t2 not in sync_events:
                sync_events[t2] = set()
            sync_events[t1].add(i)
            sync_events[t1].add(j)
            sync_events[t2].add(i)
            sync_events[t2].add(j)

    # 計算 Q
    dt_all = np.abs(t_i[:, None] - t_j[None, :])
    c_ij = np.sum(0.5 * np.minimum(dt_all, tau_max) * ((dt_all > 0) & (dt_all <= tau_max)))
    Q = c_ij / np.sqrt(n_i * n_j) if n_i * n_j > 0 else 0.0

    return min(Q, 1.0), es_ij

# 4. 主程式
#sync_matrix = np.zeros((n_grid, n_grid))
from scipy.sparse import lil_matrix

sync_matrix = lil_matrix((n_grid, n_grid), dtype=np.float32)
#es_matrix = np.zeros((n_grid, n_grid), dtype=int)
from scipy.sparse import lil_matrix

es_matrix = lil_matrix((n_grid, n_grid), dtype=np.int32)  # or float32 if needed
sync_events = {}

for i in range(n_grid):
    for j in range(i + 1, n_grid):
        Q, es_ij = compute_es_and_transactions(
            events_flat[:, i].values, events_flat[:, j].values, tau_max, i, j, sync_events
        )
        sync_matrix[i, j] = Q
        sync_matrix[j, i] = Q
        es_matrix[i, j] = es_ij
        es_matrix[j, i] = es_ij

# 5. 生成 transactions（包含日期和經緯度）
#transactions = []
transactions_with_coords = []
for time_idx in sorted(sync_events.keys()):
    if sync_events[time_idx]:
        #date = dates[time_idx]
        date = pd.to_datetime(events_flat['time'].values[time_idx])
        locations = list(sync_events[time_idx])
        lats = [grid_latitudes[loc] for loc in locations]
        lons = [grid_longitudes[loc] for loc in locations]
        #transactions.append(locations)
        transactions_with_coords.append((date, locations, lats, lons))

# 6. 儲存 transactions
#with open('./data/transactions.pkl', 'wb') as f:
#    pickle.dump(transactions, f)

# 儲存帶日期和經緯度的 transactions
transactions_df = pd.DataFrame(transactions_with_coords, columns=['date', 'locations', 'latitudes', 'longitudes'])
transactions_df.to_csv('transaction_with_coord/transactions_with_coords_community_gde.csv', index=False)