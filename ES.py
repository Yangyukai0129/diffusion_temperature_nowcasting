import xarray as xr
import numpy as np
from scipy import stats
import pickle
import pandas as pd

# 1. 讀取數據
heatwave_events = xr.open_dataset('heatwave_events_5deg(1965-2012).nc')['temperature']
tau_max = 10
n_time = heatwave_events.shape[0]
# events_flat = heatwave_events.stack(grid=['lat', 'lon']).transpose('time', 'grid')
events_flat = heatwave_events.stack(grid=['latitude', 'longitude']).transpose('valid_time', 'grid')
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
    t_i, intervals_i = compute_event_intervals(events_i)
    t_j, intervals_j = compute_event_intervals(events_j)
    n_i, n_j = len(t_i), len(t_j)
    
    if n_i == 0 or n_j == 0:
        return 0.0, 0
    
    es_ij = 0
    for alpha in range(n_i):
        for beta in range(n_j):
            t_ij = abs(t_i[alpha] - t_j[beta])
            
            if alpha == 0:
                t_i_prev = float('inf')
                t_i_next = intervals_i[0] if len(intervals_i) > 0 else float('inf')
            elif alpha == n_i - 1:
                t_i_prev = intervals_i[-1] if len(intervals_i) > 0 else float('inf')
                t_i_next = float('inf')
            else:
                t_i_prev = intervals_i[alpha - 1]
                t_i_next = intervals_i[alpha]
            
            if beta == 0:
                t_j_prev = float('inf')
                t_j_next = intervals_j[0] if len(intervals_j) > 0 else float('inf')
            elif beta == n_j - 1:
                t_j_prev = intervals_j[-1] if len(intervals_j) > 0 else float('inf')
                t_j_next = float('inf')
            else:
                t_j_prev = intervals_j[beta - 1]
                t_j_next = intervals_j[beta]
            
            tau_ij = 0.5 * min(t_i_prev, t_i_next, t_j_prev, t_j_next)
            
            if t_ij < tau_ij and t_ij <= tau_max:
                es_ij += 1
                t1 = t_i[alpha]
                t2 = t_j[beta]
                if t1 not in sync_events:
                    sync_events[t1] = set()
                if t2 not in sync_events:
                    sync_events[t2] = set()
                sync_events[t1].add(i)
                sync_events[t1].add(j)
                sync_events[t2].add(i)
                sync_events[t2].add(j)
    
    c_ij = 0.0
    for ti in t_i:
        for tj in t_j:
            dt = abs(ti - tj)
            if 0 < dt <= tau_max:
                c_ij += 0.5 * min(dt, tau_max)
    Q = c_ij / np.sqrt(n_i * n_j) if n_i * n_j > 0 else 0.0
    
    return min(Q, 1.0), es_ij

# 4. 主程式
#sync_matrix = np.zeros((n_grid, n_grid))
from scipy.sparse import lil_matrix

sync_matrix = lil_matrix((n_grid, n_grid), dtype=np.float32)
#es_matrix = np.zeros((n_grid, n_grid), dtype=int)
from scipy.sparse import lil_matrix

es_matrix = lil_matrix((n_grid, n_grid), dtype=np.int8)  # or float32 if needed
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
transactions_df.to_csv('transactions_with_coords(1965-2012).csv', index=False)