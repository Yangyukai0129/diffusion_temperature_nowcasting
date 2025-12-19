import xarray as xr
import numpy as np
import pandas as pd
from scipy.sparse import lil_matrix
from joblib import Parallel, delayed

# 1. 讀取數據
heatwave_events = xr.open_dataset(
    'heatwave_events_90threshold_5deg(1965-2024).nc'
)['t']

tau_max = 10
events_flat = heatwave_events.stack(grid=['latitude', 'longitude']).transpose('valid_time', 'grid')
n_time, n_grid = events_flat.shape

dates = heatwave_events['valid_time'].values
lat = events_flat['latitude'].values
lon = events_flat['longitude'].values

# 2. ES 函數（純 numpy 寫法）
def compute_es(events_i, events_j, tau_max):
    t_i = np.where(events_i == 1)[0]
    t_j = np.where(events_j == 1)[0]
    n_i, n_j = len(t_i), len(t_j)

    if n_i == 0 or n_j == 0:
        return 0.0, 0, np.empty((0, 2), dtype=np.int32)

    intervals_i = np.diff(t_i) if n_i > 1 else np.array([])
    intervals_j = np.diff(t_j) if n_j > 1 else np.array([])

    tau_i = np.minimum(
        np.concatenate(([np.inf], intervals_i)),
        np.concatenate((intervals_i, [np.inf]))
    )
    tau_j = np.minimum(
        np.concatenate(([np.inf], intervals_j)),
        np.concatenate((intervals_j, [np.inf]))
    )

    es_ij = 0
    pairs = []
    c_ij = 0.0

    for a in range(n_i):
        for b in range(n_j):
            t_ij = abs(t_i[a] - t_j[b])
            tau_ab = 0.5 * min(tau_i[a], tau_j[b])

            if t_ij < tau_ab and t_ij <= tau_max:
                es_ij += 1
                pairs.append((t_i[a], t_j[b]))

            if 0 < t_ij <= tau_max:
                c_ij += 0.5 * min(t_ij, tau_max)

    Q = c_ij / np.sqrt(n_i * n_j) if n_i * n_j > 0 else 0.0
    return min(Q, 1.0), es_ij, np.array(pairs, dtype=np.int32)

# 3. 包裝函數（for joblib）
def process_pair(i, j):
    Q, es_ij, pairs = compute_es(events_flat[:, i].values,
                                 events_flat[:, j].values,
                                 tau_max)
    return i, j, Q, es_ij, pairs

# 4. 並行運算
results = Parallel(n_jobs=4, backend="loky", verbose=5)(
    delayed(process_pair)(i, j) for i in range(n_grid) for j in range(i + 1, n_grid)
)

# 5. 填回矩陣 & 建 transactions
sync_matrix = lil_matrix((n_grid, n_grid), dtype=np.float32)
es_matrix = lil_matrix((n_grid, n_grid), dtype=np.int16)
sync_events = {}

for i, j, Q, es_ij, pairs in results:
    sync_matrix[i, j] = Q
    sync_matrix[j, i] = Q
    es_matrix[i, j] = es_ij
    es_matrix[j, i] = es_ij

    # 更新同步事件
    for t1, t2 in pairs:
        if t1 not in sync_events:
            sync_events[t1] = set()
        if t2 not in sync_events:
            sync_events[t2] = set()
        sync_events[t1].add(i)
        sync_events[t1].add(j)
        sync_events[t2].add(i)
        sync_events[t2].add(j)

# 6. 生成 transactions
transactions_with_coords = []
for time_idx in sorted(sync_events.keys()):
    if sync_events[time_idx]:
        date = pd.to_datetime(events_flat['valid_time'].values[time_idx])
        locations = list(sync_events[time_idx])
        lats = [lat[loc] for loc in locations]
        lons = [lon[loc] for loc in locations]
        transactions_with_coords.append((date, locations, lats, lons))

# 7. 儲存 CSV
transactions_df = pd.DataFrame(
    transactions_with_coords,
    columns=['date', 'locations', 'latitudes', 'longitudes']
)
transactions_df.to_csv('transactions_with_coords_90threshold.csv', index=False)

print("✅ ES + Transactions 計算完成並儲存")