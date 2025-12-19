import random
import re # 匯入正規表達式模組，用於解析檔名
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

# 假設這些檔案都在同一個專案目錄下
from unet_ga import UNet
from unet import train 
from data_utils import prepare_file_list, compute_mean_std, LazyWeatherDataset

# =========================================================
# <<< 步驟 1: 設定可用的資料目錄 >>>
# =========================================================
# 請在此處填入您所有預先生成好的資料夾路徑
AVAILABLE_DATA_DIRS = [
    "data/8day_1day",
    "data/7day_1day",
    "data/4day_1day",
]

# =========================================================
# <<< 步驟 2: 新增一個輔助函數來從目錄名稱解析時間步長 >>>
# =========================================================
def _parse_steps_from_dir(data_dir):
    """
    從 "data/Xday_Yday" 這樣的路徑中解析出 cond_steps 和 target_steps。
    *** 假設：1 day = 8 steps *** (您可以根據您的資料定義修改這個值)
    """
    match = re.search(r'(\d+)day_(\d+)day', data_dir)
    if not match:
        raise ValueError(f"無法從目錄 '{data_dir}' 中解析天數。請確保格式為 'Xday_Yday'。")
    
    cond_days = int(match.group(1))
    target_days = int(match.group(2))
    
    # 轉換為 steps
    DAY_TO_STEPS = 8 
    cond_steps = cond_days * DAY_TO_STEPS
    target_steps = target_days * DAY_TO_STEPS
    
    return cond_steps, target_steps

# <<< 新增: 輔助函數，計算最大允許深度 >>>
def get_max_depth_for_data(data_shape=(9, 42)):
    """
    根據輸入資料的最小邊長，計算可以進行多少次 2x 池化。
    """
    min_dim = min(data_shape)
    max_depth = 0
    while min_dim >= 2: # 只要最小邊長還能被2整除（或大於等於2），就可以再加深一層
        min_dim //= 2
        max_depth += 1
    # U-Net的depth定義是block的數量，池化次數是 depth-1。
    # 這裡我們計算的是池化次數，所以最大 depth 是 max_depth + 1
    # 為了保險起見，我們返回 max_depth，限制池化次數。
    # 例如，9 -> 4 -> 2 -> 1，可以池化3次，depth=4。
    # 我們的計算 max_depth 會是3。
    # 所以最大 depth 是 max_depth + 1
    return max_depth + 1

# =========================================================
# 步驟 3: 修改基因定義 (create_random_individual)
# =========================================================
def create_random_individual():
    """隨機生成一個包含資料目錄選擇和 U-Net 架構的個體"""
    # 1. 選擇資料目錄 (資料基因)
    data_dir = random.choice(AVAILABLE_DATA_DIRS)
    cond_steps, target_steps = _parse_steps_from_dir(data_dir)

        # <<< 關鍵修改點 >>>
    # 假設所有資料的空間維度都是 9x42，如果不是，需要動態獲取
    data_shape = (9, 42) 
    max_allowed_depth = get_max_depth_for_data(data_shape)
    
    # 從 [2, 3, ..., max_allowed_depth] 中隨機選擇一個 depth
    # 例如，對於 9x42，max_allowed_depth 是 4，所以 depth 會在 [2, 3, 4] 中選。
    # 確保我們的選擇不會超過允許的最大深度
    possible_depths = [d for d in [2, 3, 4, 5] if d <= max_allowed_depth]
    depth = random.choice(possible_depths)
    base_channels = random.choice([32, 64, 128])
    channel_mults = [1]
    for _ in range(depth - 1):
        mult = channel_mults[-1] * random.choice([1, 2])
        channel_mults.append(min(mult, 8)) 

    # 3. 組合成完整的 config (染色體)
    config = {
        # 資料基因
        "data_dir": data_dir,
        
        # 架構基因
        "depth": depth,
        "base_channels": base_channels,
        "channel_mults": channel_mults,

        # 由資料基因衍生的參數 (確保一致性)
        "cond_steps": cond_steps,
        "target_steps": target_steps,
        "in_channels": target_steps,
        "out_channels": target_steps,
        "cond_channels": cond_steps,
        "time_dim": 32, # time_dim 通常是固定的
    }
    return config

# =========================================================
# 步驟 4: 修改適應度函數 (evaluate_fitness)
# =========================================================
def evaluate_fitness(individual_config, device):
    """
    評估單一個體的適應度。現在它直接使用 config 中的 data_dir。
    """
    data_dir = individual_config["data_dir"]
    print(f"\n--- 評估個體 (資料: {data_dir}) ---")
    print(f"    架構: depth={individual_config['depth']}, base_ch={individual_config['base_channels']}")
    
    try:
        # 1. 根據 data_dir 載入資料
        train_files, _ = prepare_file_list(data_dir)
        if not train_files:
            print(f"!!! 警告: 在 '{data_dir}' 中找不到訓練檔案。")
            return -float('inf')
            
        cond_mean, cond_std, target_mean, target_std = compute_mean_std(train_files)
        train_dataset = LazyWeatherDataset(train_files, cond_mean, cond_std, target_mean, target_std)

        # 2. 建立模型與訓練 (代理)
        model = UNet(individual_config).to(device)
        
        subset_indices = random.sample(range(len(train_dataset)), k=min(len(train_dataset), 500))
        subset = Subset(train_dataset, subset_indices)
        proxy_loader = DataLoader(subset, batch_size=16, shuffle=True, num_workers=2)

        optimizer = optim.AdamW(model.parameters(), lr=1e-4)
        criterion = nn.L1Loss()
        
        train_loss_history, _, _, _ = train(
            model, proxy_loader, num_epochs=1, device=device,
            optimizer=optimizer, criterion=criterion, 
            train_loss_history=[], use_checkpoint=False
        )
        
        fitness = -train_loss_history[-1]
        
        del model, optimizer, proxy_loader, train_dataset, subset
        torch.cuda.empty_cache()

        print(f"--- 適應度: {fitness:.6f} ---")
        return fitness

    except Exception as e:
        print(f"!!! 評估失敗 {individual_config['data_dir']}: {e}")
        return -float('inf')

# =========================================================
# 步驟 5: 修改 Crossover 和 Mutate 以確保一致性
# =========================================================
def _fix_config_consistency(config):
    """在基因改變後更新依賴的參數，並檢查 depth 的有效性"""
    # 1. 更新時間步長相關參數
    cond_steps, target_steps = _parse_steps_from_dir(config["data_dir"])
    config["cond_steps"] = cond_steps
    config["target_steps"] = target_steps
    config["cond_channels"] = cond_steps
    config["in_channels"] = target_steps
    config["out_channels"] = target_steps

    # <<< 關鍵修改點 >>>
    # 2. 檢查並修正 depth
    data_shape = (9, 42) # 同樣假設固定尺寸
    max_allowed_depth = get_max_depth_for_data(data_shape)
    if config["depth"] > max_allowed_depth:
        config["depth"] = max_allowed_depth # 如果交叉/突變產生了過深的網路，將其修正為最大允許深度

    # 3. 修正 channel_mults 長度
    if len(config["channel_mults"]) != config["depth"]:
        # 如果深度被修正了，或者交叉產生了不匹配，重新生成 channel_mults
        new_mults = [1]
        for _ in range(config["depth"] - 1):
            mult = new_mults[-1] * random.choice([1, 2])
            new_mults.append(min(mult, 8))
        config["channel_mults"] = new_mults
        
    return config

def crossover(parent1, parent2):
    child = parent1.copy()
    ga_keys = ["data_dir", "depth", "base_channels", "channel_mults"]
    
    for key in ga_keys:
        if random.random() < 0.5:
            child[key] = parent2[key]

    # 交叉後，必須修復可能產生的不一致性
    child = _fix_config_consistency(child)
    return child

def mutate(individual, mutation_rate=0.2):
    mutated_individual = individual.copy()
    
    if random.random() < mutation_rate:
        mutated_individual["data_dir"] = random.choice(AVAILABLE_DATA_DIRS)
    if random.random() < mutation_rate:
        mutated_individual["depth"] = random.choice([2, 3, 4])
    if random.random() < mutation_rate:
        mutated_individual["base_channels"] = random.choice([32, 48, 64])

    # 突變後，同樣要修復不一致性
    mutated_individual = _fix_config_consistency(mutated_individual)
    return mutated_individual

def selection(population_with_fitness, num_parents):
    """錦標賽選擇"""
    parents = []
    for _ in range(num_parents):
        tournament = random.sample(population_with_fitness, k=3)
        winner = max(tournament, key=lambda x: x[1])
        parents.append(winner[0])
    return parents