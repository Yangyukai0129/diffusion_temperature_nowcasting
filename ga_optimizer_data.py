import random
import re
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

from unet_ga import UNet # 假設這是您的 UNet 架構定義
from unet import train
from data_utils import prepare_file_list, compute_mean_std, LazyWeatherDataset

# =========================================================
# <<< 步驟 1: 定義可能的資料目錄列表 >>>
# =========================================================
POSSIBLE_DATA_DIRS = [
    "data/3day_1day", "data/4day_1day", "data/5day_1day", "data/6day_1day",
    "data/7day_1day", "data/8day_1day", "data/9day_1day", "data/10day_1day",
]
# Input index s 是 3 位元，對應 8 種值 (2^3 = 8)。
POSSIBLE_DATA_DIRS_MAPPED = POSSIBLE_DATA_DIRS

# =========================================================
# <<< 步驟 2: 輔助函數來從目錄名稱解析時間步長 >>>
# =========================================================
def _parse_steps_from_dir(data_dir):
    match = re.search(r'(\d+)day_(\d+)day', data_dir)
    if not match:
        raise ValueError(f"無法從目錄 '{data_dir}' 中解析天數。請確保格式為 'Xday_Yday'。")

    cond_days = int(match.group(1))
    target_days = int(match.group(2))

    DAY_TO_STEPS = 8
    cond_steps = cond_days * DAY_TO_STEPS
    target_steps = target_days * DAY_TO_STEPS

    return cond_steps, target_steps

# <<< 輔助函數，計算最大允許深度 (用於初始化固定 UNet 架構) >>>
def get_max_depth_for_data(data_shape=(9, 42)):
    min_dim = min(data_shape)
    max_depth = 0
    while min_dim >= 2:
        min_dim //= 2
        max_depth += 1
    return max_depth # 返回池化次數

# =========================================================
# <<< 步驟 3: 定義固定的 UNet 架構參數 >>>
# =========================================================
# 這裡定義一個固定的 UNet 架構配置，這將不再是基因的一部分。
# 您可以根據經驗或預先設定的值來調整這些參數。
FIXED_UNET_CONFIG = {
    "depth": 3, # 固定的深度
    "base_channels": 64, # 固定的基礎通道數
    "channel_mults": [1, 2, 4], # 固定的通道倍數，長度應與 depth 相符
    "time_dim": 32, # 通常固定
}
# 確保 channel_mults 的長度與 depth 相符
if len(FIXED_UNET_CONFIG["channel_mults"]) != FIXED_UNET_CONFIG["depth"]:
    print("警告: FIXED_UNET_CONFIG 中的 channel_mults 長度與 depth 不符，請檢查！")
    # 可以選擇在這裡動態生成或拋出錯誤
    # For now, let's just make it match the depth as a placeholder
    fixed_channel_mults = [1]
    current_mult = 1
    for _ in range(FIXED_UNET_CONFIG["depth"] - 1):
        current_mult *= 2 # 簡單地使用 2 倍增長
        fixed_channel_mults.append(min(current_mult, 8)) # 假設最大倍數為 8
    FIXED_UNET_CONFIG["channel_mults"] = fixed_channel_mults


# =========================================================
# <<< 步驟 4: 定義基因到表型的映射 (Mapping) - 只包含 data_dir 的位元 >>>
# =========================================================
# Input index s: 3 bits (b1..b3) -> bin2int(b1..b3) ∈ {0,...,7}
# 映射到 POSSIBLE_DATA_DIRS_MAPPED 的索引
S_BITS = 3 # 已修改為 3 位元，以覆蓋 8 個資料目錄

# 染色體總長度：3 (S) = 3 bits
CHROMOSOME_LENGTH = S_BITS

def bin_to_int(binary_str):
    return int(binary_str, 2)

def int_to_bin(integer, num_bits):
    return format(integer, f'0{num_bits}b')

# =========================================================
# <<< 步驟 5: 解碼函數 (decode_chromosome) - 只從染色體解碼 data_dir >>>
# =========================================================
def decode_chromosome(chromosome_str):
    """
    將染色體字串解碼為配置字典。
    Chromosome layout: [b1..b3] (S_BITS=3)
    """
    if len(chromosome_str) != CHROMOSOME_LENGTH:
        raise ValueError(f"染色體長度不正確，預期 {CHROMOSOME_LENGTH}，實際 {len(chromosome_str)}")

    # 1. 解碼 Input index s
    s_bin = chromosome_str[0:S_BITS]
    s_idx = bin_to_int(s_bin) # s = 0..7

    # 映射到資料目錄
    s_idx = min(s_idx, len(POSSIBLE_DATA_DIRS_MAPPED) - 1)
    data_dir = POSSIBLE_DATA_DIRS_MAPPED[s_idx]
    cond_steps, target_steps = _parse_steps_from_dir(data_dir)

    # UNet 架構參數來自固定的配置
    depth = FIXED_UNET_CONFIG["depth"]
    base_channels = FIXED_UNET_CONFIG["base_channels"]
    channel_mults = FIXED_UNET_CONFIG["channel_mults"]
    time_dim = FIXED_UNET_CONFIG["time_dim"]

    config = {
        "data_dir": data_dir,
        "cond_steps": cond_steps,
        "target_steps": target_steps,
        "in_channels": target_steps, # UNet 輸入是目標序列
        "out_channels": target_steps, # UNet 輸出是目標序列
        "cond_channels": cond_steps, # 條件輸入
        "time_dim": time_dim,

        "depth": depth,
        "base_channels": base_channels,
        "channel_mults": channel_mults,
    }
    return config

# =========================================================
# <<< 步驟 6: 正規化函數 (canonicalize_chromosome) - 針對 data_dir 的正規化 >>>
# =========================================================
def canonicalize_chromosome(chromosome_str):
    """
    對於僅包含 data_dir 選擇的染色體，目前的設計不需要特別的正規化，
    因為 s_idx 會在 decode_chromosome 中被 min() 限制在有效範圍內。
    但如果需要，可以在這裡加入檢查或修正邏輯。
    """
    if len(chromosome_str) != CHROMOSOME_LENGTH:
        raise ValueError(f"染色體長度不正確，預期 {CHROMOSOME_LENGTH}，實際 {len(chromosome_str)}")

    # 這裡可以加入檢查，例如確保 s_bin 不會導致超出 POSSIBLE_DATA_DIRS_MAPPED 的索引
    # 但 decode_chromosome 已經有 min() 處理，所以在這裡可能不是必須的
    return chromosome_str


# =========================================================
# <<< 步驟 7: 建立隨機個體 (create_random_individual) >>>
# =========================================================
def create_random_individual():
    """隨機生成一個染色體字串 (只針對 data_dir)，並進行正規化。"""
    chromosome_str = ''.join(random.choice('01') for _ in range(CHROMOSOME_LENGTH))
    return canonicalize_chromosome(chromosome_str)

# =========================================================
# 步驟 8: 適應度函數 (evaluate_fitness) - 使用解碼後的 data_dir 和固定的 UNet 架構
# =========================================================
phenotype_cache = {} # 用於儲存已計算的表型適應度

def evaluate_fitness(chromosome_str, device):
    """
    評估單一個體的適應度。現在它接受染色體字串，解碼後再評估。
    使用表型緩存來避免重複計算。
    """
    canonical_chromosome = canonicalize_chromosome(chromosome_str) # 確保輸入是正規化的

    if canonical_chromosome in phenotype_cache:
        print(f"--- 從緩存中讀取適應度 (染色體: {canonical_chromosome}): {phenotype_cache[canonical_chromosome]:.6f} ---")
        return phenotype_cache[canonical_chromosome]

    # 解碼染色體以獲取配置 (此時 config 中已經包含了固定的 UNet 架構)
    individual_config = decode_chromosome(canonical_chromosome)
    
    data_dir = individual_config["data_dir"]
    
    print(f"\n--- 評估個體 (資料: {data_dir}) ---")
    print(f"    UNet 架構: depth={individual_config['depth']}, base_channels={individual_config['base_channels']}, channel_mults={individual_config['channel_mults']}")
    
    try:
        # 1. 根據個體選擇的 data_dir 載入資料
        train_files, _ = prepare_file_list(data_dir)
        if not train_files:
            print(f"!!! 警告: 在 '{data_dir}' 中找不到訓練檔案。")
            fitness = -float('inf')
            phenotype_cache[canonical_chromosome] = fitness
            return fitness
            
        computed_stats = compute_mean_std(train_files)
        cond_mean, cond_std, target_mean, target_std = computed_stats
        
        train_dataset = LazyWeatherDataset(train_files, cond_mean, cond_std, target_mean, target_std)

        # 2. 建立 UNet 模型與訓練 (代理)
        model = UNet(individual_config).to(device) 
        
        subset_indices = random.sample(range(len(train_dataset)), k=min(len(train_dataset), 500))
        subset = Subset(train_dataset, subset_indices)
        proxy_loader = DataLoader(subset, batch_size=16, shuffle=True, num_workers=2)

        optimizer = optim.AdamW(model.parameters(), lr=1e-4)
        criterion = nn.MSELoss() 
        
        train_loss_history, _, _, _, _, _ = train( 
            model, proxy_loader, num_epochs=10, device=device,
            optimizer=optimizer, criterion=criterion, 
            train_loss_history=[], use_checkpoint=False 
        )
        
        final_mse = train_loss_history[-1] 
        fitness = -final_mse**0.5  
        
        del model, optimizer, proxy_loader, train_dataset, subset
        torch.cuda.empty_cache()

        print(f"--- 適應度 (最終負 RMSE): {fitness:.6f} ---")
        phenotype_cache[canonical_chromosome] = fitness
        return fitness

    except Exception as e:
        print(f"!!! 評估失敗 (資料: {data_dir}, 架構: depth={individual_config.get('depth', 'N/A')}, base_ch={individual_config.get('base_channels', 'N/A')}): {e}")
        import traceback
        traceback.print_exc() 
        fitness = -float('inf')
        phenotype_cache[canonical_chromosome] = fitness
        return fitness

# =========================================================
# 步驟 9: 交叉和突變操作 (bit-level) - 只針對 data_dir 位元
# =========================================================
def crossover(parent1_chromosome, parent2_chromosome):
    """
    位元級 1-2 點交叉。
    """
    if len(parent1_chromosome) != CHROMOSOME_LENGTH or len(parent2_chromosome) != CHROMOSOME_LENGTH:
        raise ValueError("父染色體長度不匹配。")

    child1_chromosome_list = list(parent1_chromosome)
    child2_chromosome_list = list(parent2_chromosome)

    # 選擇 1 或 2 個交叉點
    num_crossover_points = random.choice([1, 2])
    
    if num_crossover_points == 1:
        point = random.randint(1, CHROMOSOME_LENGTH - 1)
        child1_chromosome_list[:point], child2_chromosome_list[:point] = \
            child2_chromosome_list[:point], child1_chromosome_list[:point]
    else: # 2 points
        point1 = random.randint(1, CHROMOSOME_LENGTH - 2)
        point2 = random.randint(point1 + 1, CHROMOSOME_LENGTH - 1)
        
        # 交換中間部分
        child1_chromosome_list[point1:point2], child2_chromosome_list[point1:point2] = \
            child2_chromosome_list[point1:point2], child1_chromosome_list[point1:point2]
    
    return "".join(child1_chromosome_list), "".join(child2_chromosome_list)

def mutate(chromosome_str, mutation_rate):
    """
    位元翻轉突變。
    """
    mutated_chromosome_list = list(chromosome_str)
    for i in range(CHROMOSOME_LENGTH):
        if random.random() < mutation_rate:
            mutated_chromosome_list[i] = '1' if mutated_chromosome_list[i] == '0' else '0'
    return "".join(mutated_chromosome_list)

def selection(population_with_fitness, num_parents):
    """錦標賽選擇 (與之前相同，但操作的是染色體字串)"""
    valid_population = [item for item in population_with_fitness if item[1] != -float('inf')]
    
    if not valid_population:
        print("!!! 警告: 所有個體適應度均為 -inf，將從原始種群隨機選擇父母。")
        # 這裡需要返回染色體字串，而不是整個 (config, fitness) 元組
        return random.sample([item[0] for item in population_with_fitness], k=min(num_parents, len(population_with_fitness)))

    parents = []
    k_tournament = min(3, len(valid_population)) 

    for _ in range(num_parents):
        if len(valid_population) < k_tournament: 
            tournament = valid_population
        else:
            tournament = random.sample(valid_population, k=k_tournament)
        
        winner = max(tournament, key=lambda x: x[1])
        parents.append(winner[0]) # 父母是染色體字串
    return parents

def genetic_algorithm_optimizer(
    population_size=10, 
    generations=5, 
    mutation_rate=0.05, # 位元翻轉突變率通常較低
    num_parents_to_select=4, 
    device="cuda"
):
    print(f"=== 開始遺傳演算法最佳化資料集選擇 (固定 UNet 架構, 共 {generations} 代) ===")
    
    # 步驟 1) 初始化 N 隨機染色體；正規化。
    population = [create_random_individual() for _ in range(population_size)]
    
    best_chromosome_so_far = None
    best_fitness_so_far = -float('inf')

    for gen in range(generations):
        print(f"\n--- 第 {gen+1}/{generations} 代 ---")
        population_with_fitness = []
        for chromosome in population:
            # 步驟 2.1) 解碼 (I, d, c) 並計算適應度 F。
            fitness = evaluate_fitness(chromosome, device)
            population_with_fitness.append((chromosome, fitness))
            
            if fitness > best_fitness_so_far:
                best_fitness_so_far = fitness
                best_chromosome_so_far = chromosome # 儲存染色體字串

        population_with_fitness.sort(key=lambda x: x[1], reverse=True)
        print("\n=== 本代最佳個體 ===")
        decoded_best_config = decode_chromosome(population_with_fitness[0][0])
        print(f"資料集: {decoded_best_config['data_dir']}, 適應度: {population_with_fitness[0][1]:.6f}")

        # 步驟 2.2) 錦標賽選擇 → 交配池。
        parents = selection(population_with_fitness, num_parents_to_select)

        next_population = []
        # 步驟 2.4) 通過精英策略 + 最佳後代形成下一代種群。
        num_elites = 2 # 保留最佳的精英
        for i in range(min(num_elites, len(population_with_fitness))):
            next_population.append(population_with_fitness[i][0])

        while len(next_population) < population_size:
            # 步驟 2.3) 位元級 1-2 點交叉 → 後代；位元翻轉突變。
            p1_chrom, p2_chrom = random.sample(parents, 2)
            child1_chrom, child2_chrom = crossover(p1_chrom, p2_chrom)
            
            mutated_child1_chrom = mutate(child1_chrom, mutation_rate)
            mutated_child2_chrom = mutate(child2_chrom, mutation_rate)

            # 步驟 2.4) 正規化；
            next_population.append(canonicalize_chromosome(mutated_child1_chrom))
            if len(next_population) < population_size: # 確保不超出種群大小
                next_population.append(canonicalize_chromosome(mutated_child2_chrom))
        
        population = next_population

    print("\n=== 遺傳演算法結束 ===")
    final_best_config = decode_chromosome(best_chromosome_so_far)
    print(f"最終選定的最佳配置 (固定 UNet 架構):")
    print(final_best_config)
    return final_best_config

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用設備: {device}")

    best_config = genetic_algorithm_optimizer(
        population_size=10, 
        generations=5, 
        mutation_rate=0.05, # 將突變率調整為位元級的常見值
        num_parents_to_select=4,
        device=device
    )

    print("\n最終選定的最佳配置 (資料集，固定 UNet 架構):")
    print(best_config)