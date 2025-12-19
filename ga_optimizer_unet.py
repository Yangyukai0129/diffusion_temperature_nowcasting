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
# <<< 步驟 1: 定義固定的資料目錄 >>>
# =========================================================
# 現在 data_dir 將是固定值，不再是基因的一部分
FIXED_DATA_DIR = "data/8day_1day" # 您可以在這裡選擇您想要使用的固定資料集

# =========================================================
# <<< 步驟 2: 輔助函數來從目錄名稱解析時間步長 (與之前相同) >>>
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

# <<< 輔助函數，計算最大允許深度 (來自您的第二個程式碼) >>>
def get_max_depth_for_data(data_shape=(9, 42)):
    """
    根據輸入資料的最小邊長，計算可以進行多少次 2x 池化。
    """
    min_dim = min(data_shape)
    max_depth = 0
    while min_dim >= 2:
        min_dim //= 2
        max_depth += 1
    return max_depth # 返回池化次數

# =========================================================
# <<< 步驟 3: 定義基因到表型的映射 (Mapping) - 移除 data_dir 的位元 >>>
# =========================================================
# Input index s: 不再是基因的一部分

# Depth d: 3 bits (b1..b3) -> 1 + bin2int(b1..b3) ∈ {1,...,8} (之前是 b4..b6)
D_BITS = 3

# φ: {00,01,10,11} → {16,32,64,128} (Layer-1 channels C1)
PHI_MAP = {
    '00': 16,
    '01': 32,
    '10': 64,
    '11': 128
}
PHI_INV_MAP = {v: k for k, v in PHI_MAP.items()} # 用於反向映射，突變時可能需要

# μ: {00,01,10,11} → {1,2,4,8} (multiplicative factors M)
MU_MAP = {
    '00': 1,
    '01': 2,
    '10': 4,
    '11': 8
}
MU_INV_MAP = {v: k for k, v in MU_MAP.items()} # 用於反向映射，突變時可能需要

G_BITS = 2 # g(k) 的位元數

C_MAX = 1024 # 通道數上限

# 染色體總長度：3 (D) + 2*8 (g(1)...g(8)) = 3 + 16 = 19 bits
CHROMOSOME_LENGTH = D_BITS + G_BITS * 8 # 已修改為 19 bits

def bin_to_int(binary_str):
    return int(binary_str, 2)

def int_to_bin(integer, num_bits):
    return format(integer, f'0{num_bits}b')

# =========================================================
# <<< 步驟 4: 解碼函數 (decode_chromosome) - 不再從染色體解碼 data_dir >>>
# =========================================================
def decode_chromosome(chromosome_str):
    """
    將染色體字串解碼為 UNet 配置字典。
    Chromosome layout: [b1..b3][g(1)]...[g(8)] (D_BITS=3)
    """
    if len(chromosome_str) != CHROMOSOME_LENGTH:
        raise ValueError(f"染色體長度不正確，預期 {CHROMOSOME_LENGTH}，實際 {len(chromosome_str)}")

    # data_dir, cond_steps, target_steps 來自固定的值
    data_dir = FIXED_DATA_DIR
    cond_steps = COND_STEPS
    target_steps = TARGET_STEPS

    # 1. 解碼 Depth d
    d_bin = chromosome_str[0 : D_BITS] # 修正起始索引
    depth = 1 + bin_to_int(d_bin) # d = 1..8

    # 修正 depth 以符合資料集的最大池化深度
    data_shape = (9, 42)
    max_allowed_pooling_steps = get_max_depth_for_data(data_shape)
    depth = min(depth, max_allowed_pooling_steps + 1)
    depth = max(depth, 2) # 最小深度為 2

    # 2. 解碼 Layer-1 channels C1 (g(1))
    g1_bin = chromosome_str[D_BITS : D_BITS + G_BITS] # 修正起始索引
    base_channels = PHI_MAP.get(g1_bin, 16) # 預設值以防出錯

    # 3. 解碼 multiplicative factors M (g(2)...g(8))
    channel_mults = [1] # 第一層的乘數通常是 1
    current_mult = 1
    for k in range(1, depth): # 從 g(2) 到 g(depth)，對應乘數
        start_idx = D_BITS + (k * G_BITS) # 修正起始索引
        end_idx = start_idx + G_BITS
        gk_bin = chromosome_str[start_idx : end_idx]

        mult_factor = MU_MAP.get(gk_bin, 1)
        current_mult *= mult_factor
        channel_mults.append(min(current_mult, C_MAX // base_channels if base_channels > 0 else 8))

    # 如果 channel_mults 的長度不足 depth，則補齊
    while len(channel_mults) < depth:
        channel_mults.append(channel_mults[-1] * random.choice([1, 2]))
        channel_mults[-1] = min(channel_mults[-1], C_MAX // base_channels if base_channels > 0 else 8)

    # 如果 channel_mults 的長度超過 depth，則截斷
    channel_mults = channel_mults[:depth]

    config = {
        "data_dir": data_dir,
        "cond_steps": cond_steps,
        "target_steps": target_steps,
        "in_channels": target_steps, # UNet 輸入是目標序列
        "out_channels": target_steps, # UNet 輸出是目標序列
        "cond_channels": cond_steps, # 條件輸入
        "time_dim": 32, # 通常固定

        "depth": depth,
        "base_channels": base_channels,
        "channel_mults": channel_mults,
    }
    return config

# =========================================================
# <<< 步驟 5: 正規化函數 (canonicalize_chromosome) - 移除 data_dir 相關的正規化 >>>
# =========================================================
def canonicalize_chromosome(chromosome_str):
    """
    正規化染色體：如果 d < 8，設定 g(k) = 00 for all k > d。
    並修正 depth 以確保其在允許範圍內。
    """
    original_chromosome_list = list(chromosome_str)

    # 提取深度位元
    d_bin = chromosome_str[0 : D_BITS] # 修正起始索引
    depth_val = 1 + bin_to_int(d_bin) # d = 1..8

    # 修正 depth 以符合資料集的最大池化深度
    data_shape = (9, 42)
    max_allowed_pooling_steps = get_max_depth_for_data(data_shape)
    corrected_depth = min(depth_val, max_allowed_pooling_steps + 1)
    corrected_depth = max(corrected_depth, 2)

    # 如果原始深度與修正後的深度不同，更新染色體中的深度位元
    if depth_val != corrected_depth:
        corrected_d_bin = int_to_bin(corrected_depth - 1, D_BITS)
        for i in range(D_BITS):
            original_chromosome_list[i] = corrected_d_bin[i] # 修正索引
        depth_val = corrected_depth # 更新為修正後的深度

    # 對於超出深度的 g(k) 設置為 '00'
    for k_idx in range(depth_val, 8): # k_idx 從 depth_val (實際修正後的深度) 到 7
        start_idx = D_BITS + (k_idx * G_BITS) # 修正起始索引
        end_idx = start_idx + G_BITS
        # 確保索引範圍有效
        if end_idx <= CHROMOSOME_LENGTH:
            for i in range(G_BITS):
                original_chromosome_list[start_idx + i] = '0'

    return "".join(original_chromosome_list)

# =========================================================
# <<< 步驟 6: 建立隨機個體 (create_random_individual) >>>
# =========================================================
def create_random_individual():
    """隨機生成一個染色體字串，並進行正規化。"""
    chromosome_str = ''.join(random.choice('01') for _ in range(CHROMOSOME_LENGTH))
    return canonicalize_chromosome(chromosome_str)

# =========================================================
# 步驟 7: 適應度函數 (evaluate_fitness) - 使用解碼後的 UNet 架構和固定的 data_dir
# =========================================================
phenotype_cache = {} # 用於儲存已計算的表型適應度

def evaluate_fitness(chromosome_str, device):
    """
    評估單一個體的適應度。現在它接受染色體字串，解碼後再評估。
    使用表型緩存來避免重複計算。
    """
    canonical_chromosome = canonicalize_chromosome(chromosome_str) # 確保輸入是正規化的

    if canonical_chromosome in phenotype_cache:
        print(f"--- 從緩存中讀取適應度 (染色體: {canonical_chromosome[:10]}...): {phenotype_cache[canonical_chromosome]:.6f} ---")
        return phenotype_cache[canonical_chromosome]

    # 解碼染色體以獲取配置
    individual_config = decode_chromosome(canonical_chromosome)

    data_dir = individual_config["data_dir"] # 現在 data_dir 是固定的
    
    print(f"\n--- 評估個體 (資料: {data_dir}) ---")
    print(f"    UNet 架構: depth={individual_config['depth']}, base_channels={individual_config['base_channels']}, channel_mults={individual_config['channel_mults']}")
    
    try:
        # 1. 根據固定的 data_dir 載入資料
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
        fitness = -final_mse**0.5  # 使用負 RMSE 作為適應度
        
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
# 步驟 8: 交叉和突變操作 (bit-level)
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
    print(f"=== 開始遺傳演算法最佳化 UNet 架構 (固定資料集: {FIXED_DATA_DIR}, 共 {generations} 代) ===")
    
    # 步驟 1) 初始化 N 隨機 染色體；正規化。
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
        print(f"UNet 架構: depth={decoded_best_config['depth']}, base_ch={decoded_best_config['base_channels']}, 適應度: {population_with_fitness[0][1]:.6f}")

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
    print(f"最終選定的最佳配置 (固定資料集: {FIXED_DATA_DIR}):")
    print(final_best_config)
    return final_best_config

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用設備: {device}")

    # 在這裡確保 _parse_steps_from_dir 已經被呼叫以初始化 FIXED_DATA_DIR 的步長
    COND_STEPS, TARGET_STEPS = _parse_steps_from_dir(FIXED_DATA_DIR)

    best_config = genetic_algorithm_optimizer(
        population_size=10, 
        generations=5, 
        mutation_rate=0.05, # 將突變率調整為位元級的常見值
        num_parents_to_select=4,
        device=device
    )

    print("\n最終選定的最佳配置 (UNet 架構，固定資料集):")
    print(best_config)