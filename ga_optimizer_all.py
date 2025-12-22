import random
import re
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

from unet_ga import UNet, train  # ✅ 修复：统一从 unet_ga 导入
from data_utils import prepare_file_list, compute_mean_std, LazyWeatherDataset
import numpy as np
import matplotlib.pyplot as plt

# =========================================================
# <<< 步驟 1: 定義可能的資料目錄列表 >>>
# =========================================================
POSSIBLE_DATA_DIRS = [
    "data/3day_1day", "data/4day_1day", "data/5day_1day", "data/6day_1day",
    "data/7day_1day", "data/8day_1day", "data/9day_1day", "data/10day_1day",
]
POSSIBLE_DATA_DIRS_MAPPED = POSSIBLE_DATA_DIRS

# ✅ 修复：定义为全局常量
DATA_SHAPE = (9, 42)

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

# <<< 輔助函數，計算最大允許深度 >>>
def get_max_depth_for_data(data_shape=DATA_SHAPE):  # ✅ 使用全局常量
    min_dim = min(data_shape)
    max_depth = 0
    while min_dim >= 2:
        min_dim //= 2
        max_depth += 1
    return max_depth

# =========================================================
# <<< 步驟 3: 定義基因到表型的映射 (Mapping) >>>
# =========================================================
# Input index s: 3 bits (b1..b3) -> bin2int(b1..b3) ∈ {0,...,7}
S_BITS = 3

# Depth d: 3 bits (b4..b6) -> 1 + bin2int(b4..b6) ∈ {1,...,8}
D_BITS = 3

# φ: {00,01,10,11} → {16,32,64,128} (Layer-1 channels C1)
PHI_MAP = {
    '00': 16,
    '01': 32,
    '10': 64,
    '11': 128
}

# μ: {01,10,11} → {1,2,4} (multiplicative factors M)
# 注意：'00' 只用於佔位（超出深度的 g(k)），在有效深度內會被自動修正為 '01'
MU_MAP = {
    '00': 0,  # 佔位符號（會在 canonicalize 時被修正為 '01'）
    '01': 1,  # 保持當前倍數（×1）
    '10': 2,  # 增加2倍（×2）
    '11': 4   # 增加4倍（×4）
}

G_BITS = 2  # g(k) 的位元數

C_MAX = 1024  # 通道數上限

# 染色體總長度：3 (S) + 3 (D) + 2*8 (g(1)...g(8)) = 22 bits
CHROMOSOME_LENGTH = S_BITS + D_BITS + G_BITS * 8

def bin_to_int(binary_str):
    return int(binary_str, 2)

def int_to_bin(integer, num_bits):
    return format(integer, f'0{num_bits}b')

# =========================================================
# <<< 步驟 4: 解碼函數 (decode_chromosome) >>>
# =========================================================
def decode_chromosome(chromosome_str):
    """
    將染色體字串解碼為 UNet 配置字典。
    Chromosome layout: [b1..b3][b4..b6][g(1)]...[g(8)] (S_BITS=3, D_BITS=3)
    """
    if len(chromosome_str) != CHROMOSOME_LENGTH:
        raise ValueError(f"染色體長度不正確，預期 {CHROMOSOME_LENGTH}，實際 {len(chromosome_str)}")

    # ✅ 安全保護：確保染色體已正規化，避免活躍區出現 '00'
    # chromosome_str = canonicalize_chromosome(chromosome_str)

    # 1. 解碼 Input index s
    s_bin = chromosome_str[0:S_BITS]
    s_idx = bin_to_int(s_bin)
    s_idx = min(s_idx, len(POSSIBLE_DATA_DIRS_MAPPED) - 1)
    data_dir = POSSIBLE_DATA_DIRS_MAPPED[s_idx]
    cond_steps, target_steps = _parse_steps_from_dir(data_dir)

    # 2. 解碼 Depth d
    d_bin = chromosome_str[S_BITS : S_BITS + D_BITS]
    depth = 1 + bin_to_int(d_bin)

    # 修正 depth 以符合資料集的最大池化深度
    max_allowed_pooling_steps = get_max_depth_for_data(DATA_SHAPE)  # ✅ 使用全局常量
    depth = min(depth, max_allowed_pooling_steps + 1)
    depth = max(depth, 2)

    # 3. 解碼 Layer-1 channels C1 (g(1))
    g1_bin = chromosome_str[S_BITS + D_BITS : S_BITS + D_BITS + G_BITS]
    base_channels = PHI_MAP.get(g1_bin, 16)

    # 4. 解碼 multiplicative factors M (g(2)...g(depth+1))
    channel_mults = [1]  # 第一層的乘數是 1
    current_mult = 1

    for k in range(1, depth):  # k=1..depth-1，对应 g(2)..g(depth)
        start_idx = S_BITS + D_BITS + (k * G_BITS)
        end_idx = start_idx + G_BITS
        gk_bin = chromosome_str[start_idx : end_idx]

        mult_factor = MU_MAP.get(gk_bin, 1)
        current_mult *= mult_factor
        channel_mults.append(min(current_mult, C_MAX // base_channels if base_channels > 0 else 8))

    # ✅ 修复：移除随机扩展，用确定性逻辑
    while len(channel_mults) < depth:
        # 如果基因没有提供足够的信息，用最后一个值填充
        channel_mults.append(channel_mults[-1])

    # 如果 channel_mults 的長度超過 depth，則截斷
    channel_mults = channel_mults[:depth]

    config = {
        "data_dir": data_dir,
        "cond_steps": cond_steps,
        "target_steps": target_steps,
        "in_channels": target_steps,
        "out_channels": target_steps,
        "cond_channels": cond_steps,
        "time_dim": 32,
        "depth": depth,
        "base_channels": base_channels,
        "channel_mults": channel_mults,
    }
    return config

# =========================================================
# <<< 步驟 5: 正規化函數 (canonicalize_chromosome) >>>
# =========================================================
def canonicalize_chromosome(chromosome_str):
    original_chromosome_list = list(chromosome_str)

    # 1. 取得並修正深度 (Source 6, 17)
    d_bin = chromosome_str[S_BITS : S_BITS + D_BITS]
    depth_val = 1 + bin_to_int(d_bin)
    
    max_allowed = get_max_depth_for_data(DATA_SHAPE)
    corrected_depth = max(2, min(depth_val, max_allowed + 1))

    # 更新深度位元 (Source 17)
    if depth_val != corrected_depth:
        original_chromosome_list[S_BITS : S_BITS + D_BITS] = list(int_to_bin(corrected_depth - 1, D_BITS))
        depth_val = corrected_depth

    # 2. 活躍區 (g(2) 到 g(d))：禁止 00，若出現則轉為 01 (Source 8)
    for k in range(1, depth_val): # k=1 是 g(2)
        idx = S_BITS + D_BITS + (k * G_BITS)
        if original_chromosome_list[idx : idx + 2] == ['0', '0']:
            original_chromosome_list[idx : idx + 2] = ['0', '1']

    # 3. 填充區 (g(d+1) 到 g(8))：強制補 00 (Source 9, 19)
    for k in range(depth_val, 8):
        idx = S_BITS + D_BITS + (k * G_BITS)
        original_chromosome_list[idx : idx + 2] = ['0', '0']

    return "".join(original_chromosome_list)

# =========================================================
# <<< 步驟 6: 建立隨機個體 (create_random_individual) >>>
# =========================================================
def create_random_individual():
    """隨機生成一個染色體字串，並進行正規化。"""
    chromosome_str = ''.join(random.choice('01') for _ in range(CHROMOSOME_LENGTH))
    return canonicalize_chromosome(chromosome_str)

# =========================================================
# 步驟 7: 適應度函數 (evaluate_fitness)
# =========================================================
phenotype_cache = {}

def evaluate_fitness(chromosome_str, device):
    """
    評估單一個體的適應度。
    使用表型緩存來避免重複計算。
    """
    canonical_chromosome = canonicalize_chromosome(chromosome_str)

    if canonical_chromosome in phenotype_cache:
        print(f"--- 從緩存中讀取適應度 (染色體: {canonical_chromosome[:10]}...): {phenotype_cache[canonical_chromosome]:.6f} ---")
        return phenotype_cache[canonical_chromosome]

    # 解碼染色體以獲取配置
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
    """位元級 1-2 點交叉。"""
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
    else:  # 2 points
        point1 = random.randint(1, CHROMOSOME_LENGTH - 2)
        point2 = random.randint(point1 + 1, CHROMOSOME_LENGTH - 1)

        child1_chromosome_list[point1:point2], child2_chromosome_list[point1:point2] = \
            child2_chromosome_list[point1:point2], child1_chromosome_list[point1:point2]

    return "".join(child1_chromosome_list), "".join(child2_chromosome_list)

def mutate(chromosome_str, mutation_rate):
    """位元翻轉突變。"""
    mutated_chromosome_list = list(chromosome_str)
    for i in range(CHROMOSOME_LENGTH):
        if random.random() < mutation_rate:
            mutated_chromosome_list[i] = '1' if mutated_chromosome_list[i] == '0' else '0'
    return "".join(mutated_chromosome_list)

def selection(population_with_fitness, num_parents):
    """錦標賽選擇"""
    valid_population = [item for item in population_with_fitness if item[1] != -float('inf')]

    if not valid_population:
        print("!!! 警告: 所有個體適應度均為 -inf，將從原始種群隨機選擇父母。")
        return random.sample([item[0] for item in population_with_fitness], k=min(num_parents, len(population_with_fitness)))

    parents = []
    k_tournament = min(3, len(valid_population))

    for _ in range(num_parents):
        if len(valid_population) < k_tournament:
            tournament = valid_population
        else:
            tournament = random.sample(valid_population, k=k_tournament)

        winner = max(tournament, key=lambda x: x[1])
        parents.append(winner[0])
    return parents

# =========================================================
# <<< 繪製收斂圖的函數 >>>
# =========================================================
def plot_convergence(history):
    """根據紀錄的歷史數據繪製 GA 收斂圖。"""
    generations = range(len(history['overall_best_rmse']))

    plt.figure(figsize=(12, 8))

    plt.plot(generations, history['overall_best_rmse'], 'r-o', linewidth=2, markersize=8, label='Overall Best RMSE (Elitism)')
    plt.plot(generations, history['best_rmse_per_gen'], 'g--^', alpha=0.7, label='Generation\'s Best RMSE')
    plt.plot(generations, history['avg_rmse_per_gen'], 'b:s', alpha=0.6, label='Generation\'s Average RMSE')

    plt.title('Genetic Algorithm Convergence Curve', fontsize=16)
    plt.xlabel('Generation', fontsize=12)
    plt.ylabel('RMSE (Lower is Better)', fontsize=12)
    plt.xticks(generations)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend(fontsize=10)
    plt.show()

# =========================================================
# ✅ 修复：移除重复定义，只保留一个完整版本
# =========================================================
def genetic_algorithm_optimizer(
    population_size=10,
    generations=5,
    mutation_rate=0.05,
    num_parents_to_select=4,
    device="cuda"
):
    print(f"=== 開始遺傳演算法最佳化 UNet 架構與資料集選擇 (共 {generations} 代) ===")

    # ✅ 修复：注释改为 22 位元
    # 步驟 1) 初始化 N 隨機 22 位元染色體；正規化。
    population = [create_random_individual() for _ in range(population_size)]

    best_chromosome_so_far = None
    best_fitness_so_far = -float('inf')

    # 用於儲存繪圖數據的歷史紀錄
    history = {
        "overall_best_rmse": [],
        "best_rmse_per_gen": [],
        "avg_rmse_per_gen": []
    }

    for gen in range(generations):
        print(f"\n--- 第 {gen+1}/{generations} 代 ---")
        population_with_fitness = []
        for chromosome in population:
            fitness = evaluate_fitness(chromosome, device)
            population_with_fitness.append((chromosome, fitness))

            if fitness > best_fitness_so_far:
                best_fitness_so_far = fitness
                best_chromosome_so_far = chromosome

        # 紀錄數據用於繪圖
        valid_fitness_values = [f for _, f in population_with_fitness if f != -float('inf')]

        if not valid_fitness_values:
            current_best_rmse = float('inf')
            current_avg_rmse = float('inf')
        else:
            rmse_values = [-f for f in valid_fitness_values]
            current_best_rmse = min(rmse_values)
            current_avg_rmse = np.mean(rmse_values)

        overall_best_rmse_so_far = -best_fitness_so_far

        history["overall_best_rmse"].append(overall_best_rmse_so_far)
        history["best_rmse_per_gen"].append(current_best_rmse)
        history["avg_rmse_per_gen"].append(current_avg_rmse)

        print(f"\n本代統計: 最佳RMSE={current_best_rmse:.6f}, 平均RMSE={current_avg_rmse:.6f}, 歷代最佳RMSE={overall_best_rmse_so_far:.6f}")

        population_with_fitness.sort(key=lambda x: x[1], reverse=True)
        print("\n=== 本代最佳個體 ===")
        decoded_best_config = decode_chromosome(population_with_fitness[0][0])
        print(f"資料集: {decoded_best_config['data_dir']}, UNet 架構: depth={decoded_best_config['depth']}, base_ch={decoded_best_config['base_channels']}, 適應度: {population_with_fitness[0][1]:.6f}")

        parents = selection(population_with_fitness, num_parents_to_select)
        next_population = []
        num_elites = 2
        for i in range(min(num_elites, len(population_with_fitness))):
            next_population.append(population_with_fitness[i][0])

        while len(next_population) < population_size:
            p1_chrom, p2_chrom = random.sample(parents, 2)
            child1_chrom, child2_chrom = crossover(p1_chrom, p2_chrom)
            mutated_child1_chrom = mutate(child1_chrom, mutation_rate)
            mutated_child2_chrom = mutate(child2_chrom, mutation_rate)
            next_population.append(canonicalize_chromosome(mutated_child1_chrom))
            if len(next_population) < population_size:
                next_population.append(canonicalize_chromosome(mutated_child2_chrom))

        population = next_population

    # 在演算法結束後呼叫繪圖函數
    print("\n=== 繪製收斂曲線圖 ===")
    plot_convergence(history)

    print("\n=== 遺傳演算法結束 ===")
    final_best_config = decode_chromosome(best_chromosome_so_far)
    print(f"最終選定的最佳配置: {final_best_config}, 最佳適應度: {best_fitness_so_far:.6f}")

    return final_best_config

# =========================================================
# <<< 主程式入口 >>>
# =========================================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用設備: {device}")

    # 執行 GA 最佳化
    best_config = genetic_algorithm_optimizer(
        population_size=10,
        generations=20,
        mutation_rate=0.05,
        num_parents_to_select=4,
        device=device
    )

    print("\n最終選定的最佳配置 (UNet 架構與資料集):")
    print(best_config)