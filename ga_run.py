import torch
from ga_optimizer import create_random_individual, evaluate_fitness, selection, crossover, mutate
import random

def run_ga_search():
    # --- GA 參數設定 ---
    POPULATION_SIZE = 10
    NUM_GENERATIONS = 5
    NUM_PARENTS = 4
    MUTATION_RATE = 0.2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用裝置：{device}")

    # --- GA 流程 ---
    print("初始化族群...")
    population = [create_random_individual() for _ in range(POPULATION_SIZE)]

    best_overall_individual = (None, -float('inf'))

    for generation in range(NUM_GENERATIONS):
        print(f"\n{'='*20} GENERATION {generation+1}/{NUM_GENERATIONS} {'='*20}")

        population_with_fitness = []
        for ind in population:
            fitness = evaluate_fitness(ind, device)
            population_with_fitness.append((ind, fitness))

        population_with_fitness.sort(key=lambda x: x[1], reverse=True)
        
        current_best = population_with_fitness[0]
        if current_best[1] > best_overall_individual[1]:
            best_overall_individual = current_best

        print(f"\nGeneration {generation+1} Best:")
        print(f"  - Fitness: {current_best[1]:.6f}")
        print(f"  - Config: {current_best[0]}")
        print(f"Overall Best Fitness so far: {best_overall_individual[1]:.6f}")

        if generation == NUM_GENERATIONS - 1:
            break

        parents = selection(population_with_fitness, NUM_PARENTS)
        offspring = [best_overall_individual[0]] # 精英策略: 保留歷史最佳個體
        
        while len(offspring) < POPULATION_SIZE:
            p1, p2 = random.sample(parents, 2)
            child = crossover(p1, p2)
            child = mutate(child, MUTATION_RATE)
            offspring.append(child)
        
        population = offspring

    print("\n\nGA 搜尋結束！")
    print("找到的最佳設定為：")
    print(f"  - Fitness: {best_overall_individual[1]:.6f}")
    print(f"  - Config: {best_overall_individual[0]}")

if __name__ == "__main__":
    run_ga_search()