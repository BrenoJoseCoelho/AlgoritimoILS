import os
import random
import time
import numpy as np
from scipy.sparse import coo_matrix
import sys
import argparse

def load_sparse_file(file_path):
    rows, cols, data = [], [], []
    matrix_size = 0
    with open(file_path, 'r') as file:
        for i, line in enumerate(file):
            values = line.split()
            if i == 0:
                matrix_size = int(values[0])
            elif len(values) == 3:
                row, col, value = int(values[0])-1, int(values[1])-1, int(values[2])
                rows.append(row)
                cols.append(col)
                data.append(value)
    
    return coo_matrix((data, (rows, cols)), shape=(matrix_size, matrix_size)) 

def evaluate_solution(solution, database):
    indices = list(solution)
    submatrix = database[indices, :][:, indices]
    return submatrix.sum()

def local_search(solution, database, num_components):
    best_solution = set(solution)
    best_value = evaluate_solution(best_solution, database)
    improved = True

    while improved:
        improved = False
        for candidate in range(num_components):
            if candidate not in best_solution:
                new_solution = best_solution | {candidate}
            else:
                new_solution = best_solution - {candidate}

            new_value = evaluate_solution(new_solution, database)

            if new_value > best_value:
                best_solution = new_solution
                best_value = new_value
                improved = True

        print(f"Busca local: Melhor solução até agora: {best_value}")
    return list(best_solution)

def perturb(solution, num_components, perturbation_size=60):
    new_solution = set(solution)
    for _ in range(perturbation_size):
        random_index = random.randint(0, num_components - 1)
        if random_index in new_solution:
            new_solution.remove(random_index)
        else:
            new_solution.add(random_index)
    return list(new_solution)

def ils(database, max_iterations=1000):
    num_rows, _ = database.shape
    solution = random.sample(range(num_rows), 50)
    best_solution = solution[:]
    best_value = evaluate_solution(solution, database)
    print(f"Solução inicial: {best_value}")

    for iteration in range(max_iterations):
        solution = local_search(solution, database, num_rows)
        solution_value = evaluate_solution(solution, database)
        if solution_value > best_value:
            best_solution = solution[:]
            best_value = solution_value
        solution = perturb(solution, num_rows)

        print(f"Iteração {iteration + 1}: Melhor valor: {best_value}")

    return best_solution, best_value

def process_instances(data_folder='data', file_name=None):
    if file_name:
        instances = [file_name]
    else:
        instances = [f for f in os.listdir(data_folder) if f.endswith('.sparse')]
    
    results = []
    
    for instance in instances:
        file_path = os.path.join(data_folder, instance)
        print(f"Lendo o arquivo: {file_path}")
        matrix = load_sparse_file(file_path).tocsr()

        best_values = []
        execution_times = []

        for _ in range(2):
            start_time = time.time()
            _, best_value = ils(matrix)
            end_time = time.time()

            best_values.append(best_value)
            execution_times.append(end_time - start_time)

        results.append({
            'instance': instance,
            'best_value': max(best_values),
            'average_value': np.mean(best_values),
            'average_time': np.mean(execution_times)
        })

    print("\nResultados:")
    print(f"{'Instância':<15} {'Melhor valor':<15} {'Valor médio':<15} {'Tempo médio':<15}")
    for result in results:
        print(f"{result['instance']:<15} {result['best_value']:<15} {result['average_value']:<15.2f} {result['average_time']:<15.2f}s")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Processar instâncias de arquivos .sparse.')
    parser.add_argument('file', nargs='?', default=None, help='Nome do arquivo .sparse para processar. Se não fornecido, processa todos os arquivos na pasta "data".')
    args = parser.parse_args()

    process_instances(file_name=args.file)
