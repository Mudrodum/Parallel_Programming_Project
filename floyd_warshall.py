import numpy as np
import multiprocessing as mp
from multiprocessing.shared_memory import SharedMemory
import time
import matplotlib.pyplot as plt

# Generate a random adjacency matrix for a graph with 500 nodes and 0.5 probability of connection
def generate_graph(num_nodes=500, connection_prob=0.5):
    graph = np.random.rand(num_nodes, num_nodes)
    graph = (graph < connection_prob).astype(float)
    np.fill_diagonal(graph, 0)  # Distance from a node to itself is 0
    graph[graph == 0] = float('inf')  # No connection means infinite distance
    np.fill_diagonal(graph, 0)
    return graph

# Non-parallel Floyd-Warshall Algorithm
def floyd_warshall(graph):
    num_nodes = len(graph)
    dist = graph.copy()
    for k in range(num_nodes):
        for i in range(num_nodes):
            for j in range(num_nodes):
                dist[i, j] = min(dist[i, j], dist[i, k] + dist[k, j])
    return dist

# Worker function for parallel processing
def update_dist_worker(start_row, end_row, num_nodes, shm_name, k):
    existing_shm = SharedMemory(name=shm_name)
    dist = np.ndarray((num_nodes, num_nodes), dtype=np.float64, buffer=existing_shm.buf)
    for i in range(start_row, end_row):
        dist[i, :] = np.minimum(dist[i, :], dist[i, k] + dist[k, :])

def floyd_warshall_parallel(graph, num_workers=4):
    num_nodes = len(graph)

    # Create shared memory for the distance matrix
    shm = SharedMemory(create=True, size=graph.nbytes)
    dist = np.ndarray((num_nodes, num_nodes), dtype=np.float64, buffer=shm.buf)
    np.copyto(dist, graph)

    # Create worker pool
    pool = mp.Pool(num_workers)
    chunk_size = num_nodes // num_workers

    for k in range(num_nodes):
        tasks = []
        for w in range(num_workers):
            start_row = w * chunk_size
            end_row = (w + 1) * chunk_size if w != num_workers - 1 else num_nodes
            tasks.append((start_row, end_row, num_nodes, shm.name, k))
        pool.starmap(update_dist_worker, tasks)

    # Copy results to avoid shared memory issues before cleanup
    result = dist.copy()

    # Cleanup shared memory
    pool.close()
    pool.join()
    shm.close()
    shm.unlink()
    return result

# Execution and timing
if __name__ == "__main__":
    np.random.seed(seed=42)
    # Generate graph
    graph = generate_graph()

    # Non-parallel execution
    start_time = time.time()
    dist_non_parallel = floyd_warshall(graph)
    non_parallel_time = time.time() - start_time

    print(f"Non-parallel execution time: {non_parallel_time:.2f} seconds")

    # Parallel execution
    num_workers_list = [2, 4, 8, 16]
    parallel_times = []
    dist_parallel = None

    for num_workers in num_workers_list:
        start_time = time.time()
        dist_parallel = floyd_warshall_parallel(graph, num_workers=num_workers)
        parallel_time = time.time() - start_time
        parallel_times.append(parallel_time)

        print(f"Parallel execution time with {num_workers} workers: {parallel_time:.2f} seconds")
        print(f"Speedup: {non_parallel_time / parallel_time:.2f}")

    # Calculate speedup
    speedups = [non_parallel_time / t for t in parallel_times]

    # Verify correctness
    print("Verifying results...")
    if np.allclose(dist_non_parallel, dist_parallel, atol=1e-8):
        print("Verification successful: Non-parallel and parallel results match!")
    else:
        print("Verification failed: Results do not match.")

    # Plot speedup vs. number of workers
    plt.figure(figsize=(8, 6))
    plt.plot(num_workers_list, speedups, marker='o', linestyle='-', color='b')
    plt.title("Speedup vs. Number of Workers", fontsize=14)
    plt.xlabel("Number of Workers", fontsize=12)
    plt.ylabel("Speedup (Non-parallel Time / Parallel Time)", fontsize=12)
    plt.grid(True)
    plt.xticks(num_workers_list)
    plt.tight_layout()
    plt.show()
