# Floyd-Warshall Algorithm Implementation

This repository contains both non-parallel and parallel implementations of the Floyd-Warshall algorithm for solving the all-pairs shortest path problem on a randomly generated graph. The graph has 500 nodes with a 0.5 probability of connection between each pair of nodes.

## Algorithm Details

### Non-Parallel Algorithm
The non-parallel version of the Floyd-Warshall algorithm iterates through all nodes and updates the shortest paths sequentially. It is implemented as a nested loop with a time complexity of \(O(n^3)\).

### Parallel Algorithm
The parallel version divides the computation of distance matrix updates among multiple worker processes using Python's `multiprocessing` module and shared memory. Each process computes updates for a specific chunk of rows, enabling concurrent execution.

#### Parallelization Method
- **Shared Memory**: The distance matrix is stored in shared memory using `multiprocessing.shared_memory.SharedMemory`.
- **Worker Processes**: The computation is split row-wise among worker processes.

## How to Reproduce Results

### Prerequisites
Ensure you have Python 3.8 or later installed, as the `SharedMemory` module is available starting from Python 3.8.

### Required Libraries
Install the following Python libraries if not already installed:
- `numpy`
- `matplotlib`

You can install them using pip:
```bash
pip install numpy matplotlib
```

### Installation
1. Clone the repository.
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```
2. Install the required dependencies as mentioned above.

### Running the Script
1. Execute the script directly:
   ```bash
   python floyd_warshall.py
   ```
2. The script generates a random graph (500 nodes by default, 0.5 probability of connection), runs both the non-parallel and parallel versions of the algorithm, verifies the correctness of the results, and plots the speedup achieved by parallel execution.

### Output
- **Execution Times**: The script prints the execution times for both non-parallel and parallel versions.
- **Speedup**: Speedup is calculated as the ratio of non-parallel execution time to parallel execution time.
- **Correctness Verification**: The script verifies that the results from both implementations match.
- **Speedup Plot**: A plot of speedup vs. number of workers is displayed.

## Parallelization Details

### Key Optimizations
1. **Row-Wise Parallelization**: Each worker updates a specific chunk of rows in the distance matrix for a given intermediate node.
2. **Shared Memory**: The shared memory ensures all processes have access to the same distance matrix, avoiding redundant memory copies.

### Correctness Verification
The correctness of the parallel implementation is verified by comparing its results to the non-parallel version using `np.allclose` to account for floating-point tolerances.

## Results

### Execution Times and Speedup
- **Non-parallel execution time**: 44.36 seconds
- **Parallel execution times**:
  - 2 workers: 1.12 seconds (Speedup: 42.41)
  - 4 workers: 0.94 seconds (Speedup: 50.84)
  - 8 workers: 1.14 seconds (Speedup: 41.92)
  - 16 workers: 1.66 seconds (Speedup: 28.76)

### Speedup Plot
The script generates a plot of speedup vs. number of workers:
![Figure_1](https://github.com/user-attachments/assets/09aa86ed-0b58-438f-b884-378eee94942b)


## Conclusion
The parallel implementation achieves best speedup for 4 workers. Each worker incurs a setup and communication cost, so we encounter worse performance with more than 4 workers.

