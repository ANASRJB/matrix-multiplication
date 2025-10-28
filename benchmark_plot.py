import numpy as np
import time
import matplotlib.pyplot as plt
import subprocess
import re

def matrix_multiply_python(A, B):
    """Pure Python matrix multiplication"""
    n = len(A)
    C = [[0.0 for _ in range(n)] for _ in range(n)]
    
    for i in range(n):
        for j in range(n):
            for k in range(n):
                C[i][j] += A[i][k] * B[k][j]
    return C

def matrix_multiply_numpy(A, B):
    """NumPy matrix multiplication"""
    return np.dot(A, B)

def benchmark_python(max_size=500, step=100):
    """Benchmark Python implementations"""
    sizes = list(range(100, max_size + 1, step))
    times_python = []
    times_numpy = []
    
    for size in sizes:
        # Generate random matrices
        A_py = [[np.random.random() for _ in range(size)] for _ in range(size)]
        B_py = [[np.random.random() for _ in range(size)] for _ in range(size)]
        
        A_np = np.random.rand(size, size)
        B_np = np.random.rand(size, size)
        
        # Benchmark pure Python
        start = time.time()
        matrix_multiply_python(A_py, B_py)
        end = time.time()
        times_python.append(end - start)
        
        # Benchmark NumPy
        start = time.time()
        matrix_multiply_numpy(A_np, B_np)
        end = time.time()
        times_numpy.append(end - start)
        
        print(f"Size: {size}, Python: {times_python[-1]:.4f}s, NumPy: {times_numpy[-1]:.4f}s")
    
    return sizes, times_python, times_numpy

def run_c_benchmarks():
    """Run C benchmarks and parse results"""
    print("Running OpenMP benchmark...")
    result_omp = subprocess.run(['./matrix_openmp'], capture_output=True, text=True)
    
    print("Running Pthreads benchmark...")
    result_pthreads = subprocess.run(['./matrix_pthreads'], capture_output=True, text=True)
    
    # Parse results
    def parse_output(output):
        sizes, times = [], []
        for line in output.split('\n'):
            if 'Size:' in line:
                match = re.search(r'Size: (\d+), Time: ([\d.]+)', line)
                if match:
                    sizes.append(int(match.group(1)))
                    times.append(float(match.group(2)))
        return sizes, times
    
    sizes_omp, times_omp = parse_output(result_omp.stdout)
    sizes_pthreads, times_pthreads = parse_output(result_pthreads.stdout)
    
    return sizes_omp, times_omp, sizes_pthreads, times_pthreads

def plot_results(all_results):
    """Plot comparison of all implementations"""
    plt.figure(figsize=(12, 8))
    
    # Unpack results
    (sizes_python, times_python, times_numpy, 
     sizes_omp, times_omp, sizes_pthreads, times_pthreads) = all_results
    
    # Plot all curves
    plt.plot(sizes_python, times_python, 'o-', label='Pure Python', linewidth=2)
    plt.plot(sizes_python, times_numpy, 's-', label='Python NumPy', linewidth=2)
    plt.plot(sizes_omp, times_omp, '^-', label='C OpenMP', linewidth=2)
    plt.plot(sizes_pthreads, times_pthreads, 'd-', label='C Pthreads', linewidth=2)
    
    plt.xlabel('Matrix Size')
    plt.ylabel('Execution Time (seconds)')
    plt.title('Matrix Multiplication Performance Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')  # Log scale for better visualization
    plt.tight_layout()
    plt.savefig('matrix_performance_comparison.png', dpi=300)
    plt.show()

def main():
    # Compile C programs
    print("Compiling C programs...")
    subprocess.run(['gcc', '-o', 'matrix_openmp', 'matrix_openmp.c', '-fopenmp', '-O3'])
    subprocess.run(['gcc', '-o', 'matrix_pthreads', 'matrix_pthreads.c', '-lpthread', '-O3'])
    
    # Run benchmarks
    print("Benchmarking Python implementations...")
    sizes_python, times_python, times_numpy = benchmark_python()
    
    print("\nBenchmarking C implementations...")
    sizes_omp, times_omp, sizes_pthreads, times_pthreads = run_c_benchmarks()
    
    # Plot results
    all_results = (sizes_python, times_python, times_numpy, 
                   sizes_omp, times_omp, sizes_pthreads, times_pthreads)
    plot_results(all_results)
    
    # Print summary
    print("\n=== PERFORMANCE SUMMARY ===")
    print(f"{'Implementation':<15} {'Time for size 500 (s)':<20}")
    print("-" * 40)
    if len(sizes_python) >= 5:
        print(f"{'Pure Python':<15} {times_python[4]:<20.4f}")
        print(f"{'Python NumPy':<15} {times_numpy[4]:<20.4f}")
    if len(times_omp) >= 5:
        print(f"{'C OpenMP':<15} {times_omp[4]:<20.4f}")
    if len(times_pthreads) >= 5:
        print(f"{'C Pthreads':<15} {times_pthreads[4]:<20.4f}")

if __name__ == "__main__":
    main()