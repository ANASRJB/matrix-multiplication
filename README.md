# Matrix Multiplication Performance Comparison

A comprehensive benchmarking study comparing different matrix multiplication implementations across multiple programming paradigms and optimization techniques.

## Overview

This project implements and benchmarks matrix multiplication algorithms using four different approaches:
1. **Pure Python** - Basic nested loop implementation
2. **Python with NumPy** - Optimized library implementation
3. **C with OpenMP** - Parallel implementation using OpenMP
4. **C with Pthreads** - Manual thread management implementation

## Methodology

The benchmarking approach follows a systematic methodology to evaluate performance across different matrix sizes:

### Matrix Size Variation
- **Size Range**: 100×100 to 1000×1000 matrices
- **Step Size**: Increments of 100 (100, 200, 300, ..., 1000)
- **Matrix Type**: Square matrices with random double-precision values

### Performance Measurement
- **Metric**: Execution time in seconds
- **Timing Method**: 
  - Python: `time.time()` for wall-clock time
  - OpenMP: `omp_get_wtime()` for high-precision timing
  - Pthreads: `clock()` for CPU time measurement
- **Data Collection**: Each implementation runs through the complete size range

### Visualization
- **Plotting**: Matplotlib generates performance comparison graphs
- **Scale**: Logarithmic y-axis for better visualization of performance differences
- **Output**: High-resolution PNG file (`matrix_performance_comparison.png`)


## Usage

### Prerequisites
```bash
# Required packages
pip install numpy matplotlib

# Compiler requirements
gcc (with OpenMP support)
```

### Running the Benchmark

1. **Execute the complete benchmark suite**:
   ```bash
   python benchmark_plot.py
   ```

2. **Run individual C implementations**:
   ```bash
   # Compile and run OpenMP version
   gcc -o matrix_openmp matrix_openmp.c -fopenmp -O3
   ./matrix_openmp
   
   # Compile and run Pthreads version  
   gcc -o matrix_pthreads matrix_pthreads.c -lpthread -O3
   ./matrix_pthreads
   ```

## Expected Results

The benchmark typically reveals the following performance hierarchy (fastest to slowest):

1. **C OpenMP** - Best parallel performance
2. **Python NumPy** - Optimized library performance  
3. **C Pthreads** - Manual parallelization overhead
4. **Pure Python** - Baseline sequential performance

### Performance Characteristics
- **Parallelization Benefit**: Becomes more apparent with larger matrices
- **Memory Access**: Cache performance impacts all implementations

## Output

The benchmark generates:
- **Console Output**: Real-time progress and final performance summary
- **Performance Plot**: `matrix_performance_comparison.png` with comparative analysis
- **Quantitative Data**: Execution times for each matrix size and implementation

## Research Applications

This benchmarking framework is suitable for:
- **Algorithm Analysis**: Comparing computational complexity implementations
- **Parallelization Studies**: Evaluating different parallel programming models
- **Performance Optimization**: Identifying bottlenecks and optimization opportunities  
- **Educational Purposes**: Demonstrating the impact of different programming approaches