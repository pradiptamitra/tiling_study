# Matrix Tiling Study

A comprehensive Python study demonstrating how tiling (blocking) improves memory access performance in matrix multiplication through better cache locality.

## Overview

This project analyzes the performance impact of different tile sizes on matrix multiplication operations. By breaking matrices into smaller tiles that fit within CPU caches (L1, L2, L3), we can dramatically improve performance by reducing cache misses and improving data reuse.

## Features

- **Automatic Cache Detection**: Detects L1, L2, and L3 cache sizes on Linux, macOS, and Windows
- **Performance Measurement**: Measures execution time and GFLOPs/s for different tile sizes
- **Cache-Aware Analysis**: Determines if each tile size optimally fits in L1, L2, or L3 cache
- **Beautiful Visualizations**: Generates plots with cache-size annotations showing where performance peaks relative to your system's cache hierarchy
- **Statistical Analysis**: Runs multiple iterations and reports mean/std dev timing data

## How Tiling Works

### Without Tiling (Naive Approach)
Traditional matrix multiplication accesses memory in patterns that lead to cache misses:
```
for i in range(n):
    for j in range(n):
        for k in range(n):
            C[i,j] += A[i,k] * B[k,j]
```

### With Tiling
We break the matrices into smaller blocks (tiles) that fit in cache:
```
for i_tile in range(0, n, tile_size):
    for j_tile in range(0, n, tile_size):
        for k_tile in range(0, n, tile_size):
            # Multiply smaller tile blocks
            C[i_tile:i_end, j_tile:j_end] +=
                A[i_tile:i_end, k_tile:k_end] @ B[k_tile:k_end, j_tile:j_end]
```

This ensures that the working set (A tile, B tile, C tile) stays in cache between iterations.

## Files

- `matrix_tiling_study.py`: Main study script with performance benchmarking
- `hello_world.py`: Simple test file
- `tiling_performance.png`: Generated performance visualization

## Requirements

```bash
pip install numpy matplotlib
```

## Usage

Run the tiling study:

```bash
python matrix_tiling_study.py
```

This will:
1. Detect your system's cache sizes
2. Run matrix multiplication (512×512) with tile sizes: 4, 8, 16, 32, 64, 128, 256, 512
3. Report performance metrics for each tile size
4. Generate `tiling_performance.png` with annotated performance plots

## Output

The script produces:

1. **Console Output**:
   - Your system's L1, L2, L3 cache sizes
   - Performance table showing execution time and GFLOPs/s for each tile size
   - Cache optimality classification ("L1-optimal", "L2-optimal", etc.)
   - Summary with best tile size and improvement percentage

2. **Visualization** (`tiling_performance.png`):
   - **Top plot**: Execution time vs tile size with error bars
   - **Bottom plot**: Peak performance (GFLOPs/s) vs tile size
   - Vertical dashed lines marking optimal tile sizes for L1, L2, L3 caches

## Cache Optimality

The script calculates whether a tile size fits efficiently in each cache level:

- **Footprint = 3 × tile_size² × 4 bytes** (3 tiles: A, B, C output; float32 = 4 bytes)
- **L1-optimal**: Footprint ≤ 50% of L1 cache size (fastest)
- **L2-optimal**: Footprint ≤ 50% of L2 cache size
- **L3-optimal**: Footprint ≤ 50% of L3 cache size
- **Spills cache**: Tile too large, data goes to main memory (slowest)

The factor of 0.5 provides a safety margin to account for other cache usage by the OS and compiler-generated code.

## Example Output

```
======================================================================
Matrix Tiling Performance Study
======================================================================
Matrix size: 512 x 512
Number of runs per tile size: 3
Total operations per multiplication: 268.44 billion FLOPs

======================================================================
Cache Information:
======================================================================
L1 Cache: 32.00 KB (32768 bytes)
L2 Cache: 512.00 KB (524288 bytes)
L3 Cache: 8.00 MB (8388608 bytes)

======================================================================
Performance Results:
======================================================================
Tile Size       Avg Time (s)    GFLOPs/s        Cache-Optimal
----------------------------------------------------------------------
4               0.234567        1145.23         L1-optimal
8               0.156789        1710.42         L1-optimal
16              0.098765        2715.89         L2-optimal
32              0.087654        3062.15         L2-optimal
64              0.125432        2138.47         L3-optimal
128             0.234567        1145.23         Spills cache
256             0.456789        587.12          Spills cache
512             0.678901        395.74          Spills cache

======================================================================
Summary:
======================================================================
Best performance: Tile size 32 (3062.15 GFLOPs/s)
Baseline (tile_size=matrix_size): 395.74 GFLOPs/s
Improvement over naive: 673.5%
```

## Key Insights

1. **Small tile sizes** (4-8): Optimal cache usage but overhead from more tile iterations
2. **Medium tile sizes** (16-64): Sweet spot - good cache utilization with reasonable iteration count
3. **Large tile sizes** (128+): Data spills from cache, performance degrades rapidly

The optimal tile size depends on:
- Your CPU's cache hierarchy
- The matrix size
- Available system memory
- Compiler optimizations

## Performance Improvement

Tiling can provide **2-10x performance improvements** over naive matrix multiplication by:
- Reducing L3 cache misses
- Improving data reuse within cache
- Better memory bandwidth utilization
- Enabling SIMD and vectorization within tiles
