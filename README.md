# Matrix Tiling Study

A comprehensive Python study demonstrating how tiling (blocking) improves memory access performance in matrix multiplication through better cache locality. This project compares two different tiling strategies and analyzes their performance characteristics.

## Overview

This project analyzes the performance impact of different tile sizes on matrix multiplication operations. By breaking matrices into smaller tiles that fit within CPU caches (L1, L2, L3), we can improve performance by reducing cache misses and improving data reuse.

The study implements and compares two tiling strategies:
- **Standard (C-tile-focused)**: Keeps output tiles hot in cache
- **A-tile-focused**: Keeps input A tiles hot in cache (experimental)

## Features

- **Automatic Cache Detection**: Detects L1 data cache, L2, and L3 cache sizes on Linux, macOS, and Windows
- **JIT Compilation**: Uses Numba for high-performance JIT compilation
- **Dual Strategy Comparison**: Compare standard vs A-focused tiling strategies
- **Performance Measurement**: Measures execution time for different tile sizes
- **Cache-Aware Analysis**: Determines if each tile size optimally fits in L1, L2, or L3 cache
- **Beautiful Visualizations**: Generates plots with cache-size annotations showing where performance peaks relative to your system's cache hierarchy
- **Statistical Analysis**: Runs multiple iterations and reports mean/std dev timing data

## How Tiling Works

### Standard Tiling (C-tile-focused)
The standard approach keeps the output tile in cache while streaming through inputs:
```python
for i_tile in range(0, n, tile_size):
    for j_tile in range(0, n, tile_size):
        for k_tile in range(0, n, tile_size):
            # Process this output tile completely
            # Load required A and B tiles as needed
            for i in range(i_tile, i_end):
                for j in range(j_tile, j_end):
                    for k in range(k_tile, k_end):
                        C[i, j] += A[i, k] * B[k, j]
```

**Strategy**: For each output tile C, load all required A and B tiles to complete it. This keeps the output tile hot in cache while streaming through inputs.

### A-tile-focused Tiling (Experimental)
An alternative approach that maximizes reuse of each A tile:
```python
for i_tile in range(0, n, tile_size):
    for k_tile in range(0, n, tile_size):
        # Keep this A tile "hot" in cache
        for j_tile in range(0, n, tile_size):
            # Update all C tiles influenced by this A tile
            for i in range(i_tile, i_end):
                for j in range(j_tile, j_end):
                    for k in range(k_tile, k_end):
                        C[i, j] += A[i, k] * B[k, j]
```

**Strategy**: For each A tile, keep it hot in cache and stream through all B tiles, updating the corresponding output tiles. This maximizes reuse of each A tile.

**Trade-off**: While A tiles get better reuse, C tiles may be loaded/stored multiple times, potentially causing more memory traffic.

## Files

- `matrix_tiling_study.py`: Main study script with performance benchmarking
- `tiling_performance_standard.png`: Performance visualization for standard strategy
- `tiling_performance_a_focused.png`: Performance visualization for A-focused strategy
- `tiling_performance_both.png`: Comparison visualization for both strategies

## Requirements

```bash
pip install numpy matplotlib numba
```

## Usage

The script supports three modes controlled by the `MODE` variable in `main()`:

### Run Standard Strategy Only
```python
MODE = 'standard'
```

### Run A-focused Strategy Only
```python
MODE = 'a_focused'
```

### Compare Both Strategies
```python
MODE = 'both'  # Default
```

Then run:
```bash
python matrix_tiling_study.py
```

## Configuration

Key parameters in `main()`:
- `MATRIX_SIZE`: Size of square matrices (default: 1024)
- `MODE`: Tiling strategy - 'standard', 'a_focused', or 'both'
- `tile_sizes`: List of tile sizes to test

## Output

The script produces:

1. **Console Output**:
   - Your system's L1, L2, L3 cache sizes
   - Performance table showing execution time for each tile size
   - Cache optimality classification ("L1-optimal", "L2-optimal", etc.)
   - Summary with best tile size and speedup vs baseline

2. **Visualization** (`tiling_performance_{MODE}.png`):
   - Execution time vs tile size with error bars
   - For 'both' mode: Overlaid comparison of both strategies
   - Vertical dashed lines marking optimal tile sizes for L1, L2, L3 caches
   - Color coding: Blue (Standard), Red (A-focused)

## Cache Optimality

The script calculates whether a tile size fits efficiently in each cache level:

- **Footprint = 3 × tile_size² × 4 bytes** (3 tiles: A, B, C; float32 = 4 bytes)
- **L1-optimal**: Footprint ≤ 50% of L1 data cache size (fastest)
- **L2-optimal**: Footprint ≤ 50% of L2 cache size
- **L3-optimal**: Footprint ≤ 50% of L3 cache size
- **Spills cache**: Tile too large, data goes to main memory (slowest)

The conservative factor of 0.5 accounts for:
- Cache sharing between threads and OS
- L1 instruction vs data cache split
- Cache associativity conflicts
- Working set overhead (loop counters, temporary variables)

## Example Output

```
======================================================================
Matrix Tiling Performance Study (Comparing Both Strategies)
======================================================================
Matrix size: 1024 x 1024
Number of runs per tile size: 3
Strategy: both

======================================================================
Cache Information:
======================================================================
L1 Cache: 64.00 KB (65536 bytes)
L2 Cache: 512.00 KB (524288 bytes)
L3 Cache: 8.00 MB (8388608 bytes)

======================================================================
Running Standard (C-tile-focused) strategy...
======================================================================
Tile Size       Avg Time (s)    Cache-Optimal
-----------------------------------------------------------------
1               0.123456        L1-optimal
4               0.098765        L1-optimal
8               0.087654        L1-optimal
16              0.076543        L1-optimal
32              0.065432        L2-optimal
64              0.054321        L2-optimal
128             0.067890        L3-optimal
256             0.089012        L3-optimal
512             0.123456        Spills cache
1024            0.234567        Spills cache

======================================================================
Running A-tile-focused strategy...
======================================================================
Tile Size       Avg Time (s)    Cache-Optimal
-----------------------------------------------------------------
1               0.125432        L1-optimal
4               0.099876        L1-optimal
8               0.088765        L1-optimal
16              0.077654        L1-optimal
32              0.066543        L2-optimal
64              0.055432        L2-optimal
128             0.068901        L3-optimal
256             0.090123        L3-optimal
512             0.124567        Spills cache
1024            0.235678        Spills cache

======================================================================
Summary:
======================================================================

Standard (C-tile-focused):
  Best performance: Tile size 64 (0.054321s)
  Baseline (largest tile): 0.234567s
  Speedup vs baseline: 331.8%

A-tile-focused:
  Best performance: Tile size 64 (0.055432s)
  Baseline (largest tile): 0.235678s
  Speedup vs baseline: 325.2%

Comparison:
  Standard is 2.0% faster at optimal tile size
```

## Key Insights

1. **Very small tile sizes** (1-4): Excessive tiling overhead despite perfect cache fit
2. **Small tile sizes** (8-16): Good L1 cache utilization, reduced overhead
3. **Medium tile sizes** (32-64): Sweet spot - L2 cache optimal with best performance
4. **Large tile sizes** (128+): Data spills from smaller caches, performance degrades
5. **Baseline** (tile_size = matrix_size): No tiling benefit, equivalent to naive approach

### Strategy Comparison

- **Standard (C-tile-focused)**: Generally performs better due to:
  - Each output tile written once
  - Better cache reuse of output accumulator
  - Less memory bandwidth consumption

- **A-tile-focused**: Often comparable performance, but:
  - Output tiles may be loaded/stored multiple times
  - Potential cache thrashing if many output tiles
  - May perform better on certain architectures

The optimal tile size depends on:
- Your CPU's cache hierarchy
- Cache line size and associativity
- The matrix size
- Memory bandwidth vs compute speed
- Compiler/JIT optimizations

## Performance Notes

This implementation uses Numba JIT compilation for high performance. The JIT compiler applies aggressive optimizations including:
- SIMD vectorization
- Register allocation
- Memory access pattern optimization

For very small matrices or when the working set fits entirely in cache, tiling overhead may outweigh benefits. Tiling shows greatest advantage when:
- Matrix size exceeds cache capacity
- Memory bandwidth is the bottleneck
- Working with larger matrices (2048×2048 and above)

## Implementation Details

- All matrix multiplication kernels are JIT-compiled with Numba
- L1 **data** cache is correctly detected (not instruction cache)
- Cache optimal tile size formula: `tile_size = sqrt(cache_size * 0.5 / (3 * 4))`
- Boundary handling with `min()` for non-evenly divisible matrices
- Statistical reporting with mean and standard deviation across multiple runs
