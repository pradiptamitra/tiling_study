#!/usr/bin/env python3
"""
Matrix Multiplication Tiling Study

Demonstrates how tiling (blocking) improves memory access performance
by improving cache locality. This code:
1. Detects system cache sizes (L1, L2, L3)
2. Performs matrix multiplication with varying tile sizes
3. Measures performance metrics
4. Generates visualizations with cache-size annotations
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import subprocess
import json
import os
import sys
from typing import Tuple, Dict, List
from numba import jit


@jit(nopython=True, fastmath=False, parallel=False)
def _tiled_multiply_jit(A: np.ndarray, B: np.ndarray, tile_size: int) -> np.ndarray:
    """
    Tiled matrix multiplication compiled with Numba (C-tile-focused).

    Strategy: For each output tile C, load all required A and B tiles to complete it.
    This keeps the output tile hot in cache while streaming through inputs.
    """
    n = A.shape[0]
    C = np.zeros((n, n))

    for i_tile in range(0, n, tile_size):
        for j_tile in range(0, n, tile_size):
            for k_tile in range(0, n, tile_size):
                i_end = min(i_tile + tile_size, n)
                j_end = min(j_tile + tile_size, n)
                k_end = min(k_tile + tile_size, n)

                for i in range(i_tile, i_end):
                    for j in range(j_tile, j_end):
                        for k in range(k_tile, k_end):
                            C[i, j] += A[i, k] * B[k, j]

    return C


@jit(nopython=True, fastmath=False, parallel=False)
def _tiled_multiply_a_focused_jit(A: np.ndarray, B: np.ndarray, tile_size: int) -> np.ndarray:
    """
    Tiled matrix multiplication compiled with Numba (A-tile-focused).

    Strategy: For each A tile, keep it hot in cache and stream through all B tiles,
    updating the corresponding output tiles. This maximizes reuse of each A tile.
    """
    n = A.shape[0]
    C = np.zeros((n, n))

    for i_tile in range(0, n, tile_size):
        for k_tile in range(0, n, tile_size):
            # Keep this A tile "hot" in cache
            i_end = min(i_tile + tile_size, n)
            k_end = min(k_tile + tile_size, n)

            # Iterate through all B tiles and corresponding C tiles
            for j_tile in range(0, n, tile_size):
                j_end = min(j_tile + tile_size, n)

                for i in range(i_tile, i_end):
                    for j in range(j_tile, j_end):
                        for k in range(k_tile, k_end):
                            C[i, j] += A[i, k] * B[k, j]

    return C


class CacheDetector:
    """Detect CPU cache sizes from the system."""

    @staticmethod
    def get_cache_sizes() -> Dict[str, int]:
        """
        Attempt to detect L1, L2, and L3 cache sizes.
        Returns sizes in bytes. Returns empty dict if detection fails.
        """
        cache_sizes = {}

        # Try Linux method first (most reliable)
        if sys.platform.startswith('linux'):
            return CacheDetector._get_cache_linux()
        elif sys.platform == 'darwin':  # macOS
            return CacheDetector._get_cache_macos()
        elif sys.platform.startswith('win'):
            return CacheDetector._get_cache_windows()

        return cache_sizes

    @staticmethod
    def _get_cache_linux() -> Dict[str, int]:
        """Get cache sizes on Linux using lscpu or /proc/cpuinfo."""
        cache_sizes = {}
        try:
            result = subprocess.run(['lscpu'], capture_output=True, text=True)
            for line in result.stdout.split('\n'):
                if 'L1d cache' in line:
                    cache_sizes['L1'] = int(line.split(':')[1].strip().split()[0]) * 1024
                elif 'L2 cache' in line:
                    cache_sizes['L2'] = int(line.split(':')[1].strip().split()[0]) * 1024
                elif 'L3 cache' in line:
                    cache_sizes['L3'] = int(line.split(':')[1].strip().split()[0]) * 1024
        except Exception:
            pass

        return cache_sizes

    @staticmethod
    def _get_cache_macos() -> Dict[str, int]:
        """Get cache sizes on macOS using sysctl."""
        cache_sizes = {}
        try:
            # L1 data cache
            result = subprocess.run(['sysctl', 'hw.l1dcachesize'],
                                  capture_output=True, text=True)
            if result.returncode == 0:
                cache_sizes['L1'] = int(result.stdout.split(':')[1].strip())

            # L2 cache
            result = subprocess.run(['sysctl', 'hw.l2cachesize'],
                                  capture_output=True, text=True)
            if result.returncode == 0:
                cache_sizes['L2'] = int(result.stdout.split(':')[1].strip())

            # L3 cache
            result = subprocess.run(['sysctl', 'hw.l3cachesize'],
                                  capture_output=True, text=True)
            if result.returncode == 0:
                cache_sizes['L3'] = int(result.stdout.split(':')[1].strip())
        except Exception:
            pass

        return cache_sizes

    @staticmethod
    def _get_cache_windows() -> Dict[str, int]:
        """Get cache sizes on Windows using wmic."""
        cache_sizes = {}
        try:
            result = subprocess.run(['wmic', 'cpu', 'get', 'l2cachesize,l3cachesize'],
                                  capture_output=True, text=True)
            lines = result.stdout.strip().split('\n')
            if len(lines) > 1:
                values = lines[1].split()
                if len(values) >= 2:
                    cache_sizes['L2'] = int(values[0]) * 1024
                    cache_sizes['L3'] = int(values[1]) * 1024
        except Exception:
            pass

        return cache_sizes


class MatrixTilingStudy:
    """Perform matrix multiplication tiling study."""

    def __init__(self, matrix_size: int = 1024):
        """
        Initialize the study.

        Args:
            matrix_size: Size of square matrices to multiply
        """
        self.matrix_size = matrix_size
        self.cache_sizes = CacheDetector.get_cache_sizes()
        self.results = {}

    def _tiled_multiply(self, A: np.ndarray, B: np.ndarray, tile_size: int) -> np.ndarray:
        """
        Tiled matrix multiplication with specified tile size - uses Numba JIT (C-tile-focused).

        Multiplies blocks of size tile_size x tile_size to improve cache locality.
        """
        return _tiled_multiply_jit(A, B, tile_size)

    def _tiled_multiply_a_focused(self, A: np.ndarray, B: np.ndarray, tile_size: int) -> np.ndarray:
        """
        Tiled matrix multiplication with specified tile size - uses Numba JIT (A-tile-focused).

        Keeps A tiles hot in cache while streaming through B tiles.
        """
        return _tiled_multiply_a_focused_jit(A, B, tile_size)

    def run_experiment(self, tile_sizes: List[int], num_runs: int = 3, mode: str = 'standard') -> Dict:
        """
        Run tiling experiment with various tile sizes.

        Args:
            tile_sizes: List of tile sizes to test
            num_runs: Number of runs per tile size (for averaging)
            mode: Tiling strategy - 'standard', 'a_focused', or 'both'

        Returns:
            Dictionary with results
        """
        if mode == 'both':
            return self._run_both_experiments(tile_sizes, num_runs)

        mode_name = "Standard (C-tile-focused)" if mode == 'standard' else "A-tile-focused"

        print(f"\n{'='*70}")
        print(f"Matrix Tiling Performance Study ({mode_name})")
        print(f"{'='*70}")
        print(f"Matrix size: {self.matrix_size} x {self.matrix_size}")
        print(f"Number of runs per tile size: {num_runs}")
        print(f"Strategy: {mode_name}")
        print(f"\n{'='*70}")
        print(f"Cache Information:")
        print(f"{'='*70}")

        if self.cache_sizes:
            for level, size in sorted(self.cache_sizes.items()):
                size_kb = size / 1024
                size_mb = size / (1024 * 1024)
                if size_mb >= 1:
                    print(f"{level} Cache: {size_mb:.2f} MB ({size} bytes)")
                else:
                    print(f"{level} Cache: {size_kb:.2f} KB ({size} bytes)")
        else:
            print("Could not detect cache sizes on this system.")

        # Create test matrices
        A = np.random.randn(self.matrix_size, self.matrix_size).astype(np.float32)
        B = np.random.randn(self.matrix_size, self.matrix_size).astype(np.float32)

        print(f"\n{'='*70}")
        print(f"Performance Results:")
        print(f"{'='*70}")
        print(f"{'Tile Size':<15} {'Avg Time (s)':<15} {'Cache-Optimal':<20}")
        print(f"{'-'*65}")

        results = {
            'tile_sizes': [],
            'times': [],
            'time_std': []
        }

        # Select multiply function based on mode
        multiply_func = self._tiled_multiply if mode == 'standard' else self._tiled_multiply_a_focused

        # Test each tile size
        for tile_size in tile_sizes:
            times = []

            for run in range(num_runs):
                start = time.perf_counter()
                C = multiply_func(A, B, tile_size)
                end = time.perf_counter()
                times.append(end - start)

            avg_time = np.mean(times)
            std_time = np.std(times)

            # Estimate cache optimality
            cache_optimal = self._estimate_cache_optimality(tile_size)

            results['tile_sizes'].append(tile_size)
            results['times'].append(avg_time)
            results['time_std'].append(std_time)

            print(f"{tile_size:<15} {avg_time:<15.6f} {cache_optimal:<20}")

        self.results = results
        return results

    def _run_both_experiments(self, tile_sizes: List[int], num_runs: int = 3) -> Dict:
        """
        Run experiments for both tiling strategies.

        Args:
            tile_sizes: List of tile sizes to test
            num_runs: Number of runs per tile size (for averaging)

        Returns:
            Dictionary with results for both strategies
        """
        print(f"\n{'='*70}")
        print(f"Matrix Tiling Performance Study (Comparing Both Strategies)")
        print(f"{'='*70}")
        print(f"Matrix size: {self.matrix_size} x {self.matrix_size}")
        print(f"Number of runs per tile size: {num_runs}")
        print(f"\n{'='*70}")
        print(f"Cache Information:")
        print(f"{'='*70}")

        if self.cache_sizes:
            for level, size in sorted(self.cache_sizes.items()):
                size_kb = size / 1024
                size_mb = size / (1024 * 1024)
                if size_mb >= 1:
                    print(f"{level} Cache: {size_mb:.2f} MB ({size} bytes)")
                else:
                    print(f"{level} Cache: {size_kb:.2f} KB ({size} bytes)")
        else:
            print("Could not detect cache sizes on this system.")

        # Create test matrices (same for both experiments)
        A = np.random.randn(self.matrix_size, self.matrix_size).astype(np.float32)
        B = np.random.randn(self.matrix_size, self.matrix_size).astype(np.float32)

        results = {
            'tile_sizes': [],
            'standard': {'times': [], 'time_std': []},
            'a_focused': {'times': [], 'time_std': []}
        }

        print(f"\n{'='*70}")
        print(f"Running Standard (C-tile-focused) strategy...")
        print(f"{'='*70}")
        print(f"{'Tile Size':<15} {'Avg Time (s)':<15} {'Cache-Optimal':<20}")
        print(f"{'-'*65}")

        # Run standard experiments
        for tile_size in tile_sizes:
            times = []
            for run in range(num_runs):
                start = time.perf_counter()
                C = self._tiled_multiply(A, B, tile_size)
                end = time.perf_counter()
                times.append(end - start)

            avg_time = np.mean(times)
            std_time = np.std(times)
            cache_optimal = self._estimate_cache_optimality(tile_size)

            results['tile_sizes'].append(tile_size)
            results['standard']['times'].append(avg_time)
            results['standard']['time_std'].append(std_time)

            print(f"{tile_size:<15} {avg_time:<15.6f} {cache_optimal:<20}")

        A = np.random.randn(self.matrix_size, self.matrix_size).astype(np.float32)
        B = np.random.randn(self.matrix_size, self.matrix_size).astype(np.float32)
        print(f"\n{'='*70}")
        print(f"Running A-tile-focused strategy...")
        print(f"{'='*70}")
        print(f"{'Tile Size':<15} {'Avg Time (s)':<15} {'Cache-Optimal':<20}")
        print(f"{'-'*65}")

        # Run A-focused experiments
        for i, tile_size in enumerate(tile_sizes):
            times = []
            for run in range(num_runs):
                start = time.perf_counter()
                C = self._tiled_multiply_a_focused(A, B, tile_size)
                end = time.perf_counter()
                times.append(end - start)

            avg_time = np.mean(times)
            std_time = np.std(times)
            cache_optimal = self._estimate_cache_optimality(tile_size)

            results['a_focused']['times'].append(avg_time)
            results['a_focused']['time_std'].append(std_time)

            print(f"{tile_size:<15} {avg_time:<15.6f} {cache_optimal:<20}")

        self.results = results
        return results

    def _estimate_cache_optimality(self, tile_size: int) -> str:
        """Estimate if tile size aligns well with cache."""
        if not self.cache_sizes:
            return "N/A"

        # Estimate memory footprint of one tile block computation
        # We need: tile_size x tile_size from A, tile_size x tile_size from B,
        # and tile_size x tile_size from C (output tile being accumulated)
        # All in float32 (4 bytes each)
        footprint_bytes = 3 * (tile_size * tile_size * 4)

        # Check against L1 cache
        if 'L1' in self.cache_sizes and footprint_bytes <= self.cache_sizes['L1'] * 0.5:
            return "L1-optimal"
        # Check against L2 cache
        elif 'L2' in self.cache_sizes and footprint_bytes <= self.cache_sizes['L2'] * 0.5:
            return "L2-optimal"
        # Check against L3 cache
        elif 'L3' in self.cache_sizes and footprint_bytes <= self.cache_sizes['L3'] * 0.5:
            return "L3-optimal"
        else:
            return "Spills cache"

    def plot_results(self, output_file: str = 'tiling_performance.png', mode: str = 'standard'):
        """
        Generate visualization of tiling performance.

        Args:
            output_file: Path to save the plot
            mode: Tiling strategy - 'standard', 'a_focused', or 'both'
        """
        if not self.results:
            print("No results to plot. Run experiment first.")
            return

        fig, ax = plt.subplots(figsize=(12, 6))

        # Check if we have results for both strategies
        if mode == 'both' and 'standard' in self.results:
            # Plot both strategies
            tile_sizes = self.results['tile_sizes']

            # Plot standard
            standard_times = self.results['standard']['times']
            standard_std = self.results['standard']['time_std']
            ax.errorbar(tile_sizes, standard_times, yerr=standard_std, fmt='o-', linewidth=2,
                         markersize=8, capsize=5, label='Standard (C-tile-focused)', color='blue')

            # Plot A-focused
            a_times = self.results['a_focused']['times']
            a_std = self.results['a_focused']['time_std']
            ax.errorbar(tile_sizes, a_times, yerr=a_std, fmt='s-', linewidth=2,
                         markersize=8, capsize=5, label='A-tile-focused', color='red')

            title = f'Matrix Multiplication Time vs Tile Size (Both Strategies)\n(Matrix: {self.matrix_size}x{self.matrix_size})'
        else:
            # Plot single strategy
            mode_name = "Standard (C-tile-focused)" if mode == 'standard' else "A-tile-focused"
            tile_sizes = self.results['tile_sizes']
            times = self.results['times']
            time_std = self.results['time_std']

            ax.errorbar(tile_sizes, times, yerr=time_std, fmt='o-', linewidth=2,
                         markersize=8, capsize=5, label='Execution Time')
            title = f'Matrix Multiplication Time vs Tile Size ({mode_name})\n(Matrix: {self.matrix_size}x{self.matrix_size})'

        ax.set_xlabel('Tile Size (elements)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Time (seconds)', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log')

        # Add cache size markers
        self._add_cache_markers(ax, 'time')

        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"\nPlot saved to: {output_file}")
        plt.show()

    def _add_cache_markers(self, ax, plot_type: str):
        """Add vertical lines and labels for cache sizes on the plot."""
        if not self.cache_sizes:
            return

        # Convert cache sizes to tile sizes
        # For float32: tile_size^2 * 4 bytes per matrix
        # We estimate optimal tile size as sqrt(cache_size / 12) for 3 tiles (A, B, C)

        colors = {'L1': 'red', 'L2': 'orange', 'L3': 'purple'}

        for level in ['L1', 'L2', 'L3']:
            if level in self.cache_sizes:
                cache_bytes = self.cache_sizes[level]
                # Estimate tile size that fits in cache
                # Same logic as _estimate_cache_optimality: 3 tiles * tile_size² * 4 bytes <= cache * 0.5
                optimal_tile = int(np.sqrt(cache_bytes * 0.5 / (3 * 4)))

                ax.axvline(x=optimal_tile, color=colors[level], linestyle='--',
                          linewidth=2, alpha=0.7, label=f'{level} optimal (~{optimal_tile})')

        ax.legend(loc='best', fontsize=10)


def main():
    """Main entry point."""
    # Configuration
    MODE = 'both'  # 'standard', 'a_focused', or 'both'
    MATRIX_SIZE = 2048
    # Generate tile sizes as powers of 2 from 1 to MATRIX_SIZE
    tile_sizes = [2**i for i in range(int(np.log2(MATRIX_SIZE)) + 1)]
    print(f"Testing tile sizes: {min(tile_sizes)} to {max(tile_sizes)} (powers of 2)")

    # Create and run study
    study = MatrixTilingStudy(matrix_size=MATRIX_SIZE)
    results = study.run_experiment(tile_sizes, num_runs=2, mode=MODE)

    # Generate visualization
    output_filename = f'tiling_performance_{MODE}.png'
    study.plot_results(output_filename, mode=MODE)

    # Print summary
    print(f"\n{'='*70}")
    print(f"Summary:")
    print(f"{'='*70}")

    if MODE == 'both':
        # Summary for both strategies
        standard_times = results['standard']['times']
        a_times = results['a_focused']['times']

        standard_best_idx = np.argmin(standard_times)
        a_best_idx = np.argmin(a_times)

        print(f"\nStandard (C-tile-focused):")
        print(f"  Best performance: Tile size {results['tile_sizes'][standard_best_idx]} ({standard_times[standard_best_idx]:.6f}s)")
        print(f"  Baseline (largest tile): {standard_times[-1]:.6f}s")
        if standard_times[standard_best_idx] < standard_times[-1]:
            improvement = (standard_times[-1] / standard_times[standard_best_idx] - 1) * 100
            print(f"  Speedup vs baseline: {improvement:.1f}%")

        print(f"\nA-tile-focused:")
        print(f"  Best performance: Tile size {results['tile_sizes'][a_best_idx]} ({a_times[a_best_idx]:.6f}s)")
        print(f"  Baseline (largest tile): {a_times[-1]:.6f}s")
        if a_times[a_best_idx] < a_times[-1]:
            improvement = (a_times[-1] / a_times[a_best_idx] - 1) * 100
            print(f"  Speedup vs baseline: {improvement:.1f}%")

        # Compare best of each
        print(f"\nComparison:")
        if standard_times[standard_best_idx] < a_times[a_best_idx]:
            diff = (a_times[a_best_idx] / standard_times[standard_best_idx] - 1) * 100
            print(f"  Standard is {diff:.1f}% faster at optimal tile size")
        else:
            diff = (standard_times[standard_best_idx] / a_times[a_best_idx] - 1) * 100
            print(f"  A-tile-focused is {diff:.1f}% faster at optimal tile size")
    else:
        # Summary for single strategy
        best_idx = np.argmin(results['times'])
        best_time = results['times'][best_idx]
        baseline_time = results['times'][-1]

        print(f"\nBest performance: Tile size {results['tile_sizes'][best_idx]} ({best_time:.6f}s)")
        print(f"Baseline (largest tile): {baseline_time:.6f}s")
        if best_time < baseline_time:
            improvement = (baseline_time / best_time - 1) * 100
            print(f"Speedup vs baseline: {improvement:.1f}%")


if __name__ == '__main__':
    main()
