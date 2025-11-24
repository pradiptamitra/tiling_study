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
            # L1 instruction cache
            result = subprocess.run(['sysctl', 'hw.l1icachesize'],
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

    def _naive_multiply(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """Standard matrix multiplication (ijk order)."""
        n = A.shape[0]
        C = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    C[i, j] += A[i, k] * B[k, j]
        return C

    def _tiled_multiply(self, A: np.ndarray, B: np.ndarray, tile_size: int) -> np.ndarray:
        """
        Tiled matrix multiplication with specified tile size.

        Multiplies blocks of size tile_size x tile_size to improve cache locality.
        """
        n = A.shape[0]
        C = np.zeros((n, n))

        # Process matrix in tiles
        for i_tile in range(0, n, tile_size):
            for j_tile in range(0, n, tile_size):
                for k_tile in range(0, n, tile_size):
                    # Determine actual tile boundaries (handle edge cases)
                    i_end = min(i_tile + tile_size, n)
                    j_end = min(j_tile + tile_size, n)
                    k_end = min(k_tile + tile_size, n)

                    # Multiply tiles
                    C[i_tile:i_end, j_tile:j_end] += \
                        A[i_tile:i_end, k_tile:k_end] @ B[k_tile:k_end, j_tile:j_end]

        return C

    def run_experiment(self, tile_sizes: List[int], num_runs: int = 3) -> Dict:
        """
        Run tiling experiment with various tile sizes.

        Args:
            tile_sizes: List of tile sizes to test
            num_runs: Number of runs per tile size (for averaging)

        Returns:
            Dictionary with results
        """
        print(f"\n{'='*70}")
        print(f"Matrix Tiling Performance Study")
        print(f"{'='*70}")
        print(f"Matrix size: {self.matrix_size} x {self.matrix_size}")
        print(f"Number of runs per tile size: {num_runs}")
        print(f"Total operations per multiplication: {self.matrix_size**3 * 2 / 1e9:.2f} billion FLOPs")
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

        print(f"\n{'='*70}")
        print(f"Performance Results:")
        print(f"{'='*70}")
        print(f"{'Tile Size':<15} {'Avg Time (s)':<15} {'GFLOPs/s':<15} {'Cache-Optimal':<20}")
        print(f"{'-'*70}")

        results = {
            'tile_sizes': [],
            'times': [],
            'gflops': [],
            'time_std': []
        }

        # Create test matrices
        A = np.random.randn(self.matrix_size, self.matrix_size).astype(np.float32)
        B = np.random.randn(self.matrix_size, self.matrix_size).astype(np.float32)

        # Test each tile size
        for tile_size in tile_sizes:
            times = []

            for run in range(num_runs):
                start = time.perf_counter()
                C = self._tiled_multiply(A, B, tile_size)
                end = time.perf_counter()
                times.append(end - start)

            avg_time = np.mean(times)
            std_time = np.std(times)
            gflops = (self.matrix_size ** 3 * 2 / 1e9) / avg_time

            # Estimate cache optimality
            cache_optimal = self._estimate_cache_optimality(tile_size)

            results['tile_sizes'].append(tile_size)
            results['times'].append(avg_time)
            results['time_std'].append(std_time)
            results['gflops'].append(gflops)

            print(f"{tile_size:<15} {avg_time:<15.6f} {gflops:<15.2f} {cache_optimal:<20}")

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

    def plot_results(self, output_file: str = 'tiling_performance.png'):
        """
        Generate visualization of tiling performance.

        Args:
            output_file: Path to save the plot
        """
        if not self.results:
            print("No results to plot. Run experiment first.")
            return

        tile_sizes = self.results['tile_sizes']
        times = self.results['times']
        gflops = self.results['gflops']
        time_std = self.results['time_std']

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        # Plot 1: Execution time vs tile size
        ax1.errorbar(tile_sizes, times, yerr=time_std, fmt='o-', linewidth=2,
                     markersize=8, capsize=5, label='Execution Time')
        ax1.set_xlabel('Tile Size (elements)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Time (seconds)', fontsize=12, fontweight='bold')
        ax1.set_title(f'Matrix Multiplication Time vs Tile Size\n(Matrix: {self.matrix_size}x{self.matrix_size})',
                     fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.set_xscale('log')

        # Add cache size markers
        self._add_cache_markers(ax1, 'time')

        # Plot 2: GFLOPs vs tile size
        ax2.plot(tile_sizes, gflops, 'o-', linewidth=2, markersize=8,
                color='green', label='GFLOPs/s')
        ax2.set_xlabel('Tile Size (elements)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Performance (GFLOPs/s)', fontsize=12, fontweight='bold')
        ax2.set_title(f'Peak Performance vs Tile Size',
                     fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.set_xscale('log')

        # Add cache size markers
        self._add_cache_markers(ax2, 'performance')

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
                # Estimate tile size that fits in cache (assuming 3 tiles of tile_size^2: A, B, C)
                optimal_tile = int(np.sqrt(cache_bytes / 12))

                ax.axvline(x=optimal_tile, color=colors[level], linestyle='--',
                          linewidth=2, alpha=0.7, label=f'{level} optimal (~{optimal_tile})')

        ax.legend(loc='best', fontsize=10)


def main():
    """Main entry point."""
    # Matrix size - adjust based on your system's memory
    MATRIX_SIZE = 512  # 512x512 matrix (can increase for larger systems)

    # Tile sizes to test
    # Start small and go up to the full matrix size
    tile_sizes = [4, 8, 16, 32, 64, 128, 256, 512]

    # Create and run study
    study = MatrixTilingStudy(matrix_size=MATRIX_SIZE)
    results = study.run_experiment(tile_sizes, num_runs=3)

    # Generate visualization
    study.plot_results('tiling_performance.png')

    # Print summary
    print(f"\n{'='*70}")
    print(f"Summary:")
    print(f"{'='*70}")
    best_idx = np.argmin(results['times'])
    print(f"Best performance: Tile size {results['tile_sizes'][best_idx]} "
          f"({results['gflops'][best_idx]:.2f} GFLOPs/s)")
    print(f"Baseline (tile_size=matrix_size): {results['gflops'][-1]:.2f} GFLOPs/s")
    if results['gflops'][best_idx] > results['gflops'][-1]:
        improvement = (results['gflops'][best_idx] / results['gflops'][-1] - 1) * 100
        print(f"Improvement over naive: {improvement:.1f}%")


if __name__ == '__main__':
    main()
