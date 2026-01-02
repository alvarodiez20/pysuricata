#!/usr/bin/env python3
"""
Micro-benchmarks for PySuricata performance analysis.
Tests individual components in isolation to measure time complexity.
"""

import time
import numpy as np
import pandas as pd
from typing import List, Dict, Any
import matplotlib.pyplot as plt
from pathlib import Path

# Import PySuricata components
from pysuricata.accumulators.sketches import KMV, MisraGries, ReservoirSampler
from pysuricata.accumulators.numeric import NumericAccumulator
from pysuricata.accumulators.categorical import CategoricalAccumulator
from pysuricata.accumulators.boolean import BooleanAccumulator
from pysuricata.accumulators.datetime import DatetimeAccumulator
from pysuricata.compute.consume import consume_chunk_pandas
from pysuricata.compute.core.types import ColumnKinds
from pysuricata.config import EngineConfig

def benchmark_kmv_insertions():
    """Benchmark KMV insertion performance with different k values."""
    print("🔬 Benchmarking KMV Insertion Performance")
    print("=" * 50)
    
    k_values = [256, 512, 1024, 2048, 4096]
    data_sizes = [1000, 5000, 10000, 50000, 100000]
    
    results = {}
    
    for k in k_values:
        results[k] = {}
        print(f"\n📊 Testing KMV with k={k}")
        
        for size in data_sizes:
            # Generate test data
            data = [f"value_{i}" for i in range(size)]
            
            # Benchmark KMV
            kmv = KMV(k=k)
            
            start_time = time.perf_counter()
            for value in data:
                kmv.add(value)
            end_time = time.perf_counter()
            
            total_time = end_time - start_time
            ops_per_sec = size / total_time
            
            results[k][size] = {
                'total_time': total_time,
                'ops_per_sec': ops_per_sec,
                'time_per_op': total_time / size
            }
            
            print(f"   Size {size:,}: {total_time:.4f}s ({ops_per_sec:,.0f} ops/sec)")
    
    return results

def benchmark_categorical_processing():
    """Benchmark categorical accumulator processing methods."""
    print("\n🔬 Benchmarking Categorical Processing")
    print("=" * 50)
    
    # Test data with different cardinalities
    cardinalities = [10, 100, 1000, 5000]
    chunk_sizes = [1000, 5000, 10000, 50000]
    
    results = {}
    
    for cardinality in cardinalities:
        results[cardinality] = {}
        print(f"\n📊 Testing cardinality={cardinality}")
        
        for chunk_size in chunk_sizes:
            # Generate test data
            data = [f"cat_{i % cardinality}" for i in range(chunk_size)]
            
            # Test current implementation (sequential)
            acc = CategoricalAccumulator("test_col")
            
            start_time = time.perf_counter()
            acc.update(data)
            end_time = time.perf_counter()
            
            sequential_time = end_time - start_time
            
            # Test vectorized approach (simulated)
            start_time = time.perf_counter()
            # Simulate vectorized processing
            pd.Series(data).value_counts()
            end_time = time.perf_counter()
            
            vectorized_time = end_time - start_time
            
            results[cardinality][chunk_size] = {
                'sequential_time': sequential_time,
                'vectorized_time': vectorized_time,
                'speedup': sequential_time / vectorized_time if vectorized_time > 0 else 1
            }
            
            print(f"   Chunk {chunk_size:,}: Sequential {sequential_time:.4f}s, "
                  f"Vectorized {vectorized_time:.4f}s, Speedup {sequential_time/vectorized_time:.2f}x")
    
    return results

def benchmark_memory_usage_calls():
    """Benchmark pandas memory_usage() call overhead."""
    print("\n🔬 Benchmarking Memory Usage Call Overhead")
    print("=" * 50)
    
    sizes = [1000, 5000, 10000, 50000, 100000]
    results = {}
    
    for size in sizes:
        # Create test DataFrame
        df = pd.DataFrame({
            'numeric': np.random.randn(size),
            'categorical': [f"cat_{i % 100}" for i in range(size)],
            'boolean': [i % 2 == 0 for i in range(size)]
        })
        
        # Benchmark memory_usage calls
        times = []
        
        for _ in range(10):  # Run multiple times for average
            start_time = time.perf_counter()
            for col in df.columns:
                df[col].memory_usage(deep=True)
            end_time = time.perf_counter()
            times.append(end_time - start_time)
        
        avg_time = np.mean(times)
        results[size] = avg_time
        
        print(f"   Size {size:,}: {avg_time:.4f}s per call")
    
    return results

def benchmark_finalization_sorting():
    """Benchmark finalization sorting operations."""
    print("\n🔬 Benchmarking Finalization Sorting")
    print("=" * 50)
    
    sample_sizes = [1000, 5000, 10000, 20000, 50000]
    results = {}
    
    for size in sample_sizes:
        # Generate test data
        data = np.random.randn(size)
        
        # Benchmark numpy sort
        start_time = time.perf_counter()
        sorted_data = np.sort(data)
        end_time = time.perf_counter()
        
        sort_time = end_time - start_time
        
        # Benchmark quantile computation
        start_time = time.perf_counter()
        quantiles = np.quantile(data, [0.25, 0.5, 0.75])
        end_time = time.perf_counter()
        
        quantile_time = end_time - start_time
        
        results[size] = {
            'sort_time': sort_time,
            'quantile_time': quantile_time,
            'total_time': sort_time + quantile_time
        }
        
        print(f"   Size {size:,}: Sort {sort_time:.4f}s, Quantiles {quantile_time:.4f}s")
    
    return results

def benchmark_chunk_processing():
    """Benchmark chunk processing with different configurations."""
    print("\n🔬 Benchmarking Chunk Processing")
    print("=" * 50)
    
    chunk_sizes = [10000, 25000, 50000, 100000]
    column_counts = [5, 10, 15, 20]
    
    results = {}
    
    for chunk_size in chunk_sizes:
        results[chunk_size] = {}
        print(f"\n📊 Testing chunk_size={chunk_size}")
        
        for col_count in column_counts:
            # Generate test DataFrame
            data = {}
            for i in range(col_count):
                if i % 4 == 0:
                    data[f'num_{i}'] = np.random.randn(chunk_size)
                elif i % 4 == 1:
                    data[f'cat_{i}'] = [f"cat_{j % 100}" for j in range(chunk_size)]
                elif i % 4 == 2:
                    data[f'bool_{i}'] = [j % 2 == 0 for j in range(chunk_size)]
                else:
                    data[f'dt_{i}'] = pd.date_range('2020-01-01', periods=chunk_size, freq='1min')
            
            df = pd.DataFrame(data)
            
            # Create accumulators
            kinds = ColumnKinds()
            for col in df.columns:
                if col.startswith('num_'):
                    kinds[col] = 'numeric'
                elif col.startswith('cat_'):
                    kinds[col] = 'categorical'
                elif col.startswith('bool_'):
                    kinds[col] = 'boolean'
                else:
                    kinds[col] = 'datetime'
            
            accs = {}
            for col, kind in kinds.items():
                if kind == 'numeric':
                    accs[col] = NumericAccumulator(col)
                elif kind == 'categorical':
                    accs[col] = CategoricalAccumulator(col)
                elif kind == 'boolean':
                    accs[col] = BooleanAccumulator(col)
                else:
                    accs[col] = DatetimeAccumulator(col)
            
            # Benchmark chunk processing
            start_time = time.perf_counter()
            consume_chunk_pandas(df, accs, kinds)
            end_time = time.perf_counter()
            
            processing_time = end_time - start_time
            
            results[chunk_size][col_count] = {
                'processing_time': processing_time,
                'rows_per_sec': chunk_size / processing_time,
                'cols_per_sec': col_count / processing_time
            }
            
            print(f"   {col_count} cols: {processing_time:.4f}s "
                  f"({chunk_size/processing_time:,.0f} rows/sec)")
    
    return results

def create_performance_visualizations(results: Dict[str, Any]):
    """Create visualizations of benchmark results."""
    print("\n📊 Creating Performance Visualizations")
    
    try:
        # KMV Performance Plot
        if 'kmv' in results:
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            
            for k, data in results['kmv'].items():
                sizes = list(data.keys())
                times = [data[size]['time_per_op'] * 1000000 for size in sizes]  # Convert to microseconds
                ax.plot(sizes, times, marker='o', label=f'k={k}')
            
            ax.set_xlabel('Data Size')
            ax.set_ylabel('Time per Operation (μs)')
            ax.set_title('KMV Insertion Performance')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig('kmv_performance.png', dpi=150, bbox_inches='tight')
            plt.close()
        
        # Categorical Processing Plot
        if 'categorical' in results:
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            
            for cardinality, data in results['categorical'].items():
                sizes = list(data.keys())
                speedups = [data[size]['speedup'] for size in sizes]
                ax.plot(sizes, speedups, marker='o', label=f'Cardinality={cardinality}')
            
            ax.set_xlabel('Chunk Size')
            ax.set_ylabel('Vectorization Speedup')
            ax.set_title('Categorical Processing Speedup')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig('categorical_speedup.png', dpi=150, bbox_inches='tight')
            plt.close()
        
        print("✅ Visualizations saved as PNG files")
        
    except ImportError:
        print("❌ Matplotlib not available - skipping visualizations")

def run_all_benchmarks():
    """Run all micro-benchmarks and return results."""
    print("🚀 Starting PySuricata Performance Micro-Benchmarks")
    print("=" * 60)
    
    results = {}
    
    # Run individual benchmarks
    results['kmv'] = benchmark_kmv_insertions()
    results['categorical'] = benchmark_categorical_processing()
    results['memory_usage'] = benchmark_memory_usage_calls()
    results['finalization'] = benchmark_finalization_sorting()
    results['chunk_processing'] = benchmark_chunk_processing()
    
    # Create visualizations
    create_performance_visualizations(results)
    
    # Summary
    print("\n🎯 BENCHMARK SUMMARY")
    print("=" * 60)
    
    print("\n📊 Key Findings:")
    
    # KMV findings
    if 'kmv' in results:
        kmv_data = results['kmv'][2048]  # Use k=2048 as reference
        largest_size = max(kmv_data.keys())
        time_per_op = kmv_data[largest_size]['time_per_op']
        print(f"• KMV (k=2048): {time_per_op*1000000:.2f} μs per operation")
    
    # Categorical findings
    if 'categorical' in results:
        cat_data = results['categorical'][1000]  # Use cardinality=1000
        largest_size = max(cat_data.keys())
        speedup = cat_data[largest_size]['speedup']
        print(f"• Categorical vectorization: {speedup:.2f}x speedup potential")
    
    # Memory usage findings
    if 'memory_usage' in results:
        largest_size = max(results['memory_usage'].keys())
        mem_time = results['memory_usage'][largest_size]
        print(f"• Memory usage calls: {mem_time:.4f}s overhead per call")
    
    # Finalization findings
    if 'finalization' in results:
        final_data = results['finalization'][20000]  # Use 20K sample size
        total_time = final_data['total_time']
        print(f"• Finalization (20K samples): {total_time:.4f}s per column")
    
    # Chunk processing findings
    if 'chunk_processing' in results:
        chunk_data = results['chunk_processing'][50000]  # Use 50K chunk size
        largest_cols = max(chunk_data.keys())
        rows_per_sec = chunk_data[largest_cols]['rows_per_sec']
        print(f"• Chunk processing: {rows_per_sec:,.0f} rows/sec")
    
    return results

if __name__ == "__main__":
    results = run_all_benchmarks()
    
    print(f"\n✅ All benchmarks completed!")
    print(f"📁 Results saved in current directory")
    print(f"📊 Use these results to identify optimization opportunities")
