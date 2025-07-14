"""
TinyTorch Profiler

A lightweight profiling utility for measuring performance of ML operations.
Following PyTorch's pattern with torch.profiler, this module provides
educational profiling tools for understanding ML performance.

Usage:
    from tinytorch.profiler import SimpleProfiler
    
    profiler = SimpleProfiler()
    result = profiler.profile(my_function, *args, **kwargs)
    profiler.print_result(result)

Similar to:
    torch.profiler.profile() - PyTorch's profiling context manager
    tf.profiler - TensorFlow's profiling utilities
    jax.profiler - JAX's profiling tools
"""

import time
import sys
import gc
import numpy as np
from typing import Callable, Dict, Any, Optional

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

try:
    import tracemalloc
    HAS_TRACEMALLOC = True
except ImportError:
    HAS_TRACEMALLOC = False

class SimpleProfiler:
    """
    Simple profiler for measuring individual function performance.
    
    Measures timing, memory usage, and other key metrics for a single function.
    Students collect multiple measurements and compare results themselves.
    """
    
    def __init__(self, track_memory: bool = True, track_cpu: bool = True):
        self.track_memory = track_memory and HAS_TRACEMALLOC
        self.track_cpu = track_cpu and HAS_PSUTIL
        
        if self.track_memory:
            tracemalloc.start()
    
    def _get_memory_info(self) -> Dict[str, Any]:
        """Get current memory information."""
        if not self.track_memory:
            return {}
        
        try:
            current, peak = tracemalloc.get_traced_memory()
            return {
                'current_memory_mb': current / 1024 / 1024,
                'peak_memory_mb': peak / 1024 / 1024
            }
        except:
            return {}
    
    def _get_cpu_info(self) -> Dict[str, Any]:
        """Get current CPU information."""
        if not self.track_cpu:
            return {}
        
        try:
            process = psutil.Process()
            return {
                'cpu_percent': process.cpu_percent(),
                'memory_percent': process.memory_percent(),
                'num_threads': process.num_threads()
            }
        except:
            return {}
    
    def _get_array_info(self, result: Any) -> Dict[str, Any]:
        """Get information about numpy arrays."""
        if not isinstance(result, np.ndarray):
            return {}
        
        return {
            'result_shape': result.shape,
            'result_dtype': str(result.dtype),
            'result_size_mb': result.nbytes / 1024 / 1024,
            'result_elements': result.size
        }
    
    def profile(self, func: Callable, *args, name: Optional[str] = None, warmup: bool = True, **kwargs) -> Dict[str, Any]:
        """
        Profile a single function execution with comprehensive metrics.
        
        Args:
            func: Function to profile
            *args: Arguments to pass to function
            name: Optional name for the function (defaults to func.__name__)
            warmup: Whether to do a warmup run (recommended for fair timing)
            **kwargs: Keyword arguments to pass to function
            
        Returns:
            Dictionary with comprehensive performance metrics
            
        Example:
            profiler = SimpleProfiler()
            result = profiler.profile(my_function, arg1, arg2, name="My Function")
            print(f"Time: {result['wall_time']:.4f}s")
            print(f"Memory: {result['memory_delta_mb']:.2f}MB")
        """
        func_name = name or func.__name__
        
        # Reset memory tracking
        if self.track_memory:
            tracemalloc.clear_traces()
        
        # Warm up (important for fair comparison)
        if warmup:
            try:
                warmup_result = func(*args, **kwargs)
                del warmup_result
            except:
                pass
        
        # Force garbage collection for clean measurement
        gc.collect()
        
        # Get baseline measurements
        memory_before = self._get_memory_info()
        cpu_before = self._get_cpu_info()
        
        # Time the actual execution
        start_time = time.time()
        start_cpu_time = time.process_time()
        
        result = func(*args, **kwargs)
        
        end_time = time.time()
        end_cpu_time = time.process_time()
        
        # Get post-execution measurements
        memory_after = self._get_memory_info()
        cpu_after = self._get_cpu_info()
        
        # Calculate metrics
        wall_time = end_time - start_time
        cpu_time = end_cpu_time - start_cpu_time
        
        profile_result = {
            'name': func_name,
            'wall_time': wall_time,
            'cpu_time': cpu_time,
            'cpu_efficiency': (cpu_time / wall_time) if wall_time > 0 else 0,
            'result': result
        }
        
        # Add memory metrics
        if self.track_memory and memory_before and memory_after:
            profile_result.update({
                'memory_before_mb': memory_before.get('current_memory_mb', 0),
                'memory_after_mb': memory_after.get('current_memory_mb', 0),
                'peak_memory_mb': memory_after.get('peak_memory_mb', 0),
                'memory_delta_mb': memory_after.get('current_memory_mb', 0) - memory_before.get('current_memory_mb', 0)
            })
        
        # Add CPU metrics
        if self.track_cpu and cpu_after:
            profile_result.update({
                'cpu_percent': cpu_after.get('cpu_percent', 0),
                'memory_percent': cpu_after.get('memory_percent', 0),
                'num_threads': cpu_after.get('num_threads', 1)
            })
        
        # Add array information
        profile_result.update(self._get_array_info(result))
        
        return profile_result
    
    def print_result(self, profile_result: Dict[str, Any], show_details: bool = False) -> None:
        """
        Print profiling results in a readable format.
        
        Args:
            profile_result: Result from profile() method
            show_details: Whether to show detailed metrics
        """
        name = profile_result['name']
        wall_time = profile_result['wall_time']
        
        print(f"📊 {name}: {wall_time:.4f}s")
        
        if show_details:
            if 'memory_delta_mb' in profile_result:
                print(f"   💾 Memory: {profile_result['memory_delta_mb']:.2f}MB delta, {profile_result['peak_memory_mb']:.2f}MB peak")
            if 'result_size_mb' in profile_result:
                print(f"   🔢 Output: {profile_result['result_shape']} ({profile_result['result_size_mb']:.2f}MB)")
            if 'cpu_efficiency' in profile_result:
                print(f"   ⚡ CPU: {profile_result['cpu_efficiency']:.2f} efficiency")
    
    def get_capabilities(self) -> Dict[str, bool]:
        """Get information about profiler capabilities."""
        return {
            'memory_tracking': self.track_memory,
            'cpu_tracking': self.track_cpu,
            'has_psutil': HAS_PSUTIL,
            'has_tracemalloc': HAS_TRACEMALLOC
        }

# Convenience function for quick profiling
def profile_function(func: Callable, *args, name: Optional[str] = None, 
                     show_details: bool = False, **kwargs) -> Dict[str, Any]:
    """
    Quick profiling of a single function.
    
    Args:
        func: Function to profile
        *args: Arguments to pass to function
        name: Optional name for the function
        show_details: Whether to print detailed metrics
        **kwargs: Keyword arguments to pass to function
        
    Returns:
        Dictionary with profiling results
        
    Example:
        result = profile_function(my_matmul, A, B, name="Custom MatMul", show_details=True)
        print(f"Execution time: {result['wall_time']:.4f}s")
    """
    profiler = SimpleProfiler(track_memory=True, track_cpu=True)
    result = profiler.profile(func, *args, name=name, **kwargs)
    
    if show_details:
        profiler.print_result(result, show_details=True)
    
    return result 