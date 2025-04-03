import time, psutil, numpy as np


# Decorator function for tracking time
def track_time(func):
    def inner(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        return result, time.time() - start
    return inner


# Decorator function for tracking memory usage in GB
# Currently implents resident set size, virtual memory size, and peak working set size
def track_memory(func):
    def inner(*args, **kwargs):
        proc = psutil.Process()
        pmem = proc.memory_info() # Querying process memory
        before = np.array([pmem.rss, pmem.vms, pmem.peak_wset])
        result = func(*args, **kwargs)
        pmem = proc.memory_info() # Querying process memory again
        after = np.array([pmem.rss, pmem.vms, pmem.peak_wset])
        return result, list((after - before) / 1e6)
    return inner


# Decorator function for tracking multiple metrics
# Currently only time is implemented, but this allows for more
def track(func, _metrics=None):
    def inner(*args, **kwargs):
        result, duration = track_time(func)(*args, **kwargs)
        metrics = [duration] # Could have multiple metrics
        if _metrics is None: return result, metrics
        while len(_metrics) < len(metrics): _metrics.append(0)
        for i in range(len(metrics)): _metrics[i] += metrics[i]
        return result # Void function if you supply a list
    return inner