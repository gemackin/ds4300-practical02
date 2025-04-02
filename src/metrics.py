import time, psutil


# Decorator function for tracking time
def track_time(func):
    def inner(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        return result, time.time() - start
    return inner


# Decorator function for tracking memory usage
def track_memory(func):
    def inner(*args, **kwargs):
        proc = psutil.Process()
        before = proc.memory_info().rss
        result = func   (*args, **kwargs)
        after = proc.memory_info().rss
        return result, (after - before) / 1024**2
    return inner


# Decorator function for tracking multiple metrics
# Currently only time is implemented, but this allows for more
def track(func, _metrics=None):
    def inner(*args, **kwargs):
        result, duration = track_time(func)(*args, **kwargs)
        metrics = [duration] # Could have multiple metrics
        if metrics is None: return result, metrics
        while len(_metrics) > len(metrics): _metrics.append(0)
        for i in range(len(metrics)): _metrics[i] += metrics[i]
        return result # Void function if you supply a list
    return inner