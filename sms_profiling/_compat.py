"""Optional dependencies. tqdm only draws progress bars: without it everything runs, silently."""
try:
    from tqdm import tqdm
except ImportError:
    class tqdm:                                  # noqa: N801 - minimal stand-in with the same interface
        def __init__(self, *a, **k): pass
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def update(self, n=1): pass
