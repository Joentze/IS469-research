"""Disk-based caching system for pipeline operations - result agnostic."""

import functools
import pickle
from pathlib import Path
from typing import Any, Callable, Optional


class DiskCache:
    """Universal disk-based cache for any pipeline operations."""

    def __init__(self, cache_dir: str = ".cache/"):
        """
        Initialize the disk cache.

        Args:
            cache_dir: Directory to store cache files
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_path(self, operation_name: str) -> Path:
        """Get the cache file path for a given operation name."""
        # Sanitize the name to be filesystem-safe
        safe_name = operation_name.replace("/", "_").replace("\\", "_").replace(":", "_")
        return self.cache_dir / f"{safe_name}.pkl"

    def _args_to_key(self, args: tuple, kwargs: dict) -> str:
        """
        Create a simple string key from function arguments.

        This is a lightweight identifier based on argument representation, not hashing.
        Works with any serializable types (documents, strings, numbers, etc.).
        """
        try:
            # Try to create a simple string representation
            arg_strs = []

            # Handle positional arguments
            for arg in args:
                if hasattr(arg, "__len__") and not isinstance(arg, (str, bytes)):
                    # For list-like objects, include length as proxy
                    try:
                        arg_strs.append(f"len_{len(arg)}")
                    except TypeError:
                        arg_strs.append(repr(arg)[:50])
                elif isinstance(arg, str):
                    # For strings, use first 50 chars
                    arg_strs.append(arg[:50].replace("\n", "_"))
                else:
                    # For other types, use string representation
                    arg_strs.append(str(arg)[:50])

            # Handle keyword arguments
            for key, val in sorted(kwargs.items()):
                if hasattr(val, "__len__") and not isinstance(val, (str, bytes)):
                    try:
                        arg_strs.append(f"{key}_len_{len(val)}")
                    except TypeError:
                        arg_strs.append(f"{key}_{repr(val)[:30]}")
                elif isinstance(val, str):
                    arg_strs.append(f"{key}_{val[:30].replace(chr(10), '_')}")
                else:
                    arg_strs.append(f"{key}_{str(val)[:30]}")

            return "_".join(arg_strs) if arg_strs else "noargs"
        except Exception:
            return "args_unknown"

    def get(self, operation_name: str, *args: Any, **kwargs: Any) -> Optional[Any]:
        """
        Retrieve cached result for a given operation and arguments.

        Args:
            operation_name: Name of the operation
            *args: Positional arguments to use as cache key
            **kwargs: Keyword arguments to use as cache key

        Returns:
            Cached result if available, None otherwise
        """
        cache_path = self._get_cache_path(operation_name)

        if not cache_path.exists():
            return None

        try:
            with open(cache_path, "rb") as f:
                cache_data = pickle.load(f)

            # Check if we have a cached result for these arguments
            arg_key = self._args_to_key(args, kwargs)
            if arg_key in cache_data:
                return cache_data[arg_key]
        except Exception as e:
            # Silently ignore cache read errors
            print(f"Warning: Cache read error for {operation_name}: {e}")

        return None

    def put(self, operation_name: str, result: Any, *args: Any, **kwargs: Any) -> None:
        """
        Store result in cache for a given operation and arguments.

        Args:
            operation_name: Name of the operation
            result: Result to cache (can be any type)
            *args: Positional arguments that produced this result
            **kwargs: Keyword arguments that produced this result
        """
        cache_path = self._get_cache_path(operation_name)

        try:
            # Load existing cache or create new one
            cache_data = {}
            if cache_path.exists():
                with open(cache_path, "rb") as f:
                    cache_data = pickle.load(f)

            # Add/update this result
            arg_key = self._args_to_key(args, kwargs)
            cache_data[arg_key] = result

            # Write back to disk
            with open(cache_path, "wb") as f:
                pickle.dump(cache_data, f)
        except Exception as e:
            print(f"Warning: Cache write error for {operation_name}: {e}")

    def clear(self, operation_name: Optional[str] = None) -> None:
        """
        Clear cache entries.

        Args:
            operation_name: Specific operation name to clear, or None to clear all
        """
        if operation_name is None:
            # Clear all cache files
            for file in self.cache_dir.glob("*.pkl"):
                try:
                    file.unlink()
                except Exception as e:
                    print(f"Warning: Failed to delete cache file {file}: {e}")
        else:
            # Clear specific cache file
            cache_path = self._get_cache_path(operation_name)
            if cache_path.exists():
                try:
                    cache_path.unlink()
                except Exception as e:
                    print(f"Warning: Failed to delete cache file {cache_path}: {e}")

    def info(self) -> dict:
        """Get information about the cache."""
        cache_files = list(self.cache_dir.glob("*.pkl"))
        total_size = sum(f.stat().st_size for f in cache_files)
        return {
            "cache_dir": str(self.cache_dir),
            "num_files": len(cache_files),
            "total_size_bytes": total_size,
            "files": [f.name for f in cache_files],
        }


# Global cache instance
_global_cache: Optional[DiskCache] = None


def get_global_cache(cache_dir: str = ".cache/") -> DiskCache:
    """
    Get or create the global cache instance.

    Args:
        cache_dir: Directory to store cache files

    Returns:
        Global DiskCache instance
    """
    global _global_cache
    if _global_cache is None:
        _global_cache = DiskCache(cache_dir)
    return _global_cache


def cache_result(
    operation_name: Optional[str] = None, cache_dir: str = ".cache/"
) -> Callable:
    """
    Decorator to cache function results based on arguments.

    Universal for any function with any return type.

    Usage:
        @cache_result("my_operation")
        def my_function(arg1, arg2):
            return expensive_result

        # First call: executes and caches
        result1 = my_function(arg1, arg2)

        # Second call with same args: returns from cache
        result2 = my_function(arg1, arg2)

    Args:
        operation_name: Name for the cache entry (defaults to function name)
        cache_dir: Directory to store cache files

    Returns:
        Decorator function
    """

    def decorator(func: Callable) -> Callable:
        # Use function name if no operation name provided
        name = operation_name or func.__name__

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            cache = get_global_cache(cache_dir)

            # Try to get from cache
            cached = cache.get(name, *args, **kwargs)
            if cached is not None:
                print(f"Cache hit: {name}")
                return cached

            # Call function and cache result
            result = func(*args, **kwargs)
            cache.put(name, result, *args, **kwargs)
            print(f"Cache stored: {name}")

            return result

        return wrapper

    return decorator


def clear_cache(operation_name: Optional[str] = None, cache_dir: str = ".cache/") -> None:
    """
    Clear cache entries.

    Args:
        operation_name: Specific operation name to clear, or None to clear all
        cache_dir: Cache directory
    """
    cache = get_global_cache(cache_dir)
    cache.clear(operation_name)


def cache_info(cache_dir: str = ".cache/") -> dict:
    """
    Get information about the cache.

    Args:
        cache_dir: Cache directory

    Returns:
        Cache statistics
    """
    cache = get_global_cache(cache_dir)
    return cache.info()
