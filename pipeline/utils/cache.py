"""
Cache Manager Utility
"""

import logging
import pickle
import hashlib
from pathlib import Path
from typing import Any, Optional
import os


class CacheManager:
    """Cache manager for pipeline results"""
    
    def __init__(self, cache_dir: str, enable_caching: bool = True):
        """Initialize cache manager"""
        self.cache_dir = Path(cache_dir)
        self.enable_caching = enable_caching
        self.logger = logging.getLogger(__name__)
        
        if self.enable_caching:
            self.cache_dir.mkdir(exist_ok=True)
            self.logger.info(f"Cache manager initialized at: {self.cache_dir}")
    
    def get(self, key: str) -> Optional[Any]:
        """Get cached result by key"""
        if not self.enable_caching:
            return None
        
        try:
            cache_file = self.cache_dir / f"{key}.pkl"
            if cache_file.exists():
                with open(cache_file, 'rb') as f:
                    result = pickle.load(f)
                self.logger.debug(f"Cache hit for key: {key}")
                return result
            else:
                self.logger.debug(f"Cache miss for key: {key}")
                return None
                
        except Exception as e:
            self.logger.warning(f"Failed to load cache for key {key}: {str(e)}")
            return None
    
    def set(self, key: str, value: Any) -> bool:
        """Set cached result by key"""
        if not self.enable_caching:
            return False
        
        try:
            cache_file = self.cache_dir / f"{key}.pkl"
            with open(cache_file, 'wb') as f:
                pickle.dump(value, f)
            self.logger.debug(f"Cached result for key: {key}")
            return True
            
        except Exception as e:
            self.logger.warning(f"Failed to cache result for key {key}: {str(e)}")
            return False
    
    def clear(self) -> bool:
        """Clear all cached results"""
        if not self.enable_caching:
            return True
        
        try:
            for cache_file in self.cache_dir.glob("*.pkl"):
                cache_file.unlink()
            self.logger.info("Cache cleared successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to clear cache: {str(e)}")
            return False
    
    def get_cache_size(self) -> int:
        """Get the number of cached items"""
        if not self.enable_caching:
            return 0
        
        try:
            return len(list(self.cache_dir.glob("*.pkl")))
        except Exception:
            return 0
    
    def get_cache_info(self) -> dict:
        """Get cache information"""
        return {
            "enabled": self.enable_caching,
            "cache_dir": str(self.cache_dir),
            "cache_size": self.get_cache_size(),
            "cache_size_bytes": self._get_cache_size_bytes()
        }
    
    def _get_cache_size_bytes(self) -> int:
        """Get cache size in bytes"""
        if not self.enable_caching:
            return 0
        
        try:
            total_size = 0
            for cache_file in self.cache_dir.glob("*.pkl"):
                total_size += cache_file.stat().st_size
            return total_size
        except Exception:
            return 0
