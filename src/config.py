import yaml
import argparse
from types import SimpleNamespace

class Config:
    def __init__(self, config_path=None):
        self._data = {}
        if config_path:
            with open(config_path, 'r', encoding='utf-8') as f:
                self._data = yaml.safe_load(f)
    
    def __getattr__(self, key):
        if key.startswith('_'):
            raise AttributeError(key)
        val = self._data.get(key)
        if isinstance(val, dict):
            return Config._dict_to_namespace(val)
        return val
    
    @staticmethod
    def _dict_to_namespace(d):
        ns = SimpleNamespace()
        for k, v in d.items():
            if isinstance(v, dict):
                setattr(ns, k, Config._dict_to_namespace(v))
            else:
                setattr(ns, k, v)
        return ns
    
    def to_dict(self):
        return self._data

def load_config(config_path):
    return Config(config_path)