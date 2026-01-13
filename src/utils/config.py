from __future__ import annotations

from pathlib import Path

def _deep_update(base: dict, new: dict) -> dict:
    for k, v in new.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            base[k] = _deep_update(base[k], v)
        else:
            base[k] = v
    return base

class CfgNode(dict):
    def __getattr__(self, item):
        v = self.get(item)
        if isinstance(v, dict) and not isinstance(v, CfgNode):
            v = CfgNode(v)
            self[item] = v
        return v

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def load_config(*yaml_paths: str | Path) -> CfgNode:
    import yaml

    cfg: dict = {}
    for p in yaml_paths:
        p = Path(p)
        if not p.exists():
            raise FileNotFoundError(f"Config file not found: {p}")
        data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        if not isinstance(data, dict):
            raise ValueError(f"Config must be a dict. Got {type(data)} in {p}")
        cfg = _deep_update(cfg, data)
    return CfgNode(cfg)