"""配置读写。

- JSON 落盘，容忍字段缺失/多余（便于版本升级）
- API KEY 使用 Windows DPAPI 加密（仅当前用户可解密），失败时退化为混淆 base64
"""
from __future__ import annotations

import base64
import dataclasses
import json
import logging
import os
import shutil
import typing
from pathlib import Path
from typing import Any, get_args, get_origin

from .models import (
    CONFIG_VERSION, AppConfig, BehaviorConfig, CaptureConfig, ModelEndpoint,
    OutputConfig, PersonalInfo,
)
from .paths import config_path, default_output_dir, ensure_dirs

log = logging.getLogger(__name__)

DPAPI_PREFIX = "dpapi:"
PLAIN_PREFIX = "plain:"

try:  # pragma: no cover - 平台相关
    import win32crypt  # type: ignore

    _HAS_DPAPI = True
except Exception:  # pragma: no cover
    win32crypt = None  # type: ignore
    _HAS_DPAPI = False


# ------------------------------------------------------------------ 密钥保护
def protect_secret(plain: str) -> str:
    """加密 API KEY 以便落盘。"""
    if not plain:
        return ""
    if plain.startswith((DPAPI_PREFIX, PLAIN_PREFIX)):
        return plain
    raw = plain.encode("utf-8")
    if _HAS_DPAPI:
        try:
            blob = win32crypt.CryptProtectData(raw, "AIStudent", None, None, None, 0)
            return DPAPI_PREFIX + base64.b64encode(blob).decode("ascii")
        except Exception as exc:  # pragma: no cover
            log.warning("DPAPI 加密失败，回退为本地混淆存储: %s", exc)
    return PLAIN_PREFIX + base64.b64encode(raw).decode("ascii")


def unprotect_secret(token: str) -> str:
    """还原 API KEY。"""
    if not token:
        return ""
    try:
        if token.startswith(DPAPI_PREFIX):
            blob = base64.b64decode(token[len(DPAPI_PREFIX):])
            if not _HAS_DPAPI:
                log.error("该配置由 Windows DPAPI 加密，当前环境无法解密")
                return ""
            _desc, data = win32crypt.CryptUnprotectData(blob, None, None, None, 0)
            return data.decode("utf-8")
        if token.startswith(PLAIN_PREFIX):
            return base64.b64decode(token[len(PLAIN_PREFIX):]).decode("utf-8")
        return token  # 兼容明文
    except Exception as exc:
        log.error("API KEY 解密失败: %s", exc)
        return ""


def mask_secret(value: str, keep: int = 4) -> str:
    if not value:
        return "(未设置)"
    if len(value) <= keep * 2:
        return "*" * len(value)
    return f"{value[:keep]}{'*' * 8}{value[-keep:]}"


# ------------------------------------------------------------------ 类型转换
def _convert(value: Any, tp: Any) -> Any:
    if value is None or tp is None:
        return value
    origin = get_origin(tp)
    if origin is typing.Union:  # Optional[...] / X | None
        args = [a for a in get_args(tp) if a is not type(None)]
        if value is None:
            return None
        return _convert(value, args[0]) if args else value
    if origin in (list, typing.List):
        args = get_args(tp)
        item_tp = args[0] if args else None
        if not isinstance(value, (list, tuple)):
            return []
        return [_convert(v, item_tp) for v in value]
    if origin in (dict, typing.Dict):
        args = get_args(tp)
        val_tp = args[1] if len(args) > 1 else None
        if not isinstance(value, dict):
            return {}
        return {str(k): _convert(v, val_tp) for k, v in value.items()}
    if origin is typing.Literal:
        allowed = get_args(tp)
        return value if value in allowed else (allowed[0] if allowed else value)
    if tp is bool:
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on", "是"}
        return bool(value)
    if tp is int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0
    if tp is float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0
    if tp is str:
        return "" if value is None else str(value)
    return value


def _dataclass_from_dict(cls: type, data: Any) -> Any:
    if not isinstance(data, dict) or not dataclasses.is_dataclass(cls):
        return cls()
    hints = typing.get_type_hints(cls)
    kwargs: dict[str, Any] = {}
    for f in dataclasses.fields(cls):
        if f.name in data:
            kwargs[f.name] = _convert(data[f.name], hints.get(f.name))
    try:
        return cls(**kwargs)
    except TypeError as exc:  # pragma: no cover
        log.warning("配置节 %s 解析失败(%s)，使用默认值", cls.__name__, exc)
        return cls()


# ------------------------------------------------------------------ 读写
def config_from_dict(data: dict[str, Any]) -> AppConfig:
    cfg = AppConfig(
        version=int(data.get("version") or CONFIG_VERSION),
        model=_dataclass_from_dict(ModelEndpoint, data.get("model")),
        person=_dataclass_from_dict(PersonalInfo, data.get("person")),
        capture=_dataclass_from_dict(CaptureConfig, data.get("capture")),
        output=_dataclass_from_dict(OutputConfig, data.get("output")),
        behavior=_dataclass_from_dict(BehaviorConfig, data.get("behavior")),
        prompt_overrides=dict(data.get("prompt_overrides") or {}),
        last_session_id=str(data.get("last_session_id") or ""),
    )
    cfg.model.api_key = unprotect_secret(cfg.model.api_key)
    cfg.model.asr_api_key = unprotect_secret(cfg.model.asr_api_key)
    cfg.model.vision_api_key = unprotect_secret(cfg.model.vision_api_key)
    if not cfg.output.output_dir:
        cfg.output.output_dir = str(default_output_dir())
    return cfg


def config_to_dict(cfg: AppConfig) -> dict[str, Any]:
    data = cfg.to_dict()
    data["version"] = CONFIG_VERSION
    data["model"]["api_key"] = protect_secret(cfg.model.api_key)
    data["model"]["asr_api_key"] = protect_secret(cfg.model.asr_api_key)
    data["model"]["vision_api_key"] = protect_secret(cfg.model.vision_api_key)
    return data


def load_config(path: Path | str | None = None) -> AppConfig:
    ensure_dirs()
    p = Path(path) if path else config_path()
    if not p.exists():
        cfg = AppConfig()
        cfg.output.output_dir = str(default_output_dir())
        return cfg
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except Exception as exc:
        log.error("配置文件损坏，已备份并使用默认配置: %s", exc)
        try:
            shutil.copy2(p, p.with_suffix(".corrupt.json"))
        except Exception:
            pass
        return AppConfig()
    return config_from_dict(raw)


def save_config(cfg: AppConfig, path: Path | str | None = None) -> Path:
    p = Path(path) if path else config_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    data = config_to_dict(cfg)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, p)
    try:  # 只保留一份可回溯备份
        shutil.copy2(p, p.with_suffix(".bak.json"))
    except Exception:
        pass
    return p


def export_config(cfg: AppConfig, target: Path | str, include_secrets: bool = False) -> Path:
    """导出配置（默认剔除密钥，便于分享/排错）。"""
    t = Path(target)
    data = cfg.to_dict()
    if not include_secrets:
        data["model"]["api_key"] = ""
        data["model"]["asr_api_key"] = ""
    t.parent.mkdir(parents=True, exist_ok=True)
    t.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return t
