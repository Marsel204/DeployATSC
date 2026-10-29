"""Process-local source handles. Credentials never form a public identity or digest."""
from __future__ import annotations

import re
from functools import lru_cache
import threading
from urllib.parse import unquote, urlsplit, parse_qsl
import uuid
from src.common.json_data import clone

_lock = threading.RLock()
_private_sources = {}
_handles = {}
_secret_fragments = set()
_registry_version = 0
_replacements = ()
_fragments = ()
_text_cache = {}
_KEY = re.compile(r'(?i)(password|passwd|token|secret|api[_-]?key|authorization|user[-_]?(?:id|pw)|username)')


def registry_version():
    with _lock:return _registry_version


def _changed():
    global _registry_version, _replacements, _fragments
    _registry_version += 1
    _replacements = tuple((source,_handles[source]) for source in sorted(_handles,key=len,reverse=True))
    _fragments = tuple(re.compile(r'(?<![A-Za-z0-9_])'+re.escape(secret)+r'(?![A-Za-z0-9_])')
                       for secret in sorted(_secret_fragments,key=len,reverse=True) if secret)
    _text_cache.clear()


@lru_cache(maxsize=4096)
def _secret_key(key):return bool(_KEY.fullmatch(key))
_URL = re.compile(r"[A-Za-z][A-Za-z0-9+.-]*://[^\s\"'<>]+")
_CREDENTIAL = re.compile(r'(?i)(?:password|passwd|token|secret|api[_-]?key|authorization|user[-_]?(?:id|pw)|username)\s*[=:]\s*[\"\']?([^\s\"\'&!]+)')


def private_source(value):
    value = str(value)
    return bool('://' in value or '!' in value or _CREDENTIAL.search(value))


def public_source(value):
    """Opaque random identity for remote sources/pipelines, with no hash of secrets."""
    value = str(value).strip()
    if not private_source(value):return value
    with _lock:
        if value in _handles:return _handles[value]
        handle = 'source_' + uuid.uuid4().hex
        try:
            _handles[value] = handle
            _private_sources[handle] = value
            for match in _URL.finditer(value):
                try:
                    parts = urlsplit(match.group())
                    _secret_fragments.update(unquote(part) for part in (parts.username, parts.password, parts.fragment) if part)
                    _secret_fragments.update(v for key,v in parse_qsl(parts.query) if v and re.search(r"(?i)token|key|secret|pass|auth|signature",key))
                except ValueError:pass
            _secret_fragments.update(m.group(1) for m in _CREDENTIAL.finditer(value))
        finally:
            _changed()
        return handle


def resolve_source(value):
    value = str(value).strip()
    with _lock:
        if value in _private_sources:return _private_sources[value]
    if re.fullmatch(r'source_[a-f0-9]{32}', value):
        raise ValueError('Public source handle is unavailable after restart; re-enter the source locally and review its binding')
    return value


def safe_text(value, *, fragments=True):
    text = str(value)
    with _lock:return _safe_text(text,bool(fragments))


def _safe_text(text, fragments):
    text = str(text)
    key = (text,fragments)
    cached = _text_cache.get(key)
    if cached is not None:return cached
    original = text
    for source,handle in _replacements:text = text.replace(source,handle)
    if fragments:
        for pattern in _fragments:text = pattern.sub('[redacted]',text)
    text = _CREDENTIAL.sub('[redacted credential]',_URL.sub('[redacted source]',text))
    if len(original)<=4096:
        if len(_text_cache)>=4096:_text_cache.clear()
        _text_cache[key] = text
    return text


def discover_sources(value):
    """Learn credentials for a whole view before rendering any free-text field."""
    if isinstance(value, dict):
        for key, item in value.items():
            if key in ('source', 'source_sys', 'source_acq') and item is not None:
                public_source(item)
            elif key == 'sources' and isinstance(item, dict):
                for source in item.values():public_source(source)
            elif isinstance(item,(dict,list,tuple)):discover_sources(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            if isinstance(item,(dict,list,tuple)):discover_sources(item)


def public_data(value):
    with _lock:
        discover_sources(value)
        return _public_data(value)


def _public_data(value):
    if isinstance(value, dict):
        result = {}
        for key,item in value.items():
            if _secret_key(str(key)):
                result[key] = '[redacted]'
            elif key in ('source','source_sys','source_acq') and item is not None:
                result[key] = public_source(item)
            elif key=="sources" and isinstance(item,dict):
                result[key]={role:public_source(source) for role,source in item.items()}
            elif isinstance(item,str):
                result[key]=_safe_text(item,str(key).endswith('error') or key in ('name','reference','msg'))
            elif isinstance(item,(dict,list,tuple)):result[key] = _public_data(item)
            else:result[key] = item
        return result
    if isinstance(value, (list,tuple)):
        return [_safe_text(item,False) if isinstance(item,str) else _public_data(item) if isinstance(item,(dict,list,tuple)) else item for item in value]
    # Credential fragments can equal valid domain enums (system / RED / unknown).
    # Only free text and error fields above undergo fragment redaction.
    return _safe_text(value,False) if isinstance(value,str) else value


class PublicDataCache:
    """Session-owned immutable subtrees; registry discovery always precedes cache reads."""
    def __init__(self):self.values={}

    def render(self,value,keys):
        with _lock:
            discover_sources(value)
            result=_public_data({key:item for key,item in value.items() if key not in keys})
            for key in keys:
                if key not in value:continue
                item=value[key];cached=self.values.get(key)
                if cached is None or cached[0] is not item or cached[1]!=_registry_version:
                    cached=(item,_registry_version,_public_data({key:item})[key]);self.values[key]=cached
                result[key]=clone(cached[2])
            return result
