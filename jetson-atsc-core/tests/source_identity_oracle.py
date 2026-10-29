"""Process-local source handles. Credentials never form a public identity or digest."""
from __future__ import annotations

import re
import threading
from urllib.parse import unquote, urlsplit, parse_qsl
import uuid

_lock = threading.RLock()
_private_sources = {}
_handles = {}
_secret_fragments = set()
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
        _handles[value] = handle
        _private_sources[handle] = value
        for match in _URL.finditer(value):
            try:
                parts = urlsplit(match.group())
                _secret_fragments.update(unquote(part) for part in (parts.username, parts.password, parts.fragment) if part)
                _secret_fragments.update(v for key,v in parse_qsl(parts.query) if v and re.search(r"(?i)token|key|secret|pass|auth|signature",key))
            except ValueError:pass
        _secret_fragments.update(m.group(1) for m in _CREDENTIAL.finditer(value))
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
    with _lock:
        for source in sorted(_handles, key=len, reverse=True):text = text.replace(source, _handles[source])
        if fragments:
            for secret in sorted(_secret_fragments, key=len, reverse=True):
                if secret:text = re.sub(r'(?<![A-Za-z0-9_])'+re.escape(secret)+r'(?![A-Za-z0-9_])', '[redacted]', text)
    text = _URL.sub('[redacted source]', text)
    return _CREDENTIAL.sub('[redacted credential]', text)


def discover_sources(value):
    """Learn credentials for a whole view before rendering any free-text field."""
    if isinstance(value, dict):
        for key, item in value.items():
            if key in ('source', 'source_sys', 'source_acq') and item is not None:
                public_source(item)
            elif key == 'sources' and isinstance(item, dict):
                for source in item.values():public_source(source)
            else:discover_sources(item)
    elif isinstance(value, (list, tuple)):
        for item in value:discover_sources(item)


def public_data(value):
    discover_sources(value)
    return _public_data(value)


def _public_data(value):
    if isinstance(value, dict):
        result = {}
        for key,item in value.items():
            if re.fullmatch(r'(?i)(password|passwd|token|secret|api[_-]?key|authorization|user[-_]?(?:id|pw)|username)',str(key)):
                result[key] = '[redacted]'
            elif key in ('source','source_sys','source_acq') and item is not None:
                result[key] = public_source(item)
            elif key=="sources" and isinstance(item,dict):
                result[key]={role:public_source(source) for role,source in item.items()}
            elif isinstance(item,str) and (str(key).endswith('error') or key in ('name','reference','msg')):
                result[key]=safe_text(item)
            else:result[key] = _public_data(item)
        return result
    if isinstance(value, (list,tuple)):return [_public_data(item) for item in value]
    # Credential fragments can equal valid domain enums (system / RED / unknown).
    # Only free text and error fields above undergo fragment redaction.
    return safe_text(value,fragments=False) if isinstance(value,str) else value
