"""Shared helpers for SpikeRuGPT data preparation scripts."""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from string import Formatter
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Tuple

try:
    import yaml
except ImportError as exc:  # pragma: no cover - import guard for CLI use
    raise SystemExit("PyYAML is required. Install requirements_data.txt first.") from exc


CYRILLIC_RE = re.compile(r"[\u0400-\u04FF]")
LETTER_RE = re.compile(r"[^\W\d_]", re.UNICODE)
URL_RE = re.compile(r"https?://|www\.", re.IGNORECASE)
SPACE_RE = re.compile(r"[ \t\f\v]+")
MANY_NEWLINES_RE = re.compile(r"\n{4,}")


def load_plan(path: str | os.PathLike[str]) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8-sig") as f:
        plan = yaml.safe_load(f)
    if not isinstance(plan, dict):
        raise ValueError(f"Expected mapping in {path}")
    return plan


def ensure_dir(path: str | os.PathLike[str]) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_json(path: str | os.PathLike[str], payload: Mapping[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


def append_jsonl(path: str | os.PathLike[str], payload: Mapping[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False, sort_keys=True))
        f.write("\n")


def safe_slug(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9_.-]+", "-", value)
    return value.strip("-") or "source"


def source_key(source: Mapping[str, Any]) -> str:
    dataset = source.get("dataset", "")
    config = source.get("config", "")
    split = source.get("split", "")
    return "::".join([str(dataset), str(config), str(split)])


def source_name(source: Mapping[str, Any]) -> str:
    return str(source.get("name") or safe_slug(source_key(source)))


def is_enabled(source: Mapping[str, Any]) -> bool:
    return bool(source.get("enabled", True))


def iter_stage_sources(plan: Mapping[str, Any], stage: str) -> Iterator[Dict[str, Any]]:
    section = plan.get(stage)
    if not section:
        return
    if isinstance(section, Mapping) and isinstance(section.get("sources"), list):
        for source in section["sources"]:
            yield dict(source)
        return
    if isinstance(section, list):
        for item in section:
            for source in item.get("sources", []):
                merged = dict(source)
                merged.setdefault("validation_name", item.get("name"))
                yield merged


def collect_sources(plan: Mapping[str, Any], stages: Iterable[str]) -> List[Dict[str, Any]]:
    seen = set()
    out: List[Dict[str, Any]] = []
    for stage in stages:
        for source in iter_stage_sources(plan, stage):
            key = source_key(source)
            if key in seen:
                continue
            seen.add(key)
            out.append(source)
    return out


def nested_get(row: Mapping[str, Any], path: str) -> Any:
    current: Any = row
    for part in path.split("."):
        if isinstance(current, Mapping) and part in current:
            current = current[part]
        else:
            return None
    return current


def stringify(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, list):
        parts = [stringify(x) for x in value]
        parts = [x for x in parts if x]
        return "\n".join(parts) if parts else None
    if isinstance(value, Mapping):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def first_field_text(row: Mapping[str, Any], fields: Iterable[str]) -> Optional[str]:
    for field in fields:
        text = stringify(nested_get(row, field))
        if text and text.strip():
            return text
    return None


class MissingIsBlank(dict):
    def __missing__(self, key: str) -> str:
        return ""


def format_template(template: str, row: Mapping[str, Any]) -> Optional[str]:
    values = MissingIsBlank()
    for _, field_name, _, _ in Formatter().parse(template):
        if not field_name:
            continue
        values[field_name] = stringify(nested_get(row, field_name)) or ""
    text = template.format_map(values).strip()
    return text or None


def extract_text(
    row: Mapping[str, Any],
    source: Mapping[str, Any],
    defaults: Mapping[str, Any],
) -> Optional[str]:
    if source.get("text_template"):
        return format_template(str(source["text_template"]), row)
    fields = source.get("text_fields") or defaults.get("text_fields") or ["text"]
    return first_field_text(row, fields)


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u00a0", " ").replace("\ufeff", "")
    lines = [SPACE_RE.sub(" ", line).strip() for line in text.split("\n")]
    text = "\n".join(lines).strip()
    text = MANY_NEWLINES_RE.sub("\n\n\n", text)
    return text


def cyrillic_ratio(text: str) -> float:
    letters = LETTER_RE.findall(text)
    if not letters:
        return 0.0
    return len(CYRILLIC_RE.findall(text)) / len(letters)


def repeated_line_fraction(text: str) -> float:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) < 4:
        return 0.0
    unique = len(set(lines))
    return 1.0 - (unique / len(lines))


def basic_filter_reason(
    text: str,
    source: Mapping[str, Any],
    defaults: Mapping[str, Any],
) -> Optional[str]:
    min_chars = int(source.get("min_chars", defaults.get("min_chars", 0)))
    max_chars = int(source.get("max_chars", defaults.get("max_chars", 10**12)))
    if len(text) < min_chars:
        return "too_short"
    if len(text) > max_chars:
        return "too_long"

    min_cyr = float(source.get("min_cyrillic_ratio", defaults.get("min_cyrillic_ratio", 0.0)))
    if cyrillic_ratio(text) < min_cyr:
        return "low_cyrillic_ratio"

    max_repeated = float(
        source.get("max_repeated_line_fraction", defaults.get("max_repeated_line_fraction", 1.0))
    )
    if repeated_line_fraction(text) > max_repeated:
        return "repeated_lines"

    max_urls = int(source.get("max_url_count", defaults.get("max_url_count", 10**9)))
    if len(URL_RE.findall(text)) > max_urls:
        return "too_many_urls"

    return None


def document_hash(text: str) -> str:
    normalized = normalize_text(text).casefold()
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()


def load_hf_stream(source: Mapping[str, Any]) -> Iterable[Mapping[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - import guard for CLI use
        raise SystemExit("datasets is required. Install requirements_data.txt first.") from exc

    dataset = source["dataset"]
    config = source.get("config")
    split = source.get("split", "train")
    trust_remote_code = bool(source.get("trust_remote_code", False))

    kwargs: Dict[str, Any] = {
        "split": split,
        "streaming": True,
        "trust_remote_code": trust_remote_code,
    }
    if config:
        kwargs["name"] = config
    return load_dataset(dataset, **kwargs)


def iter_clean_texts(
    source: Mapping[str, Any],
    defaults: Mapping[str, Any],
    *,
    seen_hashes: Optional[set[str]] = None,
) -> Iterator[Tuple[str, Dict[str, Any]]]:
    if source.get("kind", "hf") != "hf":
        raise NotImplementedError(f"Unsupported source kind: {source.get('kind')}")

    for row_index, row in enumerate(load_hf_stream(source)):
        text = extract_text(row, source, defaults)
        if not text:
            continue
        text = normalize_text(text)
        reason = basic_filter_reason(text, source, defaults)
        if reason:
            continue
        h = document_hash(text)
        if seen_hashes is not None:
            if h in seen_hashes:
                continue
            seen_hashes.add(h)
        meta = {
            "source": source_name(source),
            "dataset": source.get("dataset"),
            "config": source.get("config"),
            "split": source.get("split"),
            "row_index": row_index,
            "hash": h,
            "chars": len(text),
        }
        yield text, meta


def weighted_quota(total: int, sources: List[Mapping[str, Any]]) -> Dict[str, int]:
    enabled = [s for s in sources if is_enabled(s)]
    weight_sum = sum(float(s.get("weight", 1.0)) for s in enabled)
    if weight_sum <= 0:
        raise ValueError("Total source weight must be positive")
    quotas: Dict[str, int] = {}
    remaining = total
    for i, source in enumerate(enabled):
        name = source_name(source)
        if i == len(enabled) - 1:
            quota = remaining
        else:
            quota = int(total * float(source.get("weight", 1.0)) / weight_sum)
            remaining -= quota
        quotas[name] = max(0, quota)
    return quotas
