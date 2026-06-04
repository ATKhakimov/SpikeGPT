"""Shared helpers for SpikeRuGPT data preparation scripts."""

from __future__ import annotations

import hashlib
import json
import os
import re
import string
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
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b", re.IGNORECASE)
PHONE_RE = re.compile(r"(?:\+?\d[\s().-]*){9,}")
HTML_TAG_RE = re.compile(r"<[^>\n]{1,200}>")
MARKDOWN_LINK_RE = re.compile(r"!?\[([^\]]{0,200})\]\([^)]+\)")
MARKDOWN_URL_RE = re.compile(r"\b(?:https?://|www\.)\S+", re.IGNORECASE)
MARKDOWN_STYLE_RE = re.compile(r"[*_`~]{1,3}")
SPACE_RE = re.compile(r"[ \t\f\v]+")
MANY_NEWLINES_RE = re.compile(r"\n{4,}")
WORD_RE = re.compile(r"[^\W\d_]{2,}", re.UNICODE)

BOILERPLATE_LINE_RE = re.compile(
    r"(?i)\b("
    r"cookie|cookies|privacy policy|terms of use|subscribe|sign in|log in|"
    r"используем cookies|политика конфиденциальности|пользовательское соглашение|"
    r"подписаться|войти|регистрация|читать далее|главная|меню|наверх"
    r")\b"
)
SPAM_RE = re.compile(
    r"(?i)\b("
    r"casino|казино|слот(?:ы|ов)?|букмекер|ставк[аи]|betting|bet365|"
    r"промокод|фрибет|займ(?:ы|ов)?|микрозайм|кредит без отказа|"
    r"купить дешево|купить недорого|доставка по россии|seo|дорвей|"
    r"эротик|порно|adult|viagra|виагра|onlyfans"
    r")\b"
)


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


def clean_markup(text: str) -> str:
    text = MARKDOWN_LINK_RE.sub(lambda m: m.group(1), text)
    text = MARKDOWN_URL_RE.sub("", text)
    text = HTML_TAG_RE.sub(" ", text)
    text = MARKDOWN_STYLE_RE.sub("", text)
    return normalize_text(text)


def remove_boilerplate_lines(text: str) -> str:
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            lines.append("")
            continue
        if len(stripped) < 90 and BOILERPLATE_LINE_RE.search(stripped):
            continue
        lines.append(line)
    return normalize_text("\n".join(lines))


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


def alpha_ratio(text: str) -> float:
    if not text:
        return 0.0
    return len(LETTER_RE.findall(text)) / len(text)


def digit_ratio(text: str) -> float:
    if not text:
        return 0.0
    return sum(ch.isdigit() for ch in text) / len(text)


def punctuation_ratio(text: str) -> float:
    if not text:
        return 0.0
    punct = set(string.punctuation) | set("«»„“”‘’—–…№")
    return sum(ch in punct for ch in text) / len(text)


def short_line_fraction(text: str) -> float:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) < 6:
        return 0.0
    return sum(len(line) < 35 for line in lines) / len(lines)


def unique_word_fraction(text: str) -> float:
    words = [word.casefold() for word in WORD_RE.findall(text)]
    if len(words) < 50:
        return 1.0
    return len(set(words)) / len(words)


def simhash64(text: str) -> int:
    vector = [0] * 64
    features = WORD_RE.findall(text.casefold())
    for feature in features[:20000]:
        digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, "big")
        for bit in range(64):
            if value & (1 << bit):
                vector[bit] += 1
            else:
                vector[bit] -= 1
    out = 0
    for bit, score in enumerate(vector):
        if score >= 0:
            out |= 1 << bit
    return out


def hamming_distance64(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def is_near_duplicate(
    value: int,
    buckets: Dict[Tuple[int, int], List[int]],
    *,
    threshold: int,
    bands: int = 4,
) -> bool:
    band_bits = 64 // bands
    candidates: List[int] = []
    for band in range(bands):
        mask = (1 << band_bits) - 1
        key = (band, (value >> (band * band_bits)) & mask)
        candidates.extend(buckets.get(key, []))
    return any(hamming_distance64(value, candidate) <= threshold for candidate in candidates)


def add_simhash(value: int, buckets: Dict[Tuple[int, int], List[int]], *, bands: int = 4) -> None:
    band_bits = 64 // bands
    for band in range(bands):
        mask = (1 << band_bits) - 1
        key = (band, (value >> (band * band_bits)) & mask)
        buckets.setdefault(key, []).append(value)


def iter_text_chunks(
    text: str,
    source: Mapping[str, Any],
    defaults: Mapping[str, Any],
) -> Iterator[str]:
    max_chars = int(source.get("max_chars", defaults.get("max_chars", 10**12)))
    if len(text) <= max_chars:
        yield text
        return

    if not bool(source.get("chunk_long_documents", defaults.get("chunk_long_documents", False))):
        yield text
        return

    chunk_chars = int(source.get("chunk_chars", defaults.get("chunk_chars", max_chars)))
    overlap = int(source.get("chunk_overlap_chars", defaults.get("chunk_overlap_chars", 0)))
    chunk_chars = max(1000, min(chunk_chars, max_chars))
    overlap = max(0, min(overlap, chunk_chars // 4))

    current: List[str] = []
    current_len = 0
    for paragraph in re.split(r"\n{2,}", text):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        para_len = len(paragraph) + 2
        if current and current_len + para_len > chunk_chars:
            chunk = normalize_text("\n\n".join(current))
            if chunk:
                yield chunk
            if overlap and current:
                tail = []
                tail_len = 0
                for item in reversed(current):
                    if tail_len + len(item) > overlap:
                        break
                    tail.append(item)
                    tail_len += len(item) + 2
                current = list(reversed(tail))
                current_len = tail_len
            else:
                current = []
                current_len = 0

        if para_len > chunk_chars:
            sentences = re.split(r"(?<=[.!?。！？])\s+", paragraph)
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                if current and current_len + len(sentence) + 1 > chunk_chars:
                    chunk = normalize_text("\n\n".join(current))
                    if chunk:
                        yield chunk
                    current = []
                    current_len = 0
                current.append(sentence)
                current_len += len(sentence) + 1
        else:
            current.append(paragraph)
            current_len += para_len

    chunk = normalize_text("\n\n".join(current))
    if chunk:
        yield chunk


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

    min_alpha = float(source.get("min_alpha_ratio", defaults.get("min_alpha_ratio", 0.0)))
    if alpha_ratio(text) < min_alpha:
        return "low_alpha_ratio"

    min_cyr = float(source.get("min_cyrillic_ratio", defaults.get("min_cyrillic_ratio", 0.0)))
    if cyrillic_ratio(text) < min_cyr:
        return "low_cyrillic_ratio"

    max_digits = float(source.get("max_digit_ratio", defaults.get("max_digit_ratio", 1.0)))
    if digit_ratio(text) > max_digits:
        return "too_many_digits"

    max_punct = float(source.get("max_punctuation_ratio", defaults.get("max_punctuation_ratio", 1.0)))
    if punctuation_ratio(text) > max_punct:
        return "too_much_punctuation"

    max_repeated = float(
        source.get("max_repeated_line_fraction", defaults.get("max_repeated_line_fraction", 1.0))
    )
    if repeated_line_fraction(text) > max_repeated:
        return "repeated_lines"

    max_short_lines = float(
        source.get("max_short_line_fraction", defaults.get("max_short_line_fraction", 1.0))
    )
    if short_line_fraction(text) > max_short_lines:
        return "too_many_short_lines"

    max_urls = int(source.get("max_url_count", defaults.get("max_url_count", 10**9)))
    if len(URL_RE.findall(text)) > max_urls:
        return "too_many_urls"

    max_emails = int(source.get("max_email_count", defaults.get("max_email_count", 10**9)))
    if len(EMAIL_RE.findall(text)) > max_emails:
        return "too_many_emails"

    max_phones = int(source.get("max_phone_count", defaults.get("max_phone_count", 10**9)))
    if len(PHONE_RE.findall(text)) > max_phones:
        return "too_many_phones"

    min_unique = float(
        source.get("min_unique_word_fraction", defaults.get("min_unique_word_fraction", 0.0))
    )
    if unique_word_fraction(text) < min_unique:
        return "low_unique_word_fraction"

    if SPAM_RE.search(text):
        return "spam_keyword"

    return None


def record_stat(stats: Optional[Dict[str, int]], key: str, amount: int = 1) -> None:
    if stats is not None:
        stats[key] = stats.get(key, 0) + amount


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
    data_files = source.get("data_files")

    if isinstance(data_files, str) and data_files.endswith(".jsonl.zst"):
        return load_zst_jsonl_stream(dataset, data_files)

    kwargs: Dict[str, Any] = {
        "split": split,
        "streaming": True,
        "trust_remote_code": trust_remote_code,
    }
    if data_files:
        kwargs["data_files"] = data_files
        return load_dataset("json", **kwargs)
    if config:
        kwargs["name"] = config
    return load_dataset(dataset, **kwargs)


def load_zst_jsonl_stream(dataset: str, data_file: str) -> Iterable[Mapping[str, Any]]:
    try:
        import zstandard as zstd
        from huggingface_hub import hf_hub_download
    except ImportError as exc:  # pragma: no cover - import guard for CLI use
        raise SystemExit("zstandard and huggingface_hub are required for .jsonl.zst sources.") from exc

    prefix = f"hf://datasets/{dataset}/"
    filename = data_file[len(prefix) :] if data_file.startswith(prefix) else data_file
    local_path = hf_hub_download(repo_id=dataset, repo_type="dataset", filename=filename)

    def iterator() -> Iterator[Mapping[str, Any]]:
        with open(local_path, "rb") as compressed:
            reader = zstd.ZstdDecompressor().stream_reader(compressed)
            text_stream = reader
            buffer = b""
            while True:
                chunk = text_stream.read(1024 * 1024)
                if not chunk:
                    break
                buffer += chunk
                while b"\n" in buffer:
                    line, buffer = buffer.split(b"\n", 1)
                    line = line.strip()
                    if line:
                        yield json.loads(line.decode("utf-8"))
            tail = buffer.strip()
            if tail:
                yield json.loads(tail.decode("utf-8"))

    return iterator()


def iter_clean_texts(
    source: Mapping[str, Any],
    defaults: Mapping[str, Any],
    *,
    seen_hashes: Optional[set[str]] = None,
    seen_simhashes: Optional[Dict[Tuple[int, int], List[int]]] = None,
    stats: Optional[Dict[str, int]] = None,
) -> Iterator[Tuple[str, Dict[str, Any]]]:
    if source.get("kind", "hf") != "hf":
        raise NotImplementedError(f"Unsupported source kind: {source.get('kind')}")

    for row_index, row in enumerate(load_hf_stream(source)):
        record_stat(stats, "rows_seen")
        text = extract_text(row, source, defaults)
        if not text:
            record_stat(stats, "missing_text")
            continue
        text = remove_boilerplate_lines(clean_markup(normalize_text(text)))
        chunk_seen = False
        for chunk_index, chunk in enumerate(iter_text_chunks(text, source, defaults)):
            chunk_seen = True
            reason = basic_filter_reason(chunk, source, defaults)
            if reason:
                record_stat(stats, f"rejected_{reason}")
                continue
            h = document_hash(chunk)
            if seen_hashes is not None:
                if h in seen_hashes:
                    record_stat(stats, "rejected_exact_duplicate")
                    continue
                seen_hashes.add(h)
            simhash_value = None
            if seen_simhashes is not None:
                simhash_value = simhash64(chunk)
                threshold = int(source.get("simhash_threshold", defaults.get("simhash_threshold", 3)))
                if is_near_duplicate(simhash_value, seen_simhashes, threshold=threshold):
                    record_stat(stats, "rejected_near_duplicate")
                    continue
                add_simhash(simhash_value, seen_simhashes)
            record_stat(stats, "accepted_chunks")
            meta = {
                "source": source_name(source),
                "dataset": source.get("dataset"),
                "config": source.get("config"),
                "split": source.get("split"),
                "row_index": row_index,
                "chunk_index": chunk_index,
                "hash": h,
                "simhash": f"{simhash_value:016x}" if simhash_value is not None else None,
                "chars": len(chunk),
            }
            yield chunk, meta
        if not chunk_seen:
            record_stat(stats, "empty_after_cleanup")


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
