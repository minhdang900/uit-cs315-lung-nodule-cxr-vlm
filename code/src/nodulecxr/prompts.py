"""Fixed prompts, target strings and a tolerant JSON answer parser."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Optional

from nodulecxr.records import IMG_SIZE

QUESTION_LOC = ("Is there a lung nodule or mass on this chest X-ray? If yes, give the bounding "
                "box(es) in pixel coordinates. Answer in JSON.")
QUESTION_CLS = "Is there a lung nodule or mass on this chest X-ray? Answer in JSON."
ANSWER_PREFIX = '{"nodule": "'

_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.M)
_LABEL_RE = re.compile(r'"?nodule\w*"?\s*[:=]\s*"?(yes|no|true|false)\b', re.I)
_NODULE_KEY_RE = re.compile(r"^nodule", re.I)
_BOX_KEYS = ("boxes", "bounding_boxes", "bbox", "bboxes")
_DECODER = json.JSONDecoder()


def build_target(label: int, boxes, with_boxes: bool = True) -> str:
    d = {"nodule": "yes" if label else "no"}
    if with_boxes:
        d["boxes"] = [[int(round(v)) for v in b[:4]] for b in boxes] if label else []
    return json.dumps(d)


@dataclass
class ParsedAnswer:
    label: Optional[int]
    boxes: list = field(default_factory=list)
    ok: bool = False


def _clean_boxes(raw, size: int) -> list:
    if isinstance(raw, (list, tuple)) and len(raw) >= 4 and all(isinstance(v, (int, float)) for v in raw[:4]):
        raw = [raw]  # a single flat box, e.g. "bbox": [x1, y1, x2, y2]
    out = []
    for b in raw if isinstance(raw, list) else []:
        if isinstance(b, dict):
            b = b.get("bbox_2d") or b.get("bbox") or b.get("box")
        if not isinstance(b, (list, tuple)) or len(b) < 4:
            continue
        try:
            x1, y1, x2, y2 = (float(v) for v in b[:4])
        except (TypeError, ValueError):
            continue
        x1, x2 = sorted((min(max(x1, 0.0), size), min(max(x2, 0.0), size)))
        y1, y2 = sorted((min(max(y1, 0.0), size), min(max(y2, 0.0), size)))
        if x2 - x1 >= 1 and y2 - y1 >= 1:
            out.append([x1, y1, x2, y2])
    return out


def _label_value(v) -> Optional[int]:
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, str):
        s = re.sub(r"[^\w]", "", v).lower()
        if s in ("yes", "true"):
            return 1
        if s in ("no", "false"):
            return 0
    return None


def _label_from_dict(d: dict) -> Optional[int]:
    key = "nodule" if "nodule" in d else next(
        (k for k in d if isinstance(k, str) and _NODULE_KEY_RE.match(k)), None)
    return None if key is None else _label_value(d[key])


def _boxes_from_dict(d: dict):
    return next((d[k] for k in _BOX_KEYS if k in d), [])


def _dict_candidates(t: str):
    """Every JSON object that decodes starting at some '{', in order of position."""
    i = t.find("{")
    while i != -1:
        try:
            obj, _ = _DECODER.raw_decode(t, i)
        except json.JSONDecodeError:
            obj = None
        if isinstance(obj, dict):
            yield obj
        i = t.find("{", i + 1)


def parse_answer(text: str, size: int = IMG_SIZE) -> ParsedAnswer:
    t = _FENCE_RE.sub("", text or "").strip()
    for d in _dict_candidates(t):
        label = _label_from_dict(d)
        if label is not None:
            return ParsedAnswer(label, _clean_boxes(_boxes_from_dict(d), size) if label else [], True)
    m = _LABEL_RE.search(t)
    if m:
        return ParsedAnswer(1 if m.group(1).lower() in ("yes", "true") else 0, [], False)
    return ParsedAnswer(None, [], False)
