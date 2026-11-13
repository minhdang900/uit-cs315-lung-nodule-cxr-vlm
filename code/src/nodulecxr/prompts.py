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
_JSON_RE = re.compile(r"\{.*\}", re.S)
_LABEL_RE = re.compile(r'"?nodule"?\s*[:=]\s*"?(yes|no)\b', re.I)


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


def parse_answer(text: str, size: int = IMG_SIZE) -> ParsedAnswer:
    t = _FENCE_RE.sub("", text or "").strip()
    m = _JSON_RE.search(t)
    if m:
        try:
            d = json.loads(m.group(0))
        except json.JSONDecodeError:
            d = None
        if isinstance(d, dict) and str(d.get("nodule", "")).lower() in ("yes", "no"):
            label = 1 if str(d["nodule"]).lower() == "yes" else 0
            return ParsedAnswer(label, _clean_boxes(d.get("boxes", []), size) if label else [], True)
    m = _LABEL_RE.search(t)
    if m:
        return ParsedAnswer(1 if m.group(1).lower() == "yes" else 0, [], False)
    return ParsedAnswer(None, [], False)
