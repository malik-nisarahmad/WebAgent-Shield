"""Preserve source locations while converting HTML into bounded evidence."""
import hashlib
from bs4 import BeautifulSoup, Comment


def node_path(element) -> str:
    parts = []
    while element and element.name != "[document]":
        parts.append(f"{element.name}[{len(element.find_previous_siblings(element.name)) + 1}]")
        element = element.parent
    return "/" + "/".join(reversed(parts))


def extract(html: str, url: str, channel: str) -> list[dict]:
    soup = BeautifulSoup(html, "lxml")
    passages = []
    for node in soup.find_all(string=True):
        parent = node.parent
        if parent.name in {"script", "style"}:
            continue
        text = " ".join(str(node).split())
        if not text:
            continue
        path = []
        ancestor = parent
        hidden = False
        while ancestor and ancestor.name != "[document]":
            siblings = ancestor.find_previous_siblings(ancestor.name)
            path.append(f"{ancestor.name}[{len(siblings) + 1}]")
            style = ancestor.get("style", "").replace(" ", "").lower()
            hidden |= ancestor.has_attr("hidden") or "display:none" in style or "visibility:hidden" in style
            ancestor = ancestor.parent
        location = "/" + "/".join(reversed(path))
        comment = isinstance(node, Comment)
        for offset in range(0, len(text), 1600):
            piece = text[offset:offset + 1800]
            passages.append({"id": hashlib.sha256(f"{url}|{channel}|{location}|{offset}|{piece}".encode()).hexdigest()[:20],
                             "text": piece, "url": url, "channel": channel,
                             "location": location, "normalized_offset": offset,
                             "hidden": hidden, "kind": "comment" if comment else "text",
                             "eligible": channel == "rendered" and not hidden and not comment})
    for element in soup.find_all(True):
        for attr in ("content", "title", "alt", "aria-label", "placeholder"):
            value = element.get(attr)
            if value:
                passages.append({"id": f"attribute-{len(passages)}", "text": str(value)[:1800],
                                 "url": url, "channel": channel, "location": f"{node_path(element)}/@{attr}",
                                 "kind": "attribute", "hidden": False, "eligible": False})
    return passages


def inspect(passages: list[dict]) -> dict:
    """Phase 2 boundary. A trained detector will replace this in phase 3."""
    return {"detector_status": "not_trained", "findings": [], "passages": passages}


def evidence(scan: dict) -> dict:
    eligible = [p for p in scan["passages"] if p["eligible"]]
    coverage = dict(scan["coverage"])
    coverage["evidence_truncated"] = len(eligible) > 80
    coverage["partial"] = coverage.get("partial", False) or len(eligible) > 80
    return {"scan_id": scan["id"], "url": scan["url"],
            "detector_status": scan["detector_status"], "coverage": coverage,
            "passages": eligible[:80]}
