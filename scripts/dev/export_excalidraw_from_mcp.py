from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any


def estimate_text_size(text: str, font_size: int) -> tuple[float, float]:
    lines = text.split("\n") if text else [""]
    max_chars = max(len(line) for line in lines) if lines else 1
    width = max(20.0, max_chars * font_size * 0.58)
    height = max(20.0, len(lines) * font_size * 1.25)
    return width, height


def base_element(src: dict[str, Any]) -> dict[str, Any]:
    now = int(time.time() * 1000)
    return {
        "id": src.get("id", f"elem-{random.randint(100000, 999999)}"),
        "type": src.get("type", "rectangle"),
        "x": float(src.get("x", 0)),
        "y": float(src.get("y", 0)),
        "width": float(src.get("width", 100)),
        "height": float(src.get("height", 100)),
        "angle": 0,
        "strokeColor": src.get("strokeColor", "#1e1e1e"),
        "backgroundColor": src.get("backgroundColor", "transparent"),
        "fillStyle": "hachure",
        "strokeWidth": float(src.get("strokeWidth", 1)),
        "strokeStyle": "solid",
        "roughness": float(src.get("roughness", 1)),
        "opacity": int(src.get("opacity", 100)),
        "groupIds": [],
        "frameId": None,
        "roundness": None,
        "seed": random.randint(1, 2_000_000_000),
        "version": int(src.get("version", 1)),
        "versionNonce": random.randint(1, 2_000_000_000),
        "isDeleted": False,
        "boundElements": [],
        "updated": now,
        "link": None,
        "locked": False,
    }


def convert_element(src: dict[str, Any]) -> dict[str, Any]:
    kind = src.get("type", "rectangle")
    el = base_element(src)

    if kind == "text":
        text = src.get("text", "")
        font_size = int(src.get("fontSize", 20))
        width, height = estimate_text_size(text, font_size)
        el.update(
            {
                "width": width,
                "height": height,
                "backgroundColor": "transparent",
                "fillStyle": "solid",
                "roughness": 0,
                "text": text,
                "fontSize": font_size,
                "fontFamily": int(src.get("fontFamily", 1)),
                "textAlign": "left",
                "verticalAlign": "top",
                "containerId": None,
                "originalText": text,
                "lineHeight": 1.25,
                "baseline": int(height - (font_size * 0.2)),
            }
        )
        return el

    if kind in {"line", "arrow"}:
        width = float(src.get("width", 100))
        height = float(src.get("height", 0))
        el.update(
            {
                "backgroundColor": "transparent",
                "fillStyle": "solid",
                "points": [[0, 0], [width, height]],
                "lastCommittedPoint": None,
                "startBinding": None,
                "endBinding": None,
                "startArrowhead": None,
                "endArrowhead": "arrow" if kind == "arrow" else None,
                "elbowed": False,
            }
        )
        return el

    if kind in {"rectangle", "diamond", "ellipse"}:
        return el

    # Fallback for unsupported custom types
    el["type"] = "rectangle"
    return el


def parse_mcp_text_file(path: Path) -> list[dict[str, Any]]:
    content = path.read_text(encoding="utf-8")
    start = content.find("{")
    end = content.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"Could not find JSON object in {path}")
    data = json.loads(content[start : end + 1])
    return data.get("elements", [])


def parse_json_file(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("elements", [])


def main() -> None:
    root = Path.cwd()
    txt_path = root / "diagrams" / "project_flow_excalidraw_mcp_output.txt"
    json_path = root / "diagrams" / "project_flow_excalidraw_elements.excalidraw"
    out_path = root / "diagrams" / "project_flow.excalidraw"

    elements_from_txt = parse_mcp_text_file(txt_path)
    elements_from_json = parse_json_file(json_path)

    # Prefer structured json input; keep txt parse as consistency fallback.
    elements_src = elements_from_json or elements_from_txt
    if not elements_src:
        raise ValueError("No elements found in source files")

    converted = [convert_element(e) for e in elements_src]

    scene = {
        "type": "excalidraw",
        "version": 2,
        "source": "https://excalidraw.com",
        "elements": converted,
        "appState": {
            "gridSize": None,
            "viewBackgroundColor": "#ffffff",
        },
        "files": {},
    }

    out_path.write_text(json.dumps(scene, indent=2), encoding="utf-8")

    print(f"Exported Excalidraw file: {out_path}")
    print(f"Elements: {len(converted)}")


if __name__ == "__main__":
    main()
