#!/usr/bin/env python3
"""
Generate UMAP preview SVGs for Tensor Logic + Homology formalization.
Based on heyting-viz pattern from PCN/SKY PaperPacks.
"""

import json
import math
import os
import re
import sys
from pathlib import Path

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    print("Warning: numpy not installed, using fallback", file=sys.stderr)

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    print("Warning: umap-learn not installed", file=sys.stderr)


def extract_declarations(lean_dir: Path) -> list:
    """Extract declarations from Lean files."""
    decls = []

    for lean_file in lean_dir.rglob("*.lean"):
        rel_path = lean_file.relative_to(lean_dir.parent)
        module = str(rel_path).replace("/", ".").replace(".lean", "")

        # Determine module family for coloring (Tensor Logic specific)
        if "AST" in module:
            family = "AST"
        elif "Parse" in module:
            family = "Parser"
        elif "Eval" in module:
            family = "Eval"
        elif "Validate" in module:
            family = "Validate"
        elif "HomologyEncoding" in module or "HomologyFromFacts" in module:
            family = "HomologyBridge"
        elif "F2Matrix" in module:
            family = "F2Matrix"
        elif "ChainComplex" in module:
            family = "ChainComplex"
        elif "Regime" in module:
            family = "Regime"
        elif "CLI" in module or "Main" in module:
            family = "CLI"
        elif "Test" in module or "Sanity" in module:
            family = "Tests"
        else:
            family = "Core"

        with open(lean_file, "r") as f:
            content = f.read()

        patterns = [
            (r"theorem\s+(\w+)", "theorem"),
            (r"lemma\s+(\w+)", "lemma"),
            (r"def\s+(\w+)", "def"),
            (r"structure\s+(\w+)", "structure"),
            (r"inductive\s+(\w+)", "inductive"),
            (r"abbrev\s+(\w+)", "abbrev"),
            (r"instance\s+(\w+)", "instance"),
        ]

        for pattern, kind in patterns:
            for match in re.finditer(pattern, content):
                name = match.group(1)
                if name.startswith("_") or name in ["mk", "rec", "casesOn"]:
                    continue
                decls.append({
                    "name": f"{module}.{name}",
                    "short": name,
                    "kind": kind,
                    "module": module,
                    "family": family,
                })

    return decls


# Tensor Logic module family colors (HSL format)
FAMILY_COLORS = {
    "AST": ("220", "70%", "58%"),           # Blue
    "Parser": ("180", "70%", "58%"),        # Cyan
    "Eval": ("128", "70%", "58%"),          # Green
    "Validate": ("285", "70%", "58%"),      # Purple
    "HomologyBridge": ("345", "70%", "58%"),# Pink
    "F2Matrix": ("33", "70%", "58%"),       # Orange
    "ChainComplex": ("49", "70%", "58%"),   # Yellow
    "Regime": ("260", "70%", "58%"),        # Violet
    "CLI": ("0", "0%", "70%"),              # Gray
    "Tests": ("160", "50%", "50%"),         # Teal
    "Core": ("210", "50%", "60%"),          # Steel blue
}


def family_color(family: str) -> str:
    h, s, l = FAMILY_COLORS.get(family, ("210", "50%", "60%"))
    return f"hsl({h}, {s}, {l})"


def make_features(decls: list) -> list:
    """Create feature vectors for UMAP from declaration properties."""
    features = []
    for d in decls:
        # Simple features: kind encoding, name length, module depth
        kind_map = {"theorem": 0, "lemma": 1, "def": 2, "structure": 3,
                    "inductive": 4, "abbrev": 5, "instance": 6}
        kind_val = kind_map.get(d["kind"], 7)
        name_len = len(d["short"])
        module_depth = d["module"].count(".")
        family_idx = list(FAMILY_COLORS.keys()).index(d["family"]) if d["family"] in FAMILY_COLORS else 0

        features.append([kind_val, name_len, module_depth, family_idx,
                        hash(d["name"]) % 100, hash(d["module"]) % 50])
    return features


def generate_2d_svg(decls: list, coords_2d: list, output_path: Path, k_neighbors: int = 5):
    """Generate 2D UMAP preview SVG with kNN edges."""
    width, height = 1500, 900
    margin = 50
    plot_w, plot_h = 1090, 800

    # Normalize coordinates
    xs = [c[0] for c in coords_2d]
    ys = [c[1] for c in coords_2d]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    x_range = x_max - x_min or 1
    y_range = y_max - y_min or 1

    def scale_x(x):
        return margin + (x - x_min) / x_range * plot_w
    def scale_y(y):
        return margin + (y - y_min) / y_range * plot_h

    # Compute kNN edges
    edges = []
    if HAS_NUMPY:
        coords_arr = np.array(coords_2d)
        for i in range(len(coords_2d)):
            dists = np.sqrt(np.sum((coords_arr - coords_arr[i])**2, axis=1))
            neighbors = np.argsort(dists)[1:k_neighbors+1]
            for j in neighbors:
                if i < j:
                    edges.append((i, j))

    lines = [
        '<?xml version="1.0" encoding="utf-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-label="UMAP preview">',
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="#0b0f14"/>',
        f'<text x="{margin}" y="32" fill="#ffffff" font-size="20" font-family="ui-sans-serif,system-ui,Segoe UI,Roboto,Helvetica,Arial">UMAP 2D - Tensor Logic + Homology proof map</text>',
        f'<text x="{margin}" y="48" fill="#b8c7d9" font-size="12" font-family="ui-sans-serif,system-ui,Segoe UI,Roboto,Helvetica,Arial">Declarations - Colors: module family - Edges: k-NN similarity</text>',
        f'<rect x="{margin}" y="{margin}" width="{plot_w}" height="{plot_h}" fill="#0f1721" stroke="#1c2a3a" stroke-width="1"/>',
    ]

    # Draw edges
    for i, j in edges:
        x1, y1 = scale_x(coords_2d[i][0]), scale_y(coords_2d[i][1])
        x2, y2 = scale_x(coords_2d[j][0]), scale_y(coords_2d[j][1])
        lines.append(f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" stroke="#3b4b5d" stroke-opacity="0.18" stroke-width="1"/>')

    # Draw nodes
    for i, d in enumerate(decls):
        x, y = scale_x(coords_2d[i][0]), scale_y(coords_2d[i][1])
        color = family_color(d["family"])
        r = 4 if d["kind"] in ["theorem", "lemma"] else 3
        lines.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{r}" fill="{color}" opacity="0.85"/>')

    # Legend
    legend_x = width - 200
    legend_y = 80
    lines.append(f'<text x="{legend_x}" y="{legend_y}" fill="#ffffff" font-size="14" font-family="ui-sans-serif">Module Families</text>')
    for i, (fam, (h, s, l)) in enumerate(FAMILY_COLORS.items()):
        y = legend_y + 20 + i * 18
        color = f"hsl({h}, {s}, {l})"
        lines.append(f'<circle cx="{legend_x + 8}" cy="{y - 4}" r="5" fill="{color}"/>')
        lines.append(f'<text x="{legend_x + 20}" y="{y}" fill="#b8c7d9" font-size="11" font-family="ui-sans-serif">{fam}</text>')

    lines.append('</svg>')

    output_path.write_text('\n'.join(lines))
    print(f"Generated: {output_path}")


def generate_3d_svg(decls: list, coords_3d: list, output_path: Path, animated: bool = True):
    """Generate 3D UMAP preview SVG with rotation animation."""
    width, height = 1500, 900
    cx, cy = width // 2, height // 2
    scale = 250

    # Normalize coordinates
    if HAS_NUMPY:
        coords = np.array(coords_3d)
        coords = (coords - coords.mean(axis=0)) / (coords.std(axis=0) + 1e-6)
    else:
        coords = coords_3d

    lines = [
        '<?xml version="1.0" encoding="utf-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="#0b0f14"/>',
        '<text x="50" y="32" fill="#ffffff" font-size="20" font-family="ui-sans-serif,system-ui">UMAP 3D - Tensor Logic + Homology proof map</text>',
        '<text x="50" y="48" fill="#b8c7d9" font-size="12" font-family="ui-sans-serif">Rotating view - Colors: module family</text>',
    ]

    if animated:
        lines.append('<style>@keyframes rotate { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }</style>')
        lines.append(f'<g style="transform-origin: {cx}px {cy}px; animation: rotate 20s linear infinite;">')
    else:
        lines.append('<g>')

    # Project and draw (simple orthographic for static)
    for i, d in enumerate(decls):
        if HAS_NUMPY:
            x, y, z = coords[i]
        else:
            x, y, z = coords_3d[i]
        px = cx + x * scale
        py = cy + y * scale
        color = family_color(d["family"])
        r = 4 if d["kind"] in ["theorem", "lemma"] else 3
        opacity = 0.5 + 0.3 * (z + 1) / 2  # depth-based opacity
        lines.append(f'<circle cx="{px:.2f}" cy="{py:.2f}" r="{r}" fill="{color}" opacity="{opacity:.2f}"/>')

    lines.append('</g>')
    lines.append('</svg>')

    output_path.write_text('\n'.join(lines))
    print(f"Generated: {output_path}")


def main():
    script_dir = Path(__file__).parent
    lean_dir = script_dir.parent / "HeytingLean"
    output_dir = script_dir.parent / "artifacts" / "visuals"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Extracting declarations...")
    decls = extract_declarations(lean_dir)
    print(f"Found {len(decls)} declarations")

    if len(decls) < 3:
        print("Too few declarations for UMAP")
        return

    features = make_features(decls)

    if HAS_NUMPY and HAS_UMAP:
        print("Running UMAP 2D...")
        reducer_2d = umap.UMAP(n_components=2, random_state=42, n_neighbors=min(15, len(decls)-1))
        coords_2d = reducer_2d.fit_transform(np.array(features)).tolist()

        print("Running UMAP 3D...")
        reducer_3d = umap.UMAP(n_components=3, random_state=42, n_neighbors=min(15, len(decls)-1))
        coords_3d = reducer_3d.fit_transform(np.array(features)).tolist()
    else:
        print("Using random fallback (no numpy/umap)")
        import random
        random.seed(42)
        coords_2d = [[random.gauss(0, 1), random.gauss(0, 1)] for _ in decls]
        coords_3d = [[random.gauss(0, 1), random.gauss(0, 1), random.gauss(0, 1)] for _ in decls]

    generate_2d_svg(decls, coords_2d, output_dir / "tensor_logic_2d_preview.svg")
    generate_3d_svg(decls, coords_3d, output_dir / "tensor_logic_3d_preview.svg", animated=False)
    generate_3d_svg(decls, coords_3d, output_dir / "tensor_logic_3d_preview_animated.svg", animated=True)

    # Save data as JSON for interactive viewer
    data = {
        "meta": {"count": len(decls), "families": list(FAMILY_COLORS.keys())},
        "items": decls,
        "coords_2d": coords_2d,
        "coords_3d": coords_3d,
    }
    (output_dir / "tensor_logic_proofs.json").write_text(json.dumps(data, indent=2))
    print(f"Generated: {output_dir / 'tensor_logic_proofs.json'}")


if __name__ == "__main__":
    main()
