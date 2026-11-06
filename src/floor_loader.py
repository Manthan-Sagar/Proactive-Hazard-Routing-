"""
floor_loader.py
===============
Completely self-contained architectural raster and XML annotation loader.
Loads local floor plans from `proactive_hazard_routing/data/` with ZERO external dependencies.
Features:
  - XML polyline and bounding-box parsing (doors, exits, stairs, boundary)
  - Morphological free-space extraction (Paper 1)
  - 8-connected grid navigation graph construction
  - Precomputed distance potential fields
"""

import os
import xml.etree.ElementTree as ET
import numpy as np
import cv2
import networkx as nx
from PIL import Image
from skimage import morphology
from typing import Dict, List, Tuple, Optional

# Root directory of this isolated project
MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(MODULE_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")


def parse_xml(xml_content_or_path: str) -> Dict:
    """Parses XML floor annotations (polylines and boxes)."""
    if os.path.isfile(xml_content_or_path):
        with open(xml_content_or_path, 'r', encoding='utf-8') as f:
            xml_text = f.read()
    else:
        xml_text = xml_content_or_path

    root = ET.fromstring(xml_text)
    data = {
        'doors': [],
        'exits': [],
        'stairs': [],
        'lifts': [],
        'stairs_doors': [],
        'lift_doors': [],
        'boundary': []
    }

    for elem in root.iter():
        label = elem.get('label', '').lower()
        if elem.tag == 'polyline' and elem.get('points'):
            pts = [tuple(map(float, pt.split(','))) for pt in elem.get('points').split(';') if pt.strip()]
            if pts:
                mid = (sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts))
                if label == 'door':
                    data['doors'].append({'pos': mid, 'points': pts, 'type': 'door'})
                elif label == 'exit':
                    data['exits'].append({'pos': mid, 'points': pts, 'type': 'exit'})
                elif label == 'stairs door':
                    data['stairs_doors'].append({'pos': mid, 'points': pts, 'type': 'stairs_door'})
                elif label == 'lift door':
                    data['lift_doors'].append({'pos': mid, 'points': pts, 'type': 'lift_door'})
                elif label == 'boundary':
                    data['boundary'] = pts
        elif elem.tag == 'box':
            try:
                xtl, ytl = float(elem.get('xtl', 0)), float(elem.get('ytl', 0))
                xbr, ybr = float(elem.get('xbr', 0)), float(elem.get('ybr', 0))
                c = ((xtl + xbr) / 2, (ytl + ybr) / 2)
                obj = {'pos': c, 'box': (xtl, ytl, xbr, ybr), 'type': 'stairs' if 'stair' in label else 'lift'}
                if 'stair' in label:
                    data['stairs'].append(obj)
                elif 'lift' in label:
                    data['lifts'].append(obj)
            except Exception:
                pass

    return data


def detect_free_space(img: np.ndarray, annotations: Optional[Dict] = None) -> np.ndarray:
    """Extracts binary traversable space from the architectural floor plan image."""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    free_space = binary > 0

    if annotations and annotations.get('boundary'):
        mask = np.zeros_like(free_space, dtype=np.uint8)
        cv2.fillPoly(mask, [np.array(annotations['boundary'], dtype=np.int32)], 1)
        free_space = free_space & (mask > 0)

    fs_u8 = (free_space.astype(np.uint8) * 255)

    if annotations:
        for s in annotations.get('stairs', []):
            cv2.rectangle(fs_u8, (int(s['box'][0]), int(s['box'][1])), (int(s['box'][2]), int(s['box'][3])), 0, 3)
        for l in annotations.get('lifts', []):
            cv2.rectangle(fs_u8, (int(l['box'][0]), int(l['box'][1])), (int(l['box'][2]), int(l['box'][3])), 0, 3)
        for cat in ['doors', 'exits', 'stairs_doors', 'lift_doors']:
            for item in annotations.get(cat, []):
                cv2.circle(fs_u8, (int(item['pos'][0]), int(item['pos'][1])), 15, 255, -1)

    return morphology.remove_small_objects(fs_u8 > 0, min_size=30).astype(np.uint8)


def build_nav_graph(free_space: np.ndarray, annotations: Dict, floor_name: str = "Ground", step: int = 2) -> nx.Graph:
    """Builds an 8-connected grid navigation graph with soft doorway discounts and exit links."""
    h, w = free_space.shape
    G = nx.Graph()

    pixel_nodes = {}
    for y in range(0, h, step):
        for x in range(0, w, step):
            if free_space[y, x] > 0:
                nid = f"{floor_name}_pix_{x}_{y}"
                G.add_node(nid, pos=(x, y), type='free', floor=floor_name)
                pixel_nodes[(x, y)] = nid

    # 8-connectivity with Euclidean weights
    for (x, y), cn in pixel_nodes.items():
        for dx, dy in [(-step, 0), (step, 0), (0, -step), (0, step),
                       (-step, -step), (step, step), (-step, step), (step, -step)]:
            npos = (x + dx, y + dy)
            if npos in pixel_nodes:
                w_dist = step * 1.414 if (dx != 0 and dy != 0) else float(step)
                G.add_edge(cn, pixel_nodes[npos], weight=w_dist)

    # Connect exits to nearby navigable nodes
    for i, ext in enumerate(annotations.get('exits', [])):
        nid = f"{floor_name}_exit_{i}"
        G.add_node(nid, pos=ext['pos'], type='exit', floor=floor_name)
        for r in [10, 20, 30]:
            found = False
            for px, pn in pixel_nodes.items():
                d = np.hypot(ext['pos'][0] - px[0], ext['pos'][1] - px[1])
                if d < r:
                    G.add_edge(nid, pn, weight=d)
                    found = True
            if found:
                break

    return G


# Available local floor plan presets
FLOOR_PRESETS = {
    "ground": {
        "img": os.path.join(DATA_DIR, "floors", "o2_", "g.png"),
        "xml": os.path.join(DATA_DIR, "xml", "02", "g.xml"),
        "name": "Ground"
    },
    "floor1": {
        "img": os.path.join(DATA_DIR, "floors", "o2_", "1.png"),
        "xml": os.path.join(DATA_DIR, "xml", "02", "1.xml"),
        "name": "Floor1"
    },
    "penthouse": {
        "img": os.path.join(DATA_DIR, "floors", "o2_", "p.png"),
        "xml": os.path.join(DATA_DIR, "xml", "02", "p.xml"),
        "name": "Penthouse"
    },
    "office1": {
        "img": os.path.join(DATA_DIR, "floors", "o1.png"),
        "xml": os.path.join(DATA_DIR, "xml", "o1.xml"),
        "name": "Office1"
    }
}


def load_floor(floor_key: str = "ground") -> Tuple[np.ndarray, np.ndarray, nx.Graph, List[str], Dict]:
    """
    Loads a local floor plan raster and XML annotations completely from the isolated `data/` folder.
    Returns: (rgb_image, free_space_grid, nav_graph, exit_nodes, annotations_dict)
    """
    preset = FLOOR_PRESETS.get(floor_key.lower(), FLOOR_PRESETS["ground"])
    img_path = preset["img"]
    xml_path = preset["xml"]

    assert os.path.exists(img_path), f"Floor image not found at: {img_path}"
    assert os.path.exists(xml_path), f"XML annotations not found at: {xml_path}"

    img = np.array(Image.open(img_path).convert('RGB'))
    annotations = parse_xml(xml_path)
    free_space = detect_free_space(img, annotations)
    graph = build_nav_graph(free_space, annotations, floor_name=preset["name"], step=2)
    exit_nodes = [n for n, d in graph.nodes(data=True) if d.get('type') == 'exit']

    return img, free_space, graph, exit_nodes, annotations

