#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
poly_mask_annotator.py  (v1.8.2, 2025-09-09)

v1.7.9 → v1.8.2 の主な追加/変更:
- 「Type: Area / Polyline」トグルを正式追加（描画前に選択）。Polyline は Finish しても閉じません。
- JSONに open_path を保存/復元。Editのヒット判定やInsertも open/closed で分岐。
- Export：open は 1px の折れ線として index/カラーPNG 出力。
- **UI改良**：Type の直後で改行し、
  2 行目に「＋New / ✓Finish / ✗Cancel / ⌫Last point / Color / パレット」を配置して横幅を抑制。

依存: Jupyter, ipywidgets, ipycanvas, scikit-image
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

try:
    from ipywidgets import (
        VBox, HBox, Button, Dropdown, ToggleButton, ToggleButtons, Text, Checkbox,
        BoundedIntText, ColorPicker, Layout, HTML, IntSlider
    )
    from ipycanvas import MultiCanvas, hold_canvas
except Exception as e:  # pragma: no cover
    raise ImportError(
        "This module requires Jupyter with ipywidgets and ipycanvas installed.\n"
        "Install: pip install ipywidgets ipycanvas"
    ) from e

try:
    from skimage.draw import polygon as sk_polygon, line as sk_line
except Exception as e:  # pragma: no cover
    raise ImportError("This module requires scikit-image.\nInstall: pip install scikit-image") from e


# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------

ALLOWED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def _distinct_color(idx: int) -> str:
    palette = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    ]
    return palette[idx % len(palette)]


def _euclid(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _dist_point_to_segment(px: float, py: float, x1: float, y1: float, x2: float, y2: float) -> float:
    """距離（点-線分）。"""
    vx, vy = x2 - x1, y2 - y1
    wx, wy = px - x1, py - y1
    denom = vx * vx + vy * vy
    if denom <= 1e-12:
        return math.hypot(px - x1, py - y1)
    t = max(0.0, min(1.0, (wx * vx + wy * wy) / denom))
    qx, qy = x1 + t * vx, y1 + t * vy
    return math.hypot(px - qx, py - qy)


def chaikin(points: List[Tuple[float, float]], iterations: int = 2, closed: bool = True) -> List[Tuple[float, float]]:
    if len(points) < 3:
        return points[:]
    pts = points[:]
    if closed:
        pts = pts + [pts[0]]
    for _ in range(iterations):
        new_pts = []
        for i in range(len(pts) - 1):
            p, q = pts[i], pts[i + 1]
            Q = (0.75 * p[0] + 0.25 * q[0], 0.75 * p[1] + 0.25 * q[1])
            R = (0.25 * p[0] + 0.75 * q[0], 0.25 * p[1] + 0.75 * q[1])
            new_pts.extend([Q, R])
        if closed:
            new_pts.append(new_pts[0])
        pts = new_pts
    if closed and len(pts) > 1:
        pts = pts[:-1]
    return pts


def chaikin_mixed(points: List[Tuple[float, float]], sharp_vertices: set[int], iterations: int = 2, closed: bool = True) -> List[Tuple[float, float]]:
    if not points:
        return []
    if len(points) < 3 or not sharp_vertices:
        return chaikin(points, iterations=iterations, closed=closed)
    expanded: List[Tuple[float, float]] = []
    for i, p in enumerate(points):
        if i in sharp_vertices:
            # 鋭角頂点は重複させて角を残す（open/closed 共通）
            expanded.extend([p, p, p])
        else:
            expanded.append(p)
    return chaikin(expanded, iterations=iterations, closed=closed)


def build_draw_points(
    points: List[Tuple[float, float]],
    smooth: bool,
    sharp_vertices: List[int],
    closed: bool = True
) -> List[Tuple[float, float]]:
    if not smooth:
        return points
    s = set(int(i) for i in (sharp_vertices or []))
    if s:
        return chaikin_mixed(points, s, iterations=2, closed=closed)
    return chaikin(points, iterations=2, closed=closed)


# ------------------------------------------------------------
# Data classes
# ------------------------------------------------------------

@dataclass
class Region:
    label: str
    label_id: int
    color: str
    points: List[Tuple[float, float]] = field(default_factory=list)  # (x, y) in full image
    smooth: bool = False
    sharp_vertices: List[int] = field(default_factory=list)
    open_path: bool = False  # Trueなら開いた折れ線（領域ではない）

    def to_dict(self) -> Dict:
        return {
            "label": self.label,
            "label_id": self.label_id,
            "color": self.color,
            "smooth": self.smooth,
            "points": [[float(x), float(y)] for (x, y) in self.points],
            "sharp_vertices": [int(i) for i in (self.sharp_vertices or [])],
            "open_path": bool(self.open_path),
        }

    @staticmethod
    def from_dict(d: Dict) -> "Region":
        pts = [(float(x), float(y)) for x, y in d.get("points", [])]
        return Region(
            label=d.get("label", "region"),
            label_id=int(d.get("label_id", 1)),
            color=d.get("color", "#ff7f0e"),
            points=pts,
            smooth=bool(d.get("smooth", False)),
            sharp_vertices=[int(i) for i in d.get("sharp_vertices", [])],
            open_path=bool(d.get("open_path", False)),
        )


@dataclass
class Project:
    image_path: str
    image_size: Tuple[int, int]  # (H, W)
    regions: List[Region] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(
            {
                "image_path": self.image_path,
                "image_size": [int(self.image_size[0]), int(self.image_size[1])],  # (H, W)
                "regions": [r.to_dict() for r in self.regions],
            },
            ensure_ascii=False,
            indent=2,
        )

    @staticmethod
    def from_json(s: str) -> "Project":
        d = json.loads(s)
        H, W = d.get("image_size", [0, 0])
        regions = [Region.from_dict(rd) for rd in d.get("regions", [])]
        return Project(image_path=d.get("image_path", ""), image_size=(int(H), int(W)), regions=regions)


# ------------------------------------------------------------
# Main widget
# ------------------------------------------------------------

class MaskAnnotator(VBox):
    def __init__(
        self,
        folder: str,
        max_display: int = 1000,
        allow_color_preview_export: bool = True,
        enable_wheel_zoom: bool = False,
        edge_margin_px: int = 48,
        file_filter: str = "",
        recursive: bool = False,
        # Viewport options
        show_scrollbars: bool = True,
        viewport_max_height: str = "80vh",
        viewport_height: Optional[str] = None,
        # Zoom range
        min_zoom_pct: int = 10,
        max_zoom_pct: int = 400,
        # Pan via scroll
        pan_via_scroll: bool = False,
    ) -> None:
        super().__init__()

        self.folder = Path(folder)
        assert self.folder.is_dir(), f"folder not found: {folder}"
        self.max_display = int(max_display)
        self.allow_color_preview_export = bool(allow_color_preview_export)
        self.enable_wheel_zoom = bool(enable_wheel_zoom)
        self._view_margin: int = int(edge_margin_px)
        self.file_filter = (file_filter or "").strip()
        self.recursive = bool(recursive)
        self._show_scrollbars = bool(show_scrollbars)
        self._viewport_max_height = str(viewport_max_height)
        self._viewport_height = str(viewport_height) if viewport_height else None
        self._pan_via_scroll = bool(pan_via_scroll)

        # Snapshot for canceling a new region
        self._pre_draw_snapshot_json: Optional[str] = None

        self.image_paths: List[Path] = self._collect_images()
        if len(self.image_paths) == 0:
            raise AssertionError(f"no images found under: {self.folder} (filter='{self.file_filter or 'ALL'}')")

        self.idx: int = 0
        self.project: Optional[Project] = None

        # View / scale
        self._img_rgba_view: Optional[np.ndarray] = None  # (H',W',4)
        self._img_rgba_full: Optional[np.ndarray] = None  # (H,W,4)
        self._img_size_full: Tuple[int, int] = (0, 0)     # (H, W)
        self._scale: float = 1.0
        self._zoom_pct: int = 100
        self._min_zoom: int = int(min_zoom_pct)
        self._max_zoom: int = int(max_zoom_pct)

        # Pan
        self._pan_x: float = 0.0
        self._pan_y: float = 0.0
        self._panning: bool = False
        self._pan_start: Optional[Tuple[float, float]] = None
        self._last_mouse_view_xy: Optional[Tuple[float, float]] = None

        # Editing state
        self._current_points: List[Tuple[float, float]] = []
        self._current_label_text: str = "region"
        self._current_label_id: int = 1
        self._current_color: str = _distinct_color(0)
        self._current_smooth: bool = False
        self._current_open: bool = False  # 描画中の種別
        self._mode: str = "idle"  # idle | draw | edit
        self._selected_region_idx: Optional[int] = None
        self._selected_vertex_idx: Optional[int] = None
        self._dragging: bool = False

        # Undo/Redo
        self._undo_stack: List[str] = []
        self._redo_stack: List[str] = []

        # Palette
        self._palette5: List[str] = ["#E69F00", "#56B4E9", "#009E73", "#D55E00", "#0072B2"]

        # Build UI and wire events
        self._build_widgets()
        self._wire_events()
        self._load_image_at_index(self.idx)

        try:
            self.canvas[2].add_class("pm-wheel-passive")
        except Exception:
            pass

        self.children = [self.toolbar, self.canvas_box, self.status_bar, self.help_box]

    # --------------------------- UI ---------------------------
    def _make_image_label(self, p: Path) -> str:
        try:
            jp = p.with_suffix("").with_name(p.stem + ".json")
            if jp.exists():
                return f"{p.name} [JSON]"
        except Exception:
            pass
        return p.name

    def _update_dd_image_labels(self) -> None:
        try:
            opts = [(self._make_image_label(p), i) for i, p in enumerate(self.image_paths)]
            val = self.dd_image.value
            self.dd_image.options = opts
            if val is not None and 0 <= int(val) < len(opts):
                self.dd_image.value = val
            elif opts:
                self.dd_image.value = 0
        except Exception:
            pass

    def _build_widgets(self) -> None:
        self.canvas = MultiCanvas(3, width=400, height=300, layout=Layout(border="1px solid #ccc"))
        self.canvas[2].line_width = 2
        self.canvas[1].global_alpha = 0.35

        self.dd_image = Dropdown(options=[(self._make_image_label(p), i) for i, p in enumerate(self.image_paths)], value=0, description="Image", layout=Layout(width="360px"))
        self.btn_prev = Button(description="◀ Prev", layout=Layout(width="80px"))
        self.btn_next = Button(description="Next ▶", layout=Layout(width="80px"))

        self.txt_label = Text(value="region", description="Label", layout=Layout(width="220px"))
        self.id_label = BoundedIntText(value=1, min=1, max=255, description="ID", layout=Layout(width="140px"))
        self.cp_color = ColorPicker(value=_distinct_color(0), description="Color")
        self.tb_smooth = ToggleButton(value=False, description="Smooth", tooltip="Chaikin smoothing")
        # Type: Area / Polyline
        self.tb_path_type = ToggleButtons(
            options=[("Area", "closed"), ("Polyline", "open")],
            value="closed",
            description="Type",
            tooltips=["塗りつぶし領域（閉）", "折れ線（開）"],
        )

        self.btn_new = Button(description="＋ New region", button_style="info")
        self.btn_finish = Button(description="✓ Finish", button_style="success")
        self.btn_cancel = Button(description="✗ Cancel", button_style="warning")
        self.btn_pop_last = Button(description="⌫ Last point", layout=Layout(width="120px"))

        self.dd_regions = Dropdown(options=[], description="Edit", layout=Layout(width="320px"))
        self.btn_edit = ToggleButton(value=False, description="Edit mode", tooltip="move/delete")
        self.btn_del_point = Button(description="Delete point", layout=Layout(width="120px"))
        self.btn_del_region = Button(description="Delete region", button_style="danger")
        self.cp_edit_color = ColorPicker(value=_distinct_color(0), description="Edit color")

        self.tb_vertex_state = ToggleButtons(
            options=[("Smooth", "smooth"), ("Sharp", "sharp")],
            description="Vertex",
            disabled=True
        )

        self.btn_insert_point = ToggleButton(value=False, description="Insert point", tooltip="add vertex on edge")

        self.txt_edit_label = Text(value="", description="Edit label", layout=Layout(width="220px"))
        self.btn_apply_label = Button(description="Apply", layout=Layout(width="70px"))

        # Zoom UI
        self.sl_zoom = IntSlider(value=100, min=self._min_zoom, max=self._max_zoom, step=5, description="Zoom %", layout=Layout(width="240px"))

        self.btn_zoom_fit = Button(description="Fit", layout=Layout(width="60px"))
        self.btn_zoom_100 = Button(description="100%", layout=Layout(width="60px"))

        self.btn_pan = ToggleButton(value=False, description="Pan", tooltip="drag to move view")
        self.btn_pan_center = Button(description="Center", layout=Layout(width="70px"))

        self.btn_undo = Button(description="Undo", layout=Layout(width="80px"))
        self.btn_redo = Button(description="Redo", layout=Layout(width="80px"))

        self.chk_export_index = Checkbox(value=True, description="Export index mask (uint8)")
        self.chk_color_preview = Checkbox(value=True, description="Export color preview (RGB)")
        self.chk_white_bg = Checkbox(value=True, description="White background (preview)")
        self.btn_save = Button(description="Save JSON")
        self.btn_load = Button(description="Load JSON")
        self.btn_export = Button(description="Export PNG mask", button_style="primary")

        self.palette_new = HBox(self._make_palette_buttons(self._palette5, context="new"), layout=Layout(gap="4px"))

        self.palette_edit = HBox(self._make_palette_buttons(self._palette5, context="edit"), layout=Layout(gap="4px"))

        self.txt_filter = Text(value=self.file_filter, description="Filter", placeholder="e.g. *.png, *_BM.jpg", layout=Layout(width="280px"))
        self.chk_recursive = Checkbox(value=self.recursive, description="Recursive")
        self.btn_apply_filter = Button(description="Apply filter", layout=Layout(width="120px"))

        self.status_bar = HTML(value="Ready")
        self.help_box = HTML(
            value=(
                "<details><summary><b>How to use (v1.8.2)</b></summary>"
                "<div style='line-height:1.5; margin-top:0.5em'>"
                "<h4>Pan &amp; Zoom</h4>"
                f"<p>Zoom range: <b>{self._min_zoom}–{self._max_zoom}%</b>. Use slider / Fit / 100% / mouse wheel.</p>"
                "<p>Pan: toggle <b>Pan</b>. If <i>pan_via_scroll=True</i>, left-drag in the viewport scrolls without redraw.</p>"
                "<h4>Area vs Polyline</h4>"
                "<ul>"
                "<li><b>Area</b>: 従来通りの閉領域。Finish後は塗りと境界線が描かれ、Exportでは塗りを出力。</li>"
                "<li><b>Polyline</b>: 開いた折れ線。Finish後もクローズしません。Exportでは1pxの線を出力します。</li>"
                "</ul>"
                "<h4>Toolbar layout</h4>"
                "<p>Type の直後で改行し、2行目に New/Finish/Cancel/Last/Color をまとめています。</p>"
                "</div>"
                "</details>"
            )
        )

        row1 = HBox([self.dd_image, self.btn_prev, self.btn_next, self.btn_undo, self.btn_redo], layout=Layout(gap="8px", align_items="center", flex_flow="row wrap", width="100%"))

        # ---- Toolbar rows (改行レイアウト) ----
        # 1行目: Label, ID, Smooth, Type（ここで改行）
        row2a = HBox([
            self.txt_label, self.id_label, self.tb_smooth, self.tb_path_type
        ], layout=Layout(gap="8px", align_items="center", flex_flow="row wrap", width="100%"))

        # 2行目: New, Finish, Cancel, Last point, Color, パレット
        row2b = HBox([
            self.btn_new, self.btn_finish, self.btn_cancel, self.btn_pop_last, self.cp_color, self.palette_new
        ], layout=Layout(gap="8px", align_items="center", flex_flow="row wrap", width="100%"))

        # まとめて VBox に
        row2 = VBox([row2a, row2b], layout=Layout(gap="6px", width="100%"))

        row3 = HBox([
            self.dd_regions, self.btn_edit, self.btn_del_point, self.btn_del_region,
            self.cp_edit_color, self.palette_edit, self.tb_vertex_state, self.btn_insert_point,
            self.txt_edit_label, self.btn_apply_label,
        ], layout=Layout(gap="8px", align_items="center", flex_flow="row wrap", width="100%"))

        row4_elems = [self.btn_save, self.btn_load, self.btn_export]
        if self.allow_color_preview_export:
            row4_elems = [self.chk_export_index, self.chk_color_preview, self.chk_white_bg] + row4_elems
        row4 = HBox(row4_elems, layout=Layout(gap="8px", align_items="center", flex_flow="row wrap", width="100%"))

        row1.children = tuple(list(row1.children) + [
            self.txt_filter, self.chk_recursive, self.btn_apply_filter,
            self.sl_zoom, self.btn_zoom_fit, self.btn_zoom_100, self.btn_pan, self.btn_pan_center
        ])

        # Scrollable image area
        if self._show_scrollbars:
            if self._viewport_height:
                canvas_box_layout = Layout(
                    overflow='auto', overflow_x='auto', overflow_y='auto',
                    width='100%', height=self._viewport_height
                )
            else:
                canvas_box_layout = Layout(
                    overflow='auto', overflow_x='auto', overflow_y='auto',
                    width='100%', max_height=self._viewport_max_height
                )
        else:
            canvas_box_layout = Layout(overflow="visible")

        self.toolbar = VBox([row1, row2, row3, row4])
        self.canvas_box = HBox([self.canvas], layout=canvas_box_layout)
        try:
            self.canvas_box.add_class("pm-wheel-passive")
            self.canvas_box.add_class("pm-drag-scroll")
        except Exception:
            pass

    # --------------------------- Event wiring ---------------------------
    def _wire_events(self) -> None:
        # Image nav
        self.dd_image.observe(self._on_select_image, names="value")
        self.btn_prev.on_click(self._on_prev)
        self.btn_next.on_click(self._on_next)

        # New region
        self.btn_new.on_click(self._on_new_region)
        self.btn_finish.on_click(self._on_finish_click)
        self.btn_cancel.on_click(self._on_cancel_click)
        self.btn_pop_last.on_click(self._on_pop_last_click)

        # Edit
        self.btn_edit.observe(lambda ch: self._toggle_edit_mode(ch["new"]), names="value")
        self.btn_del_point.on_click(lambda b: self._delete_selected_point())
        self.btn_del_region.on_click(lambda b: self._delete_selected_region())
        self.dd_regions.observe(self._on_pick_region, names="value")
        self.cp_edit_color.observe(self._on_edit_color_change, names="value")
        self.tb_vertex_state.observe(self._on_vertex_state_change, names="value")
        self.btn_insert_point.observe(lambda ch: self._set_status("Insert mode ON" if ch.get("new", False) else "Insert mode OFF"), names="value")
        self.btn_apply_label.on_click(lambda b: self._apply_label_change())

        # Undo/Redo
        self.btn_undo.on_click(lambda b: self._undo())
        self.btn_redo.on_click(lambda b: self._redo())

        # Save/Load/Export
        self.btn_save.on_click(lambda b: self._save_json())
        self.btn_load.on_click(lambda b: self._load_json_explicit())
        self.btn_export.on_click(lambda b: self._export_masks())

        # Zoom UI
        self.sl_zoom.observe(self._on_zoom_slider, names="value")
        self.btn_zoom_fit.on_click(self._on_zoom_fit)
        self.btn_zoom_100.on_click(self._on_zoom_100)

        # Pan
        self.btn_pan.observe(lambda ch: self._on_toggle_pan(ch.get("new", False)), names="value")
        self.btn_pan_center.on_click(lambda b: self._center_pan())

        # Canvas events: top layer and root
        top = self.canvas[2]
        root = self.canvas
        for target in (top, root):
            if hasattr(target, "on_mouse_down"):
                target.on_mouse_down(self._on_mouse_down)
            if hasattr(target, "on_mouse_move"):
                target.on_mouse_move(self._on_mouse_move)
            if hasattr(target, "on_mouse_up"):
                target.on_mouse_up(self._on_mouse_up)
            if hasattr(target, "on_mouse_out"):
                target.on_mouse_out(lambda *a, **k: self._on_mouse_up(0, 0))

        # Filter
        self.btn_apply_filter.on_click(lambda b: self._apply_filter())
        try:
            if hasattr(self.txt_filter, 'on_submit'):
                self.txt_filter.on_submit(lambda _=None: self._apply_filter())
        except Exception:
            pass
        try:
            def _on_submit_trait(change):
                if change.get('name') == 'submit' and change.get('new', False):
                    self._apply_filter()
            self.txt_filter.observe(_on_submit_trait, names='submit')
        except Exception:
            pass
        self.chk_recursive.observe(lambda ch: self._apply_filter(), names='value')

        # Wheel zoom
        if self.enable_wheel_zoom:
            for target in (top, root):
                for evt in ('on_wheel', 'on_mouse_wheel'):
                    if hasattr(target, evt):
                        getattr(target, evt)(self._on_wheel)
        else:
            for target in (top, root):
                for evt in ('on_wheel', 'on_mouse_wheel'):
                    if hasattr(target, evt):
                        try:
                            getattr(target, evt)(None)  # unregister
                        except Exception:
                            pass

    # --------------------------- File listing / filter ---------------------------
    def _collect_images(self) -> List[Path]:
        txt = (getattr(self, "txt_filter", None).value if getattr(self, "txt_filter", None) else self.file_filter).strip()
        patterns = [p.strip() for p in re.split(r'[ ,\t\n]+', txt) if p.strip()]
        recursive = (getattr(self, "chk_recursive", None).value if getattr(self, "chk_recursive", None) else self.recursive)

        paths: List[Path] = []
        if not patterns:
            it = self.folder.rglob("*") if recursive else self.folder.iterdir()
            for p in it:
                try:
                    if p.is_file() and p.suffix.lower() in ALLOWED_EXTS:
                        paths.append(p)
                except Exception:
                    pass
        else:
            for pat in patterns:
                try:
                    it = self.folder.rglob(pat) if recursive else self.folder.glob(pat)
                    for p in it:
                        if p.is_file() and p.suffix.lower() in ALLOWED_EXTS:
                            paths.append(p)
                except Exception:
                    pass

        uniq: Dict[str, Path] = {str(p): p for p in paths}
        return sorted(uniq.values(), key=lambda x: x.name.lower())

    def _refresh_file_list(self, keep_current: bool = False) -> None:
        old = self.image_paths[self.idx] if getattr(self, "image_paths", None) and self.image_paths and 0 <= self.idx < len(self.image_paths) else None
        self.image_paths = self._collect_images()
        if not self.image_paths:
            self.dd_image.options = []
            self.dd_image.value = None
            self._clear_all_layers()
            self._set_status("No images match the filter")
            return

        if keep_current and old in self.image_paths:
            self.idx = self.image_paths.index(old)
        else:
            self.idx = 0

        self.dd_image.options = [(self._make_image_label(p), i) for i, p in enumerate(self.image_paths)]
        self.dd_image.value = self.idx
        self._set_status(f"{len(self.image_paths)} images listed")

    def _apply_filter(self) -> None:
        self.file_filter = (self.txt_filter.value or "").strip()
        self.recursive = bool(self.chk_recursive.value)
        self._save_autosave_if_any()
        self._refresh_file_list(keep_current=False)

    # --------------------------- Image loading / drawing ---------------------------
    def _on_prev(self, _=None): self._nav_image(-1)
    def _on_next(self, _=None): self._nav_image(+1)

    def _on_select_image(self, change) -> None:
        val = change["new"] if isinstance(change, dict) and "new" in change else self.dd_image.value
        if val is None:
            self._clear_all_layers()
            self._set_status("No image selected")
            return
        new_idx = int(val)
        self._save_autosave_if_any()
        self._load_image_at_index(new_idx)

    def _nav_image(self, step: int) -> None:
        new_idx = (self.idx + step) % len(self.image_paths)
        self.dd_image.value = new_idx

    def _load_image_at_index(self, idx: int) -> None:
        self.idx = int(idx)
        path = self.image_paths[self.idx]

        im = Image.open(path).convert("RGBA")
        W_full, H_full = im.size
        self._img_size_full = (H_full, W_full)            # (H, W)
        self._img_rgba_full = np.asarray(im, dtype=np.uint8)

        init_scale = min(self.max_display / max(W_full, H_full), 1.0)
        init_pct = max(self._min_zoom, min(self._max_zoom, int(round(init_scale * 100))))

        self._apply_zoom(init_pct, sync_slider=True)

        self._pan_x = 0.0
        self._pan_y = 0.0

        self.project = Project(image_path=str(path), image_size=self._img_size_full, regions=[])
        self._undo_stack.clear(); self._redo_stack.clear()
        self._current_points.clear(); self._mode = "idle"
        self._selected_region_idx = None; self._selected_vertex_idx = None; self._dragging = False
        self._pre_draw_snapshot_json = None

        auto_json = self._default_json_path()
        if auto_json.exists():
            try:
                with open(auto_json, "r", encoding="utf-8") as f:
                    pj = Project.from_json(f.read())
                if tuple(pj.image_size) == tuple(self._img_size_full):
                    self.project = pj
                    self._push_undo_snapshot()
                    self._sync_region_dropdown()
                    self._redraw_all()
                    self._set_status(f"Loaded existing JSON: {auto_json.name}")
                else:
                    self._sync_region_dropdown()
                    self._redraw_all()
                    self._set_status("JSON exists but image size mismatch; ignored.")
            except Exception:
                self._sync_region_dropdown()
                self._redraw_all()
                self._set_status("Found JSON but failed to load; ignored.")
        else:
            self._sync_region_dropdown()
            self._redraw_all()
            self._set_status(f"Loaded image: {path.name} ({W_full}x{H_full})")

    def _default_json_path(self) -> Path:
        img = Path(self.project.image_path) if self.project else self.image_paths[self.idx]
        return img.with_suffix("").with_name(img.stem + ".json")

    # --------------------------- Coordinates ---------------------------
    def _view_to_full(self, x: float, y: float) -> Tuple[float, float]:
        s = self._scale if self._scale > 0 else 1.0
        return ((x - self._view_margin - self._pan_x) / s, (y - self._view_margin - self._pan_y) / s)

    def _full_to_view(self, x: float, y: float) -> Tuple[float, float]:
        s = self._scale
        return (x * s + self._view_margin + self._pan_x, y * s + self._view_margin + self._pan_y)

    # --------------------------- Drawing ---------------------------
    def _clear_all_layers(self) -> None:
        for i in range(3):
            self.canvas[i].clear()

    def _redraw_all(self) -> None:
        if self._img_rgba_view is None:
            return
        with hold_canvas(self.canvas[0]):
            self.canvas[0].clear()
            self.canvas[0].put_image_data(self._img_rgba_view, int(self._view_margin + self._pan_x), int(self._view_margin + self._pan_y))
        self._redraw_regions()
        self._redraw_current()

    def _redraw_regions(self) -> None:
        c = self.canvas[1]
        c.clear()
        if not self.project:
            return
        for ridx, r in enumerate(self.project.regions):
            pts = r.points
            min_pts = 2 if r.open_path else 3
            if len(pts) < min_pts:
                continue
            draw_pts = build_draw_points(pts, r.smooth, getattr(r, "sharp_vertices", []), closed=(not r.open_path))
            draw_pts_view = [self._full_to_view(x, y) for (x, y) in draw_pts]

            c.global_alpha = 1.0
            c.stroke_style = r.color
            c.line_width = 2

            # 塗り（閉領域のみ）
            if not r.open_path:
                c.global_alpha = 0.35
                c.fill_style = r.color
                c.begin_path()
                x0, y0 = draw_pts_view[0]
                c.move_to(x0, y0)
                for (x, y) in draw_pts_view[1:]:
                    c.line_to(x, y)
                c.close_path()
                c.fill()
                c.global_alpha = 1.0

            # 境界線（open/closed 共通。open はクローズしない）
            c.begin_path()
            x0, y0 = draw_pts_view[0]
            c.move_to(x0, y0)
            for (x, y) in draw_pts_view[1:]:
                c.line_to(x, y)
            if not r.open_path:
                c.close_path()
            c.stroke()

            # 頂点ハンドル（Edit中で選択中）
            if self._mode == "edit" and self._selected_region_idx == ridx:
                for vidx, (px, py) in enumerate([self._full_to_view(px, py) for (px, py) in r.points]):
                    self._draw_handle(c, px, py, selected=(vidx == self._selected_vertex_idx))

    def _redraw_current(self) -> None:
        c = self.canvas[2]
        c.clear()
        if self._mode == "draw" and len(self._current_points) > 0:
            # プレビューは常に開いた形状で表示（閉でも描画途中でクローズしない）
            draw_pts = chaikin(self._current_points, iterations=2, closed=False) if self._current_smooth else self._current_points
            view_pts = [self._full_to_view(x, y) for (x, y) in draw_pts]
            c.stroke_style = self._current_color
            c.line_width = 2
            c.begin_path()
            x0, y0 = view_pts[0]
            c.move_to(x0, y0)
            for (x, y) in view_pts[1:]:
                c.line_to(x, y)
            c.stroke()
            fx, fy = self._full_to_view(*self._current_points[0])
            lx, ly = self._full_to_view(*self._current_points[-1])
            self._draw_handle(c, fx, fy, selected=False)
            self._draw_handle(c, lx, ly, selected=True)

    def _draw_handle(self, c, x: float, y: float, selected: bool = False) -> None:
        r = 5 if selected else 4
        c.begin_path(); c.arc(x, y, r, 0, 2 * math.pi)
        c.fill_style = "#ffffff" if selected else "#000000"
        c.fill(); c.stroke_style = "#333333"; c.line_width = 1; c.stroke()

    # --------------------------- Undo/Redo ---------------------------
    def _push_undo_snapshot(self) -> None:
        if not self.project:
            return
        self._undo_stack.append(self.project.to_json())
        if len(self._undo_stack) > 200:
            self._undo_stack.pop(0)
        self._redo_stack.clear()

    def _restore_from_json(self, s: str) -> None:
        pj = Project.from_json(s)
        self.project = pj
        self._sync_region_dropdown()
        self._redraw_all()

    def _undo(self) -> None:
        if len(self._undo_stack) == 0 or not self.project:
            self._set_status("Nothing to undo")
            return
        cur = self.project.to_json()
        prev = self._undo_stack.pop()
        self._redo_stack.append(cur)
        self._restore_from_json(prev)
        self._set_status("Undo")

    def _redo(self) -> None:
        if len(self._redo_stack) == 0 or not self.project:
            self._set_status("Nothing to redo")
            return
        cur = self.project.to_json()
        nxt = self._redo_stack.pop()
        self._undo_stack.append(cur)
        self._restore_from_json(nxt)
        self._set_status("Redo")

    # --------------------------- New region ---------------------------
    def _on_new_region(self, btn) -> None:
        if self._mode == "draw" and len(self._current_points) > 0:
            self._set_status("Already drawing; Finish or Cancel first.")
            return
        self._pre_draw_snapshot_json = self.project.to_json() if self.project else None

        self._current_label_text = (self.txt_label.value or "region").strip()
        self._current_label_id = int(self.id_label.value)
        self._current_color = self.cp_color.value
        self._current_smooth = bool(self.tb_smooth.value)
        self._current_open = (self.tb_path_type.value == "open")
        self._current_points = []
        self._mode = "draw"
        self._selected_region_idx = None
        self._selected_vertex_idx = None
        self._redraw_current()
        if self._current_open:
            self._set_status("Drawing Polyline: click to add points. Right-click/⌫ to remove last. Press Finish to keep OPEN.")
        else:
            self._set_status("Drawing Area: click to add points. Right-click/⌫ to remove last. Click near first point or press Finish to close.")

    def _on_finish_click(self, b=None): self._finish_region()
    def _on_cancel_click(self, b=None): self._cancel_region()
    def _on_pop_last_click(self, b=None): self._pop_last_point()

    def _finish_region(self) -> None:
        if self._mode != "draw":
            self._set_status("Not in drawing mode.")
            return
        min_pts = 2 if self._current_open else 3
        if len(self._current_points) < min_pts:
            self._set_status(f"Need at least {min_pts} points to finish.")
            return
        r = Region(
            label=self._current_label_text,
            label_id=self._current_label_id,
            color=self._current_color,
            points=self._current_points[:],
            smooth=self._current_smooth,
            open_path=bool(self._current_open),
        )
        if self.project is None:
            self._set_status("No active project")
            return
        self.project.regions.append(r)
        self._push_undo_snapshot()
        self._mode = "idle"
        self._current_points.clear()
        self._pre_draw_snapshot_json = None
        self._sync_region_dropdown()
        self._redraw_all()
        typ = "Polyline (open)" if r.open_path else "Area (closed)"
        self._set_status(f"Region added: {r.label} (id={r.label_id}, {typ})")

    def _cancel_region(self) -> None:
        had_points = len(self._current_points) > 0
        self._current_points.clear()
        self._mode = "idle"
        if self._pre_draw_snapshot_json:
            try:
                self._restore_from_json(self._pre_draw_snapshot_json)
            except Exception:
                pass
        self._pre_draw_snapshot_json = None
        self._redraw_current()
        self._set_status("Canceled drawing and restored pre-draw state" if had_points else "Nothing to cancel")

    def _pop_last_point(self) -> None:
        if self._mode == "draw" and len(self._current_points) > 0:
            self._current_points.pop()
            self._redraw_current()
            self._set_status("Last point removed")
        else:
            self._set_status("No point to remove")

    # --------------------------- Edit mode ---------------------------
    def _toggle_edit_mode(self, on: bool) -> None:
        self._mode = "edit" if on else "idle"
        if on and self.project and self.project.regions:
            self._selected_region_idx = 0 if self.dd_regions.value is None else int(self.dd_regions.value)
        else:
            self._selected_region_idx = None
            self._selected_vertex_idx = None
        try:
            self.tb_vertex_state.disabled = True
        except Exception:
            pass
        self._redraw_all()
        self._set_status("Edit mode ON" if on else "Edit mode OFF")

    def _on_pick_region(self, change) -> None:
        if change and "new" in change:
            self._selected_region_idx = int(change["new"]) if change["new"] is not None else None
            self._selected_vertex_idx = None
            if self._selected_region_idx is not None and self.project:
                try:
                    self.cp_edit_color.value = self.project.regions[self._selected_region_idx].color
                    self.txt_edit_label.value = self.project.regions[self._selected_region_idx].label
                except Exception:
                    pass
            try:
                self.tb_vertex_state.disabled = True
            except Exception:
                pass
            self._redraw_all()

    def _delete_selected_point(self) -> None:
        ridx = self._selected_region_idx
        vidx = self._selected_vertex_idx
        if self._mode != "edit" or ridx is None or vidx is None or not self.project:
            self._set_status("No point selected")
            return
        r = self.project.regions[ridx]
        min_pts = 2 if r.open_path else 3
        if len(r.points) <= min_pts:
            self._set_status("Not enough points to keep the shape")
            return
        del r.points[vidx]
        if hasattr(r, "sharp_vertices") and r.sharp_vertices is not None:
            r.sharp_vertices = [i if i < vidx else i - 1 for i in r.sharp_vertices if i != vidx]
        self._selected_vertex_idx = None
        self._push_undo_snapshot()
        self._redraw_all()
        self._set_status("Point deleted")

    def _delete_selected_region(self) -> None:
        ridx = self._selected_region_idx
        if ridx is None or not self.project:
            self._set_status("No region selected")
            return
        del self.project.regions[ridx]
        self._selected_region_idx = None
        self._selected_vertex_idx = None
        self._push_undo_snapshot()
        self._sync_region_dropdown()
        self._redraw_all()
        self._set_status("Region deleted")

    def _on_edit_color_change(self, change) -> None:
        if self._mode != "edit" or self._selected_region_idx is None or not self.project:
            return
        col = change.get("new") if isinstance(change, dict) else None
        if not col:
            return
        self.project.regions[self._selected_region_idx].color = col
        self._push_undo_snapshot()
        self._redraw_all()
        self._set_status("Region color updated")

    def _on_vertex_state_change(self, change) -> None:
        if self._mode != "edit" or self._selected_region_idx is None or self._selected_vertex_idx is None or not self.project:
            return
        val = change.get("new") if isinstance(change, dict) else None
        if val not in ("smooth", "sharp"):
            return
        r = self.project.regions[self._selected_region_idx]
        if not hasattr(r, "sharp_vertices") or r.sharp_vertices is None:
            r.sharp_vertices = []
        if val == "sharp":
            if self._selected_vertex_idx not in r.sharp_vertices:
                r.sharp_vertices.append(self._selected_vertex_idx)
        else:
            if self._selected_vertex_idx in r.sharp_vertices:
                r.sharp_vertices.remove(self._selected_vertex_idx)
        self._push_undo_snapshot()
        self._redraw_all()

    def _apply_label_change(self) -> None:
        if self._mode != "edit" or self._selected_region_idx is None or not self.project:
            return
        txt = (self.txt_edit_label.value or "").strip()
        if not txt:
            self._set_status("Empty label ignored")
            return
        self.project.regions[self._selected_region_idx].label = txt
        self._push_undo_snapshot()
        self._sync_region_dropdown()
        self._set_status("Label updated")

    def _apply_palette_color(self, color_hex: str, context: str) -> None:
        if context == "new":
            self.cp_color.value = color_hex
            self._set_status(f"New region color = {color_hex}")
        else:
            self.cp_edit_color.value = color_hex
            if self._mode == "edit" and self._selected_region_idx is not None and self.project:
                self.project.regions[self._selected_region_idx].color = color_hex
                self._push_undo_snapshot()
                self._redraw_all()
                self._set_status(f"Region color = {color_hex}")

    def _make_palette_buttons(self, colors: List[str], context: str) -> List[Button]:
        btns: List[Button] = []
        for col in colors:
            b = Button(description="", tooltip=col, layout=Layout(width="24px", height="24px", padding="0"))
            try:
                b.style.button_color = col
            except Exception:
                pass
            b.on_click(lambda _btn, c=col, ctx=context: self._apply_palette_color(c, ctx))
            btns.append(b)
        return btns

    def _insert_point_at(self, fx: float, fy: float) -> None:
        ridx = self._selected_region_idx
        if ridx is None or not self.project:
            self._set_status("Select a region first")
            return
        r = self.project.regions[ridx]
        if not r.points:
            r.points.append((float(fx), float(fy)))
            self._push_undo_snapshot()
            self._redraw_all()
            return

        pts = r.points
        best_i, best_d, best_proj = 0, 1e18, (fx, fy)
        n = len(pts)
        edge_count = n if not r.open_path else (n - 1)
        if edge_count <= 0:
            r.points.append((float(fx), float(fy)))
            self._push_undo_snapshot()
            self._redraw_all()
            return

        for i in range(edge_count):
            x1, y1 = pts[i]
            x2, y2 = pts[(i + 1) % n] if not r.open_path else pts[i + 1]
            vx, vy = (x2 - x1), (y2 - y1)
            wx, wy = (fx - x1), (fy - y1)
            denom = vx * vx + vy * vy + 1e-12
            t = max(0.0, min(1.0, (wx * vx + wy * wy) / denom))
            px, py = (x1 + t * vx, y1 + t * vy)
            d = math.hypot(fx - px, fy - py)
            if d < best_d:
                best_d, best_i, best_proj = d, i, (px, py)
        insert_idx = best_i + 1
        r.points.insert(insert_idx, (float(best_proj[0]), float(best_proj[1])))
        self._selected_vertex_idx = insert_idx
        self._push_undo_snapshot()
        self._redraw_all()
        if not r.open_path:
            self._set_status(f"Point inserted between {best_i} and {(best_i + 1) % n}")
        else:
            self._set_status(f"Point inserted between {best_i} and {best_i + 1}")

    # --------------------------- Mouse handlers ---------------------------
    def _on_mouse_down(self, *args, **kwargs) -> None:
        if len(args) >= 2:
            x, y = float(args[0]), float(args[1])
        else:
            return
        button = kwargs.get("button", None)
        buttons = kwargs.get("buttons", None)
        if button is None and len(args) >= 3 and isinstance(args[2], (int, float)):
            button = int(args[2])
        if buttons is None and len(args) >= 3 and isinstance(args[2], int) and button is None:
            buttons = args[2]
        right = False
        try:
            if button is not None:
                right = (int(button) == 2)
            elif buttons is not None:
                right = (int(buttons) & 2) != 0
        except Exception:
            right = False

        self._last_mouse_view_xy = (x, y)

        # Pan
        if getattr(self, "btn_pan", None) is not None and self.btn_pan.value:
            self._panning = True
            self._pan_start = (x, y)
            return

        if self._mode == "draw":
            if right:
                self._pop_last_point()
            else:
                self._handle_draw_click(x, y)
        elif self._mode == "edit":
            if getattr(self, "btn_insert_point", None) is not None and self.btn_insert_point.value:
                fx, fy = self._view_to_full(x, y)
                self._insert_point_at(fx, fy)
                return
            self._handle_edit_mousedown(x, y)

    def _on_mouse_move(self, x: float, y: float) -> None:
        if self._panning and self._last_mouse_view_xy is not None:
            dx = x - self._last_mouse_view_xy[0]
            dy = y - self._last_mouse_view_xy[1]
            self._pan_x += dx
            self._pan_y += dy
            self._last_mouse_view_xy = (x, y)
            self._clamp_pan()
            self._redraw_all()
            return

        if self._mode == "edit" and self._dragging and self._selected_region_idx is not None and self._selected_vertex_idx is not None and self.project:
            fx, fy = self._view_to_full(x, y)
            r = self.project.regions[self._selected_region_idx]
            r.points[self._selected_vertex_idx] = (float(fx), float(fy))
            self._redraw_all()

    def _on_mouse_up(self, x: float, y: float) -> None:
        if self._panning:
            self._panning = False
            return
        if self._mode == "edit" and self._dragging:
            self._dragging = False
            self._push_undo_snapshot()

    # --------------------------- Edit interactions ---------------------------
    def _handle_edit_mousedown(self, vx: float, vy: float) -> None:
        fx, fy = self._view_to_full(vx, vy)
        ridx = self._hit_region(fx, fy)
        if ridx is None:
            self._selected_region_idx = None
            self._selected_vertex_idx = None
            self._redraw_all()
            return
        self._selected_region_idx = int(ridx)
        # Pick nearest vertex (in view space)
        r = self.project.regions[self._selected_region_idx] if self.project else None
        if r and r.points:
            best = (1e18, None)
            for i, (px, py) in enumerate(r.points):
                vx_i, vy_i = self._full_to_view(px, py)
                d = _euclid((vx, vy), (vx_i, vy_i))
                if d < best[0]:
                    best = (d, i)
            if best[0] <= 8.0:
                self._selected_vertex_idx = int(best[1]) if best[1] is not None else None
                try:
                    self.tb_vertex_state.disabled = False
                except Exception:
                    pass
                self._dragging = True
            else:
                self._selected_vertex_idx = None
                try:
                    self.tb_vertex_state.disabled = True
                except Exception:
                    pass
        self._sync_region_dropdown()
        self._redraw_all()

    # --------------------------- Draw click ---------------------------
    def _handle_draw_click(self, vx: float, vy: float) -> None:
        fx, fy = self._view_to_full(vx, vy)
        # Area（閉）時のみ「最初の点をクリックでFinish」
        if (not self._current_open) and len(self._current_points) >= 3:
            first_fx, first_fy = self._current_points[0]
            if _euclid((fx, fy), (first_fx, first_fy)) <= 8.0 / max(self._scale, 1e-6):
                self._finish_region()
                return
        self._current_points.append((fx, fy))
        self._redraw_current()

    # --------------------------- Pan / zoom ---------------------------
    def _clamp_pan(self) -> None:
        if self._img_rgba_view is None:
            return
        H_full, W_full = self._img_size_full  # (H, W)
        view_W, view_H = int(round(W_full * self._scale)), int(round(H_full * self._scale))
        self._pan_x = float(max(-view_W, min(0, self._pan_x)))
        self._pan_y = float(max(-view_H, min(0, self._pan_y)))

    def _on_toggle_pan(self, on: bool) -> None:
        self._panning = False
        try:
            if self._pan_via_scroll:
                if on:
                    self.canvas_box.add_class("pm-drag-scroll-active")
                else:
                    self.canvas_box.remove_class("pm-drag-scroll-active")
        except Exception:
            pass
        self._set_status("Pan ON" if on else "Pan OFF")

    def _center_pan(self) -> None:
        self._pan_x = 0.0
        self._pan_y = 0.0
        self._redraw_all()
        self._set_status("Centered")

    def _on_zoom_slider(self, change) -> None:
        try:
            new = int(change.get("new", self._zoom_pct))
        except Exception:
            new = self._zoom_pct
        self._apply_zoom(new, sync_slider=False)
        self._set_status(f"Zoom slider: {new}%")

    def _on_zoom_fit(self, _=None) -> None:
        if self._img_size_full == (0, 0):
            return
        H, W = self._img_size_full
        fit_scale = min(self.max_display / max(W, H), 1.0)
        pct = int(round(fit_scale * 100))
        self._apply_zoom(pct, sync_slider=True)
        self._set_status(f"Zoom Fit: {pct}%")

    def _on_zoom_100(self, _=None) -> None:
        self._apply_zoom(100, sync_slider=True)
        self._set_status("Zoom 100%")

    def _set_slider_value_safely(self, pct: int) -> None:
        try:
            self.sl_zoom.unobserve(self._on_zoom_slider, names="value")
            self.sl_zoom.value = int(pct)
        finally:
            self.sl_zoom.observe(self._on_zoom_slider, names="value")

    def _resize_canvas(self, view_W: int, view_H: int) -> None:
        view_W = int(max(1, view_W))
        view_H = int(max(1, view_H))
        try:
            self.canvas.width = view_W
            self.canvas.height = view_H
        except Exception:
            pass
        for i in range(3):
            try:
                self.canvas[i].width = view_W
                self.canvas[i].height = view_H
            except Exception:
                pass
        # Prevent CSS shrink/stretch by aligning CSS size to buffer
        try:
            self.canvas.layout.width = f"{view_W}px"
            self.canvas.layout.height = f"{view_H}px"
            self.canvas.layout.min_width = f"{view_W}px"
            self.canvas.layout.min_height = f"{view_H}px"
            self.canvas.layout.flex = "0 0 auto"
        except Exception:
            pass

    def _apply_zoom(self, zoom_pct: int, anchor_view_xy: Optional[Tuple[float, float]] = None, redraw_bg_only: bool = False, sync_slider: bool = False) -> None:
        if self._img_rgba_full is None:
            return
        z = int(max(self._min_zoom, min(self._max_zoom, zoom_pct)))
        self._zoom_pct = z
        self._scale = z / 100.0

        H_full, W_full = self._img_size_full  # (H, W)
        view_W = int(round(W_full * self._scale))
        view_H = int(round(H_full * self._scale))

        self._resize_canvas(view_W + 2 * self._view_margin, view_H + 2 * self._view_margin)

        im_view = Image.fromarray(self._img_rgba_full).resize((max(1, view_W), max(1, view_H)), Image.BILINEAR)
        self._img_rgba_view = np.asarray(im_view, dtype=np.uint8)

        self._clamp_pan()

        if redraw_bg_only:
            with hold_canvas(self.canvas[0]):
                self.canvas[0].clear()
                self.canvas[0].put_image_data(self._img_rgba_view, int(self._view_margin + self._pan_x), int(self._view_margin + self._pan_y))
        else:
            self._redraw_all()

        if sync_slider:
            try:
                self._set_slider_value_safely(self._zoom_pct)
            except Exception:
                pass

    # --------------------------- Wheel ---------------------------
    def _on_wheel(self, *args, **kwargs) -> None:
        dy = 0
        if len(args) >= 4:
            dy = args[3]
        elif "delta_y" in kwargs:
            dy = kwargs["delta_y"]
        elif "dy" in kwargs:
            dy = kwargs["dy"]
        step = 10  # 10% step
        new = self._zoom_pct + (-step if dy > 0 else step)
        self._apply_zoom(new, sync_slider=True)
        self._set_status(f"Wheel zoom: {self._zoom_pct}%")

    # --------------------------- JSON / Export ---------------------------
    def _save_autosave_if_any(self) -> None:
        if not self.project:
            return
        try:
            out = self._default_json_path()
            with open(out, "w", encoding="utf-8") as f:
                f.write(self.project.to_json())
            try:
                self._update_dd_image_labels()
            except Exception:
                pass
        except Exception:
            pass

    def _save_json(self) -> None:
        if not self.project:
            self._set_status("No project to save")
            return
        try:
            out = self._default_json_path()
            with open(out, "w", encoding="utf-8") as f:
                f.write(self.project.to_json())
            try:
                self._update_dd_image_labels()
            except Exception:
                pass
            self._set_status(f"Saved JSON: {out.name}")
        except Exception as e:
            self._set_status(f"Save failed: {e}")

    def _load_json_explicit(self) -> None:
        try:
            p = self._default_json_path()
            if not p.exists():
                self._set_status("JSON not found")
                return
            with open(p, "r", encoding="utf-8") as f:
                pj = Project.from_json(f.read())
            if tuple(pj.image_size) != tuple(self._img_size_full):
                self._set_status("Image size mismatch; JSON ignored")
                return
            self.project = pj
            self._undo_stack.clear(); self._redo_stack.clear()
            self._sync_region_dropdown()
            self._redraw_all()
            try:
                self._update_dd_image_labels()
            except Exception:
                pass
            self._set_status(f"Loaded JSON: {p.name}")
        except Exception as e:
            self._set_status(f"Load failed: {e}")

    def _hex_to_rgb(self, col: str) -> Tuple[int, int, int]:
        c = col.strip()
        if c.startswith("#"):
            c = c[1:]
        if len(c) == 3:
            c = "".join(ch*2 for ch in c)
        try:
            r = int(c[0:2], 16); g = int(c[2:4], 16); b = int(c[4:6], 16)
            return (r, g, b)
        except Exception:
            return (255, 0, 0)

    def _export_masks(self) -> None:
        if not self.project:
            self._set_status("Nothing to export")
            return

        img_path = Path(self.project.image_path)
        H, W = self._img_size_full
        idx_mask = np.zeros((H, W), dtype=np.uint8)
        rgb_mask = np.full((H, W, 3), 255 if self.chk_white_bg.value else 0, dtype=np.uint8)

        def _clip_rc(rr: np.ndarray, cc: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            m = (rr >= 0) & (rr < H) & (cc >= 0) & (cc < W)
            return rr[m], cc[m]

        for r in self.project.regions:
            min_pts = 2 if r.open_path else 3
            if len(r.points) < min_pts:
                continue
            pts = build_draw_points(r.points, r.smooth, getattr(r, "sharp_vertices", []), closed=(not r.open_path))

            if not r.open_path:
                # 塗りつぶし（従来）
                xs = np.asarray([np.clip(x, 0, W - 1) for x, _ in pts], dtype=np.float32)
                ys = np.asarray([np.clip(y, 0, H - 1) for _, y in pts], dtype=np.float32)
                rr, cc = sk_polygon(ys, xs, shape=(H, W))
                idx = int(np.clip(r.label_id, 0, 255))
                idx_mask[rr, cc] = idx
                cr, cg, cb = self._hex_to_rgb(r.color)
                rgb_mask[rr, cc, 0] = cr
                rgb_mask[rr, cc, 1] = cg
                rgb_mask[rr, cc, 2] = cb
            else:
                # 1px の折れ線を描画
                idx = int(np.clip(r.label_id, 0, 255))
                cr, cg, cb = self._hex_to_rgb(r.color)
                for i in range(len(pts) - 1):
                    x1, y1 = pts[i]
                    x2, y2 = pts[i + 1]
                    rr, cc = sk_line(int(round(y1)), int(round(x1)), int(round(y2)), int(round(x2)))
                    rr, cc = _clip_rc(rr, cc)
                    idx_mask[rr, cc] = idx
                    rgb_mask[rr, cc, 0] = cr
                    rgb_mask[rr, cc, 1] = cg
                    rgb_mask[rr, cc, 2] = cb

        saved_names = []
        out_idx = img_path.with_suffix("").with_name(img_path.stem + "_mask.png")
        if getattr(self, "chk_export_index", None) is None or self.chk_export_index.value:
            try:
                Image.fromarray(idx_mask, mode="L").save(out_idx)
                saved_names.append(out_idx.name)
            except Exception as e:
                self._set_status(f"Index export failed: {e}")
                return

        out_rgb = img_path.with_suffix("").with_name(img_path.stem + "_mask_color.png")
        if getattr(self, "chk_color_preview", None) is None or self.chk_color_preview.value:
            try:
                Image.fromarray(rgb_mask, mode="RGB").save(out_rgb)
                saved_names.append(out_rgb.name)
            except Exception as e:
                self._set_status(f"Preview export failed: {e}")
                return

        if saved_names:
            self._set_status("Exported: " + ", ".join(saved_names))
        else:
            self._set_status("Nothing exported (all outputs off)")

    def _sync_region_dropdown(self) -> None:
        if not getattr(self, "dd_regions", None):
            return
        if not self.project or len(self.project.regions) == 0:
            self.dd_regions.options = []
            self.dd_regions.value = None
            self._selected_region_idx = None
            self._selected_vertex_idx = None
            try:
                if getattr(self, "tb_vertex_state", None) is not None:
                    self.tb_vertex_state.disabled = True
            except Exception:
                pass
            return

        opts = []
        for i, r in enumerate(self.project.regions):
            open_tag = ", open" if r.open_path else ""
            opts.append((f"#{i}: {r.label} (id={r.label_id}{open_tag})", i))
        self.dd_regions.options = opts

        if self._selected_region_idx is None or self._selected_region_idx >= len(opts):
            self._selected_region_idx = 0
        self.dd_regions.value = self._selected_region_idx

        try:
            if getattr(self, "cp_edit_color", None) is not None:
                self.cp_edit_color.value = self.project.regions[self._selected_region_idx].color
            if getattr(self, "tb_vertex_state", None) is not None:
                self.tb_vertex_state.disabled = True
        except Exception:
            pass

    def _hit_region(self, fx: float, fy: float) -> Optional[int]:
        """クリックによるリージョン選択。
        - closed: 点内判定（従来）
        - open  : 線分への距離しきい値で判定
        """
        if not self.project:
            return None
        tol = 6.0 / max(self._scale, 1e-6)  # 画面上およそ 6px
        for i, r in enumerate(self.project.regions):
            pts = r.points
            if len(pts) < (2 if r.open_path else 3):
                continue
            if not r.open_path:
                # 内外判定（奇偶規則）
                inside = False
                n = len(pts)
                for j in range(n):
                    x1, y1 = pts[j]
                    x2, y2 = pts[(j + 1) % n]
                    if ((y1 > fy) != (y2 > fy)) and (fx < (x2 - x1) * (fy - y1) / (y2 - y1 + 1e-12) + x1):
                        inside = not inside
                if inside:
                    return i
            else:
                # 線分への最近距離
                near = False
                for j in range(len(pts) - 1):
                    x1, y1 = pts[j]
                    x2, y2 = pts[j + 1]
                    if _dist_point_to_segment(fx, fy, x1, y1, x2, y2) <= tol:
                        near = True
                        break
                if near:
                    return i
        return None

    # --------------------------- Status ---------------------------
    def _set_status(self, msg: str) -> None:
        try:
            self.status_bar.value = msg
        except Exception:
            pass
