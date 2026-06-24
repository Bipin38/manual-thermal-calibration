#!/usr/bin/env python3
"""
Manual RGB–thermal alignment tool.

Loads left RGB and rgb_matched thermal per scene, lets you fine-tune alignment
with translation (and optional scale/rotation), then saves updated:

    thermal/rgb_matched__manual_calibrated_thermal.npy
    thermal/rgb_matched__manual_calibrated_thermal.png

Original rgb_matched_thermal.* files are left unchanged.

Input:  thermal/rgb_matched_thermal.npy
Output: thermal/rgb_matched__manual_calibrated_thermal.npy + .png

Controls
--------
Trackbars (control panel window):
    dx, dy        Translation in pixels (thermal shifted to match RGB)
    scale x100    Uniform scale around image center (100 = 1.00)
    angle x10     Rotation in tenths of a degree
    opacity       Overlay blend strength (0–100)

Keyboard (main preview window, must be focused):
    Arrow keys    Nudge thermal by 1 px (platform-dependent)
    h j k l       Fine nudge left/down/up/right (1 px)
    H J K L       Coarse nudge (5 px)
    o             Toggle overlay / side-by-side view
    e             Toggle edge overlay (helps spot misalignment)
    r             Reset transform to zero
    s             Save current scene and go to next
    n / p         Next / previous scene (without saving)
    q / Esc       Quit

Usage
-----
From repo root:
    .venv/bin/python manual_thermal_calibration.py/manual_thermal_calibrator.py --dataset Dataset
    .venv/bin/streamlit run manual_thermal_calibration.py/thermal_calibrator_app.py

From this folder:
    ../.venv/bin/python manual_thermal_calibrator.py --dataset Dataset
    ../.venv/bin/streamlit run thermal_calibrator_app.py
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import List, Optional, Tuple

import cv2
import numpy as np

CALIBRATOR_DIR = os.path.dirname(os.path.abspath(__file__))


def find_repo_root(start: str | None = None) -> str:
    """Locate dataset repo root (folder containing Dataset/ or new data/)."""
    cur = os.path.abspath(start or CALIBRATOR_DIR)
    for _ in range(6):
        for name in ("Dataset", "new data"):
            if os.path.isdir(os.path.join(cur, name)):
                return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent
    return os.path.dirname(CALIBRATOR_DIR)


REPO_ROOT = find_repo_root()
BASE_DIR = REPO_ROOT
PROGRESS_FILE = os.path.join(CALIBRATOR_DIR, "thermal_calibration_progress.json")
BACKUP_NAME = "rgb_matched_thermal_pre_manual.npy"  # legacy; no longer written

THERMAL_NPY = "rgb_matched_thermal.npy"
CALIBRATED_THERMAL_NPY = "rgb_matched__manual_calibrated_thermal.npy"
CALIBRATED_THERMAL_PNG = "rgb_matched__manual_calibrated_thermal.png"

T_MIN = 20.0
T_MAX = 60.0

# Trackbar centers (value = center + offset)
DX_CENTER = 100
DY_CENTER = 100
SCALE_CENTER = 100
ANGLE_CENTER = 0
OPACITY_DEFAULT = 55

MAX_ABS_SHIFT = 100


def natural_sort_key(s: str):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", s)]


def render_thermal_colormap(thermal: np.ndarray) -> np.ndarray:
    """Inferno BGR uint8 image using fixed 20–60 °C range."""
    valid = np.nan_to_num(thermal, nan=T_MIN, posinf=T_MAX, neginf=T_MIN)
    span = max(T_MAX - T_MIN, 1e-6)
    normalized = np.clip((valid - T_MIN) / span * 255.0, 0, 255).astype(np.uint8)
    return cv2.applyColorMap(normalized, cv2.COLORMAP_INFERNO)


def render_thermal_minmax(thermal: np.ndarray) -> np.ndarray:
    valid = np.nan_to_num(thermal, nan=0.0, posinf=0.0, neginf=0.0)
    if valid.max() == valid.min():
        normalized = np.zeros_like(valid, dtype=np.uint8)
    else:
        normalized = cv2.normalize(valid, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    return cv2.applyColorMap(normalized, cv2.COLORMAP_INFERNO)


@dataclass
class TransformParams:
    dx: float = 0.0
    dy: float = 0.0
    scale: float = 1.0
    angle_deg: float = 0.0

    def to_matrix(self, width: int, height: int) -> np.ndarray:
        cx, cy = (width - 1) / 2.0, (height - 1) / 2.0
        m = cv2.getRotationMatrix2D((cx, cy), self.angle_deg, self.scale)
        m[0, 2] += self.dx
        m[1, 2] += self.dy
        return m


def warp_thermal(thermal: np.ndarray, params: TransformParams) -> np.ndarray:
    h, w = thermal.shape[:2]
    fill = float(np.nanmedian(thermal)) if np.isfinite(thermal).any() else T_MIN
    warped = cv2.warpAffine(
        thermal.astype(np.float32),
        params.to_matrix(w, h),
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=fill,
    )
    return warped.astype(np.float32)


def discover_scenes(dataset_root: str) -> List[str]:
    scenes = []
    for entry in sorted(os.listdir(dataset_root), key=natural_sort_key):
        scene_dir = os.path.join(dataset_root, entry)
        if not os.path.isdir(scene_dir):
            continue
        rgb_path = os.path.join(scene_dir, "left_cam", "left.png")
        thermal_path = os.path.join(scene_dir, "thermal", THERMAL_NPY)
        if os.path.exists(rgb_path) and os.path.exists(thermal_path):
            scenes.append(scene_dir)
    return scenes


def load_progress() -> dict:
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"scenes": {}}


def save_progress(progress: dict) -> None:
    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        json.dump(progress, f, indent=2)


def source_thermal_path(thermal_dir: str) -> str:
    """Path to the rgb_matched thermal used as warp input."""
    return os.path.join(thermal_dir, THERMAL_NPY)


def load_source_thermal(thermal_dir: str) -> np.ndarray:
    path = source_thermal_path(thermal_dir)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing source thermal: {path}")
    return np.load(path).astype(np.float32)


def calibrated_thermal_path(thermal_dir: str) -> str:
    return os.path.join(thermal_dir, CALIBRATED_THERMAL_NPY)


def calibrated_thermal_png_path(thermal_dir: str) -> str:
    return os.path.join(thermal_dir, CALIBRATED_THERMAL_PNG)


def has_calibrated_thermal(thermal_dir: str) -> bool:
    return os.path.exists(calibrated_thermal_path(thermal_dir))


def load_calibrated_thermal(thermal_dir: str) -> np.ndarray:
    path = calibrated_thermal_path(thermal_dir)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing calibrated thermal: {path}")
    return np.load(path).astype(np.float32)


def params_equal(a: TransformParams, b: TransformParams, tol: float = 1e-3) -> bool:
    return (
        abs(a.dx - b.dx) <= tol
        and abs(a.dy - b.dy) <= tol
        and abs(a.scale - b.scale) <= tol
        and abs(a.angle_deg - b.angle_deg) <= tol
    )


def load_saved_params(scene_dir: str, progress: dict, scene_key: str) -> TransformParams:
    entry = progress.get("scenes", {}).get(scene_key, {})
    params_dict = entry.get("params")
    if params_dict:
        return TransformParams(
            dx=float(params_dict.get("dx", 0.0)),
            dy=float(params_dict.get("dy", 0.0)),
            scale=float(params_dict.get("scale", 1.0)),
            angle_deg=float(params_dict.get("angle_deg", 0.0)),
        )

    meta_path = os.path.join(scene_dir, "metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        cal = meta.get("manual_thermal_calibration", {})
        if cal:
            return TransformParams(
                dx=float(cal.get("dx", 0.0)),
                dy=float(cal.get("dy", 0.0)),
                scale=float(cal.get("scale", 1.0)),
                angle_deg=float(cal.get("angle_deg", 0.0)),
            )

    return TransformParams()


def save_calibrated_thermal(thermal: np.ndarray, thermal_dir: str) -> Tuple[str, str]:
    """Write manual calibration outputs; does not overwrite rgb_matched_thermal.*."""
    npy_path = os.path.join(thermal_dir, CALIBRATED_THERMAL_NPY)
    png_path = os.path.join(thermal_dir, CALIBRATED_THERMAL_PNG)
    np.save(npy_path, thermal.astype(np.float32))
    cv2.imwrite(png_path, render_thermal_colormap(thermal))
    return npy_path, png_path


def update_metadata(scene_dir: str, params: TransformParams) -> None:
    meta_path = os.path.join(scene_dir, "metadata.json")
    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
    meta["manual_thermal_calibration"] = {
        **asdict(params),
        "source": THERMAL_NPY,
        "outputs": {
            "npy": CALIBRATED_THERMAL_NPY,
            "png": CALIBRATED_THERMAL_PNG,
        },
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=4)


def put_help(img: np.ndarray, lines: List[str]) -> np.ndarray:
    out = img.copy()
    y = 24
    for line in lines:
        cv2.putText(out, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        y += 22
    return out


def build_preview(
    rgb_bgr: np.ndarray,
    thermal: np.ndarray,
    params: TransformParams,
    opacity: float,
    overlay_mode: bool,
    edge_mode: bool,
) -> np.ndarray:
    warped = warp_thermal(thermal, params)
    thermal_bgr = render_thermal_colormap(warped)

    if overlay_mode:
        blend = cv2.addWeighted(rgb_bgr, 1.0 - opacity, thermal_bgr, opacity, 0)
        if edge_mode:
            gray = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 80, 160)
            blend[edges > 0] = (0, 255, 255)
        preview = blend
    else:
        divider = np.zeros((rgb_bgr.shape[0], 4, 3), dtype=np.uint8)
        preview = np.hstack([rgb_bgr, divider, thermal_bgr])

    status = (
        f"dx={params.dx:+.1f}  dy={params.dy:+.1f}  "
        f"scale={params.scale:.3f}  angle={params.angle_deg:+.1f}deg  "
        f"{'OVERLAY' if overlay_mode else 'SIDE-BY-SIDE'}"
    )
    return put_help(
        preview,
        [status, "s=save+next | n/p=scene | hjkl/arrows=nudge | o/e/r | q=quit"],
    )


class ManualThermalCalibrator:
    def __init__(self, scenes: List[str], start_index: int = 0):
        self.scenes = scenes
        self.index = max(0, min(start_index, len(scenes) - 1))
        self.progress = load_progress()

        self.rgb_bgr: Optional[np.ndarray] = None
        self.source_thermal: Optional[np.ndarray] = None
        self.scene_dir: Optional[str] = None
        self.thermal_dir: Optional[str] = None

        self.params = TransformParams()
        self.opacity = OPACITY_DEFAULT / 100.0
        self.overlay_mode = True
        self.edge_mode = False
        self.dirty = False

        self.window = "Thermal Calibrator"
        self.panel = "Controls"
        cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window, 900, 900)
        cv2.namedWindow(self.panel, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.panel, 480, 320)

        cv2.createTrackbar("dx", self.panel, DX_CENTER, DX_CENTER * 2, self._on_trackbar)
        cv2.createTrackbar("dy", self.panel, DY_CENTER, DY_CENTER * 2, self._on_trackbar)
        cv2.createTrackbar("scale x100", self.panel, SCALE_CENTER, 200, self._on_trackbar)
        cv2.createTrackbar("angle x10", self.panel, ANGLE_CENTER + 100, 200, self._on_trackbar)
        cv2.createTrackbar("opacity", self.panel, OPACITY_DEFAULT, 100, self._on_trackbar)

        self.load_scene(self.index)

    def _on_trackbar(self, _=None):
        self._read_trackbars()
        self.dirty = True

    def _read_trackbars(self):
        dx = cv2.getTrackbarPos("dx", self.panel) - DX_CENTER
        dy = cv2.getTrackbarPos("dy", self.panel) - DY_CENTER
        scale = cv2.getTrackbarPos("scale x100", self.panel) / 100.0
        angle = (cv2.getTrackbarPos("angle x10", self.panel) - 100) / 10.0
        opacity = cv2.getTrackbarPos("opacity", self.panel) / 100.0
        self.params = TransformParams(dx=float(dx), dy=float(dy), scale=scale, angle_deg=angle)
        self.opacity = opacity

    def _set_trackbars(self, params: TransformParams, opacity_pct: int = OPACITY_DEFAULT):
        cv2.setTrackbarPos("dx", self.panel, int(round(params.dx)) + DX_CENTER)
        cv2.setTrackbarPos("dy", self.panel, int(round(params.dy)) + DY_CENTER)
        cv2.setTrackbarPos("scale x100", self.panel, int(round(params.scale * 100)))
        cv2.setTrackbarPos("angle x10", self.panel, int(round(params.angle_deg * 10)) + 100)
        cv2.setTrackbarPos("opacity", self.panel, opacity_pct)

    def scene_key(self) -> str:
        assert self.scene_dir is not None
        return os.path.relpath(self.scene_dir, BASE_DIR)

    def load_scene(self, index: int):
        self.index = index
        self.scene_dir = self.scenes[index]
        self.thermal_dir = os.path.join(self.scene_dir, "thermal")

        rgb_path = os.path.join(self.scene_dir, "left_cam", "left.png")
        self.rgb_bgr = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
        self.source_thermal = load_source_thermal(self.thermal_dir)

        if self.rgb_bgr is None:
            raise RuntimeError(f"Failed to read RGB image: {rgb_path}")
        if self.source_thermal.shape[:2] != self.rgb_bgr.shape[:2]:
            raise RuntimeError(
                f"Shape mismatch in {self.scene_key()}: "
                f"RGB {self.rgb_bgr.shape[:2]} vs thermal {self.source_thermal.shape[:2]}"
            )

        saved = self.progress.get("scenes", {}).get(self.scene_key(), {})
        params_dict = saved.get("params", {})
        self.params = TransformParams(
            dx=float(params_dict.get("dx", 0.0)),
            dy=float(params_dict.get("dy", 0.0)),
            scale=float(params_dict.get("scale", 1.0)),
            angle_deg=float(params_dict.get("angle_deg", 0.0)),
        )
        self._set_trackbars(self.params)
        self._read_trackbars()
        self.dirty = False

        scene_name = os.path.basename(self.scene_dir)
        print(f"\n[{index + 1}/{len(self.scenes)}] {self.scene_key()}  ({scene_name})")

    def nudge(self, ndx: int = 0, ndy: int = 0):
        self.params.dx += ndx
        self.params.dy += ndy
        self.params.dx = float(np.clip(self.params.dx, -MAX_ABS_SHIFT, MAX_ABS_SHIFT))
        self.params.dy = float(np.clip(self.params.dy, -MAX_ABS_SHIFT, MAX_ABS_SHIFT))
        self._set_trackbars(self.params)
        self._read_trackbars()
        self.dirty = True

    def reset(self):
        self.params = TransformParams()
        self._set_trackbars(self.params)
        self._read_trackbars()
        self.dirty = True

    def save_scene(self):
        assert self.source_thermal is not None and self.thermal_dir is not None
        warped = warp_thermal(self.source_thermal, self.params)
        npy_path, png_path = save_calibrated_thermal(warped, self.thermal_dir)
        update_metadata(self.scene_dir, self.params)

        self.progress.setdefault("scenes", {})[self.scene_key()] = {
            "saved": True,
            "params": asdict(self.params),
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "outputs": {
                "npy": CALIBRATED_THERMAL_NPY,
                "png": CALIBRATED_THERMAL_PNG,
            },
        }
        save_progress(self.progress)
        self.dirty = False
        print(f"  Saved {npy_path} and {png_path}")

    def render(self) -> np.ndarray:
        assert self.rgb_bgr is not None and self.source_thermal is not None
        title = f"{os.path.basename(self.scene_dir)} [{self.index + 1}/{len(self.scenes)}]"
        frame = build_preview(
            self.rgb_bgr,
            self.source_thermal,
            self.params,
            self.opacity,
            self.overlay_mode,
            self.edge_mode,
        )
        cv2.setWindowTitle(self.window, title)
        return frame

    def goto(self, delta: int):
        new_index = self.index + delta
        if 0 <= new_index < len(self.scenes):
            self.load_scene(new_index)

    @staticmethod
    def _decode_nudge(key_full: int) -> Optional[Tuple[int, int]]:
        """Map key code to (dx, dy) nudge; None if not a nudge key."""
        key = key_full & 0xFF
        step = 5 if key_full > 255 else 1

        arrows = {
            81: (-step, 0),
            83: (step, 0),
            82: (0, -step),
            84: (0, step),
            2: (-step, 0),
            3: (step, 0),
            0: (0, -step),
            1: (0, step),
            65361: (-step, 0),
            65363: (step, 0),
            65362: (0, -step),
            65364: (0, step),
        }
        if key_full in arrows:
            return arrows[key_full]

        fine = {
            ord("h"): (-1, 0),
            ord("l"): (1, 0),
            ord("k"): (0, -1),
            ord("j"): (0, 1),
            ord("H"): (-5, 0),
            ord("L"): (5, 0),
            ord("K"): (0, -5),
            ord("J"): (0, 5),
        }
        return fine.get(key)

    def run(self):
        print("Focus the preview window for keyboard shortcuts.")
        while True:
            cv2.imshow(self.window, self.render())
            key_full = cv2.waitKeyEx(20)
            key = key_full & 0xFF

            if key in (27, ord("q")):
                if self.dirty:
                    print("Unsaved changes on current scene (press s to save).")
                break
            elif key == ord("s"):
                self.save_scene()
                if self.index + 1 < len(self.scenes):
                    self.goto(1)
                else:
                    print("Last scene saved. Press q to quit.")
            elif key == ord("n"):
                self.goto(1)
            elif key == ord("p"):
                self.goto(-1)
            elif key == ord("r"):
                self.reset()
            elif key == ord("o"):
                self.overlay_mode = not self.overlay_mode
            elif key == ord("e"):
                self.edge_mode = not self.edge_mode
            else:
                nudge = self._decode_nudge(key_full)
                if nudge is not None:
                    self.nudge(*nudge)

        cv2.destroyAllWindows()


def resolve_start_index(scenes: List[str], start_scene: Optional[str]) -> int:
    if not start_scene:
        return 0
    target = start_scene.strip()
    for i, scene_dir in enumerate(scenes):
        if os.path.basename(scene_dir) == target or scene_dir.endswith(target):
            return i
    raise SystemExit(f"Start scene not found: {start_scene}")


def main():
    parser = argparse.ArgumentParser(description="Manual RGB–thermal alignment calibrator.")
    parser.add_argument(
        "--repo-root",
        default=REPO_ROOT,
        help="Repo root containing Dataset/ (default: auto-detected).",
    )
    parser.add_argument(
        "--dataset",
        default="Dataset",
        help='Dataset folder under repo root (default: "Dataset").',
    )
    parser.add_argument(
        "--start",
        default=None,
        help='Scene name to start from, e.g. "Scene_49".',
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List discovered scenes and exit.",
    )
    args = parser.parse_args()

    dataset_root = os.path.join(args.repo_root, args.dataset)
    if not os.path.isdir(dataset_root):
        raise SystemExit(f"Dataset root not found: {dataset_root}")

    scenes = discover_scenes(dataset_root)
    if not scenes:
        raise SystemExit(f"No valid scenes found under {dataset_root}")

    if args.list:
        for i, scene_dir in enumerate(scenes):
            rel = os.path.relpath(scene_dir, args.repo_root)
            print(f"{i:4d}  {rel}")
        return

    start_index = resolve_start_index(scenes, args.start)
    print(f"Repo root: {args.repo_root}")
    print(f"Found {len(scenes)} scenes in {dataset_root}")
    print(f"Progress file: {PROGRESS_FILE}")
    print(f"Source thermal: thermal/{THERMAL_NPY}")
    print(f"Output thermal: thermal/{CALIBRATED_THERMAL_NPY} + .png")

    calibrator = ManualThermalCalibrator(scenes, start_index=start_index)
    calibrator.run()


if __name__ == "__main__":
    main()
