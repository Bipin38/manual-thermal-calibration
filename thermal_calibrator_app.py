"""
RGB–Thermal Manual Calibrator (Web)
===================================
Streamlit UI for fine-tuning rgb_matched thermal alignment scene by scene.

Run from repo root:
    .venv/bin/streamlit run manual_thermal_calibration.py/thermal_calibrator_app.py

Or from this folder:
    ../.venv/bin/streamlit run thermal_calibrator_app.py
"""

from __future__ import annotations

import os
from dataclasses import asdict
from datetime import datetime, timezone

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from manual_thermal_calibrator import (
    BASE_DIR,
    REPO_ROOT,
    CALIBRATED_THERMAL_NPY,
    CALIBRATED_THERMAL_PNG,
    MAX_ABS_SHIFT,
    PROGRESS_FILE,
    THERMAL_NPY,
    TransformParams,
    calibrated_thermal_path,
    calibrated_thermal_png_path,
    discover_scenes,
    has_calibrated_thermal,
    load_calibrated_thermal,
    load_progress,
    load_saved_params,
    load_source_thermal,
    natural_sort_key,
    params_equal,
    render_thermal_colormap,
    save_calibrated_thermal,
    save_progress,
    update_metadata,
    warp_thermal,
)

st.set_page_config(
    page_title="Thermal Calibrator",
    layout="wide",
    initial_sidebar_state="expanded",
)

OPACITY_DEFAULT = 55


def get_dataset_roots() -> list[str]:
    names = []
    for entry in sorted(os.listdir(BASE_DIR), key=natural_sort_key):
        path = os.path.join(BASE_DIR, entry)
        if not os.path.isdir(path) or entry.startswith("."):
            continue
        if discover_scenes(path):
            names.append(entry)
    return names


def scene_rel_key(scene_dir: str) -> str:
    return os.path.relpath(scene_dir, BASE_DIR)


def scene_has_saved_output(scene_dir: str) -> bool:
    return has_calibrated_thermal(os.path.join(scene_dir, "thermal"))


def scene_status(scene_dir: str, progress: dict) -> str:
    if scene_has_saved_output(scene_dir):
        return "saved"
    key = scene_rel_key(scene_dir)
    if progress.get("scenes", {}).get(key, {}).get("saved"):
        return "saved"
    return "pending"


STATUS_LABEL = {
    "saved": "Calibrated",
    "pending": "Not saved",
}

SAVED_SCENE_PREFIX = "✓ "


def scene_selector_label(scene_dir: str, progress: dict) -> str:
    name = os.path.basename(scene_dir)
    if scene_status(scene_dir, progress) == "saved":
        return f"{SAVED_SCENE_PREFIX}{name}"
    return name


def parse_scene_selector_label(label: str) -> str:
    return label.removeprefix(SAVED_SCENE_PREFIX)


def load_rgb(scene_dir: str) -> np.ndarray:
    rgb_path = os.path.join(scene_dir, "left_cam", "left.png")
    rgb_bgr = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
    if rgb_bgr is None:
        raise RuntimeError(f"Cannot read RGB image: {rgb_path}")
    return rgb_bgr


def edit_state_key(scene_key: str) -> str:
    return f"edit::{scene_key}"


def get_edit_params(scene_key: str) -> dict:
    return st.session_state.setdefault(edit_state_key(scene_key), asdict(TransformParams()))


def slider_key(scene_key: str, field: str) -> str:
    return f"sl_{field}::{scene_key}"


def sync_sliders_from_edit(scene_key: str) -> None:
    """Push edit dict into slider widget keys (call only before widgets render)."""
    edit = get_edit_params(scene_key)
    st.session_state[slider_key(scene_key, "dx")] = int(round(edit["dx"]))
    st.session_state[slider_key(scene_key, "dy")] = int(round(edit["dy"]))
    st.session_state[slider_key(scene_key, "scale")] = float(edit["scale"])
    st.session_state[slider_key(scene_key, "angle")] = float(edit["angle_deg"])


def params_from_sliders(scene_key: str) -> TransformParams:
    return TransformParams(
        dx=float(st.session_state.get(slider_key(scene_key, "dx"), 0.0)),
        dy=float(st.session_state.get(slider_key(scene_key, "dy"), 0.0)),
        scale=float(st.session_state.get(slider_key(scene_key, "scale"), 1.0)),
        angle_deg=float(st.session_state.get(slider_key(scene_key, "angle"), 0.0)),
    )


def sync_scene_controls(scene_key: str, scene_dir: str, progress: dict) -> None:
    """Reload alignment values whenever the active scene changes."""
    if st.session_state.get("_active_scene_key") == scene_key:
        return

    saved_params = load_saved_params(scene_dir, progress, scene_key)
    st.session_state[edit_state_key(scene_key)] = asdict(saved_params)
    sync_sliders_from_edit(scene_key)
    st.session_state[f"opacity::{scene_key}"] = OPACITY_DEFAULT
    st.session_state["_active_scene_key"] = scene_key


def params_from_edit(scene_key: str) -> TransformParams:
    edit = get_edit_params(scene_key)
    return TransformParams(
        dx=float(edit.get("dx", 0.0)),
        dy=float(edit.get("dy", 0.0)),
        scale=float(edit.get("scale", 1.0)),
        angle_deg=float(edit.get("angle_deg", 0.0)),
    )


def apply_nudge(scene_key: str, ndx: float = 0.0, ndy: float = 0.0) -> None:
    p = params_from_sliders(scene_key)
    p.dx = float(np.clip(p.dx + ndx, -MAX_ABS_SHIFT, MAX_ABS_SHIFT))
    p.dy = float(np.clip(p.dy + ndy, -MAX_ABS_SHIFT, MAX_ABS_SHIFT))
    st.session_state[edit_state_key(scene_key)] = asdict(p)


def process_pending_action(
    scene_key: str,
    scene_dir: str,
    source_thermal: np.ndarray,
    all_scenes: list[str],
) -> bool:
    """Run queued UI actions before widgets render. Returns True if rerun needed."""
    action = st.session_state.pop("_pending_action", None)
    if not action or action.get("scene_key") != scene_key:
        if action:
            st.session_state._pending_action = action
        return False

    if action["type"] == "reset":
        st.session_state[edit_state_key(scene_key)] = asdict(TransformParams())
        sync_sliders_from_edit(scene_key)
        st.rerun()

    if action["type"] == "nudge":
        apply_nudge(scene_key, action.get("ndx", 0.0), action.get("ndy", 0.0))
        sync_sliders_from_edit(scene_key)
        st.rerun()

    if action["type"] == "save":
        try:
            params = TransformParams(**action["params"])
            npy_path, png_path = save_scene(scene_dir, source_thermal, params)
            st.session_state._save_notice = {
                "scene": os.path.basename(scene_dir),
                "npy": os.path.basename(npy_path),
                "png": os.path.basename(png_path),
            }
            st.session_state._active_scene_key = None
            if action.get("next") and st.session_state.scene_index + 1 < len(all_scenes):
                go_to_scene(all_scenes, st.session_state.scene_index + 1)
            st.rerun()
        except Exception as exc:
            st.session_state._save_error = str(exc)
            st.rerun()

    return False


def save_scene(
    scene_dir: str,
    source_thermal: np.ndarray,
    params: TransformParams,
) -> tuple[str, str]:
    thermal_dir = os.path.join(scene_dir, "thermal")
    warped = warp_thermal(source_thermal, params)
    npy_path, png_path = save_calibrated_thermal(warped, thermal_dir)
    update_metadata(scene_dir, params)

    progress = load_progress()
    key = scene_rel_key(scene_dir)
    progress.setdefault("scenes", {})[key] = {
        "saved": True,
        "params": asdict(params),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "outputs": {
            "npy": CALIBRATED_THERMAL_NPY,
            "png": CALIBRATED_THERMAL_PNG,
        },
        "source": THERMAL_NPY,
    }
    save_progress(progress)
    return npy_path, png_path


def blend_rgb_thermal(
    rgb_bgr: np.ndarray,
    thermal: np.ndarray,
    opacity: float,
    overlay_mode: bool,
    edge_mode: bool,
) -> np.ndarray:
    thermal_bgr = render_thermal_colormap(thermal)

    if overlay_mode:
        blend = cv2.addWeighted(rgb_bgr, 1.0 - opacity, thermal_bgr, opacity, 0)
        if edge_mode:
            gray = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 80, 160)
            blend[edges > 0] = (0, 255, 255)
        return blend

    divider = np.zeros((rgb_bgr.shape[0], 4, 3), dtype=np.uint8)
    return np.hstack([rgb_bgr, divider, thermal_bgr])


def resolve_display_thermal(
    scene_dir: str,
    source_thermal: np.ndarray,
    params: TransformParams,
    saved_params: TransformParams,
    force_live: bool,
) -> tuple[np.ndarray, str]:
    thermal_dir = os.path.join(scene_dir, "thermal")
    has_saved = scene_has_saved_output(scene_dir)
    editing = not params_equal(params, saved_params)

    if has_saved and not force_live and not editing:
        return load_calibrated_thermal(thermal_dir), "saved"

    return warp_thermal(source_thermal, params), "live"


def bgr_to_pil(bgr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


def progress_summary(scenes: list[str], progress: dict) -> dict:
    saved = sum(1 for s in scenes if scene_status(s, progress) == "saved")
    pending = len(scenes) - saved
    return {"total": len(scenes), "done": saved, "saved": saved, "pending": pending}


def build_scene_table(scenes: list[str], progress: dict) -> pd.DataFrame:
    rows = []
    for scene_dir in scenes:
        key = scene_rel_key(scene_dir)
        entry = progress.get("scenes", {}).get(key, {})
        params = entry.get("params", {})
        if not params and scene_has_saved_output(scene_dir):
            saved = load_saved_params(scene_dir, progress, key)
            params = asdict(saved)
        updated = entry.get("updated_at") or ""
        thermal_dir = os.path.join(scene_dir, "thermal")
        rows.append(
            {
                "scene": os.path.basename(scene_dir),
                "status": STATUS_LABEL[scene_status(scene_dir, progress)],
                "file": "yes" if scene_has_saved_output(scene_dir) else "no",
                "dx": params.get("dx"),
                "dy": params.get("dy"),
                "updated": updated[:19].replace("T", " ") if updated else "",
                "npy": os.path.basename(calibrated_thermal_path(thermal_dir)),
            }
        )
    df = pd.DataFrame(rows)
    for col in ("dx", "dy"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def go_to_scene(all_scenes: list[str], index: int) -> None:
    index = max(0, min(index, len(all_scenes) - 1))
    st.session_state.scene_index = index
    st.session_state._active_scene_key = None


def main():
    st.title("RGB–Thermal Manual Calibrator")
    st.caption(
        f"**Repo:** `{REPO_ROOT}` · **Input:** `{THERMAL_NPY}` · "
        f"**Output:** `{CALIBRATED_THERMAL_NPY}` + `.png`"
    )

    dataset_names = get_dataset_roots()
    if not dataset_names:
        st.error("No datasets with valid scenes found under the repo root.")
        st.stop()

    if "dataset_name" not in st.session_state:
        st.session_state.dataset_name = "Dataset" if "Dataset" in dataset_names else dataset_names[0]
    if "scene_index" not in st.session_state:
        st.session_state.scene_index = 0

    with st.sidebar:
        st.header("Dataset")
        dataset_name = st.selectbox(
            "Folder",
            dataset_names,
            index=dataset_names.index(st.session_state.dataset_name)
            if st.session_state.dataset_name in dataset_names
            else 0,
        )
        if dataset_name != st.session_state.dataset_name:
            st.session_state.dataset_name = dataset_name
            go_to_scene(discover_scenes(os.path.join(BASE_DIR, dataset_name)), 0)

        dataset_root = os.path.join(BASE_DIR, dataset_name)
        all_scenes = discover_scenes(dataset_root)
        progress = load_progress()
        summary = progress_summary(all_scenes, progress)

        st.progress(summary["done"] / max(summary["total"], 1))
        c1, c2, c3 = st.columns(3)
        c1.metric("Saved", summary["saved"])
        c2.metric("Pending", summary["pending"])
        c3.metric("Total", summary["total"])

        filter_mode = st.radio(
            "Show scenes",
            ["All", "Pending only", "Calibrated only"],
            horizontal=True,
        )

        filtered_scenes = all_scenes
        if filter_mode == "Pending only":
            filtered_scenes = [s for s in all_scenes if scene_status(s, progress) == "pending"]
        elif filter_mode == "Calibrated only":
            filtered_scenes = [s for s in all_scenes if scene_status(s, progress) == "saved"]

        if not filtered_scenes:
            st.warning("No scenes match the current filter.")
            st.stop()

        if st.session_state.scene_index >= len(all_scenes):
            st.session_state.scene_index = 0

        current_name = os.path.basename(all_scenes[st.session_state.scene_index])
        try:
            filtered_index = next(
                i for i, s in enumerate(filtered_scenes) if os.path.basename(s) == current_name
            )
        except StopIteration:
            filtered_index = 0
            go_to_scene(all_scenes, all_scenes.index(filtered_scenes[0]))

        scene_labels = [scene_selector_label(s, progress) for s in filtered_scenes]
        picked_label = st.selectbox(
            "Scene",
            scene_labels,
            index=filtered_index,
        )
        picked_name = parse_scene_selector_label(picked_label)
        picked_dir = next(s for s in filtered_scenes if os.path.basename(s) == picked_name)
        if all_scenes.index(picked_dir) != st.session_state.scene_index:
            go_to_scene(all_scenes, all_scenes.index(picked_dir))

        nav1, nav2 = st.columns(2)
        with nav1:
            if st.button("Previous", use_container_width=True):
                go_to_scene(all_scenes, st.session_state.scene_index - 1)
                st.rerun()
        with nav2:
            if st.button("Next", use_container_width=True):
                go_to_scene(all_scenes, st.session_state.scene_index + 1)
                st.rerun()

        with st.expander("Calibration tracker", expanded=False):
            st.dataframe(
                build_scene_table(all_scenes, progress),
                use_container_width=True,
                hide_index=True,
                height=320,
            )

    scene_dir = all_scenes[st.session_state.scene_index]
    scene_key = scene_rel_key(scene_dir)
    scene_name = os.path.basename(scene_dir)
    thermal_dir = os.path.join(scene_dir, "thermal")
    status = scene_status(scene_dir, progress)
    has_saved = scene_has_saved_output(scene_dir)

    sync_scene_controls(scene_key, scene_dir, progress)
    rgb_bgr = load_rgb(scene_dir)
    source_thermal = load_source_thermal(thermal_dir)
    saved_params = load_saved_params(scene_dir, progress, scene_key)
    process_pending_action(scene_key, scene_dir, source_thermal, all_scenes)

    if notice := st.session_state.pop("_save_notice", None):
        st.success(
            f"Saved {notice['scene']}:\n"
            f"- `{notice['npy']}`\n"
            f"- `{notice['png']}`"
        )
    if err := st.session_state.pop("_save_error", None):
        st.error(f"Save failed: {err}")

    if rgb_bgr.shape[:2] != source_thermal.shape[:2]:
        st.error(
            f"Shape mismatch in {scene_name}: RGB {rgb_bgr.shape[:2]} vs "
            f"thermal {source_thermal.shape[:2]}"
        )
        st.stop()

    top_l, top_r = st.columns([3, 1])
    with top_l:
        st.subheader(f"{scene_name}  ({st.session_state.scene_index + 1} / {len(all_scenes)})")
        badge = "💾" if has_saved else "⏳"
        st.markdown(f"**Status:** {badge} {STATUS_LABEL[status]}")
    with top_r:
        st.markdown(f"**Path:** `{scene_key}`")
        st.markdown(f"**Source:** `thermal/{THERMAL_NPY}`")
        st.markdown(f"**Output:** `thermal/{CALIBRATED_THERMAL_NPY}`")
        if has_saved:
            st.markdown("**On disk:** calibrated file found")

    ctrl_l, ctrl_r = st.columns([1, 2])

    with ctrl_l:
        st.markdown("### Alignment")

        st.slider(
            "dx (pixels)",
            -MAX_ABS_SHIFT,
            MAX_ABS_SHIFT,
            step=1,
            key=slider_key(scene_key, "dx"),
            help="Shift thermal horizontally to match RGB.",
        )
        st.slider(
            "dy (pixels)",
            -MAX_ABS_SHIFT,
            MAX_ABS_SHIFT,
            step=1,
            key=slider_key(scene_key, "dy"),
            help="Shift thermal vertically to match RGB.",
        )
        st.slider(
            "scale",
            0.95,
            1.05,
            step=0.001,
            format="%.3f",
            key=slider_key(scene_key, "scale"),
        )
        st.slider(
            "angle (degrees)",
            -5.0,
            5.0,
            step=0.1,
            key=slider_key(scene_key, "angle"),
        )
        opacity = st.slider(
            "overlay opacity %",
            0,
            100,
            key=f"opacity::{scene_key}",
        )

        params = params_from_sliders(scene_key)

        st.markdown("**Fine nudge**")
        n1, n2, n3 = st.columns(3)
        with n1:
            if st.button("← 1px", use_container_width=True):
                st.session_state._pending_action = {
                    "type": "nudge", "scene_key": scene_key, "ndx": -1, "ndy": 0,
                }
                st.rerun()
            if st.button("← 5px", use_container_width=True):
                st.session_state._pending_action = {
                    "type": "nudge", "scene_key": scene_key, "ndx": -5, "ndy": 0,
                }
                st.rerun()
        with n2:
            if st.button("↑ 1px", use_container_width=True):
                st.session_state._pending_action = {
                    "type": "nudge", "scene_key": scene_key, "ndx": 0, "ndy": -1,
                }
                st.rerun()
            if st.button("↓ 1px", use_container_width=True):
                st.session_state._pending_action = {
                    "type": "nudge", "scene_key": scene_key, "ndx": 0, "ndy": 1,
                }
                st.rerun()
        with n3:
            if st.button("1px →", use_container_width=True):
                st.session_state._pending_action = {
                    "type": "nudge", "scene_key": scene_key, "ndx": 1, "ndy": 0,
                }
                st.rerun()
            if st.button("5px →", use_container_width=True):
                st.session_state._pending_action = {
                    "type": "nudge", "scene_key": scene_key, "ndx": 5, "ndy": 0,
                }
                st.rerun()

        view_mode = st.radio("Preview mode", ["Overlay", "Side by side"], horizontal=True)
        edge_mode = st.checkbox("Show RGB edges on overlay", value=False)
        force_live = st.checkbox(
            "Live warp preview (ignore saved file)",
            value=False,
            help="When off, revisiting a saved scene shows the saved calibrated thermal.",
        )

        b1, b2, b3 = st.columns(3)
        with b1:
            if st.button("Reset", use_container_width=True):
                st.session_state._pending_action = {"type": "reset", "scene_key": scene_key}
                st.rerun()
        with b2:
            save_clicked = st.button("Save", type="primary", use_container_width=True)
        with b3:
            save_next_clicked = st.button("Save & Next", use_container_width=True)

        editing = not params_equal(params, saved_params)
        st.code(
            f"dx={params.dx:+.1f}  dy={params.dy:+.1f}  "
            f"scale={params.scale:.3f}  angle={params.angle_deg:+.1f}°",
            language=None,
        )
        if has_saved and not editing and not force_live:
            st.info("Showing saved calibrated thermal aligned with RGB.")
        elif editing:
            st.warning("Unsaved changes — preview is live warp from source.")

    with ctrl_r:
        params = params_from_sliders(scene_key)
        display_thermal, preview_mode = resolve_display_thermal(
            scene_dir,
            source_thermal,
            params,
            saved_params,
            force_live=force_live,
        )
        preview_bgr = blend_rgb_thermal(
            rgb_bgr,
            display_thermal,
            float(opacity) / 100.0,
            overlay_mode=(view_mode == "Overlay"),
            edge_mode=edge_mode,
        )
        mode_label = "Saved calibration" if preview_mode == "saved" else "Live preview"
        st.markdown(f"**{mode_label}**")
        st.image(bgr_to_pil(preview_bgr), use_column_width=True)

        p2, p3 = st.columns(2)
        with p2:
            st.caption("Left RGB")
            st.image(bgr_to_pil(rgb_bgr), use_column_width=True)
        with p3:
            caption = (
                "Saved calibrated thermal"
                if preview_mode == "saved"
                else "Warped thermal (preview)"
            )
            st.caption(caption)
            st.image(bgr_to_pil(render_thermal_colormap(display_thermal)), use_column_width=True)

        if has_saved and os.path.exists(calibrated_thermal_png_path(thermal_dir)):
            with st.expander("Saved PNG on disk"):
                st.image(calibrated_thermal_png_path(thermal_dir), use_column_width=True)

    if save_clicked or save_next_clicked:
        st.session_state._pending_action = {
            "type": "save",
            "scene_key": scene_key,
            "params": asdict(params_from_sliders(scene_key)),
            "next": bool(save_next_clicked),
        }
        st.rerun()


if __name__ == "__main__":
    main()
