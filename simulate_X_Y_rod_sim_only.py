from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.widgets import Button, CheckButtons, Slider, TextBox
from scipy import ndimage as ndi


@dataclass
class CurveResult:
    phi_deg: np.ndarray
    x: np.ndarray
    y: np.ndarray


class RodDipoleSimulator:
    """
    Simulation-only rod dipole model with optional pupil-hole masks:
    - circular inner cutout
    - measured jagged hole profile loaded from PNG
    """

    def __init__(self) -> None:
        self.alpha_deg = 25.0
        self.beta_deg = 35.0
        self.gamma_deg = 25.0
        self.phi_deg = 0.0

        self.na = 1.30
        self.n_medium = 1.333

        self.show_no_hole = True
        self.show_circular_cutout = False
        self.show_profile_cutout = False
        self.circular_cutout_fraction = 0.3017895562712728

        self.phi_samples = 361
        self.ray_theta_samples = 28
        self.ray_psi_samples = 96

        self.forward_base: CurveResult | None = None
        self.forward_circular: CurveResult | None = None
        self.forward_profile: CurveResult | None = None

        self.play_sim = False
        self.play_speed_deg = 2.0

        self.fig = None
        self.ax3d = None
        self.ax_xy = None
        self.ax_phi = None
        self.status_text = None

        self.s_alpha = None
        self.s_beta = None
        self.s_gamma = None
        self.s_phi = None
        self.s_na = None

        self.tb_cutout = None
        self.chk_hole_modes = None
        self.btn_play_sim = None
        self.btn_recompute = None
        self.timer = None

        self.hole_png_path = (Path(__file__).resolve().parent / "image_of_hole" / "cleaned_circle_with_hole.png")
        self.hole_outer_na_ref = 1.3
        self.hole_fit_pad_left_px = 160
        self.hole_fit_pad_bottom_px = 160
        self.hole_profile_mask: np.ndarray | None = None
        self.hole_mean_radius: float | None = None
        self.hole_radius_fraction_fit: float | None = None
        self.hole_load_message = ""
        self._load_hole_profile()
        if self.hole_radius_fraction_fit is not None:
            self.circular_cutout_fraction = float(self.hole_radius_fraction_fit)
        elif self.hole_mean_radius is not None:
            self.circular_cutout_fraction = float(self.hole_mean_radius)

    def _circular_inner_na(self) -> float:
        return self._inner_na_from_fraction(self.circular_cutout_fraction)

    def _fraction_from_inner_na(self, na_inner: float) -> float:
        na_outer = max(1e-12, float(self.na))
        frac = float(na_inner) / na_outer
        return float(np.clip(frac, 0.0, 0.95))

    @staticmethod
    def _unit(v: np.ndarray) -> np.ndarray:
        n = float(np.linalg.norm(v))
        if n <= 0.0:
            return np.array([0.0, 0.0, 1.0], dtype=np.float64)
        return v / n

    @staticmethod
    def _rotation_axis(alpha_deg: float, beta_deg: float) -> np.ndarray:
        a = math.radians(float(alpha_deg))
        b = math.radians(float(beta_deg))
        return np.array([math.sin(a) * math.cos(b), math.sin(a) * math.sin(b), math.cos(a)], dtype=np.float64)

    def _cone_basis(self, k: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        z = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        ref = z if abs(float(np.dot(k, z))) < 0.95 else np.array([1.0, 0.0, 0.0], dtype=np.float64)
        e1 = self._unit(np.cross(k, ref))
        e2 = self._unit(np.cross(k, e1))
        return e1, e2

    def _rod_directions(self, alpha_deg: float, beta_deg: float, gamma_deg: float, phi_rad: np.ndarray) -> np.ndarray:
        k = self._rotation_axis(alpha_deg, beta_deg)
        e1, e2 = self._cone_basis(k)
        g = math.radians(float(gamma_deg))
        cg, sg = math.cos(g), math.sin(g)
        c = np.cos(phi_rad)[:, None]
        s = np.sin(phi_rad)[:, None]
        dirs = (cg * k[None, :]) + (sg * (c * e1[None, :] + s * e2[None, :]))
        norms = np.linalg.norm(dirs, axis=1, keepdims=True)
        norms = np.where(norms <= 0.0, 1.0, norms)
        return dirs / norms

    def _load_hole_profile(self) -> None:
        if not self.hole_png_path.exists():
            self.hole_profile_mask = None
            self.hole_mean_radius = None
            self.hole_load_message = f"Hole PNG not found: {self.hole_png_path.name}"
            return

        try:
            img = plt.imread(self.hole_png_path)
            g = np.asarray(img, dtype=np.float64)
            if g.ndim == 3:
                if g.shape[2] >= 3:
                    g = 0.2126 * g[..., 0] + 0.7152 * g[..., 1] + 0.0722 * g[..., 2]
                else:
                    g = g[..., 0]
            if g.ndim != 2:
                raise RuntimeError("Expected a 2D image.")

            g = g - float(np.nanmin(g))
            maxg = float(np.nanmax(g))
            if maxg <= 0.0 or (not np.isfinite(maxg)):
                raise RuntimeError("Degenerate image (constant intensity).")
            g = g / maxg

            h, w = g.shape
            yy, xx = np.mgrid[0:h, 0:w]
            cx = 0.5 * (w - 1)
            cy = 0.5 * (h - 1)
            r_norm = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / max(1e-9, 0.5 * min(h, w))

            center_band = g[r_norm <= 0.15]
            edge_band = g[(r_norm >= 0.70) & (r_norm <= 1.00)]
            if center_band.size == 0 or edge_band.size == 0:
                raise RuntimeError("Could not estimate central/edge intensity bands.")
            cmean = float(np.mean(center_band))
            emean = float(np.mean(edge_band))
            t = 0.5 * (cmean + emean)

            center_is_darker = bool(cmean < emean)
            blocked = (g <= t) if center_is_darker else (g >= t)
            blocked = blocked & (r_norm <= 1.0)
            if not np.any(blocked):
                raise RuntimeError("No blocked region detected in hole PNG.")

            lbl, n = ndi.label(blocked)
            if n <= 0:
                raise RuntimeError("Failed to label blocked region in hole PNG.")

            best = None
            for i in range(1, n + 1):
                comp = lbl == i
                area = int(np.count_nonzero(comp))
                if area <= 8:
                    continue
                ys, xs = np.nonzero(comp)
                cdist = float(np.hypot(float(np.mean(xs)) - cx, float(np.mean(ys)) - cy))
                score = area / (1.0 + cdist)
                if (best is None) or (score > best[0]):
                    best = (score, comp)
            if best is None:
                raise RuntimeError("No valid blocked component detected in hole PNG.")

            hole = best[1]
            self.hole_profile_mask = hole.astype(bool, copy=False)

            edge = hole & (~ndi.binary_erosion(hole, structure=np.ones((3, 3), dtype=bool)))
            rr = r_norm[edge]
            rr = rr[np.isfinite(rr)]
            if rr.size == 0:
                rr = r_norm[hole]
                rr = rr[np.isfinite(rr)]
            if rr.size == 0:
                raise RuntimeError("Could not compute mean hole radius.")
            self.hole_mean_radius = float(np.clip(np.mean(rr), 0.0, 0.95))
            saved_dir, fit_frac = self._save_hole_mask_debug(g, blocked, hole, t=t, center_is_darker=center_is_darker)
            self.hole_radius_fraction_fit = float(fit_frac) if fit_frac is not None else None
            if self.hole_radius_fraction_fit is not None:
                self.circular_cutout_fraction = float(self.hole_radius_fraction_fit)
            inner_na_txt = (
                f"{self._inner_na_from_fraction(self.hole_radius_fraction_fit):.3f}"
                if self.hole_radius_fraction_fit is not None
                else "n/a"
            )
            self.hole_load_message = (
                f"Jagged hole PNG loaded ({w}x{h}), fitted inner NA={inner_na_txt}, "
                f"outer NA ref={self.hole_outer_na_ref:.2f}. "
                f"Masks saved to {saved_dir}"
            )
        except Exception as e:
            self.hole_profile_mask = None
            self.hole_mean_radius = None
            self.hole_radius_fraction_fit = None
            self.hole_load_message = f"Hole PNG load failed: {e}"

    def _fit_outer_circle_from_bright_points(
        self,
        g_norm: np.ndarray,
        t_cut: float,
        center_is_darker: bool,
        pad_left_px: int = 0,
        pad_bottom_px: int = 0,
    ) -> tuple[float, float, float, np.ndarray, np.ndarray, np.ndarray]:
        """
        Fit outer bright circle from annulus mask while excluding flat clipped segments:
        1) Build white mask from the SAME cutoff used for middle-hole detection
        2) Build outer-ring annulus mask
        3) Exclude mask pixels near original image borders (flat cutoff segments)
        4) Keep only outer-edge envelope points (farthest point per angle bin)
        5) Fit circle to those points (algebraic least squares)
        Returns:
          (cx, cy, radius, annulus_points[N,2], annulus_mask[H,W], fit_points[N,2])
        """
        pad_left_px = max(0, int(pad_left_px))
        pad_bottom_px = max(0, int(pad_bottom_px))
        g_work = np.pad(
            g_norm,
            ((0, pad_bottom_px), (pad_left_px, 0)),
            mode="constant",
            constant_values=0.0,
        )

        h, w = g_work.shape
        yy, xx = np.mgrid[0:h, 0:w]

        img = g_work.astype(np.float64, copy=False)

        cx0 = 0.5 * (w - 1)
        cy0 = 0.5 * (h - 1)
        r0 = np.sqrt((xx - cx0) ** 2 + (yy - cy0) ** 2)
        r_norm0 = r0 / max(1e-9, 0.5 * min(h, w))

        # White mask from the SAME threshold basis as middle-hole detection, but slightly harsher
        # to suppress outside-circle artifacts.
        t_up = float(np.clip(float(t_cut) + 0.08, 0.0, 1.0))
        t_dn = float(np.clip(float(t_cut) - 0.08, 0.0, 1.0))
        # If center is darker, bright class is >= t; otherwise bright class is <= t.
        white = (img >= t_up) if center_is_darker else (img <= t_dn)
        white = white & (r_norm0 >= 0.45)

        # Convert to a ring-like boundary and choose the outer-most component.
        ring = white & (~ndi.binary_erosion(white, structure=np.ones((3, 3), dtype=bool)))
        cand = ring
        lbl, n = ndi.label(ring)
        if n > 0:
            best_score = -1.0
            best_comp = None
            for i in range(1, n + 1):
                comp = lbl == i
                area = int(np.count_nonzero(comp))
                if area < 12:
                    continue
                rr = r_norm0[comp]
                mean_r = float(np.mean(rr)) if rr.size else 0.0
                score = area * (mean_r ** 2)
                if score > best_score:
                    best_score = score
                    best_comp = comp
            if best_comp is not None:
                cand = best_comp

        ys, xs = np.nonzero(cand)

        if xs.size < 16:
            # Fallback to frame-centered circle.
            pts = np.column_stack([xs.astype(np.float64), ys.astype(np.float64)])
            return cx0, cy0, 0.5 * min(h, w), pts, cand.astype(bool, copy=False), pts

        pts_all = np.column_stack([xs.astype(np.float64), ys.astype(np.float64)])
        n_all = pts_all.shape[0]
        n_use = min(3200, n_all)
        idx = np.linspace(0, n_all - 1, n_use).astype(np.int64) if n_all > n_use else np.arange(n_all, dtype=np.int64)
        pts = pts_all[idx]

        # Exclude points near the ORIGINAL (unpadded) image edges.
        # Convert padded coords -> original coords before edge filtering.
        border_margin = 15.0
        w0 = float(g_norm.shape[1])
        h0 = float(g_norm.shape[0])
        x_raw = pts[:, 0] - float(pad_left_px)
        y_raw = pts[:, 1]
        fit_keep = (
            (x_raw > border_margin)
            & (x_raw < ((w0 - 1.0) - border_margin))
            & (y_raw > border_margin)
            & (y_raw < ((h0 - 1.0) - border_margin))
        )
        fit_pts = pts[fit_keep]
        if fit_pts.shape[0] < 12:
            fit_pts = pts

        # Use only the outer edge of the blurry annulus:
        # for each angular bin around a provisional center, keep the farthest point.
        if fit_pts.shape[0] >= 24:
            c0x = float(np.mean(fit_pts[:, 0]))
            c0y = float(np.mean(fit_pts[:, 1]))
            dx0 = fit_pts[:, 0] - c0x
            dy0 = fit_pts[:, 1] - c0y
            rr0 = np.hypot(dx0, dy0)
            aa0 = np.arctan2(dy0, dx0)
            n_bins = int(max(90, min(720, fit_pts.shape[0] // 3)))
            edges = np.linspace(-np.pi, np.pi, n_bins + 1)
            bidx = np.digitize(aa0, edges) - 1
            keep_outer_idx = []
            for bi in range(n_bins):
                sel = np.where(bidx == bi)[0]
                if sel.size == 0:
                    continue
                k = int(sel[np.argmax(rr0[sel])])
                keep_outer_idx.append(k)
            if keep_outer_idx:
                fit_pts = fit_pts[np.array(keep_outer_idx, dtype=np.int64)]

        # Algebraic circle fit: x^2 + y^2 = 2*cx*x + 2*cy*y + c
        X = fit_pts[:, 0]
        Y = fit_pts[:, 1]
        A = np.column_stack((2.0 * X, 2.0 * Y, np.ones_like(X)))
        b = (X * X) + (Y * Y)
        try:
            sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            cx = float(sol[0])
            cy = float(sol[1])
            c = float(sol[2])
            rad2 = c + (cx * cx) + (cy * cy)
            radius = float(np.sqrt(max(1e-9, rad2)))
        except Exception:
            cx, cy, radius = cx0, cy0, 0.5 * min(h, w)

        return cx, cy, radius, pts, cand.astype(bool, copy=False), fit_pts

    def _save_hole_mask_debug(
        self,
        g_norm: np.ndarray,
        blocked: np.ndarray,
        hole: np.ndarray,
        t: float,
        center_is_darker: bool,
    ) -> tuple[str, float | None]:
        """
        Save requested inspectable mask artifacts and return fitted radius fraction:
        hole_radius_px / outer_radius_px.
        """
        out_dir = self.hole_png_path.parent / "mask_debug"
        out_dir.mkdir(parents=True, exist_ok=True)
        for old_name in (
            "hole_source_normalized.png",
            "hole_blocked_threshold_mask.png",
            "hole_outer_ring_points_used.png",
            "hole_annulus_farthest_points.png",
        ):
            old_path = out_dir / old_name
            if old_path.exists():
                try:
                    old_path.unlink()
                except Exception:
                    pass
        selected_path = out_dir / "hole_selected_component_mask.png"
        overlay_path = out_dir / "hole_selected_overlay.png"
        annulus_fit_points_path = out_dir / "hole_annulus_fit_points.png"
        plt.imsave(selected_path, hole.astype(np.uint8), cmap="gray", vmin=0, vmax=1)

        base = np.clip(g_norm, 0.0, 1.0)

        # Fit outer bright circle from annulus mask points.
        fit_pad_left = int(max(100, self.hole_fit_pad_left_px))
        fit_pad_bottom = int(max(100, self.hole_fit_pad_bottom_px))
        cx, cy, ring_radius_px, pts, annulus_mask, fit_pts = self._fit_outer_circle_from_bright_points(
            g_norm,
            t_cut=float(t),
            center_is_darker=bool(center_is_darker),
            pad_left_px=fit_pad_left,
            pad_bottom_px=fit_pad_bottom,
        )

        # Use padded base as the working canvas for diagnostics.
        base_pad = np.pad(base, ((0, fit_pad_bottom), (fit_pad_left, 0)), mode="constant", constant_values=0.0)
        hole_pad = np.pad(hole, ((0, fit_pad_bottom), (fit_pad_left, 0)), mode="constant", constant_values=False)

        overlay = np.stack([base_pad, base_pad, base_pad], axis=-1)
        overlay[..., 0] = np.where(hole_pad, 1.0, overlay[..., 0])
        overlay[..., 1] = np.where(hole_pad, 0.15, overlay[..., 1])
        overlay[..., 2] = np.where(hole_pad, 0.15, overlay[..., 2])

        # Save annulus fit points image (requested output).
        hp0, wp0 = base_pad.shape
        annulus_pad = annulus_mask.astype(np.uint8, copy=False)
        annulus_rgb = np.stack([annulus_pad.astype(np.float64)] * 3, axis=-1)
        if fit_pts.size > 0:
            fpx = np.clip(np.rint(fit_pts[:, 0]).astype(np.int64), 0, wp0 - 1)
            fpy = np.clip(np.rint(fit_pts[:, 1]).astype(np.int64), 0, hp0 - 1)
            annulus_rgb[fpy, fpx, 0] = 1.0
            annulus_rgb[fpy, fpx, 1] = 0.0
            annulus_rgb[fpy, fpx, 2] = 0.0
        plt.imsave(annulus_fit_points_path, np.clip(annulus_rgb, 0.0, 1.0))

        # Fit a circle to the selected hole cutout mask and compute radius fraction.
        hole_radius_px = None
        hole_edge = hole_pad & (~ndi.binary_erosion(hole_pad, structure=np.ones((3, 3), dtype=bool)))
        yh, xh = np.nonzero(hole_edge)
        if xh.size >= 12:
            xhf = xh.astype(np.float64, copy=False)
            yhf = yh.astype(np.float64, copy=False)
            Ah = np.column_stack((2.0 * xhf, 2.0 * yhf, np.ones_like(xhf)))
            bh = (xhf * xhf) + (yhf * yhf)
            try:
                solh, _, _, _ = np.linalg.lstsq(Ah, bh, rcond=None)
                cxh = float(solh[0])
                cyh = float(solh[1])
                ch = float(solh[2])
                rad2h = ch + (cxh * cxh) + (cyh * cyh)
                hole_radius_px = float(np.sqrt(max(1e-9, rad2h)))
            except Exception:
                hole_radius_px = None
        if hole_radius_px is None:
            area_hole = float(np.count_nonzero(hole_pad))
            if area_hole > 0.0:
                hole_radius_px = float(np.sqrt(area_hole / np.pi))

        # Pad so full circle can be shown even if fitted center/radius extends outside frame.
        pad_margin = 3.0
        thick = 2.0
        x_min = cx - ring_radius_px - thick - pad_margin
        x_max = cx + ring_radius_px + thick + pad_margin
        y_min = cy - ring_radius_px - thick - pad_margin
        y_max = cy + ring_radius_px + thick + pad_margin

        pad_left = max(0, int(np.ceil(-x_min)))
        pad_top = max(0, int(np.ceil(-y_min)))
        pad_right = max(0, int(np.ceil(x_max - (wp0 - 1))))
        pad_bottom = max(0, int(np.ceil(y_max - (hp0 - 1))))

        overlay_pad = np.pad(
            overlay,
            ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
            mode="constant",
            constant_values=0.0,
        )

        hp, wp = overlay_pad.shape[:2]
        yy, xx = np.mgrid[0:hp, 0:wp]
        cxp = cx + float(pad_left)
        cyp = cy + float(pad_top)
        r_px = np.sqrt((xx - cxp) ** 2 + (yy - cyp) ** 2)
        ring = np.abs(r_px - ring_radius_px) <= thick
        overlay_pad[..., 0] = np.where(ring, 1.0, overlay_pad[..., 0])
        overlay_pad[..., 1] = np.where(ring, 0.0, overlay_pad[..., 1])
        overlay_pad[..., 2] = np.where(ring, 0.0, overlay_pad[..., 2])

        plt.imsave(overlay_path, np.clip(overlay_pad, 0.0, 1.0))
        frac = None
        if hole_radius_px is not None and ring_radius_px > 1e-9:
            frac = float(np.clip(hole_radius_px / float(ring_radius_px), 0.0, 0.95))
        return str(out_dir), frac

    def _build_collection_rays(self, mode: str) -> tuple[np.ndarray, np.ndarray]:
        ratio = float(self.na) / float(self.n_medium)
        ratio = max(0.0, min(0.999999, ratio))
        theta_max = math.asin(ratio)
        if theta_max <= 1e-9:
            return np.array([[0.0, 0.0, 1.0]], dtype=np.float64), np.array([1.0], dtype=np.float64)

        theta = np.linspace(0.0, theta_max, int(max(4, self.ray_theta_samples)))
        psi = np.linspace(0.0, 2.0 * np.pi, int(max(8, self.ray_psi_samples)), endpoint=False)
        th, ps = np.meshgrid(theta, psi, indexing="ij")
        sin_th = np.sin(th)

        dirs_all = np.stack([sin_th * np.cos(ps), sin_th * np.sin(ps), np.cos(th)], axis=-1).reshape(-1, 3)
        w_all = (sin_th * (theta[1] - theta[0] if theta.size > 1 else theta_max) * (2.0 * np.pi / psi.size)).reshape(-1)

        sin_th_max = max(1e-12, math.sin(theta_max))
        rho = sin_th.reshape(-1) / sin_th_max
        ang = np.arctan2(dirs_all[:, 1], dirs_all[:, 0])

        keep = np.ones_like(rho, dtype=bool)
        if mode == "circular":
            # Explicit mapping from BFP radius fraction -> inner NA for integration:
            # NA_inner = fraction * NA_outer.
            na_inner = self._inner_na_from_fraction(self.circular_cutout_fraction)
            rho_cut = na_inner / max(1e-12, float(self.na))
            keep = rho >= float(np.clip(rho_cut, 0.0, 0.95))
        elif mode == "profile":
            keep = self._keep_mask_from_profile(rho, ang)
        if not np.any(keep):
            keep = np.ones_like(keep, dtype=bool)

        dirs = dirs_all[keep]
        w = w_all[keep]
        sw = float(np.sum(w))
        if sw <= 0.0 or (not np.isfinite(sw)):
            w = np.ones_like(w) / float(w.size)
        else:
            w = w / sw
        return dirs.astype(np.float64, copy=False), w.astype(np.float64, copy=False)

    def _inner_na_from_fraction(self, frac: float) -> float:
        f = float(np.clip(frac, 0.0, 0.95))
        return f * float(self.na)

    def _keep_mask_from_profile(self, rho: np.ndarray, ang: np.ndarray) -> np.ndarray:
        mask = self.hole_profile_mask
        if mask is None:
            return np.ones_like(rho, dtype=bool)

        ratio_now = float(self.na) / float(self.n_medium)
        ratio_now = max(0.0, min(0.999999, ratio_now))
        sin_th_max_now = max(1e-12, ratio_now)
        na_ref = min(float(self.hole_outer_na_ref), float(self.n_medium) - 1e-6)
        ratio_ref = na_ref / float(self.n_medium)
        ratio_ref = max(1e-12, min(0.999999, ratio_ref))
        rho_ref = rho * (sin_th_max_now / ratio_ref)

        h, w = mask.shape
        px = (rho_ref * np.cos(ang))
        py = (rho_ref * np.sin(ang))
        u = 0.5 * (px + 1.0) * (w - 1)
        v = 0.5 * (1.0 - py) * (h - 1)
        ix = np.clip(np.rint(u).astype(np.int64), 0, w - 1)
        iy = np.clip(np.rint(v).astype(np.int64), 0, h - 1)
        blocked = mask[iy, ix]
        return ~blocked

    @staticmethod
    def _simulate_components_from_rays(rays: np.ndarray, w: np.ndarray, dirs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        dot = dirs @ rays.T
        ex = dirs[:, 0:1] - (dot * rays[None, :, 0])
        ey = dirs[:, 1:2] - (dot * rays[None, :, 1])

        def intensity(psi_deg: float) -> np.ndarray:
            pr = math.radians(float(psi_deg))
            ax = math.cos(pr)
            ay = math.sin(pr)
            comp = (ax * ex) + (ay * ey)
            return (comp * comp) @ w

        i0 = intensity(0.0)
        i90 = intensity(90.0)
        i45 = intensity(45.0)
        i135 = intensity(135.0)
        return i0, i90, i45, i135

    def _simulate_xy_for_params(self, mode: str, alpha_deg: float, beta_deg: float, gamma_deg: float, phi_deg: np.ndarray) -> CurveResult:
        rays, w = self._build_collection_rays(mode=mode)
        dirs = self._rod_directions(alpha_deg, beta_deg, gamma_deg, np.deg2rad(phi_deg))
        i0, i90, i45, i135 = self._simulate_components_from_rays(rays, w, dirs)
        eps = 1e-15
        x = (i0 - i90) / (i0 + i90 + eps)
        y = (i45 - i135) / (i45 + i135 + eps)
        return CurveResult(phi_deg=phi_deg.astype(np.float64, copy=False), x=x, y=y)

    def _compute_forward_curves(self) -> None:
        phi_deg = np.linspace(0.0, 360.0, int(max(90, self.phi_samples)))
        self.forward_base = self._simulate_xy_for_params("none", self.alpha_deg, self.beta_deg, self.gamma_deg, phi_deg)
        self.forward_circular = None
        self.forward_profile = None
        if self.show_circular_cutout:
            self.forward_circular = self._simulate_xy_for_params("circular", self.alpha_deg, self.beta_deg, self.gamma_deg, phi_deg)
        if self.show_profile_cutout and self.hole_profile_mask is not None:
            self.forward_profile = self._simulate_xy_for_params("profile", self.alpha_deg, self.beta_deg, self.gamma_deg, phi_deg)

    def _set_status(self, msg: str) -> None:
        if self.status_text is not None:
            self.status_text.set_text(msg)
        if self.fig is not None:
            self.fig.canvas.draw_idle()

    def _read_controls(self) -> None:
        self.alpha_deg = float(self.s_alpha.val)
        self.beta_deg = float(self.s_beta.val)
        self.gamma_deg = float(self.s_gamma.val)
        self.phi_deg = float(self.s_phi.val)
        self.na = float(self.s_na.val)
        self.na = max(0.01, min(self.na, self.n_medium - 1e-6))

    def _on_geometry_slider(self, _val: float) -> None:
        self._read_controls()
        self._sync_cutout_textbox()
        self._recompute_all()

    def _on_phi_slider(self, _val: float) -> None:
        self._read_controls()
        self._draw_scene()

    def _on_toggle_play_sim(self, _event) -> None:
        self.play_sim = not bool(self.play_sim)
        if self.btn_play_sim is not None:
            self.btn_play_sim.label.set_text("Pause Sim" if self.play_sim else "Play Sim")

    def _on_recompute_button(self, _event) -> None:
        self._read_controls()
        self._recompute_all()

    def _on_toggle_hole_modes(self, _label: str) -> None:
        if self.chk_hole_modes is not None:
            s = self.chk_hole_modes.get_status()
            self.show_no_hole = bool(s[0])
            self.show_circular_cutout = bool(s[1])
            self.show_profile_cutout = bool(s[2])
        self._read_controls()
        self._recompute_all()

    def _on_cutout_submit(self, text: str) -> None:
        try:
            v = float(text)
        except Exception:
            if self.tb_cutout is not None:
                self.tb_cutout.set_val(f"{self._circular_inner_na():.3f}")
            return

        vmax = min(float(self.na), float(self.n_medium) - 1e-6)
        inner_na = float(np.clip(v, 0.0, vmax))
        self.circular_cutout_fraction = self._fraction_from_inner_na(inner_na)
        if self.tb_cutout is not None:
            self.tb_cutout.set_val(f"{self._circular_inner_na():.3f}")
        self._set_status(f"Circular cutout inner NA set to {self._circular_inner_na():.3f}")
        self._recompute_all()

    def _sync_cutout_textbox(self) -> None:
        if self.tb_cutout is None:
            return
        shown = self.tb_cutout.text.strip()
        target = f"{self._circular_inner_na():.3f}"
        if shown != target:
            self.tb_cutout.set_val(target)

    def _on_timer(self) -> None:
        if self.play_sim and self.s_phi is not None:
            nxt = (float(self.s_phi.val) + float(self.play_speed_deg)) % 360.0
            self.s_phi.set_val(nxt)

    def _draw_geometry_3d(self, alpha_deg: float, beta_deg: float, gamma_deg: float, phi_deg: float) -> None:
        self.ax3d.cla()

        k = self._rotation_axis(alpha_deg, beta_deg)
        cone = self._rod_directions(alpha_deg, beta_deg, gamma_deg, np.deg2rad(np.linspace(0.0, 360.0, 240)))
        u = self._rod_directions(alpha_deg, beta_deg, gamma_deg, np.array([math.radians(float(phi_deg))]))[0]

        # Unit sphere backdrop.
        th = np.linspace(0.0, np.pi, 60)
        ps = np.linspace(0.0, 2.0 * np.pi, 120)
        sth, cth = np.sin(th), np.cos(th)
        cps, sps = np.cos(ps), np.sin(ps)
        sx = np.outer(sth, cps)
        sy = np.outer(sth, sps)
        sz = np.outer(cth, np.ones_like(ps))
        self.ax3d.plot_surface(sx, sy, sz, color="tab:gray", alpha=0.16, linewidth=0, shade=True)

        # Direction traces/arrows.
        self.ax3d.quiver(0, 0, 0, 0, 0, 1, length=1.05, color="black", linewidth=2, arrow_length_ratio=0.08)
        self.ax3d.quiver(0, 0, 0, k[0], k[1], k[2], length=1.0, color="tab:red", linewidth=2, arrow_length_ratio=0.08)
        self.ax3d.plot(cone[:, 0], cone[:, 1], cone[:, 2], color="tab:orange", lw=1.5)
        self.ax3d.plot([0.0, u[0]], [0.0, u[1]], [0.0, u[2]], color="tab:blue", lw=3)
        self.ax3d.scatter([u[0]], [u[1]], [u[2]], color="tab:blue", s=36)

        self.ax3d.set_xlim(-1.1, 1.1)
        self.ax3d.set_ylim(-1.1, 1.1)
        self.ax3d.set_zlim(-1.1, 1.1)
        self.ax3d.set_box_aspect((1, 1, 1))
        self.ax3d.set_axis_off()

        handles = [
            Line2D([0], [0], color="black", lw=2.0, label="optic axis"),
            Line2D([0], [0], color="tab:red", lw=2.0, label="rotation axis"),
            Line2D([0], [0], color="tab:blue", lw=3.0, label="rod direction"),
        ]
        self.ax3d.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.02, 0.98), fontsize=8, frameon=True)

    def _plot_xy_trace(self, curve: CurveResult, color: str, label: str, marker: str) -> None:
        self.ax_xy.plot(curve.x, curve.y, color=color, lw=2.2, label=label)
        i = int(np.argmin(np.abs(curve.phi_deg - float(self.phi_deg))))
        self.ax_xy.scatter([curve.x[i]], [curve.y[i]], color=color, s=36, marker=marker, zorder=5)

    def _plot_phi_trace(self, curve: CurveResult, color: str, label: str) -> None:
        self.ax_phi.plot(curve.phi_deg, curve.x, color=color, lw=1.6, label=f"{label}: X")
        self.ax_phi.plot(curve.phi_deg, curve.y, color=color, lw=1.6, ls="--", label=f"{label}: Y")

    def _draw_scene(self) -> None:
        self.ax_xy.cla()
        self.ax_phi.cla()

        if self.forward_base is None:
            self._compute_forward_curves()

        if self.show_no_hole and self.forward_base is not None:
            self._plot_xy_trace(self.forward_base, color="tab:blue", label="No hole", marker="o")
            self._plot_phi_trace(self.forward_base, color="tab:blue", label="No hole")
        if self.forward_circular is not None:
            self._plot_xy_trace(self.forward_circular, color="tab:orange", label="Circular hole", marker="s")
            self._plot_phi_trace(self.forward_circular, color="tab:orange", label="Circular hole")
        if self.forward_profile is not None:
            self._plot_xy_trace(self.forward_profile, color="tab:green", label="Jagged hole", marker="^")
            self._plot_phi_trace(self.forward_profile, color="tab:green", label="Jagged hole")

        self.ax_xy.axhline(0.0, color="0.85", lw=1)
        self.ax_xy.axvline(0.0, color="0.85", lw=1)
        self.ax_xy.set_aspect("equal", adjustable="box")
        self.ax_xy.set_xlabel("X")
        self.ax_xy.set_ylabel("Y")
        self.ax_xy.set_title("Simulation XY Trajectory")
        self.ax_xy.grid(alpha=0.2)
        self.ax_xy.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0, fontsize=8)

        self.ax_phi.axvline(float(self.phi_deg), color="tab:red", linestyle="--", lw=1.2)
        self.ax_phi.set_xlim(0.0, 360.0)
        self.ax_phi.set_xlabel("phi (deg)")
        self.ax_phi.set_ylabel("value")
        self.ax_phi.grid(alpha=0.25)
        self.ax_phi.legend(loc="upper left", bbox_to_anchor=(0.0, 1.20), borderaxespad=0.0, fontsize=8, ncol=3)
        self.ax_phi.set_title("X, Y vs phi")

        self._draw_geometry_3d(self.alpha_deg, self.beta_deg, self.gamma_deg, self.phi_deg)

        info = []
        if self.hole_radius_fraction_fit is not None:
            info.append(
                f"fitted inner NA={self._inner_na_from_fraction(self.hole_radius_fraction_fit):.3f}"
            )
        if self.show_circular_cutout:
            info.append(
                f"circular inner NA={self._circular_inner_na():.3f} (outer NA={self.na:.3f})"
            )
        if self.show_profile_cutout:
            info.append("jagged=ON" if self.hole_profile_mask is not None else "jagged=OFF (png load failed)")
        if info:
            self.ax_xy.text(
                0.03,
                0.97,
                "\n".join(info),
                transform=self.ax_xy.transAxes,
                va="top",
                ha="left",
                fontsize=8,
                bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "0.8"},
            )

        self.fig.canvas.draw_idle()

    def _recompute_all(self) -> None:
        self._compute_forward_curves()
        if self.forward_circular is not None and self.forward_profile is not None:
            dx_max = float(np.max(np.abs(self.forward_circular.x - self.forward_profile.x)))
            dy_max = float(np.max(np.abs(self.forward_circular.y - self.forward_profile.y)))
            self._set_status(f"max |dX|={dx_max:.6f}, max |dY|={dy_max:.6f}")
        else:
            self._set_status(self.hole_load_message if self.hole_load_message else "Ready")
        self._draw_scene()

    def build_ui(self) -> None:
        self.fig = plt.figure(figsize=(15.2, 8.3))
        self.fig.suptitle("Rod Dipole Simulator (Simulation Only)", fontsize=14)
        self.status_text = self.fig.text(0.05, 0.945, self.hole_load_message if self.hole_load_message else "Ready", fontsize=9)

        gs = self.fig.add_gridspec(
            2,
            2,
            left=0.05,
            right=0.92,
            bottom=0.22,
            top=0.92,
            width_ratios=[1.0, 1.45],
            height_ratios=[1.35, 0.85],
            hspace=0.28,
            wspace=0.22,
        )
        self.ax3d = self.fig.add_subplot(gs[:, 0], projection="3d")
        self.ax_xy = self.fig.add_subplot(gs[0, 1])
        self.ax_phi = self.fig.add_subplot(gs[1, 1])

        ax_alpha = self.fig.add_axes([0.08, 0.19, 0.32, 0.03])
        ax_beta = self.fig.add_axes([0.08, 0.15, 0.32, 0.03])
        ax_gamma = self.fig.add_axes([0.08, 0.11, 0.32, 0.03])
        ax_phi = self.fig.add_axes([0.50, 0.19, 0.32, 0.03])
        ax_na = self.fig.add_axes([0.50, 0.15, 0.32, 0.03])

        self.s_alpha = Slider(ax_alpha, "alpha (deg)", 0.0, 90.0, valinit=self.alpha_deg, valstep=0.1)
        self.s_beta = Slider(ax_beta, "beta (deg)", 0.0, 360.0, valinit=self.beta_deg, valstep=0.1)
        self.s_gamma = Slider(ax_gamma, "gamma (deg)", 0.0, 89.0, valinit=self.gamma_deg, valstep=0.1)
        self.s_phi = Slider(ax_phi, "phi (deg)", 0.0, 360.0, valinit=self.phi_deg, valstep=0.1)
        self.s_na = Slider(ax_na, "outer NA", 0.05, self.n_medium - 0.001, valinit=self.na, valstep=0.001)

        self.s_alpha.on_changed(self._on_geometry_slider)
        self.s_beta.on_changed(self._on_geometry_slider)
        self.s_gamma.on_changed(self._on_geometry_slider)
        self.s_na.on_changed(self._on_geometry_slider)
        self.s_phi.on_changed(self._on_phi_slider)

        ax_play_sim = self.fig.add_axes([0.50, 0.095, 0.10, 0.04])
        ax_recompute = self.fig.add_axes([0.62, 0.095, 0.13, 0.04])
        self.btn_play_sim = Button(ax_play_sim, "Play Sim")
        self.btn_recompute = Button(ax_recompute, "Recompute")
        self.btn_play_sim.on_clicked(self._on_toggle_play_sim)
        self.btn_recompute.on_clicked(self._on_recompute_button)

        ax_holes = self.fig.add_axes([0.08, 0.015, 0.30, 0.07])
        ax_cut = self.fig.add_axes([0.40, 0.03, 0.10, 0.04])
        self.chk_hole_modes = CheckButtons(
            ax_holes,
            ["Show no hole", "Show circular hole", "Show jagged hole"],
            [self.show_no_hole, self.show_circular_cutout, self.show_profile_cutout],
        )
        self.tb_cutout = TextBox(ax_cut, "inner NA", initial=f"{self._circular_inner_na():.3f}")
        self.chk_hole_modes.on_clicked(self._on_toggle_hole_modes)
        self.tb_cutout.on_submit(self._on_cutout_submit)

        self.timer = self.fig.canvas.new_timer(interval=40)
        self.timer.add_callback(self._on_timer)
        self.timer.start()

        self._recompute_all()

    def show(self) -> None:
        self.build_ui()
        plt.show()


def main() -> None:
    RodDipoleSimulator().show()


if __name__ == "__main__":
    main()
