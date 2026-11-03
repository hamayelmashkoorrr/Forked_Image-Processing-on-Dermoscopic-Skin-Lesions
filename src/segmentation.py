# # # """Classical lesion segmentation: Otsu, GrabCut and active contours (Chan-Vese).

# # # Every method takes a BGR image at working resolution plus a boolean `valid` mask
# # # (False on the black dermoscope vignette) and returns a boolean lesion mask.
# # # All methods share the same cleanup and the same upsampling back to full size."""
# # # import cv2
# # # import numpy as np
# # # from scipy import ndimage as ndi
# # # from skimage.segmentation import disk_level_set, morphological_chan_vese

# # # from .config import SegmentationConfig

# # # METHODS = ("otsu", "grabcut", "active_contour")


# # # # ----------------------------------------------------------------- helpers
# # # def _odd(n: int) -> int:
# # #     n = int(n)
# # #     return n if n % 2 == 1 else n + 1


# # # def _shrink(img: np.ndarray, max_side: int) -> np.ndarray:
# # #     """Downscale so the longest side is max_side (never upscales)."""
# # #     h, w = img.shape[:2]
# # #     s = max_side / max(h, w)
# # #     if s >= 1.0:
# # #         return img
# # #     return cv2.resize(img, (max(1, round(w * s)), max(1, round(h * s))),
# # #                       interpolation=cv2.INTER_AREA)


# # # def to_full_res(mask: np.ndarray, shape_hw) -> np.ndarray:
# # #     """Upsample a boolean mask to (H, W) with linear interpolation, then re-binarise."""
# # #     h, w = shape_hw
# # #     if mask.shape == (h, w):
# # #         return mask
# # #     up = cv2.resize(mask.astype(np.uint8) * 255, (w, h), interpolation=cv2.INTER_LINEAR)
# # #     return up > 127


# # # def detect_vignette(bgr: np.ndarray, cfg: SegmentationConfig) -> np.ndarray:
# # #     """Return `valid`: True everywhere except the black dermoscope vignette."""
# # #     gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
# # #     h, w = gray.shape
# # #     dark = (gray < cfg.vignette_dark_thresh).astype(np.uint8)
# # #     n, labels, stats, _ = cv2.connectedComponentsWithStats(dark, connectivity=4)
# # #     vig = np.zeros((h, w), np.uint8)
# # #     for i in range(1, n):
# # #         x, y, bw, bh, area = stats[i]
# # #         touches_border = x == 0 or y == 0 or x + bw == w or y + bh == h
# # #         if touches_border and area >= cfg.vignette_min_area_frac * h * w:
# # #             vig[labels == i] = 1
# # #     if vig.any():
# # #         vig = cv2.dilate(vig, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
# # #     if vig.mean() > cfg.vignette_max_frac:  # degenerate (very dark image): ignore
# # #         vig[:] = 0
# # #     return vig == 0


# # # def _ring(shape, frac: float) -> np.ndarray:
# # #     h, w = shape
# # #     t = max(2, int(frac * min(h, w)))
# # #     ring = np.ones(shape, bool)
# # #     ring[t:-t, t:-t] = False
# # #     return ring


# # # def _pick_side(region: np.ndarray, valid: np.ndarray, cfg: SegmentationConfig) -> np.ndarray:
# # #     """A two-class split has no built-in polarity. The lesion is the side that covers
# # #     less of the image border ring (lesions sit away from the edges, skin does not)."""
# # #     region = region & valid
# # #     ring = _ring(region.shape, cfg.border_ring_frac) & valid
# # #     if not ring.any():
# # #         return region
# # #     return region if region[ring].mean() <= 0.5 else (~region & valid)


# # # def clean_mask(mask: np.ndarray, valid: np.ndarray, cfg: SegmentationConfig) -> np.ndarray:
# # #     """Opening, then keep the component that is large AND central AND not hugging the border/vignette."""
# # #     m = (mask & valid).astype(np.uint8)
# # #     k = _odd(max(3, round(cfg.clean_open_frac * min(m.shape))))
# # #     m = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
# # #     n, labels, stats, cents = cv2.connectedComponentsWithStats(m, connectivity=8)
# # #     if n <= 1:
# # #         return np.zeros(m.shape, bool)
# # #     h, w = m.shape
# # #     near_vig = cv2.dilate((~valid).astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
# # #     best, best_score = 1, -1.0
# # #     for i in range(1, n):
# # #         x, y, bw, bh, area = stats[i]
# # #         touches = x == 0 or y == 0 or x + bw == w or y + bh == h or near_vig[labels == i].any()
# # #         cx, cy = cents[i]
# # #         d = np.hypot((cx - w / 2) / w, (cy - h / 2) / h)       # 0 at the centre
# # #         score = area * max(0.05, 1 - 2 * d) * (0.3 if touches else 1.0)
# # #         if score > best_score:
# # #             best, best_score = i, score
# # #     return ndi.binary_fill_holes(labels == best)


# # # # ----------------------------------------------------------------- methods
# # # def segment_otsu(bgr: np.ndarray, valid: np.ndarray, cfg: SegmentationConfig) -> np.ndarray:
# # #     """Global Otsu threshold on the Lab lightness channel, computed over valid pixels only."""
# # #     L = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)[:, :, 0]
# # #     L = cv2.GaussianBlur(L, (_odd(cfg.otsu_blur_ksize),) * 2, 0)
# # #     vals = L[valid].reshape(-1, 1)
# # #     if vals.size == 0:
# # #         return np.zeros(valid.shape, bool)
# # #     thr, _ = cv2.threshold(vals, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
# # #     return _pick_side(L <= thr, valid, cfg)


# # # def _fill_vignette(bgr: np.ndarray, valid: np.ndarray, cfg: SegmentationConfig) -> np.ndarray:
# # #     """Paint the vignette with colours resampled from the skin near the edge of the valid area,
# # #     so that black pixels do not swamp GrabCut's background model."""
# # #     if valid.all() or not valid.any():
# # #         return bgr
# # #     t = max(3, int(cfg.border_ring_frac * min(valid.shape)))
# # #     inner = cv2.erode(valid.astype(np.uint8), np.ones((2 * t + 1, 2 * t + 1), np.uint8)) > 0
# # #     band = (valid & ~inner) | (_ring(valid.shape, cfg.border_ring_frac) & valid)
# # #     samples = bgr[band]
# # #     rng = np.random.default_rng(0)  # fixed seed: reproducible results
# # #     out = bgr.copy()
# # #     out[~valid] = samples[rng.integers(0, len(samples), int((~valid).sum()))]
# # #     return out


# # # def segment_grabcut(bgr: np.ndarray, valid: np.ndarray, cfg: SegmentationConfig) -> np.ndarray:
# # #     """GrabCut initialised from a centred rectangle (independent of the other methods).
# # #     Pixels outside the rectangle and on the vignette are fixed as background."""
# # #     h, w = valid.shape
# # #     x0, y0 = int(cfg.grabcut_margin_frac * w), int(cfg.grabcut_margin_frac * h)
# # #     mask = np.full((h, w), cv2.GC_BGD, np.uint8)
# # #     mask[y0:h - y0, x0:w - x0] = cv2.GC_PR_FGD
# # #     mask[~valid] = cv2.GC_BGD
# # #     if not (mask == cv2.GC_PR_FGD).any():
# # #         return np.zeros((h, w), bool)
# # #     bgd = np.zeros((1, 65), np.float64)
# # #     fgd = np.zeros((1, 65), np.float64)
# # #     cv2.grabCut(np.ascontiguousarray(_fill_vignette(bgr, valid, cfg)), mask, None, bgd, fgd,
# # #                 cfg.grabcut_iters, cv2.GC_INIT_WITH_MASK)
# # #     return (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD)


# # # def segment_active_contour(bgr: np.ndarray, valid: np.ndarray, cfg: SegmentationConfig) -> np.ndarray:
# # #     """Morphological Chan-Vese (a level-set method) started from a centred disk."""
# # #     L = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)[:, :, 0].astype(np.float32)
# # #     if cfg.ac_blur_sigma > 0:
# # #         L = cv2.GaussianBlur(L, (0, 0), cfg.ac_blur_sigma)
# # #     if valid.any() and not valid.all():
# # #         L[~valid] = np.median(L[valid])      # the vignette must not form its own class
# # #     L = (L - L.min()) / (float(np.ptp(L)) + 1e-6)
# # #     h, w = L.shape
# # #     init = disk_level_set((h, w), radius=cfg.ac_init_radius_frac * min(h, w))
# # #     region = morphological_chan_vese(L, num_iter=cfg.ac_iters, init_level_set=init,
# # #                                      smoothing=cfg.ac_smoothing).astype(bool)
# # #     return _pick_side(region, valid, cfg)


# # # # ----------------------------------------------------------------- driver
# # # def run_methods(clahe_bgr: np.ndarray, noclahe_bgr: np.ndarray, cfg: SegmentationConfig):
# # #     """Run all three methods on one image.
# # #     Returns ({method: (full_resolution_bool_mask, status)}, vignette_fraction)."""
# # #     seg_bgr = clahe_bgr if cfg.use_clahe else noclahe_bgr
# # #     H, W = seg_bgr.shape[:2]
# # #     cache = {}

# # #     def prep(side):
# # #         if side not in cache:
# # #             img = np.ascontiguousarray(_shrink(seg_bgr, side))
# # #             valid = detect_vignette(_shrink(noclahe_bgr, side), cfg)  # vignette from the un-CLAHE'd image
# # #             cache[side] = (img, valid)
# # #         return cache[side]

# # #     plan = (("otsu", segment_otsu, cfg.work_max_side),
# # #             ("grabcut", segment_grabcut, cfg.work_max_side),
# # #             ("active_contour", segment_active_contour, cfg.ac_max_side))
# # #     out = {}
# # #     for name, fn, side in plan:
# # #         img, valid = prep(side)
# # #         try:
# # #             mask = clean_mask(fn(img, valid, cfg), valid, cfg)
# # #             status = "ok" if mask.any() else "empty"
# # #             out[name] = (to_full_res(mask, (H, W)), status)
# # #         except Exception as e:  # record and keep going
# # #             out[name] = (np.zeros((H, W), bool), f"error: {e}")
# # #     vignette_frac = 1.0 - float(prep(cfg.work_max_side)[1].mean())
# # #     return out, vignette_frac









# # """Classical lesion segmentation, v2: skin-colour saliency + seeded GrabCut / level set + ensemble.

# # Why v2: the old pipeline thresholded raw lightness (Otsu), seeded GrabCut with a big centred
# # rectangle, and started Chan-Vese from a fixed disk. Each of those fails when the lesion is pale or
# # pink, when it touches the dermoscope vignette, or when it is small. Here every method shares ONE
# # signal and ONE seed:

# #   1. detect + safely shrink the black vignette           -> `valid`
# #   2. estimate the patient's skin colour from a ring of valid pixels near the frame/vignette edge
# #      (robust, trimmed Gaussian in Lab)                    -> skin model
# #   3. saliency = Mahalanobis distance of every pixel from that skin model (0..1)
# #      (polarity is built in, so no "which side is the lesion" guessing)
# #   4. otsu           : Otsu on the saliency map (relaxed a bit, GT outlines are generous)
# #      grabcut        : GrabCut seeded by (3)+(4) instead of a rectangle
# #      active_contour : morphological Chan-Vese on sqrt(saliency), initialised from the Otsu mask
# #      ensemble       : pixel-wise majority vote of the three (cf. Tamoor et al. 2023)

# # Every method takes BGR at working resolution and returns a boolean mask; all share the same
# # cleanup and the same upsampling back to full size."""
# # import cv2
# # import numpy as np
# # from scipy import ndimage as ndi
# # from skimage.segmentation import morphological_chan_vese

# # from .config import SegmentationConfig

# # METHODS = ("otsu", "grabcut", "active_contour", "ensemble")


# # # ----------------------------------------------------------------- helpers
# # def _odd(n: int) -> int:
# #     n = int(n)
# #     return n if n % 2 == 1 else n + 1


# # def _disk(r: int) -> np.ndarray:
# #     r = max(1, int(r))
# #     return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


# # def _shrink(img: np.ndarray, max_side: int) -> np.ndarray:
# #     """Downscale so the longest side is max_side (never upscales)."""
# #     h, w = img.shape[:2]
# #     s = max_side / max(h, w)
# #     if s >= 1.0:
# #         return img
# #     return cv2.resize(img, (max(1, round(w * s)), max(1, round(h * s))),
# #                       interpolation=cv2.INTER_AREA)


# # def to_full_res(mask: np.ndarray, shape_hw) -> np.ndarray:
# #     """Upsample a boolean mask to (H, W) with linear interpolation, then re-binarise."""
# #     h, w = shape_hw
# #     if mask.shape == (h, w):
# #         return mask
# #     up = cv2.resize(mask.astype(np.uint8) * 255, (w, h), interpolation=cv2.INTER_LINEAR)
# #     return up > 127


# # def detect_vignette(bgr: np.ndarray, cfg: SegmentationConfig) -> np.ndarray:
# #     """Return `valid`: True everywhere except the black dermoscope vignette.
# #     The valid area is additionally eroded a little, because the rim of the vignette is a dark,
# #     blurry gradient that would otherwise look like a 'lesion' ring."""
# #     gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
# #     h, w = gray.shape
# #     dark = (gray < cfg.vignette_dark_thresh).astype(np.uint8)
# #     n, labels, stats, _ = cv2.connectedComponentsWithStats(dark, connectivity=4)
# #     vig = np.zeros((h, w), np.uint8)
# #     for i in range(1, n):
# #         x, y, bw, bh, area = stats[i]
# #         touches_border = x == 0 or y == 0 or x + bw == w or y + bh == h
# #         if touches_border and area >= cfg.vignette_min_area_frac * h * w:
# #             vig[labels == i] = 1
# #     if vig.any():
# #         vig = cv2.dilate(vig, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
# #     if vig.mean() > cfg.vignette_max_frac:  # degenerate (very dark image): ignore
# #         vig[:] = 0
# #     valid = vig == 0
# #     if not valid.all():
# #         valid = cv2.erode(valid.astype(np.uint8),
# #                           _disk(max(2, round(cfg.vignette_margin_frac * min(h, w))))) > 0
# #     return valid


# # def clean_mask(mask: np.ndarray, valid: np.ndarray, cfg: SegmentationConfig) -> np.ndarray:
# #     """Opening, keep the component that is large AND central AND not hugging border/vignette,
# #     then close small gaps and fill holes."""
# #     m = (mask & valid).astype(np.uint8)
# #     k = _odd(max(3, round(cfg.clean_open_frac * min(m.shape))))
# #     m = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
# #     n, labels, stats, cents = cv2.connectedComponentsWithStats(m, connectivity=8)
# #     if n <= 1:
# #         return np.zeros(m.shape, bool)
# #     h, w = m.shape
# #     near_vig = cv2.dilate((~valid).astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
# #     best, best_score = 1, -1.0
# #     for i in range(1, n):
# #         x, y, bw, bh, area = stats[i]
# #         touches = x == 0 or y == 0 or x + bw == w or y + bh == h or near_vig[labels == i].any()
# #         cx, cy = cents[i]
# #         d = np.hypot((cx - w / 2) / w, (cy - h / 2) / h)       # 0 at the centre
# #         score = area * max(0.05, 1 - 2 * d) * (0.3 if touches else 1.0)
# #         if score > best_score:
# #             best, best_score = i, score
# #     out = (labels == best).astype(np.uint8)
# #     kc = _odd(max(3, round(cfg.clean_close_frac * min(h, w))))
# #     out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kc, kc)))
# #     return ndi.binary_fill_holes((out > 0) & valid)


# # # ----------------------------------------------------------------- skin model + saliency
# # def _skin_band(valid: np.ndarray, cfg: SegmentationConfig) -> np.ndarray:
# #     """Ring of valid pixels near the frame / vignette edge: mostly healthy skin."""
# #     m = min(valid.shape)
# #     pad = cv2.copyMakeBorder(valid.astype(np.uint8), 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
# #     dist = cv2.distanceTransform(pad, cv2.DIST_L2, 3)[1:-1, 1:-1]
# #     band = valid & (dist >= cfg.skin_band_inner_frac * m) & (dist <= cfg.skin_band_outer_frac * m)
# #     if band.sum() < 300:                       # tiny valid area: use everything valid
# #         band = valid.copy()
# #     return band


# # def _mahal2(X: np.ndarray, mu: np.ndarray, icov: np.ndarray) -> np.ndarray:
# #     d = X - mu
# #     return np.einsum("...i,ij,...j->...", d, icov, d)


# # def _fit_skin(samples: np.ndarray, cfg: SegmentationConfig):
# #     """Trimmed Gaussian fit: repeatedly drop the band pixels that fit worst (hairs, a lesion that
# #     pokes into the ring) so they do not contaminate the skin model."""
# #     X = samples.reshape(-1, 3).astype(np.float64)
# #     reg = np.eye(3) * cfg.skin_cov_reg
# #     keep = np.ones(len(X), bool)
# #     for _ in range(cfg.skin_trim_iters):
# #         mu = np.median(X[keep], axis=0)
# #         cov = np.cov(X[keep].T) + reg
# #         d2 = _mahal2(X, mu, np.linalg.inv(cov))
# #         keep = d2 <= np.quantile(d2, cfg.skin_keep_frac)
# #     mu = X[keep].mean(axis=0)
# #     cov = np.cov(X[keep].T) + reg
# #     return mu, np.linalg.inv(cov)


# # def prepare(img_bgr: np.ndarray, ref_bgr: np.ndarray, cfg: SegmentationConfig) -> dict:
# #     """Everything the three methods share: valid mask, skin ring, saliency map, Otsu seed."""
# #     valid = detect_vignette(ref_bgr, cfg)                       # vignette from the un-CLAHE'd image
# #     h, w = valid.shape
# #     ctx = {"img": img_bgr, "valid": valid, "band": valid.copy(),
# #            "sal": np.zeros((h, w), np.float32), "thr": 1.0, "otsu": np.zeros((h, w), bool)}
# #     if not valid.any():
# #         return ctx
# #     lab = cv2.cvtColor(img_bgr.astype(np.float32) / 255.0, cv2.COLOR_BGR2LAB)  # L 0..100, a/b signed
# #     if cfg.sal_blur_sigma > 0:
# #         lab = cv2.GaussianBlur(lab, (0, 0), cfg.sal_blur_sigma)
# #     band = _skin_band(valid, cfg)
# #     mu, icov = _fit_skin(lab[band], cfg)
# #     d = np.sqrt(np.maximum(_mahal2(lab, mu, icov), 0)).astype(np.float32)
# #     d[~valid] = 0
# #     hi = max(float(np.percentile(d[valid], 99.5)), 1e-3)
# #     sal = np.clip(d / hi, 0, 1)
# #     u8 = np.round(sal * 255).astype(np.uint8)
# #     thr, _ = cv2.threshold(u8[valid].reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
# #     thr = max(thr * cfg.otsu_relax, 255.0 * cfg.sal_floor / hi)   # never below the skin-noise floor
# #     ctx.update(band=band, sal=sal, thr=float(thr) / 255.0, otsu=(u8 > thr) & valid)
# #     return ctx


# # # ----------------------------------------------------------------- methods
# # def segment_otsu(ctx: dict, cfg: SegmentationConfig) -> np.ndarray:
# #     """Otsu threshold on the skin-colour saliency map (already computed in `prepare`)."""
# #     return ctx["otsu"]


# # def _fill_vignette(bgr: np.ndarray, valid: np.ndarray, band: np.ndarray) -> np.ndarray:
# #     """Paint the vignette with skin colours resampled from the skin ring so that black pixels
# #     do not swamp GrabCut's background model."""
# #     if valid.all() or not band.any():
# #         return bgr
# #     samples = bgr[band]
# #     rng = np.random.default_rng(0)  # fixed seed: reproducible results
# #     out = bgr.copy()
# #     out[~valid] = samples[rng.integers(0, len(samples), int((~valid).sum()))]
# #     return out


# # def segment_grabcut(ctx: dict, cfg: SegmentationConfig) -> np.ndarray:
# #     """GrabCut seeded from the saliency mask:
# #     sure FG = eroded seed, probable FG = seed (+ small margin), probable BG = wider margin,
# #     sure BG = vignette, far-away pixels and pixels that look exactly like skin."""
# #     seed, valid, sal = ctx["otsu"], ctx["valid"], ctx["sal"]
# #     h, w = valid.shape
# #     m = min(h, w)
# #     seed_u8 = seed.astype(np.uint8)
# #     if not seed.any():
# #         return seed
# #     sure = cv2.erode(seed_u8, _disk(cfg.gc_sure_fg_frac * m)) > 0
# #     if not sure.any():
# #         sure = cv2.erode(seed_u8, _disk(max(1, cfg.gc_sure_fg_frac * m / 3))) > 0
# #     if not sure.any():
# #         sure = seed
# #     pfg = cv2.dilate(seed_u8, _disk(cfg.gc_prob_fg_frac * m)) > 0
# #     pbg = cv2.dilate(seed_u8, _disk(cfg.gc_prob_bg_frac * m)) > 0
# #     mask = np.full((h, w), cv2.GC_BGD, np.uint8)
# #     mask[pbg] = cv2.GC_PR_BGD
# #     mask[pfg] = cv2.GC_PR_FGD
# #     mask[seed] = cv2.GC_PR_FGD
# #     mask[(sal < cfg.gc_sure_bg_sal * ctx["thr"]) & ~pfg] = cv2.GC_BGD
# #     mask[sure] = cv2.GC_FGD
# #     mask[~valid] = cv2.GC_BGD
# #     if not (mask == cv2.GC_BGD).any() or not (mask == cv2.GC_FGD).any():
# #         return seed
# #     bgd = np.zeros((1, 65), np.float64)
# #     fgd = np.zeros((1, 65), np.float64)
# #     img = np.ascontiguousarray(_fill_vignette(ctx["img"], valid, ctx["band"]))
# #     try:
# #         cv2.grabCut(img, mask, None, bgd, fgd, cfg.grabcut_iters, cv2.GC_INIT_WITH_MASK)
# #     except cv2.error:
# #         return seed
# #     return (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD)


# # def segment_active_contour(ctx: dict, cfg: SegmentationConfig) -> np.ndarray:
# #     """Morphological Chan-Vese (level set) on sqrt(saliency), started from the Otsu mask.
# #     The sqrt compresses the very dark core so heterogeneous lesions are not cut down to it."""
# #     seed, valid, sal = ctx["otsu"], ctx["valid"], ctx["sal"]
# #     if not seed.any():
# #         return seed
# #     m = min(valid.shape)
# #     init = cv2.erode(seed.astype(np.uint8), _disk(cfg.ac_init_erode_frac * m)) > 0
# #     if not init.any():
# #         init = seed
# #     f = np.power(sal, cfg.ac_gamma).astype(np.float32)
# #     region = morphological_chan_vese(f, num_iter=cfg.ac_iters, init_level_set=init.astype(np.int8),
# #                                      smoothing=cfg.ac_smoothing).astype(bool) & valid
# #     outside = valid & ~region
# #     if region.any() and outside.any() and f[region].mean() < f[outside].mean():
# #         region = outside                       # polarity safeguard: lesion = the more salient side
# #     return region


# # # ----------------------------------------------------------------- driver
# # def run_methods(clahe_bgr: np.ndarray, noclahe_bgr: np.ndarray, cfg: SegmentationConfig):
# #     """Run all methods on one image.
# #     Returns ({method: (full_resolution_bool_mask, status)}, vignette_fraction)."""
# #     seg_bgr = clahe_bgr if cfg.use_clahe else noclahe_bgr
# #     H, W = seg_bgr.shape[:2]
# #     img = np.ascontiguousarray(_shrink(seg_bgr, cfg.work_max_side))
# #     ref = _shrink(noclahe_bgr, cfg.work_max_side)
# #     ctx = prepare(img, ref, cfg)
# #     valid = ctx["valid"]

# #     fns = {"otsu": segment_otsu, "grabcut": segment_grabcut, "active_contour": segment_active_contour}
# #     cleaned, out = {}, {}
# #     for name, fn in fns.items():
# #         try:
# #             cleaned[name] = clean_mask(fn(ctx, cfg), valid, cfg)
# #             status = "ok" if cleaned[name].any() else "empty"
# #             out[name] = (to_full_res(cleaned[name], (H, W)), status)
# #         except Exception as e:  # record and keep going
# #             cleaned[name] = np.zeros(valid.shape, bool)
# #             out[name] = (np.zeros((H, W), bool), f"error: {e}")
# #     try:
# #         votes = sum(cleaned[n].astype(np.uint8) for n in fns)
# #         ens = clean_mask(votes >= cfg.ensemble_min_votes, valid, cfg)
# #         out["ensemble"] = (to_full_res(ens, (H, W)), "ok" if ens.any() else "empty")
# #     except Exception as e:
# #         out["ensemble"] = (np.zeros((H, W), bool), f"error: {e}")
# #     return out, 1.0 - float(valid.mean())







# """Classical lesion segmentation, v4: field-of-view detection + skin-colour saliency + seeded GrabCut / level set.

# What changed relative to v3 (each item fixes a failure seen in the 8 worst images):
#   1. FIELD OF VIEW instead of "black pixels": the dermoscope rim is a SOFT dark gradient (gray ~100-120, not < 35),
#      so v3 detected no vignette at all (vignette_frac = 0) and segmented the bright-to-dark rim crescent instead of
#      the lesion. v4 takes the convex hull of the bright region as the field of view and excludes the rest.
#   2. RADIAL illumination model for the pass-2 skin fit (1, x, y, r^2, r^4 around the field centre) instead of a
#      quadratic surface: handles the light falloff toward the rim, so no more fake "lesion" pieces at the border.
#   3. POLARITY check: if the pass-1 seed is brighter AND less coloured than the rest of the field, it is healthy skin
#      (the lesion fills the field and contaminated the skin ring) -> the skin model is fitted on it instead.
#   4. WHOLE-FIELD fallback: if colour contrast is tiny and the above-threshold pixels are fragmented, no lesion/skin
#      boundary exists to be found; the lesion is then (almost) the whole field, which is what the ground truth shows.
#   5. HYSTERESIS growth of the seed toward weaker saliency (grow_low_frac; 0 disables) for diffuse lesions.

# Pipeline shared by every method
#   1. field of view (above)                                              -> `valid`
#   2. pass 1: skin colour from a ring near the field edge -> saliency (Mahalanobis distance in Lab) -> Otsu seed
#   3. pass 2: re-fit the skin model on everything OUTSIDE the seed with a radial illumination surface
#   4. thin-structure suppression (grey opening) removes hairs, ruler ticks and ink from the saliency map
#   5. seed sanity checks, polarity check, whole-field fallback, hysteresis growth
#   6. cleanup keeps the best central component AND large neighbouring fragments

#   otsu           : Otsu on the saliency map (relaxed a bit, GT outlines are generous)
#   grabcut        : GrabCut seeded by the saliency mask instead of a rectangle
#   active_contour : morphological Chan-Vese on sqrt(saliency), initialised from the Otsu mask
#   ensemble       : pixel-wise majority vote of the three"""
# import cv2
# import numpy as np
# from scipy import ndimage as ndi
# from skimage.segmentation import morphological_chan_vese
# from .config import SegmentationConfig

# METHODS = ("otsu", "grabcut", "active_contour", "ensemble")

# def _odd(n):
#     n = int(n); return n if n % 2 == 1 else n + 1

# def _disk(r):
#     r = max(1, int(r)); return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*r+1, 2*r+1))

# def _shrink(img, max_side):
#     h, w = img.shape[:2]; s = max_side / max(h, w)
#     if s >= 1.0: return img
#     return cv2.resize(img, (max(1, round(w*s)), max(1, round(h*s))), interpolation=cv2.INTER_AREA)

# def to_full_res(mask, shape_hw):
#     h, w = shape_hw
#     if mask.shape == (h, w): return mask
#     up = cv2.resize(mask.astype(np.uint8)*255, (w, h), interpolation=cv2.INTER_LINEAR)
#     return up > 127

# def detect_vignette(bgr, cfg):
#     """Dermoscope field of view = convex hull of the bright region, so the SOFT dark rim and corners are
#     excluded even when they are far from black. Returns (valid, vignette_fraction).
#     Guards against mistaking a dark lesion for vignette: the excluded part must be dark, small, and cover
#     at least two image corners (a lesion in one corner is not excluded)."""
#     gray = cv2.GaussianBlur(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), (0, 0), 2).astype(np.float32)
#     h, w = gray.shape
#     ref = float(np.percentile(gray, 75))
#     valid = np.ones((h, w), bool)
#     bright = (gray >= cfg.fov_rel_thr*ref).astype(np.uint8)
#     n, lab, st, _ = cv2.connectedComponentsWithStats(bright, connectivity=8)
#     if n > 1:
#         big = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
#         cnts, _ = cv2.findContours((lab == big).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#         m = np.zeros((h, w), np.uint8)
#         cv2.fillConvexPoly(m, cv2.convexHull(max(cnts, key=cv2.contourArea)), 1)
#         hull = m > 0
#         excl = ~hull
#         if cfg.fov_min_frac < hull.mean() < 0.985 and excl.any():
#             cy, cx = max(2, int(0.03*h)), max(2, int(0.03*w))
#             corners = [hull[cy, cx], hull[cy, w-1-cx], hull[h-1-cy, cx], hull[h-1-cy, w-1-cx]]
#             if sum(not c for c in corners) >= 2 and gray[excl].mean() < 0.6*ref:
#                 valid = hull
#     # classic fallback: pure-black pixels connected to the border
#     dark = (gray < cfg.vignette_dark_thresh).astype(np.uint8)
#     n2, l2, s2, _ = cv2.connectedComponentsWithStats(dark, connectivity=4)
#     for k in range(1, n2):
#         x, y, bw, bh, area = s2[k]
#         if (x == 0 or y == 0 or x + bw == w or y + bh == h) and area >= cfg.vignette_min_area_frac*h*w:
#             valid &= ~(cv2.dilate((l2 == k).astype(np.uint8), np.ones((7, 7), np.uint8)) > 0)
#     vig_frac = 1.0 - float(valid.mean())
#     if vig_frac > cfg.vignette_max_frac:
#         valid[:] = True; vig_frac = 0.0
#     if not valid.all():
#         valid = cv2.erode(valid.astype(np.uint8), _disk(max(2, round(cfg.vignette_margin_frac*min(h, w))))) > 0
#     return valid, vig_frac

# def clean_mask(mask, valid, cfg):
#     m = (mask & valid).astype(np.uint8)
#     k = _odd(max(3, round(cfg.clean_open_frac*min(m.shape))))
#     m = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
#     n, labels, stats, cents = cv2.connectedComponentsWithStats(m, connectivity=8)
#     if n <= 1: return np.zeros(m.shape, bool)
#     h, w = m.shape
#     near_vig = cv2.dilate((~valid).astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
#     touching = {}
#     best, best_score = 1, -1.0
#     for i in range(1, n):
#         x, y, bw, bh, area = stats[i]
#         touching[i] = x == 0 or y == 0 or x + bw == w or y + bh == h or near_vig[labels == i].any()
#         cx, cy = cents[i]
#         d = np.hypot((cx - w/2)/w, (cy - h/2)/h)
#         score = area * max(0.05, 1 - 2*d) * (0.3 if touching[i] else 1.0)
#         if score > best_score: best, best_score = i, score
#     keep = labels == best
#     if n > 2 and cfg.merge_min_area_frac > 0:
#         dist = ndi.distance_transform_edt(~keep)
#         for i in range(1, n):
#             if i == best or touching[i] or stats[i, 4] < cfg.merge_min_area_frac*stats[best, 4]: continue
#             comp = labels == i
#             if dist[comp].min() <= cfg.merge_dist_frac*min(h, w): keep |= comp
#     out = keep.astype(np.uint8)
#     kc = _odd(max(3, round(cfg.clean_close_frac*min(h, w))))
#     out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kc, kc)))
#     return ndi.binary_fill_holes((out > 0) & valid)

# def _skin_band(valid, cfg):
#     m = min(valid.shape)
#     pad = cv2.copyMakeBorder(valid.astype(np.uint8), 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
#     dist = cv2.distanceTransform(pad, cv2.DIST_L2, 3)[1:-1, 1:-1]
#     band = valid & (dist >= cfg.skin_band_inner_frac*m) & (dist <= cfg.skin_band_outer_frac*m)
#     if band.sum() < 300: band = valid.copy()
#     return band

# def _design(h, w, poly, valid=None):
#     """poly=False: constant colour. poly=True: smooth RADIAL illumination model
#     [1, x, y, r^2, r^4] centred on the dermoscope field (handles the soft falloff toward the rim)."""
#     if not poly: return np.ones((h, w, 1), np.float64)
#     ys, xs = np.mgrid[0:h, 0:w]
#     if valid is not None and valid.any():
#         cy, cx = ndi.center_of_mass(valid); R = np.sqrt(valid.sum()/np.pi)
#     else:
#         cy, cx, R = h/2, w/2, max(h, w)/2
#     x = (xs - cx)/R; y = (ys - cy)/R; r2 = x*x + y*y
#     return np.stack([np.ones_like(x), x, y, r2, r2*r2], axis=-1).astype(np.float64)

# def _maha(R, icov): return np.einsum("...i,ij,...j->...", R, icov, R)

# def _fit_skin(lab, skin, cfg, poly, keep_frac, valid=None):
#     h, w = skin.shape
#     A = _design(h, w, poly, valid)
#     idx = np.flatnonzero(skin.ravel())
#     if len(idx) > 30000: idx = idx[:: len(idx)//30000]
#     Ai = A.reshape(-1, A.shape[-1])[idx]
#     Yi = lab.reshape(-1, 3)[idx].astype(np.float64)
#     reg = np.eye(3)*cfg.skin_cov_reg
#     keep = np.ones(len(idx), bool)
#     for _ in range(cfg.skin_trim_iters + 1):
#         coef = np.linalg.lstsq(Ai[keep], Yi[keep], rcond=None)[0]
#         R = Yi - Ai @ coef
#         cov = np.cov(R[keep].T) + reg
#         icov = np.linalg.inv(cov)
#         keep = _maha(R, icov) <= np.quantile(_maha(R, icov), keep_frac)
#     lo, hi = np.percentile(Yi[keep], 1, axis=0), np.percentile(Yi[keep], 99, axis=0)
#     span = hi - lo
#     return A, coef, icov, lo - 0.25*span, hi + 0.25*span

# def _saliency(lab, valid, model, cfg):
#     A, coef, icov, lo, hi_ = model
#     bg = np.clip(A @ coef, lo, hi_)
#     d = np.sqrt(np.maximum(_maha(lab - bg, icov), 0)).astype(np.float32)
#     d[~valid] = 0
#     r = round(cfg.thin_open_frac*min(valid.shape))
#     if r >= 1: d = cv2.morphologyEx(d, cv2.MORPH_OPEN, _disk(r))
#     hi = max(float(np.percentile(d[valid], 99.5)), 1e-3)
#     return np.clip(d/hi, 0, 1).astype(np.float32), hi

# def _seed_from_sal(sal, valid, hi, cfg):
#     u8 = np.round(sal*255).astype(np.uint8)
#     t_otsu, _ = cv2.threshold(u8[valid].reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#     floor = 255.0*cfg.sal_floor/hi
#     nvalid = max(int(valid.sum()), 1)
#     def make(f, fl):
#         t = max(t_otsu*f, fl)
#         return clean_mask((u8 > t) & valid, valid, cfg), t
#     seed, t = make(cfg.otsu_relax, floor)
#     frac = seed.sum()/nvalid
#     if frac > cfg.max_seed_frac:
#         for f in (1.0, 1.3, 1.6):
#             s2, t2 = make(f, floor)
#             if s2.any() and s2.sum()/nvalid <= cfg.max_seed_frac:
#                 seed, t = s2, t2; break
#     elif frac < cfg.min_seed_frac:
#         for f, fl in ((0.5, floor), (0.35, 0.6*floor)):
#             s2, t2 = make(f, fl)
#             if s2.sum()/nvalid >= cfg.min_seed_frac:
#                 seed, t = s2, t2; break
#     return seed, float(t)/255.0

# def _looks_like_skin(lab, seed, valid):
#     """True if the seed is brighter AND less chromatic than the rest of the field (=> it is healthy skin,
#     i.e. the lesion fills most of the field and the skin model was fitted on the lesion)."""
#     rest = valid & ~seed
#     if seed.sum() < 50 or rest.sum() < 50: return False
#     L = lab[..., 0]; C = np.hypot(lab[..., 1], lab[..., 2])
#     sL, sC = max(float(L[valid].std()), 1e-3), max(float(C[valid].std()), 1e-3)
#     dark_score = (L[rest].mean() - L[seed].mean())/sL            # >0: seed darker than rest
#     chroma_score = (C[seed].mean() - C[rest].mean())/sC          # >0: seed more coloured than rest
#     return (dark_score + chroma_score) < -0.3



# def _field_filling(sal, valid, hi, cfg):
#     """True when there is no coherent lesion/skin contrast at all (flat, fragmented saliency): the lesion then
#     typically fills (almost) the whole dermoscope field."""
#     if not cfg.fallback_whole_field or hi >= cfg.fallback_max_hi: return False
#     u8 = np.round(sal*255).astype(np.uint8)
#     t, _ = cv2.threshold(u8[valid].reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#     m = ((u8 > t) & valid).astype(np.uint8)
#     n, _, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
#     if n <= 1: return True
#     return st[1:, cv2.CC_STAT_AREA].max()/max(int(m.sum()), 1) < cfg.fallback_max_cc


# def _grow(seed, sal, thr, valid, cfg):
#     """Hysteresis: add weaker-saliency pixels that are connected to the seed and close to it, so the contour
#     reaches the generous outline annotators draw around diffuse lesions."""
#     if cfg.grow_low_frac <= 0 or not seed.any(): return seed
#     h, w = seed.shape
#     low = (sal > cfg.grow_low_frac*thr) & valid
#     reach = cv2.dilate(seed.astype(np.uint8), _disk(cfg.grow_max_dilate_frac*min(h, w))) > 0
#     cand = (low & reach) | seed
#     n, lab = cv2.connectedComponents(cand.astype(np.uint8), connectivity=8)
#     ids = np.unique(lab[seed]); ids = ids[ids > 0]
#     grown = np.isin(lab, ids)
#     return clean_mask(grown, valid, cfg)


# def prepare(img_bgr, ref_bgr, cfg):
#     valid, vig_frac = detect_vignette(ref_bgr, cfg)
#     h, w = valid.shape
#     ctx = {"vig_frac": vig_frac, "img": img_bgr, "valid": valid, "band": valid.copy(), "sal": np.zeros((h, w), np.float32), "thr": 1.0, "otsu": np.zeros((h, w), bool)}
#     if not valid.any(): return ctx
#     lab = cv2.cvtColor(img_bgr.astype(np.float32)/255.0, cv2.COLOR_BGR2LAB)
#     if cfg.sal_blur_sigma > 0: lab = cv2.GaussianBlur(lab, (0, 0), cfg.sal_blur_sigma)
#     band = _skin_band(valid, cfg)
#     sal, hi = _saliency(lab, valid, _fit_skin(lab, band, cfg, False, cfg.skin_keep_frac), cfg)
#     sal0, hi0 = sal, hi
#     seed, thr = _seed_from_sal(sal, valid, hi, cfg)
#     inverted = False
#     if cfg.fix_polarity and seed.any() and _looks_like_skin(lab, seed, valid):
#         inverted = True
#         skin = cv2.erode(seed.astype(np.uint8), _disk(0.01*min(h, w))) > 0
#     elif seed.any():
#         skin = valid & ~(cv2.dilate(seed.astype(np.uint8), _disk(cfg.refit_dilate_frac*min(h, w))) > 0)
#     else:
#         skin = None
#     if cfg.refit_skin and skin is not None and skin.sum() >= max(300, cfg.refit_min_frac*valid.sum()):
#         sal2, hi2 = _saliency(lab, valid, _fit_skin(lab, skin, cfg, cfg.use_poly_bg, cfg.poly_keep_frac, valid), cfg)
#         seed2, thr2 = _seed_from_sal(sal2, valid, hi2, cfg)
#         if seed2.any(): sal, seed, thr = sal2, seed2, thr2
#     elif inverted:                     # could not refit: use the complement of the skin-like seed
#         seed = clean_mask(valid & ~cv2.dilate(seed.astype(np.uint8), _disk(0.01*min(h, w))).astype(bool), valid, cfg)
#     if _field_filling(sal0, valid, hi0, cfg):
#         seed = clean_mask(valid, valid, cfg); ctx["fallback"] = True
#     else:
#         seed = _grow(seed, sal, thr, valid, cfg)
#     ctx.update(band=band, sal=sal, thr=thr, otsu=seed, inverted=inverted)
#     return ctx

# def segment_otsu(ctx, cfg): return ctx["otsu"]

# def _fill_vignette(bgr, valid, band):
#     if valid.all() or not band.any(): return bgr
#     samples = bgr[band]; rng = np.random.default_rng(0); out = bgr.copy()
#     out[~valid] = samples[rng.integers(0, len(samples), int((~valid).sum()))]
#     return out

# def segment_grabcut(ctx, cfg):
#     seed, valid, sal = ctx["otsu"], ctx["valid"], ctx["sal"]
#     h, w = valid.shape; m = min(h, w); seed_u8 = seed.astype(np.uint8)
#     if not seed.any(): return seed
#     sure = cv2.erode(seed_u8, _disk(cfg.gc_sure_fg_frac*m)) > 0
#     if not sure.any(): sure = cv2.erode(seed_u8, _disk(max(1, cfg.gc_sure_fg_frac*m/3))) > 0
#     if not sure.any(): sure = seed
#     pfg = cv2.dilate(seed_u8, _disk(cfg.gc_prob_fg_frac*m)) > 0
#     pbg = cv2.dilate(seed_u8, _disk(cfg.gc_prob_bg_frac*m)) > 0
#     mask = np.full((h, w), cv2.GC_BGD, np.uint8)
#     mask[pbg] = cv2.GC_PR_BGD; mask[pfg] = cv2.GC_PR_FGD; mask[seed] = cv2.GC_PR_FGD
#     mask[(sal < cfg.gc_sure_bg_sal*ctx["thr"]) & ~pfg] = cv2.GC_BGD
#     mask[sure] = cv2.GC_FGD; mask[~valid] = cv2.GC_BGD
#     if not (mask == cv2.GC_BGD).any() or not (mask == cv2.GC_FGD).any(): return seed
#     bgd = np.zeros((1, 65), np.float64); fgd = np.zeros((1, 65), np.float64)
#     img = np.ascontiguousarray(_fill_vignette(ctx["img"], valid, ctx["band"]))
#     try: cv2.grabCut(img, mask, None, bgd, fgd, cfg.grabcut_iters, cv2.GC_INIT_WITH_MASK)
#     except cv2.error: return seed
#     return (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD)

# def segment_active_contour(ctx, cfg):
#     seed, valid, sal = ctx["otsu"], ctx["valid"], ctx["sal"]
#     if not seed.any() or ctx.get("fallback"): return seed
#     m = min(valid.shape)
#     init = cv2.erode(seed.astype(np.uint8), _disk(cfg.ac_init_erode_frac*m)) > 0
#     if not init.any(): init = seed
#     f = np.power(sal, cfg.ac_gamma).astype(np.float32)
#     region = morphological_chan_vese(f, num_iter=cfg.ac_iters, init_level_set=init.astype(np.int8), smoothing=cfg.ac_smoothing).astype(bool) & valid
#     outside = valid & ~region
#     if region.any() and outside.any() and f[region].mean() < f[outside].mean(): region = outside
#     return region

# def run_methods(clahe_bgr, noclahe_bgr, cfg):
#     seg_bgr = clahe_bgr if cfg.use_clahe else noclahe_bgr
#     H, W = seg_bgr.shape[:2]
#     img = np.ascontiguousarray(_shrink(seg_bgr, cfg.work_max_side))
#     ref = _shrink(noclahe_bgr, cfg.work_max_side)
#     ctx = prepare(img, ref, cfg); valid = ctx["valid"]
#     fns = {"otsu": segment_otsu, "grabcut": segment_grabcut, "active_contour": segment_active_contour}
#     cleaned, out = {}, {}
#     for name, fn in fns.items():
#         try:
#             cleaned[name] = clean_mask(fn(ctx, cfg), valid, cfg)
#             status = "ok" if cleaned[name].any() else "empty"
#             out[name] = (to_full_res(cleaned[name], (H, W)), status)
#         except Exception as e:
#             cleaned[name] = np.zeros(valid.shape, bool); out[name] = (np.zeros((H, W), bool), f"error: {e}")
#     votes = sum(cleaned[n].astype(np.uint8) for n in fns)
#     ens = clean_mask(votes >= cfg.ensemble_min_votes, valid, cfg)
#     out["ensemble"] = (to_full_res(ens, (H, W)), "ok" if ens.any() else "empty")
#     return out, float(ctx.get("vig_frac", 0.0))






"""Classical lesion segmentation, v5: v4 + calibration-sticker exclusion + reproducible GrabCut.

What changed relative to v4:
  1. STICKER / FOREIGN-OBJECT EXCLUSION: wide-field images (roughly ISIC_0000557 onwards) carry a colour-calibration
     sticker (teal, blue, green or yellow disc). It is the most colourful thing in the picture, so every method
     segmented the sticker instead of the lesion (and it also poisoned the skin-colour model, which flipped the
     polarity check and labelled the skin as lesion). `detect_stickers` finds such blobs by ABSOLUTE chromaticity
     (outside what skin/lesions can have) AND by distance from the image's own median skin colour (so blue-tinted
     dermoscopy images are not affected), and removes them from `valid`, exactly like the vignette.
  2. GrabCut is seeded (cv2.setRNGSeed) so repeated runs give identical masks.

What changed in v4 relative to v3 (each item fixes a failure seen in the 8 worst images):
  1. FIELD OF VIEW instead of "black pixels": the dermoscope rim is a SOFT dark gradient (gray ~100-120, not < 35),
     so v3 detected no vignette at all (vignette_frac = 0) and segmented the bright-to-dark rim crescent instead of
     the lesion. v4 takes the convex hull of the bright region as the field of view and excludes the rest.
  2. RADIAL illumination model for the pass-2 skin fit (1, x, y, r^2, r^4 around the field centre) instead of a
     quadratic surface: handles the light falloff toward the rim, so no more fake "lesion" pieces at the border.
  3. POLARITY check: if the pass-1 seed is brighter AND less coloured than the rest of the field, it is healthy skin
     (the lesion fills the field and contaminated the skin ring) -> the skin model is fitted on it instead.
  4. WHOLE-FIELD fallback: if colour contrast is tiny and the above-threshold pixels are fragmented, no lesion/skin
     boundary exists to be found; the lesion is then (almost) the whole field, which is what the ground truth shows.
  5. HYSTERESIS growth of the seed toward weaker saliency (grow_low_frac; 0 disables) for diffuse lesions.

Pipeline shared by every method
  1. field of view + sticker exclusion (above)                          -> `valid`
  2. pass 1: skin colour from a ring near the field edge -> saliency (Mahalanobis distance in Lab) -> Otsu seed
  3. pass 2: re-fit the skin model on everything OUTSIDE the seed with a radial illumination surface
  4. thin-structure suppression (grey opening) removes hairs, ruler ticks and ink from the saliency map
  5. seed sanity checks, polarity check, whole-field fallback, hysteresis growth
  6. cleanup keeps the best central component AND large neighbouring fragments

  otsu           : Otsu on the saliency map (relaxed a bit, GT outlines are generous)
  grabcut        : GrabCut seeded by the saliency mask instead of a rectangle
  active_contour : morphological Chan-Vese on sqrt(saliency), initialised from the Otsu mask
  ensemble       : pixel-wise majority vote of the three"""
import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage.segmentation import morphological_chan_vese
from .config import SegmentationConfig

METHODS = ("otsu", "grabcut", "active_contour", "ensemble")

def _odd(n):
    n = int(n); return n if n % 2 == 1 else n + 1

def _disk(r):
    r = max(1, int(r)); return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*r+1, 2*r+1))

def _shrink(img, max_side):
    h, w = img.shape[:2]; s = max_side / max(h, w)
    if s >= 1.0: return img
    return cv2.resize(img, (max(1, round(w*s)), max(1, round(h*s))), interpolation=cv2.INTER_AREA)

def to_full_res(mask, shape_hw):
    h, w = shape_hw
    if mask.shape == (h, w): return mask
    up = cv2.resize(mask.astype(np.uint8)*255, (w, h), interpolation=cv2.INTER_LINEAR)
    return up > 127

def detect_stickers(bgr, valid, cfg):
    """Colour-calibration stickers (teal / blue / green / yellow discs), coloured ink and similar foreign objects.
    A pixel is a candidate if its ABSOLUTE chromaticity is outside what skin and lesions can have
    (teal/green: a < sticker_a_max, blue: b < sticker_b_max, yellow: b > sticker_b_min, in OpenCV Lab units)
    AND it also differs clearly from this image's own median skin colour (so blue-tinted images, whose
    background is blue, are not flagged). Only large blobs survive; holes (e.g. a pen mark on the sticker) are filled."""
    h, w = valid.shape; m = min(h, w)
    if valid.sum() < 100: return np.zeros((h, w), bool)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    a = cv2.GaussianBlur(lab[..., 1] - 128, (0, 0), 2)
    b = cv2.GaussianBlur(lab[..., 2] - 128, (0, 0), 2)
    a0, b0 = np.median(a[valid]), np.median(b[valid])
    rel = np.hypot(a - a0, b - b0)
    cand = valid & ((a < cfg.sticker_a_max) | (b < cfg.sticker_b_max) | (b > cfg.sticker_b_min)) & (rel > cfg.sticker_min_rel)
    cand = cv2.morphologyEx(cand.astype(np.uint8), cv2.MORPH_CLOSE, _disk(0.015*m))
    n, lb, st, _ = cv2.connectedComponentsWithStats(cand, connectivity=8)
    keep = np.zeros((h, w), bool)
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] >= cfg.sticker_min_area_frac*h*w: keep |= lb == i
    if not keep.any(): return keep
    keep = ndi.binary_fill_holes(keep)
    keep = cv2.dilate(keep.astype(np.uint8), _disk(cfg.sticker_margin_frac*m)) > 0
    if keep.mean() > cfg.sticker_max_frac: return np.zeros((h, w), bool)   # implausible: do not trust
    return keep

def detect_vignette(bgr, cfg):
    gray = cv2.GaussianBlur(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), (0, 0), 2).astype(np.float32)
    h, w = gray.shape
    ref = float(np.percentile(gray, 75))
    valid = np.ones((h, w), bool)
    bright = (gray >= cfg.fov_rel_thr*ref).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(bright, connectivity=8)
    if n > 1:
        big = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
        cnts, _ = cv2.findContours((lab == big).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        m = np.zeros((h, w), np.uint8)
        cv2.fillConvexPoly(m, cv2.convexHull(max(cnts, key=cv2.contourArea)), 1)
        hull = m > 0
        excl = ~hull
        if cfg.fov_min_frac < hull.mean() < 0.985 and excl.any():
            cy, cx = max(2, int(0.03*h)), max(2, int(0.03*w))
            corners = [hull[cy, cx], hull[cy, w-1-cx], hull[h-1-cy, cx], hull[h-1-cy, w-1-cx]]
            if sum(not c for c in corners) >= 2 and gray[excl].mean() < 0.6*ref:
                valid = hull
    dark = (gray < cfg.vignette_dark_thresh).astype(np.uint8)
    n2, l2, s2, _ = cv2.connectedComponentsWithStats(dark, connectivity=4)
    for k in range(1, n2):
        x, y, bw, bh, area = s2[k]
        if (x == 0 or y == 0 or x + bw == w or y + bh == h) and area >= cfg.vignette_min_area_frac*h*w:
            valid &= ~(cv2.dilate((l2 == k).astype(np.uint8), np.ones((7, 7), np.uint8)) > 0)
    if cfg.sticker_detect:
        valid &= ~detect_stickers(bgr, valid, cfg)
    vig_frac = 1.0 - float(valid.mean())
    if vig_frac > cfg.vignette_max_frac:
        valid[:] = True; vig_frac = 0.0
    if not valid.all():
        valid = cv2.erode(valid.astype(np.uint8), _disk(max(2, round(cfg.vignette_margin_frac*min(h, w))))) > 0
    return valid, vig_frac

def clean_mask(mask, valid, cfg):
    m = (mask & valid).astype(np.uint8)
    k = _odd(max(3, round(cfg.clean_open_frac*min(m.shape))))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    n, labels, stats, cents = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 1: return np.zeros(m.shape, bool)
    h, w = m.shape
    near_vig = cv2.dilate((~valid).astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    touching = {}
    best, best_score = 1, -1.0
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        touching[i] = x == 0 or y == 0 or x + bw == w or y + bh == h or near_vig[labels == i].any()
        cx, cy = cents[i]
        d = np.hypot((cx - w/2)/w, (cy - h/2)/h)
        score = area * max(0.05, 1 - 2*d) * (0.3 if touching[i] else 1.0)
        if score > best_score: best, best_score = i, score
    keep = labels == best
    if n > 2 and cfg.merge_min_area_frac > 0:
        dist = ndi.distance_transform_edt(~keep)
        for i in range(1, n):
            if i == best or touching[i] or stats[i, 4] < cfg.merge_min_area_frac*stats[best, 4]: continue
            comp = labels == i
            if dist[comp].min() <= cfg.merge_dist_frac*min(h, w): keep |= comp
    out = keep.astype(np.uint8)
    kc = _odd(max(3, round(cfg.clean_close_frac*min(h, w))))
    out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kc, kc)))
    return ndi.binary_fill_holes((out > 0) & valid)

def _skin_band(valid, cfg):
    m = min(valid.shape)
    pad = cv2.copyMakeBorder(valid.astype(np.uint8), 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    dist = cv2.distanceTransform(pad, cv2.DIST_L2, 3)[1:-1, 1:-1]
    band = valid & (dist >= cfg.skin_band_inner_frac*m) & (dist <= cfg.skin_band_outer_frac*m)
    if band.sum() < 300: band = valid.copy()
    return band

def _design(h, w, poly, valid=None):
    if not poly: return np.ones((h, w, 1), np.float64)
    ys, xs = np.mgrid[0:h, 0:w]
    if valid is not None and valid.any():
        cy, cx = ndi.center_of_mass(valid); R = np.sqrt(valid.sum()/np.pi)
    else:
        cy, cx, R = h/2, w/2, max(h, w)/2
    x = (xs - cx)/R; y = (ys - cy)/R; r2 = x*x + y*y
    return np.stack([np.ones_like(x), x, y, r2, r2*r2], axis=-1).astype(np.float64)

def _maha(R, icov): return np.einsum("...i,ij,...j->...", R, icov, R)

def _fit_skin(lab, skin, cfg, poly, keep_frac, valid=None):
    h, w = skin.shape
    A = _design(h, w, poly, valid)
    idx = np.flatnonzero(skin.ravel())
    if len(idx) > 30000: idx = idx[:: len(idx)//30000]
    Ai = A.reshape(-1, A.shape[-1])[idx]
    Yi = lab.reshape(-1, 3)[idx].astype(np.float64)
    reg = np.eye(3)*cfg.skin_cov_reg
    keep = np.ones(len(idx), bool)
    for _ in range(cfg.skin_trim_iters + 1):
        coef = np.linalg.lstsq(Ai[keep], Yi[keep], rcond=None)[0]
        R = Yi - Ai @ coef
        cov = np.cov(R[keep].T) + reg
        icov = np.linalg.inv(cov)
        keep = _maha(R, icov) <= np.quantile(_maha(R, icov), keep_frac)
    lo, hi = np.percentile(Yi[keep], 1, axis=0), np.percentile(Yi[keep], 99, axis=0)
    span = hi - lo
    return A, coef, icov, lo - 0.25*span, hi + 0.25*span

def _saliency(lab, valid, model, cfg):
    A, coef, icov, lo, hi_ = model
    bg = np.clip(A @ coef, lo, hi_)
    d = np.sqrt(np.maximum(_maha(lab - bg, icov), 0)).astype(np.float32)
    d[~valid] = 0
    r = round(cfg.thin_open_frac*min(valid.shape))
    if r >= 1: d = cv2.morphologyEx(d, cv2.MORPH_OPEN, _disk(r))
    hi = max(float(np.percentile(d[valid], 99.5)), 1e-3)
    return np.clip(d/hi, 0, 1).astype(np.float32), hi

def _seed_from_sal(sal, valid, hi, cfg):
    u8 = np.round(sal*255).astype(np.uint8)
    t_otsu, _ = cv2.threshold(u8[valid].reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    floor = 255.0*cfg.sal_floor/hi
    nvalid = max(int(valid.sum()), 1)
    def make(f, fl):
        t = max(t_otsu*f, fl)
        return clean_mask((u8 > t) & valid, valid, cfg), t
    seed, t = make(cfg.otsu_relax, floor)
    frac = seed.sum()/nvalid
    if frac > cfg.max_seed_frac:
        for f in (1.0, 1.3, 1.6):
            s2, t2 = make(f, floor)
            if s2.any() and s2.sum()/nvalid <= cfg.max_seed_frac:
                seed, t = s2, t2; break
    elif frac < cfg.min_seed_frac:
        for f, fl in ((0.5, floor), (0.35, 0.6*floor)):
            s2, t2 = make(f, fl)
            if s2.sum()/nvalid >= cfg.min_seed_frac:
                seed, t = s2, t2; break
    return seed, float(t)/255.0

def _looks_like_skin(lab, seed, valid):
    rest = valid & ~seed
    if seed.sum() < 50 or rest.sum() < 50: return False
    L = lab[..., 0]; C = np.hypot(lab[..., 1], lab[..., 2])
    sL, sC = max(float(L[valid].std()), 1e-3), max(float(C[valid].std()), 1e-3)
    dark_score = (L[rest].mean() - L[seed].mean())/sL
    chroma_score = (C[seed].mean() - C[rest].mean())/sC
    return (dark_score + chroma_score) < -0.3

def _field_filling(sal, valid, hi, cfg):
    if not cfg.fallback_whole_field or hi >= cfg.fallback_max_hi: return False
    u8 = np.round(sal*255).astype(np.uint8)
    t, _ = cv2.threshold(u8[valid].reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    m = ((u8 > t) & valid).astype(np.uint8)
    n, _, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 1: return True
    return st[1:, cv2.CC_STAT_AREA].max()/max(int(m.sum()), 1) < cfg.fallback_max_cc

def _grow(seed, sal, thr, valid, cfg):
    if cfg.grow_low_frac <= 0 or not seed.any(): return seed
    h, w = seed.shape
    low = (sal > cfg.grow_low_frac*thr) & valid
    reach = cv2.dilate(seed.astype(np.uint8), _disk(cfg.grow_max_dilate_frac*min(h, w))) > 0
    cand = (low & reach) | seed
    n, lab = cv2.connectedComponents(cand.astype(np.uint8), connectivity=8)
    ids = np.unique(lab[seed]); ids = ids[ids > 0]
    grown = np.isin(lab, ids)
    return clean_mask(grown, valid, cfg)

def prepare(img_bgr, ref_bgr, cfg):
    valid, vig_frac = detect_vignette(ref_bgr, cfg)
    h, w = valid.shape
    ctx = {"vig_frac": vig_frac, "img": img_bgr, "valid": valid, "band": valid.copy(), "sal": np.zeros((h, w), np.float32), "thr": 1.0, "otsu": np.zeros((h, w), bool)}
    if not valid.any(): return ctx
    lab = cv2.cvtColor(img_bgr.astype(np.float32)/255.0, cv2.COLOR_BGR2LAB)
    if cfg.sal_blur_sigma > 0: lab = cv2.GaussianBlur(lab, (0, 0), cfg.sal_blur_sigma)
    band = _skin_band(valid, cfg)
    sal, hi = _saliency(lab, valid, _fit_skin(lab, band, cfg, False, cfg.skin_keep_frac), cfg)
    sal0, hi0 = sal, hi
    seed, thr = _seed_from_sal(sal, valid, hi, cfg)
    inverted = False
    if cfg.fix_polarity and seed.any() and _looks_like_skin(lab, seed, valid):
        inverted = True
        skin = cv2.erode(seed.astype(np.uint8), _disk(0.01*min(h, w))) > 0
    elif seed.any():
        skin = valid & ~(cv2.dilate(seed.astype(np.uint8), _disk(cfg.refit_dilate_frac*min(h, w))) > 0)
    else:
        skin = None
    if cfg.refit_skin and skin is not None and skin.sum() >= max(300, cfg.refit_min_frac*valid.sum()):
        sal2, hi2 = _saliency(lab, valid, _fit_skin(lab, skin, cfg, cfg.use_poly_bg, cfg.poly_keep_frac, valid), cfg)
        seed2, thr2 = _seed_from_sal(sal2, valid, hi2, cfg)
        if seed2.any(): sal, seed, thr = sal2, seed2, thr2
    elif inverted:
        seed = clean_mask(valid & ~cv2.dilate(seed.astype(np.uint8), _disk(0.01*min(h, w))).astype(bool), valid, cfg)
    if _field_filling(sal0, valid, hi0, cfg):
        seed = clean_mask(valid, valid, cfg); ctx["fallback"] = True
    else:
        seed = _grow(seed, sal, thr, valid, cfg)
    ctx.update(band=band, sal=sal, thr=thr, otsu=seed, inverted=inverted)
    return ctx

def segment_otsu(ctx, cfg): return ctx["otsu"]

def _fill_vignette(bgr, valid, band):
    if valid.all() or not band.any(): return bgr
    samples = bgr[band]; rng = np.random.default_rng(0); out = bgr.copy()
    out[~valid] = samples[rng.integers(0, len(samples), int((~valid).sum()))]
    return out

def segment_grabcut(ctx, cfg):
    seed, valid, sal = ctx["otsu"], ctx["valid"], ctx["sal"]
    h, w = valid.shape; m = min(h, w); seed_u8 = seed.astype(np.uint8)
    if not seed.any(): return seed
    sure = cv2.erode(seed_u8, _disk(cfg.gc_sure_fg_frac*m)) > 0
    if not sure.any(): sure = cv2.erode(seed_u8, _disk(max(1, cfg.gc_sure_fg_frac*m/3))) > 0
    if not sure.any(): sure = seed
    pfg = cv2.dilate(seed_u8, _disk(cfg.gc_prob_fg_frac*m)) > 0
    pbg = cv2.dilate(seed_u8, _disk(cfg.gc_prob_bg_frac*m)) > 0
    mask = np.full((h, w), cv2.GC_BGD, np.uint8)
    mask[pbg] = cv2.GC_PR_BGD; mask[pfg] = cv2.GC_PR_FGD; mask[seed] = cv2.GC_PR_FGD
    mask[(sal < cfg.gc_sure_bg_sal*ctx["thr"]) & ~pfg] = cv2.GC_BGD
    mask[sure] = cv2.GC_FGD; mask[~valid] = cv2.GC_BGD
    if not (mask == cv2.GC_BGD).any() or not (mask == cv2.GC_FGD).any(): return seed
    bgd = np.zeros((1, 65), np.float64); fgd = np.zeros((1, 65), np.float64)
    img = np.ascontiguousarray(_fill_vignette(ctx["img"], valid, ctx["band"]))
    cv2.setRNGSeed(0)                  # GrabCut's colour-model init is random: seed it so results are reproducible
    try: cv2.grabCut(img, mask, None, bgd, fgd, cfg.grabcut_iters, cv2.GC_INIT_WITH_MASK)
    except cv2.error: return seed
    return (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD)

def segment_active_contour(ctx, cfg):
    seed, valid, sal = ctx["otsu"], ctx["valid"], ctx["sal"]
    if not seed.any() or ctx.get("fallback"): return seed
    m = min(valid.shape)
    init = cv2.erode(seed.astype(np.uint8), _disk(cfg.ac_init_erode_frac*m)) > 0
    if not init.any(): init = seed
    f = np.power(sal, cfg.ac_gamma).astype(np.float32)
    region = morphological_chan_vese(f, num_iter=cfg.ac_iters, init_level_set=init.astype(np.int8), smoothing=cfg.ac_smoothing).astype(bool) & valid
    outside = valid & ~region
    if region.any() and outside.any() and f[region].mean() < f[outside].mean(): region = outside
    return region

def run_methods(clahe_bgr, noclahe_bgr, cfg):
    seg_bgr = clahe_bgr if cfg.use_clahe else noclahe_bgr
    H, W = seg_bgr.shape[:2]
    img = np.ascontiguousarray(_shrink(seg_bgr, cfg.work_max_side))
    ref = _shrink(noclahe_bgr, cfg.work_max_side)
    ctx = prepare(img, ref, cfg); valid = ctx["valid"]
    fns = {"otsu": segment_otsu, "grabcut": segment_grabcut, "active_contour": segment_active_contour}
    cleaned, out = {}, {}
    for name, fn in fns.items():
        try:
            cleaned[name] = clean_mask(fn(ctx, cfg), valid, cfg)
            status = "ok" if cleaned[name].any() else "empty"
            out[name] = (to_full_res(cleaned[name], (H, W)), status)
        except Exception as e:
            cleaned[name] = np.zeros(valid.shape, bool); out[name] = (np.zeros((H, W), bool), f"error: {e}")
    votes = sum(cleaned[n].astype(np.uint8) for n in fns)
    ens = clean_mask(votes >= cfg.ensemble_min_votes, valid, cfg)
    out["ensemble"] = (to_full_res(ens, (H, W)), "ok" if ens.any() else "empty")
    return out, float(ctx.get("vig_frac", 0.0))