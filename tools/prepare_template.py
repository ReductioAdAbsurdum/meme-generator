#!/usr/bin/env python3
"""Build the meme template from the original parliament screenshot.

What it does
  1. Removes the two-line caption at the bottom (font-matched mask + inpainting).
  2. Finds the inner edge of the white border of the photo the man is holding,
     and builds a warp grid (Coons patch) that follows the bent paper.
  3. Builds the replace mask (photo area minus the fingertip that overlaps it),
     plus a shading map and a glare map so inserted images look printed.
  4. Fills the photo area with blank paper and writes the assets used by the site (docs/).

Outputs (in ../docs/assets):
  template-data.js    geometry (warp grid, caption position) plus two embedded images:
                      the cleaned template (WebP, photo area blank) and the maps
                      (PNG, cropped to the photo bbox: R = replace mask, G = shading, B = glare).
                      Embedded as data URIs so the page also works when opened from disk.
  og.jpg              1280x725 preview image for shared links

Requirements: numpy, pillow, scipy, opencv-contrib-python-headless; macOS Arial Bold font.
Usage: python3 tools/prepare_template.py [screenshot.png]   (default: tools/original-screenshot.png)
"""
import base64
import io
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage as ndi
from scipy.ndimage import gaussian_filter1d, median_filter

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "docs" / "assets"
FONT_PATH = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
FONT_SIZE = 76
CROP_TOP, CROP_BOTTOM = 2, 4          # black rows in the screenshot
CAPTION = [("Миленко Јованов", 68, 1590), ("Александар Вучић - Србија побеђује", 64, 1678)]
DESK_OBJECT = (640, 1585, 880, 1662)   # white paper + pen on the desk that touches the letters


# ----------------------------------------------------------------------------- caption removal
def caption_mask(img):
    """Render the caption with the matching font, fit every word, return a dilated mask."""
    H, W = img.shape[:2]
    mn = img.min(axis=2).astype(int)
    target = mn > 190
    excl = np.zeros_like(target)
    x0, y0, x1, y1 = DESK_OBJECT
    excl[y0:y1, x0:x1] = True
    font = ImageFont.truetype(FONT_PATH, FONT_SIZE)

    def word_mask(text):
        l, t, r, b = font.getbbox(text, anchor="ls")
        pad = 4
        im = Image.new("L", (r - l + 2 * pad, b - t + 2 * pad), 0)
        ImageDraw.Draw(im).text((pad - l, pad - t), text, font=font, fill=255, anchor="ls")
        return np.asarray(im), (pad - l, pad - t)   # origin inside the small image

    glyphs = np.zeros((H, W), np.uint8)
    for text, ox, base in CAPTION:
        words = text.split(" ")
        for i, word in enumerate(words):
            prefix = " ".join(words[:i]) + (" " if i else "")
            wx = ox + font.getlength(prefix)
            m, (px, py) = word_mask(word)
            mb = m > 127
            best = (-1.0, 0, 0)
            for dy in range(-4, 5):
                for dx in range(-8, 9):
                    gx, gy = int(round(wx + dx - px)), base + dy - py
                    h, w = mb.shape
                    t = target[gy:gy + h, gx:gx + w]
                    v = ~excl[gy:gy + h, gx:gx + w]
                    s = (mb & t & v).sum() / max(((mb | t) & v).sum(), 1)
                    if s > best[0]:
                        best = (s, gx, gy)
            _, gx, gy = best
            h, w = m.shape
            glyphs[gy:gy + h, gx:gx + w] = np.maximum(glyphs[gy:gy + h, gx:gx + w], m)

    g = (glyphs > 40).astype(np.uint8)
    mask = cv2.dilate(g, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))
    near = cv2.dilate(g, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))) > 0
    extra = ((mn > 150) & near).astype(np.uint8)
    extra[excl] = 0
    return np.maximum(mask, cv2.dilate(extra, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))))


def remove_caption(img, mask):
    y0, y1, x0, x1 = 1480, img.shape[0] - CROP_BOTTOM, 0, 1520
    crop = img[y0:y1, x0:x1].copy()
    orig = crop.astype(np.float32)
    m = mask[y0:y1, x0:x1]
    h, w = m.shape

    # luma: frequency-selective reconstruction
    bgr = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
    fsr = np.zeros_like(bgr)
    cv2.xphoto.inpaint(bgr, (1 - m) * 255, fsr, cv2.xphoto.INPAINT_FSR_FAST)
    ycc_f = cv2.cvtColor(cv2.cvtColor(fsr, cv2.COLOR_BGR2RGB), cv2.COLOR_RGB2YCrCb).astype(np.float32)
    ycc_o = cv2.cvtColor(crop, cv2.COLOR_RGB2YCrCb).astype(np.float32)
    Yf = ycc_f[..., 0]

    # chroma: weighted average of valid neighbours with similar luma (no invented hues)
    m_c = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    valid = (m_c == 0).astype(np.float32)
    C = ycc_o[..., 1:3]
    R, step, sig_s, sig_r = 30, 3, 12.0, 12.0
    Yp = np.pad(Yf, R, mode="edge")
    Cp = np.pad(C, ((R, R), (R, R), (0, 0)), mode="edge")
    Vp = np.pad(valid, R)
    num = np.zeros((h, w, 2), np.float32)
    den = np.zeros((h, w), np.float32)
    for dy in range(-R, R + 1, step):
        for dx in range(-R, R + 1, step):
            ws = np.exp(-(dx * dx + dy * dy) / (2 * sig_s ** 2))
            sl = (slice(R + dy, R + dy + h), slice(R + dx, R + dx + w))
            wgt = ws * (np.exp(-((Yp[sl] - Yf) ** 2) / (2 * sig_r ** 2)) + 1e-3) * Vp[sl]
            num += wgt[..., None] * Cp[sl]
            den += wgt
    Cfill = num / np.maximum(den, 1e-8)[..., None]
    g_num = cv2.GaussianBlur(C * valid[..., None], (0, 0), 25)
    g_den = cv2.GaussianBlur(valid, (0, 0), 25)
    Cfill = np.where((den < 1e-6)[..., None], g_num / np.maximum(g_den, 1e-6)[..., None], Cfill)
    Cout = np.where((m_c > 0)[..., None], Cfill, C)
    Yout = np.where(m > 0, Yf, ycc_o[..., 0])
    res = cv2.cvtColor(np.dstack([Yout, Cout]).clip(0, 255).astype(np.uint8), cv2.COLOR_YCrCb2RGB).astype(np.float32)

    # desk front (horizontal structures): dark rim vs wood, anisotropic normalized convolution
    yy, xx = np.mgrid[0:h, 0:w]
    X, Y = xx + x0, yy + y0
    left_pts = np.array([(890, 1586), (893, 1611), (905, 1618), (911, 1626), (917, 1630),
                         (930, 1637), (938, 1646), (948, 1653)], float)
    xl = np.interp(Y, left_pts[:, 1], left_pts[:, 0], left=890, right=1e9)
    yb = 1656.5 + 0.0108 * (X - 960)
    rim = ((X >= xl) & (Y < yb) & (Y > 1588)).astype(np.float32)
    validm = (m == 0).astype(np.float32) * (orig.min(axis=2) < 120)
    fill_desk = np.zeros_like(orig)
    for cls in (rim, 1 - rim):
        V = validm * cls
        out_c = np.zeros_like(orig)
        got = np.zeros((h, w), bool)
        for sx, sy in ((22, 2.0), (45, 3.0), (90, 5.0), (200, 10.0)):
            n = cv2.GaussianBlur(orig * V[..., None], (0, 0), sigmaX=sx, sigmaY=sy)
            d = cv2.GaussianBlur(V, (0, 0), sigmaX=sx, sigmaY=sy)
            ok = (d > 0.02) & ~got
            out_c[ok] = (n / np.maximum(d, 1e-6)[..., None])[ok]
            got |= ok
        fill_desk += out_c * cv2.GaussianBlur(cls, (0, 0), 1.1)[..., None]
    deskw = np.clip((X - 868) / 18.0, 0, 1).astype(np.float32) * (Y > 1586)
    fill = res * (1 - deskw[..., None]) + fill_desk * deskw[..., None]
    fill += np.random.default_rng(11).normal(0, 1.5, fill.shape).astype(np.float32)
    mf = cv2.GaussianBlur(m.astype(np.float32), (0, 0), 0.8)
    out = img.copy()
    out[y0:y1, x0:x1] = (orig * (1 - mf[..., None]) + fill * mf[..., None]).clip(0, 255).astype(np.uint8)
    return out


# ----------------------------------------------------------------------------- photo geometry
def border_scan(img):
    a = img.astype(int)
    mn, mx = a.min(axis=2), a.max(axis=2)
    white = (mn >= 190) & ((mx - mn) <= 75)

    def first_run(vals, idx):
        i, n = 0, len(vals)
        while i < n:
            if vals[i]:
                j = i
                while j < n and vals[j]:
                    j += 1
                if j - i >= 3:
                    return idx[i], idx[j - 1]
                i = j
            else:
                i += 1
        return None

    def scan(horizontal, fixed, a0, a1, guess=None, win=None):
        out = {}
        for f in fixed:
            idx = np.arange(a0, a1, 1 if a1 > a0 else -1)
            if guess is not None:
                idx = idx[np.abs(idx - guess(f)) <= win]
            vals = white[f, idx] if horizontal else white[idx, f]
            r = first_run(vals, idx)
            if r:
                out[f] = r
        return out

    return {
        "left": scan(True, range(500, 1372), 0, 320),
        "right": scan(True, range(430, 1326), 899, 520),
        "top": scan(False, range(50, 668), 380, 620, guess=lambda x: 499 - 0.109 * (x - 50), win=22),
        "bottom": scan(False, range(140, 790), 1430, 1200),
    }


def build_grid(scan, NU=41, NV=57):
    def smooth_side(d, keys, typical_w, sign, max_w):
        ks = np.array(sorted(k for k in keys if k in d))
        outer = np.array([d[k][0] for k in ks], float)
        inner = np.array([d[k][1] for k in ks], float)
        w = np.abs(inner - outer) + 1
        bad = (w > max_w) | (w < 3)
        inner[bad] = outer[bad] + sign * (typical_w - 1)
        return ks, gaussian_filter1d(median_filter(inner, 21, mode="nearest"), 4)

    def robust_poly(d, keys, deg, max_w):
        ks = np.array([k for k in keys if k in d], float)
        outer = np.array([d[int(k)][0] for k in ks], float)
        inner = np.array([d[int(k)][1] for k in ks], float)
        w = np.abs(inner - outer) + 1
        ok = (w <= max_w) & (w >= 4)
        ks, inner = ks[ok], inner[ok]
        for _ in range(6):
            p = np.polyfit(ks, inner, deg)
            r = inner - np.polyval(p, ks)
            s = 1.4826 * np.median(np.abs(r - np.median(r))) + 1e-6
            keep = np.abs(r) < max(2.5 * s, 1.0)
            ks, inner = ks[keep], inner[keep]
        return p

    lk, li = smooth_side(scan["left"], range(500, 1372), 11, +1, 16)
    rk, ri = smooth_side(scan["right"], range(430, 1326), 14, -1, 18)
    p_top = robust_poly(scan["top"], [x for x in range(56, 662) if not (288 <= x <= 378)], 2, 16)
    p_bot = robust_poly(scan["bottom"], range(146, 786), 2, 13)
    li, ri = li + 0.5, ri - 0.5

    def extrap(ks, vals, q):
        q = np.asarray(q, float)
        lo, hi = np.polyfit(ks[:40], vals[:40], 1), np.polyfit(ks[-40:], vals[-40:], 1)
        return np.where(q < ks[0], np.polyval(lo, q), np.where(q > ks[-1], np.polyval(hi, q), np.interp(q, ks, vals)))

    f_left = lambda y: extrap(lk, li, y)
    f_right = lambda y: extrap(rk, ri, y)
    f_top = lambda x: np.polyval(p_top, np.asarray(x, float)) + 0.5
    f_bot = lambda x: np.polyval(p_bot, np.asarray(x, float)) - 0.5

    def corner(fh, fv, x, y):
        for _ in range(50):
            y = float(fh(x))
            x = float(fv(y))
        return np.array([x, y])

    TL, TR = corner(f_top, f_left, 55, 498), corner(f_top, f_right, 658, 431)
    BL, BR = corner(f_bot, f_left, 141, 1370), corner(f_bot, f_right, 773, 1322)

    def curve(f, a, b, horizontal, n=400):
        t = np.linspace(a, b, n)
        return np.stack([t, f(t)] if horizontal else [f(t), t], 1)

    def resample(c, n):
        d = np.r_[0, np.cumsum(np.hypot(*np.diff(c, axis=0).T))]
        t = np.linspace(0, d[-1], n)
        return np.stack([np.interp(t, d, c[:, 0]), np.interp(t, d, c[:, 1])], 1), d[-1]

    top_c, bot_c = curve(f_top, TL[0], TR[0], True), curve(f_bot, BL[0], BR[0], True)
    left_c, right_c = curve(f_left, TL[1], BL[1], False), curve(f_right, TR[1], BR[1], False)
    for c, a_, b_ in ((top_c, TL, TR), (bot_c, BL, BR), (left_c, TL, BL), (right_c, TR, BR)):
        c[0], c[-1] = a_, b_
    T, lt = resample(top_c, NU)
    B, lb = resample(bot_c, NU)
    L, ll = resample(left_c, NV)
    R, lr = resample(right_c, NV)
    u = np.linspace(0, 1, NU)[None, :, None]
    v = np.linspace(0, 1, NV)[:, None, None]
    grid = ((1 - v) * T[None] + v * B[None] + (1 - u) * L[:, None] + u * R[:, None]
            - ((1 - u) * (1 - v) * TL + u * (1 - v) * TR + (1 - u) * v * BL + u * v * BR))
    return grid, (lt + lb) / (ll + lr)


def finger_mask(img):
    a = img.astype(int)
    H, W = a.shape[:2]
    x0, y0, x1, y1 = 280, 400, 392, 512
    box = a[y0:y1, x0:x1]
    rb, mnb = box[..., 0] - box[..., 2], box.min(axis=2)
    core = (rb > 38) & (mnb > 105)
    lab, _ = ndi.label(core)
    core = ndi.binary_fill_holes(lab == lab[480 - y0, 340 - x0])
    near = cv2.dilate(core.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))) > 0
    finger = ndi.binary_fill_holes(core | ((mnb < 75) & near))
    finger = cv2.morphologyEx(finger.astype(np.uint8), cv2.MORPH_CLOSE,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    finger = cv2.GaussianBlur(finger.astype(np.float32), (0, 0), 1.6) > 0.5   # smooth the contour
    full = np.zeros((H, W), np.float32)
    full[y0:y1, x0:x1] = finger
    return cv2.GaussianBlur(full, (0, 0), 0.8)


def region_alpha(grid, shape):
    H, W = shape
    poly = np.concatenate([grid[0, :], grid[:, -1], grid[-1, ::-1], grid[::-1, 0]])
    SS = 4
    im = Image.new("L", (W * SS, H * SS), 0)
    ImageDraw.Draw(im).polygon([(x * SS + SS / 2 - 0.5, y * SS + SS / 2 - 0.5) for x, y in poly], fill=255)
    return np.asarray(im.resize((W, H), Image.BOX)).astype(np.float32) / 255


def uv_map(grid, shape):
    """Rasterize the grid triangles -> per-pixel (u, v); NaN outside."""
    H, W = shape
    NV, NU = grid.shape[:2]
    U = np.full((H, W), np.nan, np.float32)
    V = np.full((H, W), np.nan, np.float32)
    us, vs = np.linspace(0, 1, NU), np.linspace(0, 1, NV)

    def tri(p0, p1, p2, t0, t1, t2):
        xa, xb = int(np.floor(min(p0[0], p1[0], p2[0]))) - 1, int(np.ceil(max(p0[0], p1[0], p2[0]))) + 1
        ya, yb = int(np.floor(min(p0[1], p1[1], p2[1]))) - 1, int(np.ceil(max(p0[1], p1[1], p2[1]))) + 1
        gx, gy = np.meshgrid(np.arange(xa, xb + 1), np.arange(ya, yb + 1))
        d = (p1[1] - p2[1]) * (p0[0] - p2[0]) + (p2[0] - p1[0]) * (p0[1] - p2[1])
        l0 = ((p1[1] - p2[1]) * (gx - p2[0]) + (p2[0] - p1[0]) * (gy - p2[1])) / d
        l1 = ((p2[1] - p0[1]) * (gx - p2[0]) + (p0[0] - p2[0]) * (gy - p2[1])) / d
        l2 = 1 - l0 - l1
        ins = (l0 >= -1e-3) & (l1 >= -1e-3) & (l2 >= -1e-3)
        yy, xx = gy[ins], gx[ins]
        U[yy, xx] = l0[ins] * t0[0] + l1[ins] * t1[0] + l2[ins] * t2[0]
        V[yy, xx] = l0[ins] * t0[1] + l1[ins] * t1[1] + l2[ins] * t2[1]

    for j in range(NV - 1):
        for i in range(NU - 1):
            a_, b_, c_, d_ = grid[j, i], grid[j, i + 1], grid[j + 1, i + 1], grid[j + 1, i]
            ta, tb, tc, td = (us[i], vs[j]), (us[i + 1], vs[j]), (us[i + 1], vs[j + 1]), (us[i], vs[j + 1])
            tri(a_, b_, c_, ta, tb, tc)
            tri(a_, c_, d_, ta, tc, td)
    # extend a few pixels outwards (nearest) so feathered edge pixels have coordinates
    nan = np.isnan(U)
    _, lab = cv2.distanceTransformWithLabels(nan.astype(np.uint8), cv2.DIST_L2, 5,
                                             labelType=cv2.DIST_LABEL_PIXEL)
    src = np.argwhere(~nan)
    lut = np.zeros((lab.max() + 1, 2), int)
    lut[lab[~nan]] = src
    near = lut[lab]
    dist = cv2.distanceTransform(nan.astype(np.uint8), cv2.DIST_L2, 5)
    fillm = nan & (dist <= 4)
    U[fillm] = U[near[fillm][:, 0], near[fillm][:, 1]]
    V[fillm] = V[near[fillm][:, 0], near[fillm][:, 1]]
    return U, V


def shade_and_glare(img, grid, U, V, finger):
    """Shading from the white border brightness (Coons blend), plus procedural glossy glare."""
    Yl = (0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]).astype(np.float32)

    def edge_samples(pts, sign, offset, smooth):
        out = []
        for k in range(len(pts)):
            t = pts[min(k + 1, len(pts) - 1)] - pts[max(k - 1, 0)]
            t = t / (np.linalg.norm(t) + 1e-9)
            p = pts[k] + np.array([-t[1], t[0]]) * sign * offset
            x, y = int(round(p[0])), int(round(p[1]))
            out.append(np.median(Yl[y - 1:y + 2, x - 1:x + 2]) if finger[y, x] < 0.05 else np.nan)
        out = np.array(out)
        ok = ~np.isnan(out)
        out = np.interp(np.arange(len(out)), np.flatnonzero(ok), out[ok])
        return gaussian_filter1d(np.clip(out, 200, 252) / 252.0, smooth, mode="nearest")

    # top/bottom smoothed more: their small dips would show up as vertical stripes
    T = edge_samples(grid[0], -1, 5, 3.0)
    B = edge_samples(grid[-1], +1, 4.5, 3.0)
    L = edge_samples(grid[:, 0], +1, 5, 1.5)
    R = edge_samples(grid[:, -1], -1, 6.5, 1.5)
    NV, NU = grid.shape[:2]
    u = np.linspace(0, 1, NU)[None, :]
    v = np.linspace(0, 1, NV)[:, None]
    S = ((1 - v) * T[None] + v * B[None] + (1 - u) * L[:, None] + u * R[:, None]
         - ((1 - u) * (1 - v) * T[0] + u * (1 - v) * T[-1] + (1 - u) * v * B[0] + u * v * B[-1]))
    S = np.clip(1 - (1 - S) * 1.6, 0.72, 1.0).astype(np.float32)
    ok = ~np.isnan(U)
    Uc, Vc = np.nan_to_num(U, nan=0.5), np.nan_to_num(V, nan=0.5)
    shade = cv2.remap(S, (Uc * (NU - 1)).astype(np.float32), (Vc * (NV - 1)).astype(np.float32),
                      cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    def stroke(p0, p1, sigma, i0, i1):
        d = np.array(p1) - np.array(p0)
        ln = np.linalg.norm(d)
        t = ((Uc - p0[0]) * d[0] + (Vc - p0[1]) * d[1]) / ln ** 2
        tc = np.clip(t, 0, 1)
        dist = np.hypot(Uc - (p0[0] + tc * d[0]), (Vc - (p0[1] + tc * d[1])) * 1.4)
        fall = np.exp(-0.5 * (dist / sigma) ** 2)
        ends = np.clip(t * 12, 0, 1) * np.clip((1 - t) * 2.5, 0, 1)
        return fall * ends * (i0 + (i1 - i0) * tc)

    vc = 0.505 + 0.05 * Uc - 0.026 * Uc ** 2 + 0.004 * np.sin(Uc * 9.0)
    fold = np.exp(-0.5 * ((Vc - vc) / 0.017) ** 2) * (0.55 - 0.22 * Uc) * (1 + 0.18 * np.sin(Uc * 13 + 1.3))
    g = [fold,
         stroke((-0.02, 0.182), (0.62, 0.045), 0.010, 0.50, 0.12),
         stroke((-0.02, 0.110), (0.40, 0.028), 0.015, 0.26, 0.05),
         0.12 * np.exp(-0.5 * (((Uc - 0.10) / 0.28) ** 2 + ((Vc - 0.10) / 0.16) ** 2))]
    glare = 1 - np.prod([1 - np.clip(x, 0, 1) for x in g], axis=0)
    glare = cv2.GaussianBlur(glare.astype(np.float32), (0, 0), 1.5)
    shade[~ok], glare[~ok] = 1.0, 0.0
    return shade, glare


# ----------------------------------------------------------------------------- main
def data_uri(img, fmt, **kw):
    buf = io.BytesIO()
    img.save(buf, fmt, **kw)
    mime = {"WEBP": "image/webp", "PNG": "image/png"}[fmt]
    return f"data:{mime};base64," + base64.b64encode(buf.getvalue()).decode()


def og_image(tpl, data, size=(1280, 725)):
    """Template with a sample caption, used as the link preview image."""
    im = tpl.resize(size, Image.LANCZOS)
    s = size[0] / tpl.width
    draw = ImageDraw.Draw(im)
    font = ImageFont.truetype(FONT_PATH, round(FONT_SIZE * s))
    x = round(data["caption"]["x"] * s)
    lines = ("Zumbirator", "Ubaci svoju sliku. Da se zumbira!")
    for text, base in zip(lines, data["caption"]["baselines"]):
        draw.text((x, round(base * s)), text, font=font, fill="white", anchor="ls",
                  stroke_width=max(1, round(FONT_SIZE * s * 0.05)), stroke_fill=(16, 12, 28))
    return im


def main(src):
    img = np.asarray(Image.open(src).convert("RGB")).copy()
    H, W = img.shape[:2]

    print("1/5 caption removal")
    clean = remove_caption(img, caption_mask(img))

    print("2/5 photo geometry")
    grid, aspect = build_grid(border_scan(img))
    finger = finger_mask(img)
    alpha = region_alpha(grid, (H, W)) * (1 - finger)

    print("3/5 uv, shading, glare")
    U, V = uv_map(grid, (H, W))
    shade, glare = shade_and_glare(img, grid, U, V, finger)

    print("4/5 blank paper")
    paper = np.array([247, 243, 240], np.float32)
    blank = paper[None, None, :] * shade[..., None]
    blank = 255 - (255 - blank) * (1 - 0.35 * glare[..., None])
    a3 = alpha[..., None]
    clean = (clean.astype(np.float32) * (1 - a3) + blank * a3).clip(0, 255).astype(np.uint8)

    print("5/5 writing assets")
    clean = clean[CROP_TOP:H - CROP_BOTTOM]
    TH = clean.shape[0]
    alpha, shade, glare = (x[CROP_TOP:H - CROP_BOTTOM] for x in (alpha, shade, glare))
    grid = grid - np.array([0, CROP_TOP])
    ys, xs = np.nonzero(alpha > 0)
    pad = 6
    bx0, by0 = max(xs.min() - pad, 0), max(ys.min() - pad, 0)
    bx1, by1 = min(xs.max() + pad + 1, W), min(ys.max() + pad + 1, TH)
    maps = np.dstack([alpha, shade, glare])[by0:by1, bx0:bx1]
    maps = (maps * 255).round().clip(0, 255).astype(np.uint8)

    ASSETS.mkdir(parents=True, exist_ok=True)
    tpl = Image.fromarray(clean)
    data = {
        "width": W, "height": TH,
        "bbox": {"x": int(bx0), "y": int(by0), "w": int(bx1 - bx0), "h": int(by1 - by0)},
        "grid": {"nu": int(grid.shape[1]), "nv": int(grid.shape[0]),
                 "points": [round(float(c), 2) for c in grid.reshape(-1)]},
        "aspect": round(float(aspect), 4),
        "caption": {"x": 66, "baselines": [CAPTION[0][2] - CROP_TOP, CAPTION[1][2] - CROP_TOP],
                    "fontSize": FONT_SIZE, "maxWidth": W - 66 - 70},
    }
    og_image(tpl, data).save(ASSETS / "og.jpg", quality=86, optimize=True, progressive=True)
    data["image"] = data_uri(tpl, "WEBP", quality=90, method=6)
    data["maps"] = data_uri(Image.fromarray(maps, "RGB"), "PNG", optimize=True)
    js = ("/* Generated by tools/prepare_template.py - do not edit by hand. */\n"
          "window.MEME_TEMPLATE = " + json.dumps(data, separators=(",", ":")) + ";\n")
    (ASSETS / "template-data.js").write_text(js)
    print("bbox", data["bbox"], "aspect", data["aspect"], "template", tpl.size)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else str(Path(__file__).resolve().parent / "original-screenshot.png"))
