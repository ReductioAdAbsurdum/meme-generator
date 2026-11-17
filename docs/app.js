/* Meme generator: warps the user's image onto the held paper, adds the caption, exports.
   Works straight from file:// because the template is embedded in assets/template-data.js. */
(() => {
  'use strict';

  const T = window.MEME_TEMPLATE;
  const $ = (id) => document.getElementById(id);
  if (!T || !T.image || !T.maps) {
    $('stageHint').textContent = 'Nedostaje assets/template-data.js (pokreni tools/prepare_template.py).';
    return;
  }

  // ---------------------------------------------------------------- constants
  const BB = T.bbox, BW = BB.w, BH = BB.h, NPX = BW * BH;
  const NU = T.grid.nu, NV = T.grid.nv, GP = T.grid.points;
  const ASPECT = T.aspect;                     // width / height of the photo area on the paper
  const FONT = 'Arial, "Helvetica Neue", Helvetica, "Liberation Sans", Arimo, sans-serif';
  const VIDEO_SCALE = 1920 / T.width;          // caption is drawn at broadcast resolution, then upscaled
  const CAPTION_TOP = 1440;                    // top of the caption band, template px
  const MAX_LEVEL = 3072;                      // longest side of the working copy of the user image
  const ZOOM_MIN = 1, ZOOM_MAX = 5;
  const PAPER = [247, 243, 240];
  const STORE_KEY = 'meme-generator:v1';
  const IS_MAC = /Mac|iPhone|iPad|iPod/.test(navigator.platform || navigator.userAgent);
  const PASTE_KEY = IS_MAC ? '⌘V' : 'Ctrl+V';
  const TOUCH = window.matchMedia('(pointer: coarse)').matches;

  const gp = (j, i) => [GP[(j * NU + i) * 2], GP[(j * NU + i) * 2 + 1]];
  const TL = gp(0, 0), TR = gp(0, NU - 1), BL = gp(NV - 1, 0), BR = gp(NV - 1, NU - 1);
  const dist = (a, b) => Math.hypot(b[0] - a[0], b[1] - a[1]);
  const REGION_W = (dist(TL, TR) + dist(BL, BR)) / 2;
  const REGION_H = (dist(TL, BL) + dist(TR, BR)) / 2;
  // average paper axes (px per unit of u / v); the inverse turns a drag in px into a uv delta
  const EU = [(TR[0] - TL[0] + BR[0] - BL[0]) / 2, (TR[1] - TL[1] + BR[1] - BL[1]) / 2];
  const EV = [(BL[0] - TL[0] + BR[0] - TR[0]) / 2, (BL[1] - TL[1] + BR[1] - TR[1]) / 2];
  const DET = EU[0] * EV[1] - EV[0] * EU[1];
  const INV = [EV[1] / DET, -EV[0] / DET, -EU[1] / DET, EU[0] / DET];

  const clamp = (v, lo, hi) => (v < lo ? lo : v > hi ? hi : v);
  const clamp01 = (v) => (v < 0 ? 0 : v > 1 ? 1 : v);

  // ---------------------------------------------------------------- DOM
  const canvas = $('meme');
  const ctx = canvas.getContext('2d');
  canvas.width = T.width;
  canvas.height = T.height;
  canvas.tabIndex = 0;
  const ui = {
    file: $('file'), pick: $('pick'), clear: $('clearImg'), zoom: $('zoom'), zoomOut: $('zoomOut'),
    reset: $('reset'), fits: [...document.querySelectorAll('input[name="fit"]')],
    line1: $('line1'), line2: $('line2'), cyr: $('cyr'), size: $('size'), sizeOut: $('sizeOut'),
    realism: $('realism'), realOut: $('realOut'),
    dlPng: $('dlPng'), dlJpg: $('dlJpg'), copy: $('copy'), share: $('share'),
    drop: $('drop'), toast: $('toast'), busy: $('busy'), hint: $('stageHint'),
    focus: $('focusBtn'), wrap: $('canvasWrap'),
  };

  // ---------------------------------------------------------------- state
  const state = {
    src: null, srcUrl: null, level: null,
    fit: 'cover', zoom: 1, cx: 0, cy: 0,
    realism: 0.6,
    line1: 'Ime i prezime', line2: 'Tvoj tekst ovde', cyr: false, textScale: 1,
  };
  restoreSettings();

  let baseImg = null, basePx = null, alphaA, shadeA, glareA, U, V, active, photo;
  let placeholderSrc = null, blankSrc = null;
  const buf = new Float32Array(NPX * 3);
  const tmp = new Float32Array(NPX * 3);

  // ---------------------------------------------------------------- template setup
  function loadImage(src) {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => resolve(img);
      img.onerror = () => reject(new Error('image load failed'));
      img.src = src;
    });
  }

  async function init() {
    const [base, maps] = await Promise.all([loadImage(T.image), loadImage(T.maps)]);
    baseImg = base;
    const c = document.createElement('canvas');
    c.width = BW;
    c.height = BH;
    const cc = c.getContext('2d', { willReadFrequently: true });
    cc.drawImage(base, -BB.x, -BB.y);
    basePx = cc.getImageData(0, 0, BW, BH).data;
    cc.clearRect(0, 0, BW, BH);
    cc.drawImage(maps, 0, 0);
    const md = cc.getImageData(0, 0, BW, BH).data;
    alphaA = new Float32Array(NPX);
    shadeA = new Float32Array(NPX);
    glareA = new Float32Array(NPX);
    for (let k = 0, o = 0; k < NPX; k++, o += 4) {
      alphaA[k] = md[o] / 255;
      shadeA[k] = md[o + 1] / 255;
      glareA[k] = md[o + 2] / 255;
    }
    buildUV();
    photo = new ImageData(BW, BH);
    placeholderSrc = textureSource(makePlaceholder());
    blankSrc = textureSource(makeBlank());
    syncUI();
    bindEvents();
    renderNow(3);
  }

  // Per-pixel (u, v) on the paper, by rasterizing the warp grid's triangles.
  function buildUV() {
    U = new Float32Array(NPX).fill(-1);
    V = new Float32Array(NPX).fill(-1);
    const P = (j, i) => [GP[(j * NU + i) * 2] - BB.x, GP[(j * NU + i) * 2 + 1] - BB.y];
    for (let j = 0; j < NV - 1; j++) {
      for (let i = 0; i < NU - 1; i++) {
        const a = P(j, i), b = P(j, i + 1), c = P(j + 1, i + 1), d = P(j + 1, i);
        const u0 = i / (NU - 1), u1 = (i + 1) / (NU - 1), v0 = j / (NV - 1), v1 = (j + 1) / (NV - 1);
        triangle(a, b, c, u0, v0, u1, v0, u1, v1);
        triangle(a, c, d, u0, v0, u1, v1, u0, v1);
      }
    }
    // grow a few px outwards so anti-aliased edge pixels and the blur have coordinates
    for (let pass = 0; pass < 4; pass++) {
      const U2 = U.slice(), V2 = V.slice();
      for (let y = 0; y < BH; y++) {
        for (let x = 0; x < BW; x++) {
          const k = y * BW + x;
          if (U[k] >= 0) continue;
          let su = 0, sv = 0, n = 0;
          for (let dy = -1; dy <= 1; dy++) {
            const yy = y + dy;
            if (yy < 0 || yy >= BH) continue;
            for (let dx = -1; dx <= 1; dx++) {
              const xx = x + dx;
              if (xx < 0 || xx >= BW) continue;
              const q = yy * BW + xx;
              if (U[q] >= 0) { su += U[q]; sv += V[q]; n++; }
            }
          }
          if (n) { U2[k] = su / n; V2[k] = sv / n; }
        }
      }
      U = U2;
      V = V2;
    }
    let count = 0;
    for (let k = 0; k < NPX; k++) if (U[k] >= 0) count++;
    active = new Int32Array(count);
    for (let k = 0, n = 0; k < NPX; k++) if (U[k] >= 0) active[n++] = k;
  }

  function triangle(a, b, c, ua, va, ub, vb, uc, vc) {
    const minX = Math.max(0, Math.floor(Math.min(a[0], b[0], c[0])) - 1);
    const maxX = Math.min(BW - 1, Math.ceil(Math.max(a[0], b[0], c[0])) + 1);
    const minY = Math.max(0, Math.floor(Math.min(a[1], b[1], c[1])) - 1);
    const maxY = Math.min(BH - 1, Math.ceil(Math.max(a[1], b[1], c[1])) + 1);
    const d = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1]);
    if (Math.abs(d) < 1e-9) return;
    const e = -1e-3;
    for (let y = minY; y <= maxY; y++) {
      for (let x = minX; x <= maxX; x++) {
        const l0 = ((b[1] - c[1]) * (x - c[0]) + (c[0] - b[0]) * (y - c[1])) / d;
        const l1 = ((c[1] - a[1]) * (x - c[0]) + (a[0] - c[0]) * (y - c[1])) / d;
        const l2 = 1 - l0 - l1;
        if (l0 >= e && l1 >= e && l2 >= e) {
          const k = y * BW + x;
          U[k] = l0 * ua + l1 * ub + l2 * uc;
          V[k] = l0 * va + l1 * vb + l2 * vc;
        }
      }
    }
  }

  // ---------------------------------------------------------------- image sources
  // A source = pixels + mapping (u, v) -> pixel coords: x = ox + u * su, y = oy + v * sv.
  function textureSource(cnv) {
    const d = cnv.getContext('2d').getImageData(0, 0, cnv.width, cnv.height);
    return { px: d.data, w: cnv.width, h: cnv.height, ox: -0.5, oy: -0.5, su: cnv.width, sv: cnv.height, grade: false };
  }

  function cropSize() {
    const { w, h } = state.src;
    let cw, ch;
    if (state.fit === 'stretch') { cw = w; ch = h; }
    else if ((w / h > ASPECT) === (state.fit === 'cover')) { ch = h; cw = h * ASPECT; }
    else { cw = w; ch = w / ASPECT; }
    return { cw: cw / state.zoom, ch: ch / state.zoom };
  }

  function clampCenter() {
    const { w, h } = state.src;
    const { cw, ch } = cropSize();
    const lim = (size, full, v) => clamp(v, Math.min(size / 2, full - size / 2), Math.max(size / 2, full - size / 2));
    state.cx = lim(cw, w, state.cx);
    state.cy = lim(ch, h, state.cy);
  }

  function userSource() {
    const { cw, ch } = cropSize();
    const L = ensureLevel(cw, ch);
    const x0 = state.cx - cw / 2, y0 = state.cy - ch / 2;
    return { px: L.px, w: L.w, h: L.h, ox: x0 * L.sx - 0.5, oy: y0 * L.sy - 0.5, su: cw * L.sx, sv: ch * L.sy, grade: true };
  }

  // Working copy of the user image at a resolution that matches the zoom (avoids aliasing).
  function ensureLevel(cw, ch) {
    const s = state.src;
    const scale = Math.min(1, Math.max((1.4 * REGION_W) / cw, (1.4 * REGION_H) / ch), MAX_LEVEL / Math.max(s.w, s.h));
    const L = state.level;
    if (L && Math.abs(L.scale - scale) / scale < 0.12) return L;
    state.level = buildLevel(s.img, s.w, s.h, scale);
    return state.level;
  }

  function buildLevel(img, w, h, scale) {
    const tw = Math.max(1, Math.round(w * scale)), th = Math.max(1, Math.round(h * scale));
    let cur = img, cw = w, ch = h;
    while (cw > tw * 2 && ch > th * 2) {             // halve step by step for a clean downscale
      let nw = Math.max(tw, Math.round(cw / 2)), nh = Math.max(th, Math.round(ch / 2));
      const area = nw * nh;
      if (area > 12e6) { const f = Math.sqrt(12e6 / area); nw = Math.round(nw * f); nh = Math.round(nh * f); }
      const c = document.createElement('canvas');
      c.width = nw;
      c.height = nh;
      const x = c.getContext('2d');
      x.imageSmoothingEnabled = true;
      x.imageSmoothingQuality = 'high';
      x.drawImage(cur, 0, 0, nw, nh);
      cur = c; cw = nw; ch = nh;
    }
    const c = document.createElement('canvas');
    c.width = tw;
    c.height = th;
    const x = c.getContext('2d', { willReadFrequently: true });
    x.fillStyle = `rgb(${PAPER})`;                  // transparent parts print as paper
    x.fillRect(0, 0, tw, th);
    x.imageSmoothingEnabled = true;
    x.imageSmoothingQuality = 'high';
    x.drawImage(cur, 0, 0, tw, th);
    return { px: x.getImageData(0, 0, tw, th).data, w: tw, h: th, sx: tw / w, sy: th / h, scale };
  }

  function makeTextureCanvas() {
    const c = document.createElement('canvas');
    c.width = 640;
    c.height = Math.round(640 / ASPECT);
    const x = c.getContext('2d');
    x.fillStyle = `rgb(${PAPER})`;
    x.fillRect(0, 0, c.width, c.height);
    return [c, x];
  }

  function makeBlank() { return makeTextureCanvas()[0]; }

  function makePlaceholder() {
    const [c, x] = makeTextureCanvas();
    const w = c.width, h = c.height, cx = w / 2;
    x.lineJoin = 'round';
    x.lineCap = 'round';
    x.setLineDash([26, 18]);
    x.lineWidth = 7;
    x.strokeStyle = '#a7abb1';
    roundRect(x, 44, 44, w - 88, h - 88, 34);
    x.stroke();
    x.setLineDash([]);
    const cy = h * 0.38, s = 140;
    x.lineWidth = 11;
    x.strokeStyle = '#8a8f96';
    roundRect(x, cx - s, cy - s * 0.75, s * 2, s * 1.5, 24);
    x.stroke();
    x.beginPath();
    x.arc(cx + s * 0.45, cy - s * 0.3, s * 0.17, 0, Math.PI * 2);
    x.stroke();
    x.beginPath();
    x.moveTo(cx - s + 18, cy + s * 0.55);
    x.lineTo(cx - s * 0.3, cy - s * 0.08);
    x.lineTo(cx + s * 0.18, cy + s * 0.38);
    x.lineTo(cx + s * 0.46, cy + s * 0.12);
    x.lineTo(cx + s - 18, cy + s * 0.55);
    x.stroke();
    const [l1, l2, l3] = TOUCH
      ? ['Dodirni da', 'izabereš sliku', '']
      : ['Klikni ili prevuci', 'sliku ovde', `ili je nalepi (${PASTE_KEY})`];
    x.textAlign = 'center';
    x.fillStyle = '#5c6168';
    x.font = `bold 56px ${FONT}`;
    x.fillText(l1, cx, h * 0.64);
    x.fillText(l2, cx, h * 0.64 + 68);
    x.fillStyle = '#8a8f96';
    x.font = `36px ${FONT}`;
    x.fillText(l3, cx, h * 0.64 + 136);
    return c;
  }

  function roundRect(x, l, t, w, h, r) {
    x.beginPath();
    x.moveTo(l + r, t);
    x.arcTo(l + w, t, l + w, t + h, r);
    x.arcTo(l + w, t + h, l, t + h, r);
    x.arcTo(l, t + h, l, t, r);
    x.arcTo(l, t, l + w, t, r);
    x.closePath();
  }

  // ---------------------------------------------------------------- rendering
  // 1) sample the source through the uv map, 2) soften like the video, 3) light + composite.
  function samplePhoto(src) {
    const { px, w, h, ox, oy, su, sv, grade } = src;
    const wm1 = w - 1, hm1 = h - 1, W4 = w * 4;
    const [pr, pg, pb] = PAPER;
    for (let n = 0, N = active.length; n < N; n++) {
      const k = active[n];
      const x = ox + U[k] * su, y = oy + V[k] * sv;
      // coverage < 1 only where the crop goes past the image ("Uklopi"): fill with paper
      let cov = 1;
      if (x < 0 || y < 0 || x > wm1 || y > hm1) {
        cov = clamp01(x + 1) * clamp01(w - x) * clamp01(y + 1) * clamp01(h - y);
      }
      let r = pr, g = pg, b = pb;
      if (cov > 0) {
        const xc = x < 0 ? 0 : x > wm1 ? wm1 : x, yc = y < 0 ? 0 : y > hm1 ? hm1 : y;
        const x0 = xc | 0, y0 = yc | 0;
        const x1 = x0 < wm1 ? x0 + 1 : x0, y1 = y0 < hm1 ? y0 + 1 : y0;
        const fx = xc - x0, fy = yc - y0;
        const i00 = y0 * W4 + x0 * 4, i10 = y0 * W4 + x1 * 4, i01 = y1 * W4 + x0 * 4, i11 = y1 * W4 + x1 * 4;
        const w00 = (1 - fx) * (1 - fy), w10 = fx * (1 - fy), w01 = (1 - fx) * fy, w11 = fx * fy;
        let sr = px[i00] * w00 + px[i10] * w10 + px[i01] * w01 + px[i11] * w11;
        let sg = px[i00 + 1] * w00 + px[i10 + 1] * w10 + px[i01 + 1] * w01 + px[i11 + 1] * w11;
        let sb = px[i00 + 2] * w00 + px[i10 + 2] * w10 + px[i01 + 2] * w01 + px[i11 + 2] * w11;
        if (grade) {                                   // printed + filmed: a bit flatter, a bit less saturated
          const l = 0.299 * sr + 0.587 * sg + 0.114 * sb;
          sr = (sr * 0.92 + l * 0.08) * 0.9 + 14;
          sg = (sg * 0.92 + l * 0.08) * 0.9 + 14;
          sb = (sb * 0.92 + l * 0.08) * 0.9 + 14;
        }
        r = sr * cov + pr * (1 - cov);
        g = sg * cov + pg * (1 - cov);
        b = sb * cov + pb * (1 - cov);
      }
      const o = k * 3;
      buf[o] = r;
      buf[o + 1] = g;
      buf[o + 2] = b;
    }
    blur5();
  }

  // separable [1 4 6 4 1] / 16 blur of buf (matches the softness of the upscaled video frame)
  function blur5() {
    for (let y = 0; y < BH; y++) {
      const row = y * BW;
      for (let x = 0; x < BW; x++) {
        const o = (row + x) * 3;
        const a = (row + (x > 1 ? x - 2 : 0)) * 3, b = (row + (x > 0 ? x - 1 : 0)) * 3;
        const c = (row + (x < BW - 1 ? x + 1 : BW - 1)) * 3, d = (row + (x < BW - 2 ? x + 2 : BW - 1)) * 3;
        tmp[o] = (buf[a] + 4 * buf[b] + 6 * buf[o] + 4 * buf[c] + buf[d]) * 0.0625;
        tmp[o + 1] = (buf[a + 1] + 4 * buf[b + 1] + 6 * buf[o + 1] + 4 * buf[c + 1] + buf[d + 1]) * 0.0625;
        tmp[o + 2] = (buf[a + 2] + 4 * buf[b + 2] + 6 * buf[o + 2] + 4 * buf[c + 2] + buf[d + 2]) * 0.0625;
      }
    }
    const S = BW * 3;
    for (let y = 0; y < BH; y++) {
      const a = (y > 1 ? y - 2 : 0) * S, b = (y > 0 ? y - 1 : 0) * S;
      const c = (y < BH - 1 ? y + 1 : BH - 1) * S, d = (y < BH - 2 ? y + 2 : BH - 1) * S, m = y * S;
      for (let i = 0; i < S; i++) {
        buf[m + i] = (tmp[a + i] + 4 * tmp[b + i] + 6 * tmp[m + i] + 4 * tmp[c + i] + tmp[d + i]) * 0.0625;
      }
    }
  }

  function composite() {
    const out = photo.data;
    out.set(basePx);
    const k = state.realism;
    for (let n = 0, N = active.length; n < N; n++) {
      const p = active[n];
      const a = alphaA[p];
      if (a <= 0) continue;
      const sh = 1 - k + k * shadeA[p];          // shading of the bent paper (multiply)
      const gl = 1 - k * glareA[p];              // glossy reflections (screen)
      const o3 = p * 3, o4 = p * 4;
      const r = 255 - (255 - buf[o3] * sh) * gl;
      const g = 255 - (255 - buf[o3 + 1] * sh) * gl;
      const b = 255 - (255 - buf[o3 + 2] * sh) * gl;
      out[o4] = basePx[o4] + (r - basePx[o4]) * a;
      out[o4 + 1] = basePx[o4 + 1] + (g - basePx[o4 + 1]) * a;
      out[o4 + 2] = basePx[o4 + 2] + (b - basePx[o4 + 2]) * a;
    }
  }

  const capCanvas = document.createElement('canvas');

  function captionLines() {
    let a = state.line1.trim(), b = state.line2.trim();
    if (state.cyr) { a = toCyrillic(a); b = toCyrillic(b); }
    return [a, b];
  }

  // Lower-third caption in the broadcast style (white Arial Bold with a thin dark edge).
  function drawCaption() {
    const [l1, l2] = captionLines();
    if (!l1 && !l2) return;
    const s = VIDEO_SCALE;
    const cw = Math.round(T.width * s), ch = Math.round((T.height - CAPTION_TOP) * s);
    if (capCanvas.width !== cw || capCanvas.height !== ch) { capCanvas.width = cw; capCanvas.height = ch; }
    const c = capCanvas.getContext('2d');
    c.setTransform(1, 0, 0, 1, 0, 0);
    c.clearRect(0, 0, cw, ch);
    c.setTransform(s, 0, 0, s, 0, -CAPTION_TOP * s);
    const [b1, b2] = T.caption.baselines;
    const size = T.caption.fontSize * state.textScale;
    const rows = l1 && l2 ? [[l1, b2 - (b2 - b1) * state.textScale], [l2, b2]] : [[l1 || l2, b2]];
    c.textBaseline = 'alphabetic';
    c.textAlign = 'left';
    c.lineJoin = 'round';
    for (const [text, y] of rows) {
      let fs = size;
      c.font = `bold ${fs}px ${FONT}`;
      const width = c.measureText(text).width;
      if (width > T.caption.maxWidth) {
        fs *= T.caption.maxWidth / width;
        c.font = `bold ${fs}px ${FONT}`;
      }
      c.shadowColor = 'rgba(0, 0, 0, 0.35)';
      c.shadowBlur = 3;
      c.shadowOffsetX = 0;
      c.shadowOffsetY = 1;
      c.lineWidth = fs * 0.05;
      c.strokeStyle = 'rgba(16, 12, 28, 0.85)';
      c.strokeText(text, T.caption.x, y);
      c.shadowColor = 'transparent';
      c.fillStyle = '#ffffff';
      c.fillText(text, T.caption.x, y);
    }
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = 'low';
    ctx.drawImage(capCanvas, 0, CAPTION_TOP, T.width, T.height - CAPTION_TOP);
  }

  // level: 1 = caption only, 2 = + lighting/composite, 3 = + resample the image
  let pending = 0, raf = 0;
  function scheduleRender(level) {
    pending = Math.max(pending, level);
    if (!raf) raf = requestAnimationFrame(() => { raf = 0; const p = pending; pending = 0; renderNow(p); });
  }

  function renderNow(level, opts = {}) {
    if (!baseImg) return;
    if (level >= 3) samplePhoto(opts.blank ? blankSrc : state.src ? userSource() : placeholderSrc);
    if (level >= 2) composite();
    ctx.drawImage(baseImg, 0, 0);
    ctx.putImageData(photo, BB.x, BB.y);
    drawCaption();
  }

  // ---------------------------------------------------------------- user image
  function isImageFile(f) {
    return f && (/^image\//.test(f.type) || /\.(png|jpe?g|gif|webp|avif|bmp|heic|heif)$/i.test(f.name || ''));
  }

  async function loadFile(file) {
    if (!isImageFile(file)) { toast('To nije slika — probaj PNG, JPG ili WebP.'); return; }
    ui.busy.hidden = false;
    const url = URL.createObjectURL(file);
    try {
      const img = await loadImage(url);
      if (img.decode) await img.decode().catch(() => {});
      const w = img.naturalWidth, h = img.naturalHeight;
      if (!w || !h) throw new Error('empty image');
      if (state.srcUrl) URL.revokeObjectURL(state.srcUrl);
      state.src = { img, w, h };
      state.srcUrl = url;
      state.level = null;
      state.zoom = 1;
      state.cx = w / 2;
      state.cy = h / 2;
      updateImageUI();
      scheduleRender(3);
    } catch (err) {
      URL.revokeObjectURL(url);
      toast('Ne mogu da otvorim ovu sliku.');
    } finally {
      ui.busy.hidden = true;
    }
  }

  function clearImage() {
    if (state.srcUrl) URL.revokeObjectURL(state.srcUrl);
    state.src = null;
    state.srcUrl = null;
    state.level = null;
    updateImageUI();
    scheduleRender(3);
  }

  function resetView() {
    if (!state.src) return;
    state.zoom = 1;
    state.cx = state.src.w / 2;
    state.cy = state.src.h / 2;
    updateImageUI();
    scheduleRender(3);
  }

  function zoomAt(uv, z) {
    if (!state.src) return;
    z = clamp(z, ZOOM_MIN, ZOOM_MAX);
    const before = cropSize();
    const px = state.cx + (uv.u - 0.5) * before.cw, py = state.cy + (uv.v - 0.5) * before.ch;
    state.zoom = z;
    const after = cropSize();
    state.cx = px - (uv.u - 0.5) * after.cw;
    state.cy = py - (uv.v - 0.5) * after.ch;
    clampCenter();
    ui.zoom.value = Math.round(z * 100);
    ui.zoomOut.value = `${ui.zoom.value}%`;
    scheduleRender(3);
  }

  function panBy(dx, dy) {
    if (!state.src) return;
    const du = INV[0] * dx + INV[1] * dy, dv = INV[2] * dx + INV[3] * dy;
    const { cw, ch } = cropSize();
    state.cx -= du * cw;
    state.cy -= dv * ch;
    clampCenter();
    scheduleRender(3);
  }

  // ---------------------------------------------------------------- export
  function stamp() {
    const d = new Date(), p = (n) => String(n).padStart(2, '0');
    return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())}-${p(d.getHours())}${p(d.getMinutes())}${p(d.getSeconds())}`;
  }

  async function exportBlob(type, quality) {
    if (raf) { cancelAnimationFrame(raf); raf = 0; }
    const blank = !state.src;                  // no image: export blank paper, not the hint
    renderNow(blank ? 3 : Math.max(pending, 1), { blank });
    pending = 0;
    try {
      return await new Promise((resolve, reject) =>
        canvas.toBlob((b) => (b ? resolve(b) : reject(new Error('toBlob failed'))), type, quality));
    } finally {
      if (blank) renderNow(3);
    }
  }

  function downloadBlob(blob, name) {
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = name;
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(a.href), 5000);
  }

  async function doExport(type, ext, quality) {
    try {
      const blob = await exportBlob(type, quality);
      downloadBlob(blob, `meme-${stamp()}.${ext}`);
      toast(`Preuzeto: meme.${ext}`);
    } catch (err) {
      toast('Izvoz nije uspeo.');
    }
  }

  async function copyImage() {
    if (!navigator.clipboard || !window.ClipboardItem) {
      toast('Ovaj pregledač ne podržava kopiranje slike — koristi Preuzmi.');
      return;
    }
    try {
      await navigator.clipboard.write([new ClipboardItem({ 'image/png': exportBlob('image/png') })]);
      toast('Slika je kopirana — nalepi je gde želiš.');
    } catch (err) {
      toast('Kopiranje nije uspelo — koristi Preuzmi.');
    }
  }

  async function shareImage() {
    try {
      const blob = await exportBlob('image/png');
      const file = new File([blob], `meme-${stamp()}.png`, { type: 'image/png' });
      await navigator.share({ files: [file] });
    } catch (err) {
      if (err && err.name !== 'AbortError') toast('Deljenje nije uspelo.');
    }
  }

  // ---------------------------------------------------------------- Serbian Latin -> Cyrillic
  const CYR = {
    'DŽ': 'Џ', 'Dž': 'Џ', 'dž': 'џ', 'LJ': 'Љ', 'Lj': 'Љ', 'lj': 'љ', 'NJ': 'Њ', 'Nj': 'Њ', 'nj': 'њ',
    'Ǆ': 'Џ', 'ǅ': 'Џ', 'ǆ': 'џ', 'Ǉ': 'Љ', 'ǈ': 'Љ', 'ǉ': 'љ', 'Ǌ': 'Њ', 'ǋ': 'Њ', 'ǌ': 'њ',
    'A': 'А', 'B': 'Б', 'V': 'В', 'G': 'Г', 'D': 'Д', 'Đ': 'Ђ', 'E': 'Е', 'Ž': 'Ж', 'Z': 'З', 'I': 'И',
    'J': 'Ј', 'K': 'К', 'L': 'Л', 'M': 'М', 'N': 'Н', 'O': 'О', 'P': 'П', 'R': 'Р', 'S': 'С', 'T': 'Т',
    'Ć': 'Ћ', 'U': 'У', 'F': 'Ф', 'H': 'Х', 'C': 'Ц', 'Č': 'Ч', 'Š': 'Ш',
    'a': 'а', 'b': 'б', 'v': 'в', 'g': 'г', 'd': 'д', 'đ': 'ђ', 'e': 'е', 'ž': 'ж', 'z': 'з', 'i': 'и',
    'j': 'ј', 'k': 'к', 'l': 'л', 'm': 'м', 'n': 'н', 'o': 'о', 'p': 'п', 'r': 'р', 's': 'с', 't': 'т',
    'ć': 'ћ', 'u': 'у', 'f': 'ф', 'h': 'х', 'c': 'ц', 'č': 'ч', 'š': 'ш',
  };
  function toCyrillic(text) {
    const s = text.normalize('NFC');
    let out = '';
    for (let i = 0; i < s.length; i++) {
      const two = s.slice(i, i + 2);
      if (two.length === 2 && CYR[two]) { out += CYR[two]; i++; continue; }
      out += CYR[s[i]] || s[i];
    }
    return out;
  }

  // ---------------------------------------------------------------- settings + UI
  function restoreSettings() {
    try {
      const s = JSON.parse(localStorage.getItem(STORE_KEY) || 'null');
      if (!s || typeof s !== 'object') return;
      if (typeof s.line1 === 'string') state.line1 = s.line1;
      if (typeof s.line2 === 'string') state.line2 = s.line2;
      if (typeof s.cyr === 'boolean') state.cyr = s.cyr;
      if (Number.isFinite(s.textScale)) state.textScale = clamp(s.textScale, 0.7, 1.3);
      if (Number.isFinite(s.realism)) state.realism = clamp(s.realism, 0, 1);
    } catch (err) { /* storage unavailable: keep defaults */ }
  }

  function saveSettings() {
    try {
      localStorage.setItem(STORE_KEY, JSON.stringify({
        line1: state.line1, line2: state.line2, cyr: state.cyr, textScale: state.textScale, realism: state.realism,
      }));
    } catch (err) { /* ignore */ }
  }

  function syncUI() {
    ui.line1.value = state.line1;
    ui.line2.value = state.line2;
    ui.cyr.checked = state.cyr;
    ui.size.value = Math.round(state.textScale * 100);
    ui.sizeOut.value = `${ui.size.value}%`;
    ui.realism.value = Math.round(state.realism * 100);
    ui.realOut.value = `${ui.realism.value}%`;
    updateImageUI();
  }

  function updateImageUI() {
    const has = !!state.src;
    ui.pick.textContent = has ? 'Promeni sliku' : 'Izaberi sliku';
    ui.clear.hidden = !has;
    ui.zoom.disabled = !has;
    ui.reset.disabled = !has;
    for (const r of ui.fits) { r.disabled = !has; r.checked = r.value === state.fit; }
    ui.zoom.value = Math.round(state.zoom * 100);
    ui.zoomOut.value = `${ui.zoom.value}%`;
    canvas.classList.toggle('has-image', has);     // only block page scrolling while there is an image to drag
    if (TOUCH) {
      ui.hint.textContent = has ? 'Prevuci sliku prstom da je pomeriš · dva prsta za zum' : 'Dodirni papir da izabereš sliku';
    } else {
      ui.hint.textContent = has
        ? 'Prevuci sliku po papiru da je pomeriš · točkić miša za zum'
        : `Klikni na papir ili prevuci sliku bilo gde na stranicu · možeš i da je nalepiš (${PASTE_KEY})`;
    }
  }

  let toastTimer = 0;
  function toast(msg) {
    ui.toast.textContent = msg;
    ui.toast.classList.add('show');
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => ui.toast.classList.remove('show'), 2800);
  }

  function canvasPoint(e) {
    const r = canvas.getBoundingClientRect();
    return { x: ((e.clientX - r.left) * canvas.width) / r.width, y: ((e.clientY - r.top) * canvas.height) / r.height };
  }

  function uvAt(p) {
    const x = Math.round(p.x - BB.x), y = Math.round(p.y - BB.y);
    if (x < 0 || y < 0 || x >= BW || y >= BH) return null;
    const k = y * BW + x;
    return alphaA[k] > 0.3 ? { u: U[k], v: V[k] } : null;
  }

  function bindEvents() {
    ui.pick.addEventListener('click', () => ui.file.click());
    ui.file.addEventListener('change', () => {
      const f = ui.file.files && ui.file.files[0];
      if (f) loadFile(f);
      ui.file.value = '';
    });
    ui.clear.addEventListener('click', clearImage);
    for (const r of ui.fits) {
      r.addEventListener('change', () => { if (r.checked) { state.fit = r.value; resetView(); } });
    }
    ui.zoom.addEventListener('input', () => zoomAt({ u: 0.5, v: 0.5 }, ui.zoom.value / 100));
    ui.reset.addEventListener('click', resetView);

    ui.line1.addEventListener('input', () => { state.line1 = ui.line1.value; saveSettings(); scheduleRender(1); });
    ui.line2.addEventListener('input', () => { state.line2 = ui.line2.value; saveSettings(); scheduleRender(1); });
    ui.cyr.addEventListener('change', () => { state.cyr = ui.cyr.checked; saveSettings(); scheduleRender(1); });
    ui.size.addEventListener('input', () => {
      state.textScale = ui.size.value / 100;
      ui.sizeOut.value = `${ui.size.value}%`;
      saveSettings();
      scheduleRender(1);
    });
    ui.realism.addEventListener('input', () => {
      state.realism = ui.realism.value / 100;
      ui.realOut.value = `${ui.realism.value}%`;
      saveSettings();
      scheduleRender(2);
    });

    const setFocus = (on) => {
      ui.wrap.classList.toggle('focus', on);
      ui.focus.setAttribute('aria-pressed', String(on));
      ui.focus.textContent = on ? 'Ceo kadar' : 'Uvećaj papir';
    };
    setFocus(window.matchMedia('(max-width: 640px)').matches);   // phones start zoomed on the paper
    ui.focus.addEventListener('click', () => setFocus(!ui.wrap.classList.contains('focus')));

    ui.dlPng.addEventListener('click', () => doExport('image/png', 'png'));
    ui.dlJpg.addEventListener('click', () => doExport('image/jpeg', 'jpg', 0.92));
    ui.copy.addEventListener('click', copyImage);
    try {
      const probe = new File([new Blob(['x'], { type: 'image/png' })], 'x.png', { type: 'image/png' });
      if (navigator.canShare && navigator.canShare({ files: [probe] })) {
        ui.share.hidden = false;
        ui.share.addEventListener('click', shareImage);
      }
    } catch (err) { /* no Web Share */ }

    // drag & drop anywhere on the page, paste from the clipboard
    const hasFiles = (e) => e.dataTransfer && [...e.dataTransfer.types].includes('Files');
    let depth = 0;
    window.addEventListener('dragenter', (e) => { if (hasFiles(e)) { e.preventDefault(); depth++; ui.drop.hidden = false; } });
    window.addEventListener('dragover', (e) => { if (hasFiles(e)) { e.preventDefault(); e.dataTransfer.dropEffect = 'copy'; } });
    window.addEventListener('dragleave', (e) => { if (hasFiles(e) && --depth <= 0) { depth = 0; ui.drop.hidden = true; } });
    window.addEventListener('drop', (e) => {
      if (!hasFiles(e)) return;
      e.preventDefault();
      depth = 0;
      ui.drop.hidden = true;
      const f = [...e.dataTransfer.files].find(isImageFile);
      if (f) loadFile(f); else toast('Prevuci sliku (PNG, JPG, WebP…).');
    });
    window.addEventListener('paste', (e) => {
      const items = (e.clipboardData && e.clipboardData.items) || [];
      for (const it of items) {
        if (it.kind === 'file' && it.type.startsWith('image/')) {
          e.preventDefault();
          loadFile(it.getAsFile());
          return;
        }
      }
    });

    // pan (drag), pinch and wheel zoom on the paper
    const ptrs = new Map();
    let gesture = null, suppressClick = false;
    const pair = () => [...ptrs.values()];
    const pinchDist = () => { const [a, b] = pair(); return Math.hypot(a.p.x - b.p.x, a.p.y - b.p.y); };

    canvas.addEventListener('pointerdown', (e) => {
      if (!state.src) return;
      const p = canvasPoint(e);
      if (!ptrs.size && !uvAt(p)) return;
      e.preventDefault();
      canvas.focus({ preventScroll: true });
      canvas.setPointerCapture(e.pointerId);
      ptrs.set(e.pointerId, { p, cx: e.clientX, cy: e.clientY });
      if (ptrs.size === 1) {
        gesture = { type: 'pan', moved: 0 };
      } else if (ptrs.size === 2) {
        const [a, b] = pair();
        const mid = { x: (a.p.x + b.p.x) / 2, y: (a.p.y + b.p.y) / 2 };
        gesture = { type: 'pinch', d0: pinchDist(), z0: state.zoom, uv: uvAt(mid) || { u: 0.5, v: 0.5 }, moved: 99 };
      }
      canvas.classList.add('dragging');
    });
    canvas.addEventListener('pointermove', (e) => {
      const p = canvasPoint(e);
      const prev = ptrs.get(e.pointerId);
      if (!prev) {
        canvas.style.cursor = uvAt(p) ? (state.src ? 'grab' : 'pointer') : '';
        return;
      }
      ptrs.set(e.pointerId, { p, cx: e.clientX, cy: e.clientY });
      if (gesture && gesture.type === 'pan' && ptrs.size === 1) {
        gesture.moved += Math.abs(e.clientX - prev.cx) + Math.abs(e.clientY - prev.cy);
        panBy(p.x - prev.p.x, p.y - prev.p.y);
      } else if (gesture && gesture.type === 'pinch' && ptrs.size === 2 && gesture.d0 > 0) {
        zoomAt(gesture.uv, (gesture.z0 * pinchDist()) / gesture.d0);
      }
    });
    const endPointer = (e) => {
      if (!ptrs.has(e.pointerId)) return;
      ptrs.delete(e.pointerId);
      if (gesture && gesture.moved > 4) suppressClick = true;
      if (ptrs.size === 1) gesture = { type: 'pan', moved: 99 };
      if (!ptrs.size) { gesture = null; canvas.classList.remove('dragging'); }
    };
    canvas.addEventListener('pointerup', endPointer);
    canvas.addEventListener('pointercancel', endPointer);
    canvas.addEventListener('click', (e) => {
      if (suppressClick) { suppressClick = false; return; }
      if (!state.src && uvAt(canvasPoint(e))) ui.file.click();
    });
    canvas.addEventListener('wheel', (e) => {
      if (!state.src) return;
      const uv = uvAt(canvasPoint(e));
      if (!uv) return;
      e.preventDefault();
      const speed = e.ctrlKey ? 0.01 : e.deltaMode === 1 ? 0.05 : 0.0015;   // ctrl = trackpad pinch
      zoomAt(uv, state.zoom * Math.exp(-e.deltaY * speed));
    }, { passive: false });
    canvas.addEventListener('keydown', (e) => {
      if (!state.src) {
        if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); ui.file.click(); }
        return;
      }
      const step = e.shiftKey ? 60 : 20;
      const moves = { ArrowLeft: [-step, 0], ArrowRight: [step, 0], ArrowUp: [0, -step], ArrowDown: [0, step] };
      if (moves[e.key]) { e.preventDefault(); panBy(...moves[e.key]); }
      else if (e.key === '+' || e.key === '=') { e.preventDefault(); zoomAt({ u: 0.5, v: 0.5 }, state.zoom * 1.1); }
      else if (e.key === '-') { e.preventDefault(); zoomAt({ u: 0.5, v: 0.5 }, state.zoom / 1.1); }
    });
  }

  init().catch((err) => {
    console.error(err);
    ui.hint.textContent = 'Šablon nije mogao da se učita.';
  });
})();
