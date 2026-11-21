// Web worker for color_lab.html: downloads only the SCI extension of JWST i2d FITS files from the
// public STScI S3 mirror (CORS enabled, Range requests), block-averages it in JS, and hands the small
// array to Python (Pyodide, color_lab.py) for hole filling, reprojection and color composition.
importScripts('https://cdn.jsdelivr.net/pyodide/v0.27.7/full/pyodide.js');

const S3 = 'https://stpubdata.s3.amazonaws.com/jwst/public/';
const BLOCK = 2880;
// CDS main site down since Nov 2026 (cybersecurity incident), Harvard mirror unreachable; IDIA (South Africa) works
const SIMBAD_HOSTS = ['simbad.idia.ac.za', 'simbad.cfa.harvard.edu', 'simbad.harvard.edu', 'simbad.cds.unistra.fr'];
const SIMBAD_MAX = 5000;
let py = null;
let cacheKey = null;  // settings that require re-reading layers from scratch
let t0 = performance.now();  // clock shared with color_lab.step(), reset per compose

function post(type, payload) { self.postMessage({ type, ...payload }); }
function log(msg) { post('log', { msg }); }
function step(msg) { log(`[${((performance.now() - t0) / 1000).toFixed(1).padStart(6)}s] ${msg}`); }

function memInfo() {
  let heap = null;
  try { heap = py._module.HEAP8.length; } catch (e) { }
  return { heapMB: heap ? Math.round(heap / 1048576) : null };
}

function s3url(fname) {
  if (!fname.startsWith('jw')) {  // HST pipeline product, e.g. jfq8jq010_drc.fits
    const obs = fname.split('_')[0];
    return `https://stpubdata.s3.amazonaws.com/hst/public/${obs.slice(0, 4)}/${obs}/${fname}`;
  }
  const m = fname.match(/^(jw\d{5})-([oc]\d+)_/);
  if (!m) throw new Error('cannot derive S3 path for ' + fname);
  return `${S3}${m[1]}/L3/t/${m[2]}/${fname}`;
}

async function getRange(url, start, end) {  // inclusive end
  const r = await fetch(url, { headers: { Range: `bytes=${start}-${end}` } });
  if (r.status === 404 || r.status === 403) throw new Error('not on S3 (yet?): ' + url.split('/').pop());
  if (r.status !== 206) throw new Error(`HTTP ${r.status} for ${url}`);
  return new Uint8Array(await r.arrayBuffer());
}

// Read one FITS header starting at byte `offset`. Returns {cards, text, dataStart, dataSize}.
async function readHeader(url, offset) {
  let text = '';
  let pos = offset;
  const chunk = BLOCK * 16;
  while (true) {
    const bytes = await getRange(url, pos, pos + chunk - 1);
    const t = new TextDecoder('ascii').decode(bytes);
    for (let i = 0; i + 80 <= t.length; i += 80) {
      const card = t.slice(i, i + 80);
      text += card;
      if (card.slice(0, 8).trim() === 'END') {
        const hdrLen = Math.ceil(text.length / BLOCK) * BLOCK;
        const cards = {};
        for (let j = 0; j < text.length; j += 80) {
          const c = text.slice(j, j + 80);
          if (c.slice(8, 10) === '= ') {
            let v = c.slice(10).split('/')[0].trim();
            if (v.startsWith("'")) v = c.slice(10).match(/'(.*?)'/)?.[1]?.trim() ?? v;
            cards[c.slice(0, 8).trim()] = v;
          }
        }
        const naxis = parseInt(cards.NAXIS || '0');
        let n = naxis > 0 ? 1 : 0;
        for (let a = 1; a <= naxis; a++) n *= parseInt(cards['NAXIS' + a]);
        const dataSize = n * Math.abs(parseInt(cards.BITPIX)) / 8;
        return { cards, text, dataStart: offset + hdrLen, dataSize };
      }
    }
    pos += bytes.length;
    if (bytes.length < chunk) throw new Error('END card not found');
  }
}

const MAMA = ['SBC', 'FUV-MAMA', 'NUV-MAMA'];  // photon-counting HST detectors, same as color_lab.MAMA
const LOW_WEIGHT = 0.5;  // same as color_lab.LOW_WEIGHT
let FILTERS = {};  // fname -> [filter label, wavelength nm], from the page (HST names carry no filter)
const shortName = (f) => { if (FILTERS[f]) return FILTERS[f][0]; const m = [...f.matchAll(/[-_](f\d{3,4}[a-z]*\d?)/gi)]; return m.length ? m[m.length - 1][1].toLowerCase() : f; };

// Download SCI and block-average (nan-mean) by an integer factor so the image is <= maxMP megapixels.
// cosmic: remove cosmic-ray hits at full resolution strip by strip (Python color_lab.clean_strip_buf), before the
// block-average, where a hit is still much sharper than a star; skipped when the header says the image needs none.
// Returns the binned data and, when cleaned, the binned hit mask (fraction of removed pixels per block).
async function readSci(fname, maxMP, cosmic) {
  const url = s3url(fname);
  let off = 0, hdr = null, moving = false, detector = '';
  for (let ext = 0; ext < 6; ext++) {
    hdr = await readHeader(url, off);
    if (ext === 0) {
      detector = (hdr.cards.DETECTOR || '').trim();  // moving target (comet, asteroid, planet): HST MTFLAG = T, JWST TARGTYPE = MOVING
      const mt = (hdr.cards.MTFLAG || '').trim().toUpperCase(), tt = (hdr.cards.TARGTYPE || '').trim().toUpperCase();
      moving = mt === 'T' || mt === 'TRUE' || tt === 'MOVING';
    }
    if (ext > 0 && hdr.cards.EXTNAME === 'SCI') break;
    off = hdr.dataStart + Math.ceil(hdr.dataSize / BLOCK) * BLOCK;
    hdr = null;
  }
  if (!hdr) throw new Error('no SCI extension in ' + fname);
  if (hdr.cards.BITPIX !== '-32') throw new Error('expected BITPIX=-32, got ' + hdr.cards.BITPIX);
  const nx = parseInt(hdr.cards.NAXIS1), ny = parseInt(hdr.cards.NAXIS2);
  const f = maxMP ? Math.max(1, Math.ceil(Math.sqrt(nx * ny / (maxMP * 1e6)))) : 1;  // same as color_lab.block_factor
  const ox = Math.ceil(nx / f), oy = Math.ceil(ny / f);
  // HST pixels without data (dead rows, bad columns: weight 0 in the drizzled product) become NaN before the
  // block-average, so they leave no dark lines: exactly 0, or for photon-counting detectors (0 counts is real)
  // weight 0 in the WHT extension that follows SCI (same as color_lab.no_data)
  const hst = !fname.split('/').pop().toLowerCase().startsWith('jw');
  let wht = null;
  if (hst && MAMA.includes(detector.toUpperCase())) {
    const w = await readHeader(url, hdr.dataStart + Math.ceil(hdr.dataSize / BLOCK) * BLOCK);
    if (w.cards.EXTNAME === 'WHT' && w.cards.NAXIS1 === hdr.cards.NAXIS1 && w.cards.NAXIS2 === hdr.cards.NAXIS2 && w.cards.BITPIX === '-32') {
      wht = w.dataStart;
      step(`${shortName(fname)}: ${detector} counts photons, zeros are real: reading the weight map (WHT, ${(w.dataSize / 1048576).toFixed(0)} MB) for missing data`);
    }
  }
  let clean = false;
  if (cosmic) {
    const cl = py.pyimport('color_lab');
    const why = cl.skip_cosmic_text(fname, hdr.text, detector);
    const margin = cl.CR_MARGIN;
    cl.destroy();
    clean = !why ? margin : false;  // rows borrowed from the neighbouring strips
    if (why && why !== 'JWST') step(`${shortName(fname)}: no speck removal (${why})`);
  }
  step(`${shortName(fname)}: downloading SCI ${nx}x${ny} (${(hdr.dataSize / 1048576).toFixed(0)} MB), block-average /${f} → ${ox}x${oy}`);
  const sum = new Float64Array(ox * oy), cnt = new Uint32Array(ox * oy);
  const hit = clean ? new Uint32Array(ox * oy) : null;
  let hits = 0, tClean = 0;
  // strips of whole output rows, ~32 MB each
  const rowsPerStrip = Math.max(f, Math.floor(32 * 1048576 / (nx * 4) / f) * f);
  const tDl = performance.now();
  for (let y0 = 0; y0 < ny; y0 += rowsPerStrip) {
    const y1 = Math.min(ny, y0 + rowsPerStrip);
    const a = clean ? Math.max(0, y0 - clean) : y0, b = clean ? Math.min(ny, y1 + clean) : y1;
    const bytes = await getRange(url, hdr.dataStart + a * nx * 4, hdr.dataStart + b * nx * 4 - 1);
    const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    const rows = new Float32Array((b - a) * nx);
    for (let i = 0; i < rows.length; i++) rows[i] = dv.getFloat32(i * 4, false);  // FITS is big-endian
    if (wht !== null) {
      const wb = await getRange(url, wht + a * nx * 4, wht + b * nx * 4 - 1);
      const wv = new DataView(wb.buffer, wb.byteOffset, wb.byteLength);
      const w = new Float32Array(rows.length);
      for (let i = 0; i < rows.length; i++) w[i] = wv.getFloat32(i * 4, false);
      // below LOW_WEIGHT x the median weight: mostly 0 counts (dead rows, frame edge), as color_lab.no_data
      const pos = []; for (let i = 0; i < w.length; i += 7) if (w[i] > 0) pos.push(w[i]);
      pos.sort((p, q) => p - q);
      const lim = pos.length ? LOW_WEIGHT * pos[pos.length >> 1] : 0;
      for (let i = 0; i < rows.length; i++) if (!(w[i] > 0 && w[i] >= lim)) rows[i] = NaN;
    } else if (hst) {
      for (let i = 0; i < rows.length; i++) if (rows[i] === 0) rows[i] = NaN;
    }
    let data = rows, mask = null, base = (y0 - a) * nx;  // base: first pixel of row y0 in data
    if (clean) {
      const t = performance.now();
      const cl = py.pyimport('color_lab');
      const res = cl.clean_strip_buf(new Uint8Array(rows.buffer), b - a, nx, y0 - a, b - y1);
      const [cb, mb, n] = res.toJs(); res.destroy(); cl.destroy();
      data = new Float32Array(cb.slice().buffer); mask = mb; hits += n; base = 0;
      tClean += performance.now() - t;
    }
    for (let y = y0; y < y1; y++) {
      const orow = Math.floor(y / f) * ox;
      const r = base + (y - y0) * nx, mr = (y - y0) * nx;
      for (let x = 0; x < nx; x++) {
        const v = data[r + x];
        const o = orow + Math.floor(x / f);
        if (v === v) { sum[o] += v; cnt[o]++; }  // not NaN
        if (mask && mask[mr + x]) hit[o]++;
      }
    }
    post('progress', { fname, frac: y1 / ny });
  }
  const out = new Float32Array(ox * oy);
  for (let i = 0; i < out.length; i++) out[i] = cnt[i] ? sum[i] / cnt[i] : NaN;
  let crmask = null;
  if (clean) {  // fraction of removed pixels per block (the last row / column of blocks may be partial)
    crmask = new Float32Array(ox * oy);
    for (let j = 0; j < oy; j++) {
      const h = Math.min(f, ny - j * f);
      for (let i = 0; i < ox; i++) crmask[j * ox + i] = hit[j * ox + i] / (h * Math.min(f, nx - i * f));
    }
    step(`${shortName(fname)}: removed specks (cosmic rays, hot pixels) at full resolution, ${hits} pixels (${(hits / (nx * ny) * 100).toFixed(2)}%), ${(tClean / 1000).toFixed(1)}s`);
  }
  step(`${shortName(fname)}: downloaded in ${((performance.now() - tDl) / 1000).toFixed(1)}s`);
  return { data: out, ny: oy, nx: ox, header: hdr.text, factor: f, moving, crmask };
}

async function init() {
  const t0 = performance.now();
  log('loading Python (Pyodide) + numpy, scipy, astropy…');
  py = await loadPyodide();
  py.setStdout({ batched: (s) => log(s) });
  py.setStderr({ batched: (s) => log('stderr: ' + s) });
  await py.loadPackage(['numpy', 'scipy', 'astropy']);
  const code = await (await fetch('color_lab.py?v=' + Date.now())).text();
  py.FS.writeFile('/home/pyodide/color_lab.py', code);
  py.runPython('import color_lab');
  log(`Color Lab ${py.runPython('color_lab.VERSION')}: Python ready in ${((performance.now() - t0) / 1000).toFixed(1)} s`);
  post('ready', memInfo());
}

const ready = init().catch(e => post('error', { msg: 'init failed: ' + e.message }));

self.onmessage = async (ev) => {
  const { cmd, id } = ev.data;
  await ready;
  let cl = null;
  try {
    if (cmd === 'compose') {
      const { red, green, blue, maxMP, fill, cosmic, opts } = ev.data;
      let names = [...new Set([...red, ...green, ...blue])];
      const ref = names.includes(opts.ref) ? opts.ref : names[0];  // reference grid (same rule as compose())
      names = [ref, ...names.filter(n => n !== ref)];
      cl = py.pyimport('color_lab');
      FILTERS = { ...FILTERS, ...(opts.filters || {}) };
      const pf = py.toPy(opts.filters || {}); cl.set_filters(pf); pf.destroy();
      t0 = performance.now(); cl.reset_clock();
      const key = JSON.stringify({ maxMP, fill, cosmic: !!cosmic });
      if (cacheKey !== key) {
        if (cacheKey !== null) log('size / fill holes / cosmic rays changed: processing from scratch');
        cl.clear(); cacheKey = key;
      }
      cl.forget(py.toPy(names));
      log(`---- compose: R=${red.map(shortName)} G=${green.map(shortName)} B=${blue.map(shortName)}, reference grid ${shortName(names[0])}`);
      for (const n of names) {
        if (cl.has_layer(n)) { step(`${shortName(n)}: using cached layer`); continue; }
        const L = await readSci(n, maxMP, !!cosmic);
        cl.add_layer(n, new Uint8Array(L.data.buffer), L.ny, L.nx, L.header, L.factor, fill, L.crmask ? new Uint8Array(L.crmask.buffer) : null);
        cl.set_moving(n, L.moving);
        if (L.moving) step(`${shortName(n)}: moving target`);
        post('mem', memInfo());
      }
      const pyOpts = py.toPy(opts);
      const res = cl.compose(py.toPy(red), py.toPy(green), py.toPy(blue), pyOpts);
      pyOpts.destroy();
      const [bytesProxy, h, w] = res.toJs();
      const info = cl.ref_info(ref);
      const [refFactor, crval1, crval2, pixscale] = info.toJs(); info.destroy();
      const li = cl.layer_info();
      const layers = li.toJs({ dict_converter: Object.fromEntries }); li.destroy();
      const rgba = new Uint8ClampedArray(bytesProxy);  // copy out of WASM
      res.destroy();
      self.postMessage({ type: 'image', id, w, h, rgba, ref, refFactor, pixscale, layers, crval: [crval1, crval2], ...memInfo() }, [rgba.buffer]);
    }
    else if (cmd === 'autoalign') {
      // nudges (arcsec) that line up the loaded layers' point sources with an anchor layer (color_lab.auto_align)
      const { names, ref, align } = ev.data;
      cl = py.pyimport('color_lab');
      cl.reset_clock();
      const res = cl.auto_align(py.toPy(names), ref, align);
      const shifts = res.toJs({ dict_converter: Object.fromEntries }); res.destroy();
      post('autoalign', { id, shifts });
    }
    else if (cmd === 'simbad') {
      // objects from SIMBAD (first mirror that answers, SIMBAD_HOSTS) on the grid of the reference layer
      const { ref } = ev.data;
      cl = py.pyimport('color_lab');
      if (!cl.has_layer(ref)) throw new Error('reference layer not loaded');
      const fp = cl.footprint(ref); const [ra, dec, rad] = fp.toJs(); fp.destroy();
      const adql = `SELECT TOP ${SIMBAD_MAX} main_id, ra, dec, otype FROM basic ` +
        `WHERE CONTAINS(POINT('ICRS', ra, dec), CIRCLE('ICRS', ${ra}, ${dec}, ${rad})) = 1 AND ra IS NOT NULL`;
      let json = null, lastErr = null;
      for (const host of SIMBAD_HOSTS) {
        try {
          const url = `https://${host}/simbad/sim-tap/sync?` + new URLSearchParams({ request: 'doQuery', lang: 'adql', format: 'json', query: adql });
          const r = await fetch(url, { signal: AbortSignal.timeout(25000) });
          if (!r.ok) throw new Error(`HTTP ${r.status}`);
          json = await r.json();
          log(`SIMBAD (${host}): ${json.data.length} objects within ${(rad * 60).toFixed(1)}′ of ${ra.toFixed(4)}, ${dec.toFixed(4)}`);
          break;
        } catch (e) { lastErr = e; log(`SIMBAD ${host} failed: ${e.message}`); }
      }
      if (!json) throw new Error('SIMBAD unavailable: ' + (lastErr && lastErr.message));
      const rows = json.data;
      const pr = cl.sky_to_pix(ref, py.toPy(rows.map(r => r[1])), py.toPy(rows.map(r => r[2])));
      const [xs, ys] = pr.toJs(); pr.destroy();
      const objects = [];
      rows.forEach((r, i) => { if (xs[i] === xs[i]) objects.push({ id: String(r[0]).trim(), otype: String(r[3] || '').trim(), x: xs[i], y: ys[i] }); });
      post('simbad', { id, ref, objects, truncated: rows.length >= SIMBAD_MAX });
    }
    else throw new Error('unknown command ' + cmd);
  } catch (e) {
    if (cmd === 'simbad') post('simbad-error', { id, msg: e.message });
    else post('error', { id, msg: e.message });
  } finally {
    if (cl) cl.destroy();
  }
};
