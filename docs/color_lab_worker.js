// Web worker for color_lab.html: downloads only the SCI extension of JWST i2d FITS files from the
// public STScI S3 mirror (CORS enabled, Range requests), block-averages it in JS, and hands the small
// array to Python (Pyodide, color_lab.py) for hole filling, reprojection and color composition.
importScripts('https://cdn.jsdelivr.net/pyodide/v0.27.7/full/pyodide.js');

const S3 = 'https://stpubdata.s3.amazonaws.com/jwst/public/';
const BLOCK = 2880;
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

const shortName = (f) => { const m = [...f.matchAll(/[-_](f\d{3,4}[a-z]*\d?)/gi)]; return m.length ? m[m.length - 1][1].toLowerCase() : f; };

// Download SCI and block-average (nan-mean) by an integer factor so the image is <= maxMP megapixels.
async function readSci(fname, maxMP) {
  const url = s3url(fname);
  let off = 0, hdr = null;
  for (let ext = 0; ext < 6; ext++) {
    hdr = await readHeader(url, off);
    if (ext > 0 && hdr.cards.EXTNAME === 'SCI') break;
    off = hdr.dataStart + Math.ceil(hdr.dataSize / BLOCK) * BLOCK;
    hdr = null;
  }
  if (!hdr) throw new Error('no SCI extension in ' + fname);
  if (hdr.cards.BITPIX !== '-32') throw new Error('expected BITPIX=-32, got ' + hdr.cards.BITPIX);
  const nx = parseInt(hdr.cards.NAXIS1), ny = parseInt(hdr.cards.NAXIS2);
  const f = maxMP ? Math.max(1, Math.ceil(Math.sqrt(nx * ny / (maxMP * 1e6)))) : 1;  // same as color_lab.block_factor
  const ox = Math.ceil(nx / f), oy = Math.ceil(ny / f);
  step(`${shortName(fname)}: downloading SCI ${nx}x${ny} (${(hdr.dataSize / 1048576).toFixed(0)} MB), block-average /${f} → ${ox}x${oy}`);
  const sum = new Float64Array(ox * oy), cnt = new Uint32Array(ox * oy);
  // strips of whole output rows, ~32 MB each
  const rowsPerStrip = Math.max(f, Math.floor(32 * 1048576 / (nx * 4) / f) * f);
  const tDl = performance.now();
  for (let y0 = 0; y0 < ny; y0 += rowsPerStrip) {
    const y1 = Math.min(ny, y0 + rowsPerStrip);
    const bytes = await getRange(url, hdr.dataStart + y0 * nx * 4, hdr.dataStart + y1 * nx * 4 - 1);
    const dv = new DataView(bytes.buffer);
    for (let y = y0; y < y1; y++) {
      const orow = Math.floor(y / f) * ox;
      const base = (y - y0) * nx * 4;
      for (let x = 0; x < nx; x++) {
        const v = dv.getFloat32(base + x * 4, false);  // FITS is big-endian
        if (v === v) {  // not NaN
          const o = orow + Math.floor(x / f);
          sum[o] += v; cnt[o]++;
        }
      }
    }
    post('progress', { fname, frac: y1 / ny });
  }
  const out = new Float32Array(ox * oy);
  for (let i = 0; i < out.length; i++) out[i] = cnt[i] ? sum[i] / cnt[i] : NaN;
  step(`${shortName(fname)}: downloaded in ${((performance.now() - tDl) / 1000).toFixed(1)}s`);
  return { data: out, ny: oy, nx: ox, header: hdr.text, factor: f };
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
  log(`Python ready in ${((performance.now() - t0) / 1000).toFixed(1)} s`);
  post('ready', memInfo());
}

const ready = init().catch(e => post('error', { msg: 'init failed: ' + e.message }));

self.onmessage = async (ev) => {
  const { cmd, id } = ev.data;
  await ready;
  let cl = null;
  try {
    if (cmd === 'compose') {
      const { red, green, blue, maxMP, fill, opts } = ev.data;
      let names = [...new Set([...red, ...green, ...blue])];
      const ref = names.includes(opts.ref) ? opts.ref : names[0];  // reference grid (same rule as compose())
      names = [ref, ...names.filter(n => n !== ref)];
      cl = py.pyimport('color_lab');
      t0 = performance.now(); cl.reset_clock();
      const key = JSON.stringify({ maxMP, fill });
      if (cacheKey !== key) {
        if (cacheKey !== null) log('size / fill holes changed: processing from scratch');
        cl.clear(); cacheKey = key;
      }
      cl.forget(py.toPy(names));
      log(`---- compose: R=${red.map(shortName)} G=${green.map(shortName)} B=${blue.map(shortName)}, reference grid ${shortName(names[0])}`);
      for (const n of names) {
        if (cl.has_layer(n)) { step(`${shortName(n)}: using cached layer`); continue; }
        const L = await readSci(n, maxMP);
        cl.add_layer(n, new Uint8Array(L.data.buffer), L.ny, L.nx, L.header, L.factor, fill);
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
  } catch (e) {
    post('error', { id, msg: e.message });
  } finally {
    if (cl) cl.destroy();
  }
};
