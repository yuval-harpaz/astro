// Shared sky helpers for jwst_latest_release.html, hst_latest_release.html and color_lab.html:
// coordinate parsing / formatting, angular distance, cone filters, SIMBAD name lookup and autocomplete.
// SIMBAD is queried on the Harvard mirror (CORS enabled; the CDS main site was down in Nov 2026).
const Sky = (() => {
  const SIMBAD = 'https://simbad.harvard.edu/simbad/sim-script';
  const d2r = Math.PI / 180;

  // Decimal degrees as in the CRVAL fields ("13.647655, -37.653867")
  const fmt = (ra, dec, digits = 5) => `${(+ra).toFixed(digits)}, ${(+dec).toFixed(digits)}`;

  // Angular distance in degrees
  function sep(ra1, dec1, ra2, dec2) {
    const s = Math.sin(dec1 * d2r) * Math.sin(dec2 * d2r) + Math.cos(dec1 * d2r) * Math.cos(dec2 * d2r) * Math.cos((ra1 - ra2) * d2r);
    return Math.acos(Math.min(1, Math.max(-1, s))) / d2r;
  }

  // Parse a position. Returns {ra, dec} in degrees or null.
  //   default: two decimal numbers, any separator      "13.647655, -37.653867"  "13.65 -37.65"  "13.65;-37.65"
  //   sexagesimal (RA in hours) when it has : h m s d ° ' " or six numbers:
  //     "00:54:53.5 -37:41:04"  "00h54m53.5s -37d41m04s"  "00 54 53.5 -37 41 04"
  function parse(text) {
    const t = (text || '').trim();
    if (!t) return null;
    const nums = t.match(/[+-]?\d+(?:\.\d*)?|[+-]?\.\d+/g) || [];
    const leftovers = t.replace(/[+-]?\d+(?:\.\d*)?|[+-]?\.\d+/g, '').replace(/[\s,;:|/hmsd°'"′″]/gi, '');
    if (leftovers) return null;  // words: not a position
    const sexa = /[:hmsd°'"′″]/i.test(t) || nums.length === 6;
    let ra, dec;
    if (!sexa && nums.length === 2) {
      [ra, dec] = nums.map(Number);
    } else if (sexa && nums.length === 6) {
      const n = nums.map(Number);
      const neg = /^-/.test(nums[3]) || /-\s*0*\.?0*$/.test(nums[3]);
      ra = (Math.abs(n[0]) + n[1] / 60 + n[2] / 3600) * 15;
      dec = (Math.abs(n[3]) + n[4] / 60 + n[5] / 3600) * (neg || n[3] < 0 ? -1 : 1);
    } else return null;
    if (!(ra >= 0 && ra < 360 && dec >= -90 && dec <= 90)) return null;
    return { ra, dec };
  }

  // SIMBAD script query -> [{name, ra, dec, otype}]
  async function script(lines) {
    const body = ['output console=off script=off error=off', 'format object "%IDLIST(1)|%COO(d;A D)|%OTYPE(S)"', ...lines].join('\n');
    const r = await fetch(`${SIMBAD}?${new URLSearchParams({ script: body })}`);
    if (!r.ok) throw new Error(`SIMBAD HTTP ${r.status}`);
    const out = [];
    for (const line of (await r.text()).split('\n')) {
      const p = line.split('|');
      if (p.length !== 3) continue;
      const c = p[1].trim().split(/\s+/).map(Number);
      if (c.length === 2 && c.every(isFinite)) out.push({ name: p[0].replace(/\s+/g, ' ').trim(), ra: c[0], dec: c[1], otype: p[2].trim() });
    }
    return out;
  }
  const cleanName = (s) => s.replace(/[\n\r"*]/g, ' ').trim();
  // Resolve a name ("ngc628", "Pillars of Creation", "M74") -> {name, ra, dec, otype} or null
  async function resolve(name) {
    const r = await script([cleanName(name)]);
    return r[0] || null;
  }
  // Name completions for what was typed so far
  async function complete(prefix, limit = 8) {
    const p = cleanName(prefix);
    if (p.length < 2) return [];
    return script([`set limit ${limit}`, `query id wildcard ${p}*`]);
  }

  // Autocomplete box under an <input>. sources: async (text) => [{label, hint, value}] lists, merged in order.
  // onPick(item) is called with the chosen item.
  function autocomplete(input, sources, onPick) {
    const box = document.createElement('div');
    box.className = 'sky-ac';
    Object.assign(box.style, { position: 'absolute', zIndex: 60, display: 'none', background: '#1f2833', border: '1px solid #45a29e',
      borderRadius: '4px', maxHeight: '320px', overflowY: 'auto', fontSize: '13px', minWidth: '260px', boxShadow: '0 4px 12px #000a' });
    document.body.appendChild(box);
    let items = [], active = -1, seq = 0, timer = null;
    const place = () => { const r = input.getBoundingClientRect(); Object.assign(box.style, { left: (r.left + scrollX) + 'px', top: (r.bottom + scrollY + 2) + 'px', width: Math.max(260, r.width) + 'px' }); };
    const hide = () => { box.style.display = 'none'; active = -1; };
    const render = () => {
      box.innerHTML = '';
      items.forEach((it, i) => {
        const row = document.createElement('div');
        Object.assign(row.style, { padding: '5px 8px', cursor: 'pointer', display: 'flex', justifyContent: 'space-between', gap: '10px',
          background: i === active ? '#45a29e' : 'transparent', color: i === active ? '#0b0c10' : '#c5c6c7' });
        row.innerHTML = `<span></span><span style="opacity:.7;white-space:nowrap"></span>`;
        row.children[0].textContent = it.label; row.children[1].textContent = it.hint || '';
        row.onmousedown = (e) => { e.preventDefault(); hide(); onPick(it); };
        box.appendChild(row);
      });
      if (items.length) { place(); box.style.display = 'block'; } else hide();
    };
    input.setAttribute('autocomplete', 'off');
    input.addEventListener('input', () => {
      clearTimeout(timer);
      const text = input.value, my = ++seq;
      if (text.trim().length < 2 || parse(text)) { items = []; render(); return; }
      timer = setTimeout(async () => {
        items = [];
        for (const src of sources) {
          let got = [];
          try { got = await src(text); } catch (e) { got = []; }
          if (my !== seq) return;
          const seen = new Set(items.map(i => i.label.toLowerCase()));
          items.push(...got.filter(g => !seen.has(g.label.toLowerCase())));
          active = -1; render();
        }
      }, 250);
    });
    input.addEventListener('keydown', (e) => {
      if (box.style.display === 'none') return;
      if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
        e.preventDefault();
        active = (active + (e.key === 'ArrowDown' ? 1 : -1) + items.length) % items.length; render();
      } else if (e.key === 'Enter' && active >= 0) {
        e.preventDefault(); e.stopImmediatePropagation(); const it = items[active]; hide(); onPick(it);
      } else if (e.key === 'Escape') hide();
    });
    input.addEventListener('blur', () => setTimeout(hide, 150));
    window.addEventListener('resize', () => { if (box.style.display !== 'none') place(); });
    return { hide };
  }

  // Does an image (center s_ra, s_dec, radius s_rad in degrees) reach within radiusDeg of a position?
  function near(item, pos, radiusDeg) {
    const ra = parseFloat(item.s_ra), dec = parseFloat(item.s_dec);
    if (!isFinite(ra) || !isFinite(dec)) return false;
    return sep(ra, dec, pos.ra, pos.dec) <= radiusDeg + (parseFloat(item.s_rad) || 0);
  }

  // Wire a search box: positions (Sky.parse) and SIMBAD picks call onPos({ra, dec, label}); text calls onText(text).
  // localNames: async (text) => [names] from our own lists, offered first in the autocomplete.
  function searchBox({ input, button, localNames, onText, onPos }) {
    const go = () => {
      const t = input.value.trim();
      if (!t) return;
      const p = parse(t);
      if (p) onPos({ ...p, label: fmt(p.ra, p.dec) }); else onText(t);
    };
    button.onclick = go;
    input.addEventListener('keydown', (e) => { if (e.key === 'Enter') go(); });
    autocomplete(input, [
      async (t) => (await localNames(t)).slice(0, 8).map(n => ({ label: n, hint: 'in list', local: true })),
      async (t) => (await complete(t)).map(s => ({ label: s.name, hint: `${s.otype} · SIMBAD`, ...s })),
    ], (it) => {
      input.value = it.label;
      if (it.local) onText(it.label); else onPos({ ra: it.ra, dec: it.dec, label: `${it.name} (${fmt(it.ra, it.dec)})` });
    });
  }

  // Images near a position, nearest first, with item._sep (degrees from the position)
  function cone(items, pos, radiusDeg) {
    return items.filter(it => near(it, pos, radiusDeg))
      .map(it => (it._sep = sep(+it.s_ra, +it.s_dec, pos.ra, pos.dec), it))
      .sort((a, b) => a._sep - b._sep);
  }

  return { fmt, sep, parse, resolve, complete, autocomplete, near, searchBox, cone };
})();
