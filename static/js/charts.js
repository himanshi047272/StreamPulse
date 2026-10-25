/* Tiny dependency-free SVG chart helpers (work offline). */
const Charts = (() => {
  const esc = s => String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;");
  const nice = (max) => { const p = Math.pow(10, Math.floor(Math.log10(max || 1))); const f = max / p; return (f <= 1 ? 1 : f <= 2 ? 2 : f <= 5 ? 5 : 10) * p; };

  function donut(segs, center, sub, size = 190) {
    const total = segs.reduce((a, s) => a + s.value, 0) || 1, r = 70, c = 2 * Math.PI * r;
    let off = 0, arcs = "";
    segs.forEach(s => {
      const len = (s.value / total) * c;
      arcs += `<circle r="${r}" cx="100" cy="100" fill="none" stroke="${s.color}" stroke-width="26" stroke-dasharray="${len} ${c - len}" stroke-dashoffset="${-off}" transform="rotate(-90 100 100)"/>`;
      off += len;
    });
    if (!segs.some(s => s.value)) arcs = `<circle r="${r}" cx="100" cy="100" fill="none" stroke="#eef1f8" stroke-width="26"/>`;
    return `<svg viewBox="0 0 200 200" width="${size}" height="${size}" style="display:block;margin:0 auto">${arcs}
      <text x="100" y="${sub ? 100 : 108}" text-anchor="middle" font-size="26" font-weight="800" fill="#1b2437">${esc(center)}</text>
      ${sub ? `<text x="100" y="120" text-anchor="middle" font-size="11" fill="#7b8499">${esc(sub)}</text>` : ""}</svg>`;
  }

  function groupedBars(cats, series, o = {}) {
    const W = 560, H = o.height || 250, L = 40, R = 8, T = 10, B = 34, min = o.min ?? 0;
    const max = o.max ?? 1, ticks = 5, iw = W - L - R, ih = H - T - B;
    const gw = iw / cats.length, bw = Math.min(16, (gw * 0.72) / series.length);
    let s = `<svg viewBox="0 0 ${W} ${H}" width="100%">`;
    for (let i = 0; i <= ticks; i++) {
      const v = min + (max - min) * i / ticks, y = T + ih - (ih * i / ticks);
      s += `<line x1="${L}" x2="${W - R}" y1="${y}" y2="${y}" stroke="#edf0f6"/><text x="${L - 7}" y="${y + 4}" text-anchor="end" font-size="10.5" fill="#8a93a8">${(v * (o.pct ? 100 : 1)).toFixed(o.pct ? 0 : 2)}${o.pct ? "%" : ""}</text>`;
    }
    cats.forEach((c, i) => {
      const x0 = L + gw * i + (gw - bw * series.length) / 2;
      series.forEach((se, k) => {
        const v = se.values[i] ?? 0, h = Math.max(0, ih * (v - min) / (max - min));
        s += `<rect x="${x0 + k * bw}" y="${T + ih - h}" width="${bw - 2}" height="${h}" rx="4" fill="${se.color}"><title>${esc(c)} · ${esc(se.name)}: ${(v * 100).toFixed(1)}%</title></rect>`;
      });
      s += `<text x="${L + gw * i + gw / 2}" y="${H - 12}" text-anchor="middle" font-size="10.5" fill="#6b7590">${esc(c)}</text>`;
    });
    return s + "</svg>";
  }

  function line(o) {
    const W = 760, H = o.height || 300, L = 46, R = 14, T = 14, B = 38, n = o.labels.length;
    let all = [];
    o.series.forEach(s => s.values.forEach(v => v != null && all.push(v)));
    (o.bands || []).forEach(b => { b.lo.forEach(v => v != null && all.push(v)); b.hi.forEach(v => v != null && all.push(v)); });
    let lo = Math.min(...all), hi = Math.max(...all); const pad = (hi - lo) * 0.08 || 0.05; lo = Math.max(0, lo - pad); hi = hi + pad;
    const X = i => L + (W - L - R) * i / Math.max(n - 1, 1), Y = v => T + (H - T - B) * (1 - (v - lo) / (hi - lo));
    let s = `<svg viewBox="0 0 ${W} ${H}" width="100%">`;
    for (let i = 0; i <= 5; i++) { const v = lo + (hi - lo) * i / 5, y = Y(v); s += `<line x1="${L}" x2="${W - R}" y1="${y}" y2="${y}" stroke="#edf0f6"/><text x="${L - 7}" y="${y + 4}" text-anchor="end" font-size="10.5" fill="#8a93a8">${(v * 100).toFixed(0)}%</text>`; }
    (o.shades || []).forEach(sh => { s += `<rect x="${X(sh.from)}" y="${T}" width="${X(sh.to) - X(sh.from)}" height="${H - T - B}" fill="${sh.color}" opacity=".5"/><text x="${(X(sh.from) + X(sh.to)) / 2}" y="${T + 12}" text-anchor="middle" font-size="10" font-weight="600" fill="#6b7590">${sh.label}</text>`; });
    (o.bands || []).forEach(b => {
      let up = "", dn = [];
      b.lo.forEach((v, i) => { if (v != null && b.hi[i] != null) { up += `${up ? "L" : "M"}${X(i)},${Y(b.hi[i])}`; dn.unshift(`L${X(i)},${Y(v)}`); } });
      if (up) s += `<path d="${up}${dn.join("")}Z" fill="${b.color}" opacity=".13"/>`;
    });
    o.series.forEach(se => {
      let d = "", pen = false;
      se.values.forEach((v, i) => { if (v == null) { pen = false; return; } d += `${pen ? "L" : "M"}${X(i)},${Y(v)}`; pen = true; });
      s += `<path d="${d}" fill="none" stroke="${se.color}" stroke-width="${se.width || 2}" ${se.dash ? `stroke-dasharray="${se.dash}"` : ""} stroke-linejoin="round" stroke-linecap="round"/>`;
      if (se.dots) se.values.forEach((v, i) => { if (v != null) s += `<circle cx="${X(i)}" cy="${Y(v)}" r="2.6" fill="${se.color}"><title>${esc(o.labels[i])} · ${esc(se.name)}: ${(v * 100).toFixed(1)}%</title></circle>`; });
    });
    (o.vlines || []).forEach(v => { s += `<line x1="${X(v.x)}" x2="${X(v.x)}" y1="${T}" y2="${H - B}" stroke="#9aa5bd" stroke-dasharray="4 4"/>`; });
    const step = Math.ceil(n / 12);
    o.labels.forEach((l, i) => { if (i % step === 0) s += `<text x="${X(i)}" y="${H - 14}" text-anchor="middle" font-size="10" fill="#8a93a8">${esc(l)}</text>`; });
    return s + "</svg>";
  }

  function roc(curves, size = 330) {
    const W = 340, P = 38, s0 = W - P - 12;
    const X = v => P + s0 * v, Y = v => 12 + s0 * (1 - v);
    let s = `<svg viewBox="0 0 ${W} ${W}" width="100%" style="max-width:${size + 90}px">`;
    for (let i = 0; i <= 4; i++) {
      const v = i / 4; s += `<line x1="${X(0)}" x2="${X(1)}" y1="${Y(v)}" y2="${Y(v)}" stroke="#edf0f6"/><line y1="${Y(0)}" y2="${Y(1)}" x1="${X(v)}" x2="${X(v)}" stroke="#edf0f6"/>
      <text x="${P - 7}" y="${Y(v) + 4}" text-anchor="end" font-size="10" fill="#8a93a8">${v.toFixed(2)}</text><text x="${X(v)}" y="${W - 14}" text-anchor="middle" font-size="10" fill="#8a93a8">${v.toFixed(2)}</text>`;
    }
    s += `<line x1="${X(0)}" y1="${Y(0)}" x2="${X(1)}" y2="${Y(1)}" stroke="#b9c2d6" stroke-dasharray="4 4"/>`;
    curves.forEach(c => { s += `<path d="${c.pts.map((p, i) => `${i ? "L" : "M"}${X(p[0])},${Y(p[1])}`).join("")}" fill="none" stroke="${c.color}" stroke-width="${c.bold ? 3 : 1.8}" opacity="${c.bold ? 1 : .8}"/>`; });
    s += `<text x="${W / 2}" y="${W - 1}" text-anchor="middle" font-size="10.5" fill="#6b7590">False positive rate</text>`;
    return s + "</svg>";
  }

  return { donut, groupedBars, line, roc, esc };
})();
