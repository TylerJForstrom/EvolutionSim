/* Adaptive Ecosystem Lab — demo logic. Vanilla JS, no deps.
   - Animates pre-recorded trained / random ecosystem playbacks on a canvas.
   - Draws the real training-score curve (original collapse vs fixed run) as SVG. */

const SPECIES = [
  { key: "herbivore", name: "Herbivores", color: "#34d399", r: 4.2 },
  { key: "predator", name: "Predators", color: "#f87171", r: 5.4 },
  { key: "decomposer", name: "Decomposers", color: "#a78bfa", r: 3.8 },
  { key: "pollinator", name: "Pollinators", color: "#fbbf24", r: 3.8 },
  { key: "engineer", name: "Engineers", color: "#22d3ee", r: 4.6 },
];

const state = {
  data: { trained: null, random: null },
  mode: "trained",
  tick: 0,
  playing: true,
  speed: 1, // ticks per frame-second multiplier
  acc: 0,
  last: 0,
};

const $ = (id) => document.getElementById(id);

async function loadJSON(path) {
  const r = await fetch(path);
  if (!r.ok) throw new Error(`failed to load ${path}`);
  return r.json();
}

// ---------------------------------------------------------------- ecosystem
const canvas = $("eco");
const ctx = canvas.getContext("2d");
const W = canvas.width, H = canvas.height;

function curData() { return state.data[state.mode]; }

function drawVeg(d, tick) {
  const v = d.veg;
  const frame = v.frames[Math.min(v.frames.length - 1, Math.floor(tick / v.every))];
  if (!frame) return;
  const cw = W / v.w, ch = H / v.h;
  for (let j = 0; j < v.h; j++) {
    for (let i = 0; i < v.w; i++) {
      const val = frame[j * v.w + i] / 9;
      if (val <= 0.02) continue;
      ctx.fillStyle = `rgba(52,211,153,${(0.04 + val * 0.13).toFixed(3)})`;
      ctx.fillRect(i * cw, j * ch, cw + 0.6, ch + 0.6);
    }
  }
}

function drawTick(d, tick) {
  ctx.clearRect(0, 0, W, H);
  // soft base
  ctx.fillStyle = "#070a11";
  ctx.fillRect(0, 0, W, H);
  drawVeg(d, tick);

  const arr = d.ticks[tick] || [];
  const ps = d.posScale;
  const sx = W / d.grid.w, sy = H / d.grid.h;
  // draw predators last so they read on top
  for (let pass = 0; pass < 2; pass++) {
    for (let k = 0; k < arr.length; k += 3) {
      const type = arr[k];
      const isPred = type === 1;
      if ((pass === 0) === isPred) continue;
      const sp = SPECIES[type];
      const x = (arr[k + 1] / ps) * sx;
      const y = (arr[k + 2] / ps) * sy;
      ctx.beginPath();
      ctx.arc(x, y, sp.r, 0, Math.PI * 2);
      ctx.fillStyle = sp.color;
      ctx.shadowColor = sp.color;
      ctx.shadowBlur = isPred ? 9 : 5;
      ctx.fill();
    }
  }
  ctx.shadowBlur = 0;
}

function updateLegend(d, tick) {
  const pops = d.pops[tick] || [0, 0, 0, 0, 0];
  const el = $("legend");
  el.innerHTML = SPECIES.map((sp, i) =>
    `<div class="leg-row">
       <span class="leg-dot" style="background:${sp.color};color:${sp.color}"></span>
       <span class="leg-name">${sp.name}</span>
       <span class="leg-count">${pops[i]}</span>
     </div>`
  ).join("");
}

function render() {
  const d = curData();
  if (!d) return;
  drawTick(d, state.tick);
  updateLegend(d, state.tick);
  $("tickNow").textContent = state.tick;
  $("scrub").value = state.tick;
}

function loop(ts) {
  const d = curData();
  if (d) {
    if (!state.last) state.last = ts;
    const dt = (ts - state.last) / 1000;
    state.last = ts;
    if (state.playing) {
      state.acc += dt * 30 * state.speed; // ~30 ticks/sec at 1x
      while (state.acc >= 1) {
        state.acc -= 1;
        state.tick = (state.tick + 1) % d.ticks.length;
      }
      render();
    }
  }
  requestAnimationFrame(loop);
}

// ------------------------------------------------------------------- chart
function smooth(arr, win) {
  const out = arr.slice();
  for (let i = 0; i < arr.length; i++) {
    let s = 0, n = 0;
    for (let j = -win; j <= win; j++) {
      const v = arr[i + j];
      if (v == null || i + j < 0 || i + j >= arr.length) continue;
      s += v; n++;
    }
    out[i] = n ? s / n : arr[i];
  }
  return out;
}

function buildChart(curves) {
  const svg = $("chart");
  const Wc = 900, Hc = 380, ml = 50, mr = 16, mt = 18, mb = 36;
  const x0 = ml, x1 = Wc - mr, y0 = mt, y1 = Hc - mb;
  const U = 5000, yMin = -90, yMax = 210;
  const xOf = (u) => x0 + (u / U) * (x1 - x0);
  const yOf = (s) => y0 + (1 - (s - yMin) / (yMax - yMin)) * (y1 - y0);

  const parts = [];
  // grid + y labels
  for (const s of [200, 100, 0, -50]) {
    const y = yOf(s);
    parts.push(`<line x1="${x0}" y1="${y}" x2="${x1}" y2="${y}" stroke="${s === 0 ? "rgba(255,255,255,0.22)" : "rgba(255,255,255,0.07)"}" stroke-dasharray="${s === 0 ? "4 4" : ""}"/>`);
    parts.push(`<text x="${x0 - 8}" y="${y + 4}" text-anchor="end" fill="#98a2b8" font-size="12" font-family="ui-monospace,monospace">${s}</text>`);
  }
  // x labels
  for (const u of [0, 1000, 2000, 3000, 4000, 5000]) {
    parts.push(`<text x="${xOf(u)}" y="${y1 + 24}" text-anchor="middle" fill="#98a2b8" font-size="12" font-family="ui-monospace,monospace">${u}</text>`);
  }
  parts.push(`<text x="${(x0 + x1) / 2}" y="${Hc - 4}" text-anchor="middle" fill="#6b7488" font-size="12">training update</text>`);

  // line builder that breaks on null (NaN)
  const pathFor = (c) => {
    const ev = smooth(c.eval, 2);
    let dpath = "", pen = false;
    for (let i = 0; i < c.updates.length; i++) {
      const v = ev[i];
      if (v == null) { pen = false; continue; }
      const X = xOf(c.updates[i]).toFixed(1), Y = yOf(v).toFixed(1);
      dpath += (pen ? "L" : "M") + X + " " + Y + " ";
      pen = true;
    }
    return dpath;
  };

  if (curves.fixed)
    parts.push(`<path d="${pathFor(curves.fixed)}" fill="none" stroke="#34d399" stroke-width="2.2" opacity="0.95"/>`);
  parts.push(`<path d="${pathFor(curves.original)}" fill="none" stroke="#f87171" stroke-width="2.2" opacity="0.95"/>`);

  // peak marker on original
  const o = curves.original;
  let peak = -1e9, peakU = 0;
  for (let i = 0; i < o.updates.length; i++) if (o.eval[i] != null && o.eval[i] > peak) { peak = o.eval[i]; peakU = o.updates[i]; }
  const pX = xOf(peakU), pY = yOf(peak);
  parts.push(`<circle cx="${pX}" cy="${pY}" r="4" fill="#fff"/>`);
  parts.push(`<text x="${pX + 8}" y="${pY - 6}" fill="#e8edf6" font-size="13" font-weight="700">peak ${peak.toFixed(1)}</text>`);
  // collapse annotation
  const cY = yOf(-55);
  parts.push(`<text x="${xOf(3200)}" y="${cY}" fill="#f87171" font-size="13" font-weight="600">collapse → negative</text>`);

  svg.innerHTML = parts.join("");
}

// ----------------------------------------------------------------- controls
function setMode(mode) {
  state.mode = mode;
  $("modeTrained").classList.toggle("active", mode === "trained");
  $("modeRandom").classList.toggle("active", mode === "random");
  $("modeTrained").setAttribute("aria-pressed", mode === "trained");
  $("modeRandom").setAttribute("aria-pressed", mode === "random");
  render();
}
function setPlaying(p) {
  state.playing = p;
  $("playBtn").textContent = p ? "❚❚" : "▶";
  $("playBtn").setAttribute("aria-label", p ? "Pause" : "Play");
}

function wire() {
  $("modeTrained").onclick = () => setMode("trained");
  $("modeRandom").onclick = () => setMode("random");
  $("playBtn").onclick = () => setPlaying(!state.playing);
  $("scrub").oninput = (e) => { state.tick = +e.target.value; setPlaying(false); render(); };
  document.querySelectorAll(".spd").forEach((b) => {
    b.onclick = () => {
      document.querySelectorAll(".spd").forEach((x) => x.classList.remove("active"));
      b.classList.add("active");
      state.speed = +b.dataset.s;
    };
  });
}

// --------------------------------------------------------------------- init
async function init() {
  wire();
  try {
    const [trained, random, curves] = await Promise.all([
      loadJSON("data/playback_trained.json"),
      loadJSON("data/playback_random.json"),
      loadJSON("data/training_curve.json"),
    ]);
    state.data.trained = trained;
    state.data.random = random;
    $("tickMax").textContent = trained.ticks.length;
    $("scrub").max = trained.ticks.length - 1;
    buildChart(curves);
    render();
    requestAnimationFrame(loop);
  } catch (e) {
    console.error(e);
    $("legend").innerHTML = `<div class="leg-row" style="color:#f87171">Couldn't load demo data.</div>`;
  }
}

init();
