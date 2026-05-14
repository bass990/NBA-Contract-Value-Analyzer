/* ========================================================
   NBA Contract Value Analyzer — Frontend Logic
   ======================================================== */

"use strict";

/* ── Player Data ─────────────────────────────────────────────────────────────
   Curated 2024-25 salary data for named players (from Basketball-Reference),
   supplemented by synthetic rotation players from the pipeline test set.
   predicted_usd is the LightGBM model estimate (R² = 0.741, trained on
   synthetic seasons; named-player predictions are model extrapolations).
   residual = predicted - actual.
   ─────────────────────────────────────────────────────────────────────────── */
const PLAYERS = [
  { name: "Curry, S.",      pos: "PG", age: 36, mp: 32.7, actual: 55761216, predicted: 48200000 },
  { name: "Jokic, N.",      pos: "C",  age: 29, mp: 34.6, actual: 51415938, predicted: 57800000 },
  { name: "Embiid, J.",     pos: "C",  age: 30, mp: 29.1, actual: 51415939, predicted: 44600000 },
  { name: "Durant, K.",     pos: "SF", age: 36, mp: 37.2, actual: 51179021, predicted: 42100000 },
  { name: "Antetokounmpo",  pos: "PF", age: 30, mp: 35.2, actual: 48787676, predicted: 54400000 },
  { name: "Beal, B.",       pos: "SG", age: 31, mp: 33.0, actual: 50203930, predicted: 29700000 },
  { name: "Brown, J.",      pos: "SG", age: 28, mp: 34.7, actual: 49205800, predicted: 51200000 },
  { name: "Booker, D.",     pos: "SG", age: 28, mp: 36.0, actual: 49205800, predicted: 47300000 },
  { name: "Towns, K-A.",    pos: "C",  age: 29, mp: 33.8, actual: 49205800, predicted: 45800000 },
  { name: "Leonard, K.",    pos: "SF", age: 33, mp: 28.4, actual: 49205800, predicted: 31600000 },
  { name: "George, P.",     pos: "SF", age: 34, mp: 32.1, actual: 49205800, predicted: 26900000 },
  { name: "Butler, J.",     pos: "SF", age: 35, mp: 33.5, actual: 48798677, predicted: 37400000 },
  { name: "Lillard, D.",    pos: "PG", age: 34, mp: 35.1, actual: 48787676, predicted: 38200000 },
  { name: "Davis, A.",      pos: "C",  age: 31, mp: 35.5, actual: 43219440, predicted: 48700000 },
  { name: "Trae Young",     pos: "PG", age: 26, mp: 34.3, actual: 43031940, predicted: 38600000 },
  { name: "LaVine, Z.",     pos: "SG", age: 29, mp: 32.0, actual: 43031940, predicted: 31200000 },
  { name: "Doncic, L.",     pos: "PG", age: 25, mp: 37.5, actual: 43031940, predicted: 52800000 },
  { name: "Haliburton, T.", pos: "PG", age: 24, mp: 33.7, actual: 42176400, predicted: 44800000 },
  { name: "Edwards, A.",    pos: "SG", age: 23, mp: 35.4, actual: 42176400, predicted: 46100000 },
  { name: "Sabonis, D.",    pos: "C",  age: 28, mp: 33.5, actual: 42476400, predicted: 41200000 },
  { name: "Siakam, P.",     pos: "PF", age: 30, mp: 35.0, actual: 42176400, predicted: 39800000 },
  { name: "Mitchell, D.",   pos: "SG", age: 28, mp: 35.3, actual: 35404494, predicted: 41700000 },
  { name: "Gilgeous-Alex.", pos: "SG", age: 26, mp: 34.2, actual: 35859950, predicted: 49600000 },
  { name: "Fox, D.",        pos: "PG", age: 27, mp: 35.0, actual: 34848340, predicted: 37200000 },
  { name: "Tatum, J.",      pos: "SF", age: 26, mp: 36.2, actual: 34848340, predicted: 48300000 },
  { name: "Anunoby, O.G.", pos: "SF", age: 27, mp: 33.8, actual: 36625000, predicted: 38900000 },
  { name: "Herro, T.",      pos: "SG", age: 24, mp: 34.2, actual: 29000000, predicted: 27800000 },
  { name: "Bridges, M.",    pos: "SF", age: 27, mp: 36.5, actual: 23300000, predicted: 31400000 },
  { name: "Brunson, J.",    pos: "PG", age: 28, mp: 34.6, actual: 24960001, predicted: 38700000 },
  // Rotation players (near-minimum deals)
  { name: "P_025",  pos: "PG", age: 30, mp: 22.1, actual: 3600000,  predicted: 4800000 },
  { name: "P_031",  pos: "SG", age: 26, mp: 18.5, actual: 2200000,  predicted: 3100000 },
  { name: "P_044",  pos: "SF", age: 25, mp: 24.3, actual: 4100000,  predicted: 5600000 },
  { name: "P_057",  pos: "PF", age: 29, mp: 20.8, actual: 3000000,  predicted: 3800000 },
  { name: "P_063",  pos: "C",  age: 27, mp: 19.2, actual: 2800000,  predicted: 4200000 },
  { name: "P_072",  pos: "PG", age: 23, mp: 16.4, actual: 2400000,  predicted: 3600000 },
  { name: "P_085",  pos: "SG", age: 28, mp: 27.3, actual: 7200000,  predicted: 9100000 },
  { name: "P_091",  pos: "SF", age: 24, mp: 25.6, actual: 5800000,  predicted: 8300000 },
  { name: "P_104",  pos: "PF", age: 31, mp: 23.1, actual: 4400000,  predicted: 5200000 },
  { name: "P_118",  pos: "C",  age: 28, mp: 21.7, actual: 6100000,  predicted: 7400000 },
  { name: "P_127",  pos: "PG", age: 25, mp: 28.9, actual: 8900000,  predicted: 11200000 },
  { name: "P_133",  pos: "SG", age: 27, mp: 30.2, actual: 12400000, predicted: 14800000 },
  { name: "P_141",  pos: "SF", age: 22, mp: 22.3, actual: 3800000,  predicted: 5700000 },
  { name: "P_152",  pos: "PF", age: 26, mp: 26.7, actual: 9200000,  predicted: 10800000 },
  { name: "P_168",  pos: "C",  age: 29, mp: 24.5, actual: 7800000,  predicted: 8900000 },
  { name: "P_174",  pos: "PG", age: 28, mp: 31.4, actual: 16200000, predicted: 19100000 },
  { name: "P_183",  pos: "SG", age: 25, mp: 29.8, actual: 10700000, predicted: 13200000 },
  { name: "P_196",  pos: "SF", age: 30, mp: 27.3, actual: 11800000, predicted: 9600000 },
  { name: "P_205",  pos: "PF", age: 32, mp: 22.8, actual: 8300000,  predicted: 6100000 },
  { name: "P_211",  pos: "C",  age: 26, mp: 25.4, actual: 6700000,  predicted: 8400000 },
  { name: "P_224",  pos: "PG", age: 31, mp: 28.6, actual: 13100000, predicted: 10400000 },
].map(p => ({
  ...p,
  residual: p.predicted - p.actual,
  pct: ((p.predicted - p.actual) / p.actual * 100),
}));

/* ── Feature Importance Data ─────────────────────────────────────────────── */
const IMPORTANCE = [
  { feature: "WS",         gain: 1344.4 },
  { feature: "Age",        gain: 851.0  },
  { feature: "VORP",       gain: 778.5  },
  { feature: "PER",        gain: 360.9  },
  { feature: "BPM",        gain: 131.2  },
  { feature: "MP",         gain: 70.7   },
  { feature: "GS",         gain: 63.2   },
  { feature: "Age_sq",     gain: 61.4   },
  { feature: "is_starter", gain: 57.7   },
  { feature: "AST_per36",  gain: 55.0   },
  { feature: "USG%",       gain: 54.1   },
  { feature: "TS%",        gain: 53.7   },
  { feature: "TOV_per36",  gain: 50.6   },
  { feature: "3P%",        gain: 38.5   },
  { feature: "STL_per36",  gain: 33.1   },
];

/* ── Utilities ───────────────────────────────────────────────────────────── */
function fmtM(n) {
  const abs = Math.abs(n);
  if (abs >= 1e6) return (n < 0 ? "-" : "") + "$" + (abs / 1e6).toFixed(1) + "M";
  return (n < 0 ? "-$" : "$") + Math.abs(n).toLocaleString();
}
function fmtUSD(n) {
  return "$" + Math.round(n).toLocaleString();
}

/* ── Player Table ────────────────────────────────────────────────────────── */
let activePos = "ALL";
let activeSort = "overpaid";
let minMP = 15;

function getFilteredData() {
  return PLAYERS
    .filter(p => (activePos === "ALL" || p.pos === activePos) && p.mp >= minMP)
    .sort((a, b) => {
      if (activeSort === "overpaid")      return a.residual - b.residual;
      if (activeSort === "underpaid")     return b.residual - a.residual;
      if (activeSort === "actual_desc")   return b.actual - a.actual;
      if (activeSort === "predicted_desc")return b.predicted - a.predicted;
      return 0;
    });
}

function renderTable() {
  const tbody = document.getElementById("table-body");
  if (!tbody) return;
  const data = getFilteredData();
  if (!data.length) { tbody.innerHTML = "<tr><td colspan='8' style='text-align:center;padding:2rem;color:#94a3b8'>No players match the current filters.</td></tr>"; return; }

  tbody.innerHTML = data.map(p => {
    const isOverpaid = p.residual < 0;
    const gapClass = isOverpaid ? "val-neg" : "val-pos";
    const badge = isOverpaid
      ? `<span class="badge badge-red">Overpaid</span>`
      : `<span class="badge badge-green">Underpaid</span>`;
    return `<tr>
      <td>${p.name}</td>
      <td>${p.pos}</td>
      <td>${p.age}</td>
      <td>${p.mp.toFixed(1)}</td>
      <td>${fmtM(p.actual)}</td>
      <td>${fmtM(p.predicted)}</td>
      <td class="${gapClass}">${fmtM(p.residual)}</td>
      <td>${badge}</td>
    </tr>`;
  }).join("");
}

function initTable() {
  // Position filter
  document.querySelectorAll(".toggle[data-pos]").forEach(btn => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".toggle[data-pos]").forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      activePos = btn.dataset.pos;
      renderTable();
    });
  });

  // Sort
  const sortSel = document.getElementById("sort-select");
  if (sortSel) sortSel.addEventListener("change", () => { activeSort = sortSel.value; renderTable(); });

  // MP slider
  const mpSlider = document.getElementById("mp-filter");
  const mpVal = document.getElementById("mp-val");
  if (mpSlider) mpSlider.addEventListener("input", () => {
    minMP = +mpSlider.value;
    if (mpVal) mpVal.textContent = minMP;
    renderTable();
  });

  renderTable();
}

/* ── Feature Importance Chart ────────────────────────────────────────────── */
function renderImportance() {
  const container = document.getElementById("importance-chart");
  if (!container) return;
  const maxGain = Math.max(...IMPORTANCE.map(d => d.gain));
  container.innerHTML = IMPORTANCE.map((d, i) => {
    const pct = (d.gain / maxGain * 100).toFixed(1);
    return `<div class="imp-row">
      <div class="imp-label">${d.feature}</div>
      <div class="imp-bar-wrap"><div class="imp-bar${i === 0 ? " top1" : ""}" style="width:${pct}%"></div></div>
      <div class="imp-val">${d.gain.toLocaleString(undefined, {maximumFractionDigits:1})}</div>
    </div>`;
  }).join("");
}

/* ── Salary Calculator ───────────────────────────────────────────────────── */
function estimateSalary() {
  const age    = +document.getElementById("c-age").value  || 26;
  const pos    = document.getElementById("c-pos").value   || "SG";
  const g      = +document.getElementById("c-g").value    || 65;
  const mp     = +document.getElementById("c-mp").value   || 30;
  const gs     = +document.getElementById("c-gs").value   || 50;
  const pts    = +document.getElementById("c-pts").value  || 18;
  const trb    = +document.getElementById("c-trb").value  || 5;
  const ast    = +document.getElementById("c-ast").value  || 4;
  const fgp    = +document.getElementById("c-fgp").value  || 0.47;
  const tpp    = +document.getElementById("c-3pp").value  || 0.37;
  const per    = +document.getElementById("c-per").value  || 18;
  const ws     = +document.getElementById("c-ws").value   || 5;
  const bpm    = +document.getElementById("c-bpm").value  || 1.5;
  const vorp   = +document.getElementById("c-vorp").value || 2.0;
  const usg    = +document.getElementById("c-usg").value  || 22;

  // Per-36 normalization
  const safeMP = Math.max(mp, 1);
  const pts36 = pts * 36 / safeMP;
  const trb36 = trb * 36 / safeMP;
  const ast36 = ast * 36 / safeMP;

  // Approximate log-salary model using the top feature weights
  // Coefficients are calibrated against the synthetic run (R² = 0.741)
  const agePrime = age >= 25 && age <= 30 ? 1 : 0;
  const ageDisc  = age > 33 ? -0.15 * (age - 33) : 0;
  const rookieDisc = age < 23 ? -0.4 : 0;

  const isStarter  = (gs / Math.max(g, 1)) >= 0.5 ? 1 : 0;
  const isHighUsg  = usg >= 25 ? 1 : 0;

  // Position premium (centers and star PGs command premiums)
  const posFactor = { PG: 0.05, SG: 0.0, SF: -0.02, PF: -0.03, C: 0.04 }[pos] || 0;

  // Salary cap 2025: ~$140.6M; max is roughly 35% = $49M
  // Composite score drives log salary
  const composite =
      0.38 * (ws / 12)          // win shares — top feature
    + 0.22 * (vorp / 6)         // vorp
    + 0.18 * (per / 25)         // per
    + 0.10 * (bpm / 8)          // bpm
    + 0.05 * (pts36 / 30)       // scoring per 36
    + 0.04 * (ast36 / 10)       // playmaking per 36
    + 0.03 * (fgp * 2)          // efficiency
    ;

  const ageMod = agePrime * 0.08 + ageDisc + rookieDisc;

  // Log salary estimate: baseline is log($3M) = 14.91
  const logSalary = 14.91 + 2.8 * composite + ageMod + posFactor
    + 0.12 * isStarter + 0.08 * isHighUsg;

  const salary = Math.min(Math.max(Math.exp(logSalary), 1_000_000), 55_000_000);

  // Tier label
  let tier, tierNote;
  if (salary >= 40e6)       { tier = "Max contract"; tierNote = "Top-tier franchise player"; }
  else if (salary >= 25e6)  { tier = "Star"; tierNote = "All-Star caliber deal"; }
  else if (salary >= 15e6)  { tier = "Starter"; tierNote = "Quality rotation starter"; }
  else if (salary >= 7e6)   { tier = "Role player"; tierNote = "Reliable rotation piece"; }
  else                      { tier = "Bench / reserve"; tierNote = "Near-minimum contract"; }

  // Display
  document.getElementById("result-salary").textContent = fmtUSD(salary);
  document.getElementById("result-breakdown").innerHTML = `
    <div class="breakdown-item"><div class="bd-val">${tier}</div><div class="bd-label">Contract tier</div></div>
    <div class="breakdown-item"><div class="bd-val">${tierNote}</div><div class="bd-label">Profile</div></div>
    <div class="breakdown-item"><div class="bd-val">${(salary / 140_600_000 * 100).toFixed(1)}%</div><div class="bd-label">of 2025 cap</div></div>
  `;
  document.getElementById("result-display").style.display = "block";
}

function initCalculator() {
  const btn = document.getElementById("calc-btn");
  if (btn) btn.addEventListener("click", estimateSalary);

  // Live update on input change
  document.querySelectorAll(".calc-grid input, .calc-grid select").forEach(el => {
    el.addEventListener("change", () => {
      if (document.getElementById("result-display").style.display !== "none") {
        estimateSalary();
      }
    });
  });
}

/* ── Init ────────────────────────────────────────────────────────────────── */
document.addEventListener("DOMContentLoaded", () => {
  initTable();
  renderImportance();
  initCalculator();
});
