// Front end for the hybrid recommender. Plain JavaScript, no build step.
"use strict";

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];

const COLORS = {
  cf: "var(--series-cf)",
  content: "var(--series-content)",
  popularity: "var(--series-pop)",
};
const SIGNAL_NAMES = { cf: "Collaborative", content: "Content", popularity: "Popularity" };
// One colour per model, the same as in the report figures.
const MODEL_COLORS = {
  "Hybrid adaptive": "var(--series-hybrid)",
  "CF only (ALS)": "var(--series-cf)",
  "Popularity (demographic)": "var(--series-pop)",
  "Content only": "var(--series-content)",
};

const state = { mode: "existing", rated: [], meta: null, results: null, slice: null };

function esc(text) {
  return String(text ?? "").replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  })[c]);
}

const fmt = (x, digits = 3) => (x == null || Number.isNaN(x) ? "–" : Number(x).toFixed(digits));
const pct = (x) => `${Math.round(x * 100)}%`;

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  let body = null;
  try { body = await response.json(); } catch { /* not JSON */ }
  if (!response.ok) throw new Error(body?.error || `Request failed (${response.status})`);
  return body;
}

/* ----------------------------------------------------------- routing */

function route() {
  const view = location.hash === "#evaluation" ? "evaluation" : "recommend";
  $("#view-recommend").hidden = view !== "recommend";
  $("#view-evaluation").hidden = view !== "evaluation";
  $$(".nav a").forEach((a) => {
    if (a.dataset.view === view) a.setAttribute("aria-current", "page");
    else a.removeAttribute("aria-current");
  });
  if (view === "evaluation" && !state.results) loadResults();
}
window.addEventListener("hashchange", route);

/* ------------------------------------------------------ recommend view */

function setMode(mode) {
  state.mode = mode;
  $$(".segmented [data-mode]").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.mode === mode)));
  $("#mode-existing").hidden = mode !== "existing";
  $("#mode-new").hidden = mode !== "new";
}
$$(".segmented [data-mode]").forEach((b) => b.addEventListener("click", () => setMode(b.dataset.mode)));

async function loadMeta() {
  try {
    state.meta = await api("/api/meta");
  } catch (error) {
    $("#examples").innerHTML = `<li><p class="error-text">${esc(error.message)}</p></li>`;
    return;
  }
  const m = state.meta;
  $("#examples").innerHTML = m.examples.map((ex) => `
    <li><button type="button" data-user="${ex.user_id}">
      <span class="ex-label">${esc(ex.label)}</span>
      <span class="ex-meta">#${ex.user_id} · ${ex.ratings} rating${ex.ratings === 1 ? "" : "s"}</span>
    </button></li>`).join("");
  $$("#examples [data-user]").forEach((b) => b.addEventListener("click", () => {
    $("#user-id").value = b.dataset.user;
    recommend();
  }));
  $("#model").innerHTML = Object.entries(m.models)
    .map(([value, label]) => `<option value="${value}">${esc(label)}</option>`).join("");
  if (m.demographics) {
    $("#demographics").hidden = false;
    $("#age").innerHTML += m.age_groups.map((g) => `<option value="${g.value}">${esc(g.label)}</option>`).join("");
  }
}

// --- search for the new-user mode
let searchTimer = null;
$("#search").addEventListener("input", (event) => {
  clearTimeout(searchTimer);
  const query = event.target.value;
  searchTimer = setTimeout(() => runSearch(query), 160);
});
$("#search").addEventListener("keydown", (event) => {
  if (event.key === "Escape") closeSearch();
  if (event.key === "ArrowDown") { event.preventDefault(); $("#search-results button")?.focus(); }
});
document.addEventListener("click", (event) => {
  if (!event.target.closest(".search")) closeSearch();
});

function closeSearch() {
  $("#search-results").hidden = true;
  $("#search").setAttribute("aria-expanded", "false");
}

async function runSearch(query) {
  const list = $("#search-results");
  if (query.trim().length < 2) { closeSearch(); return; }
  let movies = [];
  try { movies = await api(`/api/search?q=${encodeURIComponent(query)}`); } catch { /* ignore */ }
  list.innerHTML = movies.length
    ? movies.map((m) => `<li role="option"><button type="button" data-id="${m.item_id}">
        ${esc(m.title)} <span class="muted">${m.year ?? ""} · ${esc(m.genres.slice(0, 3).join(", "))}</span>
      </button></li>`).join("")
    : `<li><button type="button" disabled><span class="muted">No movie matches “${esc(query)}”.</span></button></li>`;
  list.hidden = false;
  $("#search").setAttribute("aria-expanded", "true");
  $$("button[data-id]", list).forEach((b, i, all) => {
    const movie = movies.find((m) => String(m.item_id) === b.dataset.id);
    b.addEventListener("click", () => addRated(movie));
    b.addEventListener("keydown", (event) => {
      if (event.key === "ArrowDown") { event.preventDefault(); all[i + 1]?.focus(); }
      if (event.key === "ArrowUp") { event.preventDefault(); (all[i - 1] || $("#search")).focus(); }
      if (event.key === "Escape") { closeSearch(); $("#search").focus(); }
    });
  });
}

function addRated(movie) {
  if (!state.rated.some((r) => r.item_id === movie.item_id)) {
    state.rated.push({ ...movie, rating: 4 });
  }
  $("#search").value = "";
  closeSearch();
  renderRated();
  $("#search").focus();
}

function renderRated() {
  $("#rated-empty").hidden = state.rated.length > 0;
  $("#rated").innerHTML = state.rated.map((m, idx) => `
    <li>
      <span class="title">${esc(m.title)} <span class="ex-meta">${m.year ?? ""}</span></span>
      <span class="stars" role="radiogroup" aria-label="Your rating for ${esc(m.title)}">
        ${[1, 2, 3, 4, 5].map((s) => `<button type="button" role="radio" aria-checked="${s === m.rating}"
          aria-label="${s} star${s > 1 ? "s" : ""}" class="${s <= m.rating ? "on" : ""}" data-idx="${idx}" data-star="${s}">★</button>`).join("")}
      </span>
      <button type="button" class="remove" aria-label="Remove ${esc(m.title)}" data-remove="${idx}">×</button>
    </li>`).join("");
  $$("#rated [data-star]").forEach((b) => b.addEventListener("click", () => {
    state.rated[Number(b.dataset.idx)].rating = Number(b.dataset.star);
    renderRated();
  }));
  $$("#rated [data-remove]").forEach((b) => b.addEventListener("click", () => {
    state.rated.splice(Number(b.dataset.remove), 1);
    renderRated();
  }));
}

$("#controls").addEventListener("submit", (event) => { event.preventDefault(); recommend(); });

function payload() {
  const base = {
    model: $("#model").value || "adaptive",
    k: Number($("#k").value),
    diversify: $("#diversify").checked,
  };
  if (state.mode === "existing") {
    const raw = $("#user-id").value.trim();
    if (!/^\d+$/.test(raw)) {
      throw new Error(raw ? "A user id is a whole number, like 42." : "Enter a user id, or pick one of the examples.");
    }
    return { ...base, user_id: Number(raw) };
  }
  return {
    ...base,
    ratings: state.rated.map((m) => ({ item_id: m.item_id, rating: m.rating })),
    gender: $("#gender").value || null,
    age: $("#age").value || null,
  };
}

async function recommend() {
  const results = $("#results");
  const errorBox = $("#user-error");
  errorBox.hidden = true;
  $("#user-id").removeAttribute("aria-invalid");
  let body;
  try {
    body = payload();
  } catch (error) {
    errorBox.textContent = error.message;
    errorBox.hidden = false;
    $("#user-id").setAttribute("aria-invalid", "true");
    $("#user-id").focus();
    return;
  }

  const button = $("#go");
  button.disabled = true;
  button.textContent = "Scoring…";
  results.setAttribute("aria-busy", "true");
  results.innerHTML = `<p class="summary-line">Scoring ${state.meta ? state.meta.movies.toLocaleString() : "every"} movies…</p>
    <ul class="skeleton">${"<li></li>".repeat(6)}</ul>`;
  try {
    const [data, user] = await Promise.all([
      api("/api/recommend", { method: "POST", body: JSON.stringify(body) }),
      body.user_id != null ? api(`/api/user/${body.user_id}`) : Promise.resolve(null),
    ]);
    renderRecommendations(data, user);
  } catch (error) {
    results.innerHTML = `<div class="state"><h2>That did not work.</h2><p>${esc(error.message)}</p></div>`;
    if (state.mode === "existing") {
      errorBox.textContent = error.message;
      errorBox.hidden = false;
      $("#user-id").setAttribute("aria-invalid", "true");
    }
  } finally {
    button.disabled = false;
    button.textContent = "Recommend";
    results.removeAttribute("aria-busy");
  }
}

// Weights are already a proper split (they add up to 1), so they map straight to bar widths.
function partsBar(shares, cls = "parts") {
  return `<div class="${cls}" aria-hidden="true">${["cf", "content", "popularity"]
    .filter((k) => shares[k] > 0.005)
    .map((k) => `<span style="width:${(shares[k] * 100).toFixed(1)}%;background:${COLORS[k]}"></span>`)
    .join("")}</div>`;
}

function legend() {
  return `<div class="legend">${["cf", "content", "popularity"].map((k) =>
    `<span class="key"><i style="background:${COLORS[k]}"></i>${SIGNAL_NAMES[k]}</span>`).join("")}</div>`;
}

function renderRecommendations(data, user) {
  const recs = data.recommendations;
  const n = data.history_size;
  const historyText = n === 0 ? "No ratings in history" : `${n} rating${n === 1 ? "" : "s"} in history`;
  let readout;

  if (data.mean_weights != null) {
    const w = data.mean_weights;
    readout = `
      <div class="big-number">${pct(w.cf)}</div>
      <div class="readout-text"><strong>${historyText}.</strong> That is how much the ${esc(data.model_label.replace("Hybrid, ", "").toLowerCase())} hybrid trusts collaborative filtering for the movies people actually watch. Content gets ${pct(w.content)} and popularity ${pct(w.popularity)}.</div>
      <div class="blend">${partsBar(w, "blend-bar")}${legend()}</div>`;
  } else {
    readout = `
      <div class="big-number">${n}</div>
      <div class="readout-text"><strong>${esc(data.model_label)}.</strong> ${historyText.toLowerCase()}. Single-signal models have no blend to show; switch to a hybrid in Options to compare.</div>`;
  }

  const summary = user
    ? `<p class="summary-line"><strong>${data.hits} of ${recs.length}</strong> are movies this user went on to rate 4★ or more (out of ${user.liked_later} such movies in the hidden test period).</p>`
    : "";

  const rows = recs.map((r) => {
    const right = r.weights
      ? `<div class="rec-parts">${partsBar(r.weights)}
           <div class="parts-text">${r.ratings_in_train.toLocaleString()} rating${r.ratings_in_train === 1 ? "" : "s"} · CF ${pct(r.weights.cf)}</div></div>`
      : `<div class="rec-parts"><div class="parts-text">${r.ratings_in_train.toLocaleString()} ratings</div></div>`;
    const tip = r.parts
      ? ` title="Trust: CF ${pct(r.weights.cf)}, content ${pct(r.weights.content)}, popularity ${pct(r.weights.popularity)}. Contribution to the score: CF ${fmt(r.parts.cf, 2)}, content ${fmt(r.parts.content, 2)}, popularity ${fmt(r.parts.popularity, 2)}"`
      : "";
    return `<li class="rec"${tip}>
      <div class="rec-rank">${String(r.rank).padStart(2, "0")}</div>
      <div>
        <div class="rec-title">${esc(r.title)} <span class="year">${r.year ?? ""}</span>${r.liked_later ? '<span class="liked">✓ liked later</span>' : ""}</div>
        <div class="rec-meta">${esc(r.genres.join(" · ")) || "No genre listed"}</div>
      </div>
      ${right}
    </li>`;
  }).join("");

  const history = user && user.history.length ? `
    <div class="history">
      <span class="label">Training history${user.profile.age ? ` · ${esc(user.profile.gender)}, ${esc(user.profile.age)}` : ""}</span>
      <ol>${user.history.map((h) => `<li><span>${esc(h.title)}</span><span class="r">${"★".repeat(Math.round(h.rating))}</span></li>`).join("")}</ol>
      ${user.n_ratings > user.history.length ? `<p class="hint">Showing ${user.history.length} of ${user.n_ratings}, highest rated first.</p>` : ""}
    </div>` : "";

  $("#results").innerHTML = `
    <div class="readout">${readout}</div>
    ${summary}
    ${recs.length ? `<ol class="rec-list">${rows}</ol>` : `<div class="state"><h2>Nothing left to recommend.</h2><p>This viewer has rated every movie we know.</p></div>`}
    ${history}`;
}

/* ----------------------------------------------------- evaluation view */

async function loadResults() {
  const body = $("#eval-body");
  try {
    state.results = await api("/api/results");
  } catch (error) {
    body.innerHTML = `<div class="state"><h2>No results yet.</h2><p>${esc(error.message)}</p></div>`;
    return;
  }
  renderEvaluation();
}

function findSlice(prefix) {
  return Object.keys(state.results.slices).find((name) => name.startsWith(prefix));
}

function renderEvaluation() {
  const r = state.results;
  const k = r.config.k;
  const slices = r.slices;
  const cold = slices[findSlice("cold users (")];
  const warm = slices[findSlice("warm users")];
  const items = slices[findSlice("cold items")];
  const all = slices["all users"];
  const ndcg = `ndcg@${k}`;
  const d = r.dataset;

  $("#eval-lede").textContent = `${d.name.toUpperCase()}: ${d.eval_users.toLocaleString()} test users, ${d.cold_eval_users.toLocaleString()} of them cold, and ${d.cold_items.toLocaleString()} movies with fewer than 5 training ratings.`;

  const tile = (label, value, caption) => `<div class="stat"><span class="label">${label}</span>
    <div class="value">${value}</div><div class="caption">${caption}</div></div>`;
  const lift = (a, b) => (b > 0 ? `${a >= b ? "+" : ""}${Math.round((a / b - 1) * 100)}%` : "–");

  const sliceButtons = [
    ["All users", "all users"], ["Warm users", findSlice("warm users")],
    ["Cold users", findSlice("cold users (")], ["Cold movies", findSlice("cold items")],
  ];
  state.slice = state.slice || sliceButtons[2][1];

  $("#eval-body").innerHTML = `
    <div class="eval-head">
      ${tile(`Cold users · NDCG@${k}`, fmt(cold["Hybrid adaptive"][ndcg]), `${lift(cold["Hybrid adaptive"][ndcg], cold["CF only (ALS)"][ndcg])} vs. CF only`)}
      ${tile(`Cold movies · NDCG@${k}`, fmt(items["Hybrid adaptive"][ndcg]), `${lift(items["Hybrid adaptive"][ndcg], items["CF only (ALS)"][ndcg])} vs. CF only`)}
      ${tile(`Warm users · NDCG@${k}`, fmt(warm["Hybrid adaptive"][ndcg]), `${lift(warm["Hybrid adaptive"][ndcg], warm["CF only (ALS)"][ndcg])} vs. CF only`)}
      ${tile("Catalog coverage", pct(all["Hybrid adaptive"].coverage), `vs. ${pct(all["Popularity"].coverage)} for popularity`)}
    </div>

    <section class="eval-section">
      <div class="aside">
        <h2>Every model, every slice.</h2>
        <p>Precision, recall and NDCG at ${k} measure ranking quality against the movies each user rated 4★ or more in the hidden test period.</p>
        <p>Coverage, novelty and diversity describe the lists themselves. The best value in each column is underlined.</p>
      </div>
      <div>
        <div class="segmented slice-tabs" role="group" aria-label="Slice">
          ${sliceButtons.map(([label, key]) => `<button type="button" data-slice="${esc(key)}" aria-pressed="${key === state.slice}">${label}</button>`).join("")}
        </div>
        <div class="table-scroll" id="slice-table"></div>
      </div>
    </section>

    <section class="eval-section">
      <div class="aside">
        <h2>Accuracy by history length.</h2>
        <p>NDCG@${k} for users grouped by how many ratings they have in training. Left of the line are the cold-start users.</p>
        <p>Pure collaborative filtering collapses at zero; the hybrid follows popularity there and hands over to CF as history grows.</p>
      </div>
      <div class="chart" id="history-chart"></div>
    </section>

    <section class="eval-section">
      <div class="aside">
        <h2>Where it still fails.</h2>
        <p>From the automated failure analysis. The full write-up is in REPORT.md.</p>
      </div>
      <ol class="findings" id="findings"></ol>
    </section>`;

  $$("[data-slice]").forEach((b) => b.addEventListener("click", () => {
    state.slice = b.dataset.slice;
    $$("[data-slice]").forEach((x) => x.setAttribute("aria-pressed", String(x.dataset.slice === state.slice)));
    renderSliceTable();
  }));
  renderSliceTable();
  renderHistoryChart();
  renderFindings();
}

function renderSliceTable() {
  const k = state.results.config.k;
  const data = state.results.slices[state.slice];
  const columns = [
    [`precision@${k}`, `P@${k}`, 3], [`recall@${k}`, `R@${k}`, 3], [`ndcg@${k}`, `NDCG@${k}`, 3],
    [`hit@${k}`, `Hit@${k}`, 3], ["coverage", "Coverage", 3], ["novelty", "Novelty", 2], ["diversity", "Diversity", 3],
  ];
  const best = {};
  columns.forEach(([key]) => { best[key] = Math.max(...Object.values(data).map((m) => m[key] ?? -Infinity)); });
  const rows = Object.entries(data).map(([model, m]) => `
    <tr class="${model === "Hybrid adaptive" ? "ours" : ""}">
      <td>${MODEL_COLORS[model] ? `<span class="swatch" style="background:${MODEL_COLORS[model]}"></span>` : `<span class="swatch"></span>`}${esc(model)}</td>
      ${columns.map(([key, , digits]) => `<td class="${m[key] === best[key] ? "best" : ""}">${fmt(m[key], digits)}</td>`).join("")}
    </tr>`).join("");
  const users = Object.values(data)[0]?.users ?? 0;
  $("#slice-table").innerHTML = `<table>
    <caption class="visually-hidden">Metrics for ${esc(state.slice)}</caption>
    <thead><tr><th>Model</th>${columns.map(([, label]) => `<th>${label}</th>`).join("")}</tr></thead>
    <tbody>${rows}</tbody></table>
    <p class="hint">${users.toLocaleString()} users in this slice.</p>`;
}

function renderHistoryChart() {
  const rows = state.results.analysis.by_history;
  const models = Object.keys(MODEL_COLORS).filter((m) => m in rows[0]);
  const host = $("#history-chart");
  const compact = host.clientWidth < 560;               // phones: no end labels, legend carries identity
  const width = compact ? 420 : 760, height = compact ? 300 : 340;
  const margin = { top: 16, right: compact ? 12 : 150, bottom: 44, left: 44 };
  const innerW = width - margin.left - margin.right;
  const innerH = height - margin.top - margin.bottom;
  const maxY = Math.max(...rows.flatMap((r) => models.map((m) => r[m] ?? 0)));
  // Clean tick step (0.05, 0.1, 0.2, ...) giving at most 6 ticks.
  const step = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1].find((st) => maxY / st <= 5) || 1;
  const yMax = Math.ceil(maxY / step) * step || step;
  const x = (i) => margin.left + (i / (rows.length - 1)) * innerW;
  const y = (v) => margin.top + innerH - (v / yMax) * innerH;
  const ticks = Array.from({ length: Math.round(yMax / step) + 1 }, (_, i) => i * step);

  const lines = models.map((m) => {
    const d = rows.map((r, i) => `${i ? "L" : "M"}${x(i).toFixed(1)},${y(r[m] ?? 0).toFixed(1)}`).join("");
    const last = rows[rows.length - 1][m] ?? 0;
    return `<path d="${d}" fill="none" stroke="${MODEL_COLORS[m]}" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>
      ${rows.map((r, i) => `<circle cx="${x(i)}" cy="${y(r[m] ?? 0)}" r="4" fill="${MODEL_COLORS[m]}" stroke="var(--bone)" stroke-width="2"/>`).join("")}
      ${compact ? "" : `<text class="end-label" data-model="${esc(m)}" x="${x(rows.length - 1) + 12}" y="${y(last) + 4}">${esc(m.replace(" (demographic)", "").replace(" (ALS)", ""))} ${fmt(last, 2)}</text>`}`;
  }).join("");

  const coldX = (x(4) + x(5)) / 2;
  const svg = `<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="NDCG by number of training ratings, per model">
    <g class="grid">${ticks.map((t) => `<line x1="${margin.left}" x2="${margin.left + innerW}" y1="${y(t)}" y2="${y(t)}"/>`).join("")}</g>
    <g class="axis">${ticks.map((t) => `<text x="${margin.left - 10}" y="${y(t) + 4}" text-anchor="end">${t.toFixed(step < 0.1 ? 2 : 1)}</text>`).join("")}
      ${rows.map((r, i) => (compact && i % 2 && i > 4) ? "" : `<text x="${x(i)}" y="${height - margin.bottom + 22}" text-anchor="middle">${esc(r.history)}</text>`).join("")}
      <text x="${margin.left + innerW / 2}" y="${height - 4}" text-anchor="middle">Training ratings per user</text>
      <text x="${coldX - 8}" y="${margin.top + 10}" text-anchor="end">cold</text>
      <text x="${coldX + 8}" y="${margin.top + 10}">warm</text></g>
    <line x1="${coldX}" x2="${coldX}" y1="${margin.top}" y2="${margin.top + innerH}" stroke="var(--rule-strong)" stroke-width="1"/>
    <line id="crosshair" x1="0" x2="0" y1="${margin.top}" y2="${margin.top + innerH}" stroke="var(--ink)" stroke-width="1" visibility="hidden"/>
    ${lines}
    <rect id="hover-zone" x="${margin.left}" y="${margin.top}" width="${innerW}" height="${innerH}" fill="transparent"/>
  </svg>`;

  host.innerHTML = `${svg}<div class="tooltip" id="tooltip" role="status"></div>
    <div class="legend">${models.map((m) => `<span class="key"><i style="background:${MODEL_COLORS[m]}"></i>${esc(m)}</span>`).join("")}</div>
    <details style="margin-top:16px"><summary class="hint" style="cursor:pointer">Show as table</summary>
      <div class="table-scroll"><table><thead><tr><th>Ratings</th><th>Users</th>${models.map((m) => `<th>${esc(m)}</th>`).join("")}</tr></thead>
      <tbody>${rows.map((r) => `<tr><td>${esc(r.history)}</td><td>${r.users}</td>${models.map((m) => `<td>${fmt(r[m])}</td>`).join("")}</tr>`).join("")}</tbody></table></div>
    </details>`;

  // Avoid end-label collisions: if two labels sit within 14px, drop the lower-priority ones to the legend.
  const labels = $$(".end-label", host).sort((a, b) => Number(a.getAttribute("y")) - Number(b.getAttribute("y")));
  let lastY = -Infinity;
  labels.forEach((label) => {
    const ly = Number(label.getAttribute("y"));
    if (ly - lastY < 14) label.setAttribute("y", String(lastY + 14));
    lastY = Number(label.getAttribute("y"));
  });

  const svgEl = $("svg", host);
  const tooltip = $("#tooltip", host);
  const cross = $("#crosshair", host);
  const showAt = (clientX) => {
    const box = svgEl.getBoundingClientRect();
    const scale = width / box.width;
    const px = (clientX - box.left) * scale;
    const i = Math.max(0, Math.min(rows.length - 1, Math.round(((px - margin.left) / innerW) * (rows.length - 1))));
    cross.setAttribute("x1", x(i)); cross.setAttribute("x2", x(i)); cross.setAttribute("visibility", "visible");
    const row = rows[i];
    tooltip.innerHTML = `<strong>${esc(row.history)} rating${row.history === "1" ? "" : "s"}</strong> · ${row.users} users
      ${models.slice().sort((a, b) => (row[b] ?? 0) - (row[a] ?? 0)).map((m) => `<div class="tt-row"><span><i style="background:${MODEL_COLORS[m]}"></i>${esc(m)}</span><span>${fmt(row[m])}</span></div>`).join("")}`;
    tooltip.style.display = "block";
    // Keep the tooltip inside the chart: centre it on the hovered column, clamp at the edges.
    const half = tooltip.offsetWidth / 2;
    const left = Math.min(Math.max(x(i) / scale, half), box.width - half);
    tooltip.style.left = `${left}px`;
    tooltip.style.top = `${(margin.top + innerH) / scale + 8}px`;
  };
  const hide = () => { tooltip.style.display = "none"; cross.setAttribute("visibility", "hidden"); };
  const zone = $("#hover-zone", host);
  zone.addEventListener("mousemove", (e) => showAt(e.clientX));
  zone.addEventListener("mouseleave", hide);
  zone.addEventListener("touchstart", (e) => showAt(e.touches[0].clientX), { passive: true });
}

function renderFindings() {
  const a = state.results.analysis;
  const findings = [];
  const pb = a.popularity_bias.models["Hybrid adaptive"];
  if (pb) {
    findings.push(["Long-tail movies are rarely surfaced.",
      `Recall on the most-watched “head” movies is ${pct(pb.recall_head)}, on the long tail ${pct(pb.recall_tail)}. ${pct(pb.share_of_recs_from_head)} of recommendations come from the head.`]);
  }
  const ms = a.mainstreamness;
  if (ms?.length) {
    const niche = ms[0], main = ms[ms.length - 1];
    const k = state.results.config.k;
    const worse = niche.recall < main.recall;
    findings.push([worse ? "Niche taste: fewer of your favourites are found." : "Niche taste is not the weak spot.",
      `Recall@${k} is ${fmt(niche.recall)} for the most niche fifth of warm users and ${fmt(main.recall)} for the most mainstream fifth (NDCG@${k}: ${fmt(niche.ndcg)} vs ${fmt(main.ndcg)}).`]);
  }
  if (a.genres?.length) {
    const low = a.genres.slice(0, 3).map((g) => `${g.genre} (${pct(g.recall)})`).join(", ");
    findings.push(["Some genres are systematically missed.", `Lowest recall of liked movies: ${low}.`]);
  }
  if (a.cold_user_ratings?.length) {
    const rows = a.cold_user_ratings.map((r) => `${r["their ratings"]}: ${fmt(r.ndcg)}`).join("; ");
    findings.push(["A cold user’s few ratings can mislead.", `NDCG by the user's average training rating, ${rows}.`]);
  }
  if (a.hardest_cold_items?.length) {
    const m = a.hardest_cold_items[0];
    findings.push(["Metadata cannot tell a hit from a flop.",
      `“${m.title}” (${m.genres}) was liked by ${m.fans_in_test} test users yet reached only ${pct(m.hit_rate)} of them. Content alone cannot tell it apart from ${m.nearest_by_content[0]}.`]);
  }
  $("#findings").innerHTML = findings.map(([title, text]) => `<li><div><strong>${esc(title)}</strong><p>${esc(text)}</p></div></li>`).join("");
}

/* ---------------------------------------------------------------- boot */

route();
loadMeta();
renderRated();
