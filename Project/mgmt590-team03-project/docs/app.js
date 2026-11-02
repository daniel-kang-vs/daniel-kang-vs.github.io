/* Source-backed dashboard interactions. No model inference or invented outcomes. */
const palette = ["#4f46e5", "#0891b2", "#b45309"];
const num = (value, digits = 0) =>
  Number(value).toLocaleString(undefined, { maximumFractionDigits: digits });
const pct = (value, digits = 1) => `${num(value, digits)}%`;
const money = (value) => "$" + num(value, 2);
const sum = (values) => values.reduce((a, b) => a + b, 0);
const mean = (values) => (values.length ? sum(values) / values.length : null);
const el = (tag, text, className) => {
  const node = document.createElement(tag);
  if (text != null) node.textContent = text;
  if (className) node.className = className;
  return node;
};
const value = (id) => document.getElementById(id)?.value;
let DATA,
  instances = [],
  exportRows = [];
Chart.defaults.color = "#596277";
Chart.defaults.font.family = "Figtree, sans-serif";
Chart.defaults.borderColor = "#e6e9f2";

function table(headers, rows, caption) {
  const wrap = el("div", null, "table-wrap");
  const t = el("table");
  t.append(el("caption", caption));
  const head = el("thead"),
    hrow = el("tr");
  headers.forEach((h) => {
    const c = el("th", h);
    c.scope = "col";
    hrow.append(c);
  });
  head.append(hrow);
  t.append(head);
  const body = el("tbody");
  rows.forEach((row) => {
    const tr = el("tr");
    row.forEach((v, i) => {
      const c = el(i ? "td" : "th", v == null ? "Unavailable" : v);
      if (!i) c.scope = "row";
      tr.append(c);
    });
    body.append(tr);
  });
  t.append(body);
  wrap.append(t);
  return wrap;
}

function metric(label, number, note) {
  const card = el("div", null, "metric");
  card.append(
    el("div", label, "label"),
    el("strong", number),
    el("small", note),
  );
  document.getElementById("metrics").append(card);
}

function draw(spec) {
  const horizontal = spec.horizontal ?? spec.labels.length > 6;
  const card = el("article", null, "chart-card" + (spec.wide ? " wide" : ""));
  card.append(el("h3", spec.title), el("p", spec.caption));
  const box = el("div", null, "chart-box");
  if (horizontal)
    box.style.height = Math.max(280, spec.labels.length * 28 + 50) + "px";
  const canvas = el("canvas");
  canvas.setAttribute("role", "img");
  canvas.setAttribute(
    "aria-label",
    spec.title + ". Exact values available below.",
  );
  box.append(canvas);
  card.append(box);
  const details = el("details");
  details.append(el("summary", "View exact values"));
  const rows = spec.labels.map((label, i) => [
    label,
    ...spec.series.map((s) =>
      s.values[i] == null
        ? "Unavailable"
        : num(s.values[i], spec.unit === "ROC-AUC" ? 4 : 2),
    ),
  ]);
  details.append(
    table(
      [spec.category || "Category", ...spec.series.map((s) => s.label)],
      rows,
      spec.unit,
    ),
  );
  card.append(details);
  document.getElementById("charts").append(card);
  spec.labels.forEach((label, i) =>
    spec.series.forEach((s) =>
      exportRows.push([
        spec.title,
        spec.scope || "Current filters",
        label,
        s.label,
        s.values[i],
        spec.unit,
      ]),
    ),
  );
  const numericAxis = {
    beginAtZero: true,
    title: { display: true, text: spec.unit },
    ...(spec.max ? { max: spec.max } : {}),
  };
  const categoricalAxis = {
    grid: { display: false },
    ticks: { autoSkip: false, maxRotation: 0, font: { size: 11 } },
  };
  instances.push(
    new Chart(canvas, {
      type: "bar",
      data: {
        labels: spec.labels,
        datasets: spec.series.map((s, i) => ({
          label: s.label,
          data: s.values,
          backgroundColor: palette[i % palette.length],
          borderRadius: 3,
        })),
      },
      options: {
        indexAxis: horizontal ? "y" : "x",
        animation: false,
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: spec.series.length > 1, position: "bottom" },
        },
        scales: horizontal
          ? { x: numericAxis, y: categoricalAxis }
          : { x: categoricalAxis, y: numericAxis },
      },
    }),
  );
}

function gridTable(title, headers, rows, note) {
  const card = el("article", null, "chart-card wide");
  card.append(el("h3", title), el("p", note), table(headers, rows, title));
  document.getElementById("charts").append(card);
}

function bank() {
  const o = DATA.overview,
    scores = DATA.charts[2].series[0].values;
  metric(
    "Cross-validation ROC-AUC",
    num(mean(scores), 3),
    "Mean of five recorded folds",
  );
  metric(
    "Bankrupt firms",
    pct((o.positives / o.rows) * 100, 2),
    `${num(o.positives)} of ${num(o.rows)} firms`,
  );
  metric(
    "Missing financial values",
    pct((o.missing / (o.rows * o.features)) * 100),
    `${num(o.missing)} of ${num(o.rows * o.features)} cells`,
  );
  metric(
    "Fold score range",
    `${num(Math.min(...scores), 3)}–${num(Math.max(...scores), 3)}`,
    "Variation across validation splits",
  );
  document.getElementById("scope").textContent =
    "Training snapshot · 10,000 firms · 64 financial ratios · recorded raw-input XGBoost validation";
  document.getElementById("finding").textContent =
    "A model that predicts every firm as solvent would be 95.27% accurate. Ranking performance and the minority class matter more than headline accuracy here; the operating cutoff is not yet validated.";
  draw({ ...DATA.charts[2], max: 1, scope: "Five stored folds" });
  const fields = [...DATA.fields]
    .sort((a, b) => b.missing - a.missing)
    .slice(0, Number(value("fieldCount")));
  draw({
    title: "Which financial ratios need the most scrutiny?",
    caption:
      "Missing values as a share of 10,000 training firms. Attr identifiers follow the supplied dataset; no economic labels are inferred.",
    labels: fields.map((r) => r.field),
    series: [
      {
        label: "Missing share",
        values: fields.map((r) => (100 * r.missing) / o.rows),
      },
    ],
    unit: "% of firms",
    max: 100,
    horizontal: true,
  });
}

function ncaa() {
  const rows = DATA.records.filter(
    (r) =>
      (value("season") === "all" || r.season === value("season")) &&
      (value("conference") === "all" || r.conference === value("conference")),
  );
  const picked = rows.filter((r) => r.selected),
    seasons = new Set(rows.map((r) => r.season));
  const nets = picked
    .filter((r) => r.net != null)
    .map((r) => r.net)
    .sort((a, b) => a - b);
  const median = nets.length
    ? (nets[Math.floor((nets.length - 1) / 2)] +
        nets[Math.floor(nets.length / 2)]) /
      2
    : null;
  metric(
    "Team-seasons",
    num(rows.length),
    `${seasons.size} season${seasons.size === 1 ? "" : "s"} in scope`,
  );
  metric("Selected", num(picked.length), "Overall Seed is populated");
  metric(
    "Selection rate",
    rows.length ? pct((100 * picked.length) / rows.length) : "—",
    "Selected / team-seasons in view",
  );
  metric(
    "Selected teams’ median NET",
    median == null ? "—" : num(median, 1),
    `${nets.length} nonmissing NET ranks · lower is better`,
  );
  const scope = `${value("season") === "all" ? "All five seasons" : value("season")} · ${value("conference") === "all" ? "All conferences" : value("conference")}`;
  document.getElementById("scope").textContent =
    scope + " · Supplied training extract, not the complete national field";
  const otherNets = rows
    .filter((r) => !r.selected && r.net != null)
    .map((r) => r.net)
    .sort((a, b) => a - b);
  const otherMedian = otherNets.length
    ? (otherNets[Math.floor((otherNets.length - 1) / 2)] +
        otherNets[Math.floor(otherNets.length / 2)]) /
      2
    : null;
  document.getElementById("finding").textContent = !rows.length
    ? "No team-seasons match these filters. Reset or choose another season and conference."
    : median != null && otherMedian != null
      ? `Selected teams have a median NET rank of ${num(median, 1)}, compared with ${num(otherMedian, 1)} for nonselected teams in this scope (lower is better). This is a descriptive association, not a validated prediction.`
      : "This scope does not contain both selected and nonselected teams with NET ranks. Widen the filters before comparing those groups.";
  if (!rows.length) {
    document
      .getElementById("charts")
      .append(
        el(
          "div",
          "No matching data. No rate is calculated for an empty population.",
          "empty-state",
        ),
      );
    return;
  }
  function groups(key) {
    const map = new Map();
    rows.forEach((r) => {
      if (!map.has(r[key])) map.set(r[key], { total: 0, selected: 0 });
      const g = map.get(r[key]);
      g.total++;
      g.selected += Number(r.selected);
    });
    return [...map].map(([name, g]) => ({
      name,
      ...g,
      rate: (100 * g.selected) / g.total,
    }));
  }
  const years = groups("season").sort((a, b) => a.name.localeCompare(b.name));
  draw({
    title: "Selection rate by season",
    horizontal: true,
    caption:
      "Selected team-seasons / all team-seasons in the filtered extract. Coverage differs from the full tournament field.",
    labels: years.map((r) => `${r.name} (n=${r.total})`),
    series: [{ label: "Selection rate", values: years.map((r) => r.rate) }],
    unit: "% selected",
    max: 100,
    scope,
  });
  const conferences = groups("conference").sort(
    (a, b) => b.rate - a.rate || b.total - a.total,
  );
  draw({
    title: "Conference comparison",
    caption:
      "Top 12 rates in this scope; denominator shown in every label. Small samples can produce extreme rates.",
    labels: conferences
      .slice(0, 12)
      .map((r) => `${r.name || "Unspecified"} (n=${r.total})`),
    series: [
      {
        label: "Selection rate",
        values: conferences.slice(0, 12).map((r) => r.rate),
      },
    ],
    unit: "% selected",
    max: 100,
    horizontal: true,
    scope,
  });
  const ordered = [...rows].sort(
    (a, b) =>
      (a.seed ?? Infinity) - (b.seed ?? Infinity) ||
      a.team.localeCompare(b.team),
  );
  gridTable(
    "Inspect the teams",
    ["Team", "Season", "Conference", "Selected", "Overall seed", "NET rank"],
    ordered
      .slice(0, 50)
      .map((r) => [
        r.team,
        r.season,
        r.conference,
        r.selected ? "Yes" : "No",
        r.seed,
        r.net,
      ]),
    `First ${Math.min(50, rows.length)} of ${rows.length} matching team-seasons, ordered by overall seed. Overall seed is national order, not a 1–16 bracket line. Download includes every matching row.`,
  );
  ordered.forEach((r) =>
    exportRows.push(
      [
        "Team records",
        scope,
        `${r.team} · ${r.season} · ${r.conference}`,
        "Selected",
        Number(r.selected),
        "0/1",
      ],
      [
        "Team records",
        scope,
        `${r.team} · ${r.season} · ${r.conference}`,
        "Overall seed",
        r.seed,
        "national rank",
      ],
      [
        "Team records",
        scope,
        `${r.team} · ${r.season} · ${r.conference}`,
        "NET rank",
        r.net,
        "rank",
      ],
    ),
  );
}

function baby() {
  const source = DATA.charts[1];
  let rows = source.labels.map((name, i) => ({
    name,
    baseline: source.series[0].values[i],
    optimized: source.series[1].values[i],
    penalty: source.series[1].values[i] - source.series[0].values[i],
  }));
  rows = rows.filter(
    (r) => value("product") === "all" || r.name === value("product"),
  );
  rows.sort(
    value("order") === "name"
      ? (a, b) => a.name.localeCompare(b.name)
      : (a, b) => b.penalty - a.penalty || a.name.localeCompare(b.name),
  );
  const baseline = sum(rows.map((r) => r.baseline)),
    optimized = sum(rows.map((r) => r.optimized)),
    penalty = optimized - baseline;
  metric(
    "Products in view",
    num(rows.length),
    "One modeled shipment per product",
  );
  metric(
    "Seven-box billable weight",
    `${num(optimized, 1)} lb`,
    `${num(baseline, 1)} lb independent best fit`,
  );
  metric(
    "Added billable weight",
    `${num(penalty, 2)} lb`,
    "Seven-box solution minus best fit",
  );
  metric(
    "Weight premium",
    pct((100 * penalty) / baseline, 2),
    "Added weight / independent best fit",
  );
  document.getElementById("scope").textContent =
    (value("product") === "all" ? "All 17 products" : value("product")) +
    " · Fixed seven-box solution · Filtering does not rerun the optimizer";
  document.getElementById("finding").textContent =
    value("product") === "all"
      ? "Lotus Travel Crib and City Mini GT2 account for 64.7% of added billable weight. They are the first products to investigate for packaging exceptions; any catalog change needs a new optimization run."
      : penalty > 0.001
        ? `${rows[0].name} adds ${num(penalty, 3)} lb under the shared catalog (${pct((100 * penalty) / baseline, 2)} over its independent best fit). Check shipment volume and carrier rates before translating this into money.`
        : `${rows[0].name} uses no additional billable weight under the seven-box solution. This does not establish a dollar or handling-cost saving.`;
  const scope = value("product") === "all" ? "All products" : value("product");
  draw({
    title: "Where does standardization add weight?",
    caption:
      "Sorted by the selected order. Zero means the shared catalog matches the independent best feasible box.",
    labels: rows.map((r) => r.name),
    series: [
      {
        label: "Additional billable weight",
        values: rows.map((r) => r.penalty),
      },
    ],
    unit: "lb",
    horizontal: true,
    scope,
  });
  draw({
    title: "Shared catalog versus independent best fit",
    caption:
      "Two alternatives for the same products. Billable weight is a cost proxy; shipment frequency is not included.",
    labels: rows.map((r) => r.name),
    series: [
      { label: "Independent best fit", values: rows.map((r) => r.baseline) },
      { label: "Seven-box solution", values: rows.map((r) => r.optimized) },
    ],
    unit: "lb",
    horizontal: true,
    scope,
  });
}

function taxi() {
  const r = DATA.slots[Number(value("slot"))],
    labels = ["A · baseline", "B · fixed fleet", "C · adjustable fleet"];
  const costs = ["total_cost_A", "total_cost_B", "total_cost_C"].map((k) =>
    Number(r[k]),
  );
  const best = Math.min(...costs),
    delta = costs[0] - best;
  const slotName =
    document.getElementById("slot").selectedOptions[0].textContent;
  metric("Baseline modeled cost", money(costs[0]), slotName);
  metric(
    "Lowest saved scenario cost",
    money(best),
    labels[costs.indexOf(best)],
  );
  metric(
    "Difference from baseline",
    money(delta),
    `${pct((delta / costs[0]) * 100, 3)} lower · modeled comparison`,
  );
  metric(
    "Adjustable scenario fleet",
    num(r.F_star),
    "Model C · rounded vehicle equivalent",
  );
  document.getElementById("scope").textContent =
    slotName +
    " · KPI cards and the first chart use this slot; the model benchmark below covers all 35 scenarios.";
  document.getElementById("finding").textContent =
    Number(value("slot")) === 0
      ? "The smaller fleet in scenario C does not produce the lowest total cost in the saved Monday comparison. Compare the full objective, including the rental adjustment, before recommending a fleet change."
      : `The best saved cost is only ${money(delta)} below baseline (${pct((delta / costs[0]) * 100, 3)}). A small modeled difference is not evidence of meaningful realized savings.`;
  draw({
    title: "Selected slot: compare the full cost",
    caption:
      "A uses flat idle costs; B uses borough idle costs. C also adjusts fleet size. Rental adjustment is included in total cost.",
    labels,
    series: [{ label: "Total modeled cost", values: costs }],
    unit: "USD / slot",
    scope: slotName,
  });
  draw({
    ...DATA.charts[0],
    labels: ["Empirical", "LightGBM + opt.", "GLM + opt.", "Decision-focused"],
    scope: "All 35 scenarios",
    caption:
      "Fixed global benchmark: mean missed-trip and idle-vehicle cost across all 35 day × time-window scenarios. Lower is better.",
  });
  const selected = DATA.scenarios.filter((s) => s.model === value("model"));
  const buckets = [...new Set(selected.map((s) => s.bucket))].sort();
  const max = Math.max(...DATA.scenarios.map((s) => s.cost));
  const card = el("article", null, "chart-card wide");
  card.append(
    el("h3", "When is the selected model most exposed?"),
    el(
      "p",
      "Modeled cost by weekday and time window. Darker cells indicate higher costs on a shared scale across all four models; hover for exact USD. This is a separate global run, not the selected stakeholder scenario. Windows differ in length; these totals are not hourly rates.",
    ),
  );
  const wrap = el("div", null, "table-wrap"),
    t = el("table", null, "heatmap");
  t.append(
    el(
      "caption",
      document.getElementById("model").selectedOptions[0].textContent,
    ),
  );
  const head = el("thead"),
    hr = el("tr");
  ["Time window", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"].forEach(
    (s) => {
      const th = el("th", s);
      th.scope = "col";
      hr.append(th);
    },
  );
  head.append(hr);
  t.append(head);
  const body = el("tbody");
  buckets.forEach((bucket) => {
    const tr = el("tr"),
      th = el("th", bucket);
    th.scope = "row";
    tr.append(th);
    for (let day = 1; day <= 7; day++) {
      const row = selected.find((s) => s.bucket === bucket && s.day === day);
      const td = el("td", row ? "$" + num(row.cost / 1000, 1) + "k" : "—");
      if (row) {
        const strength = row.cost / max;
        td.style.background = `rgba(79,70,229,${0.06 + 0.75 * strength})`;
        td.style.color = strength > 0.58 ? "white" : "#20263b";
        td.title = money(row.cost);
        exportRows.push([
          "Model scenario detail",
          value("model"),
          `${bucket} · day ${day}`,
          "Modeled cost",
          row.cost,
          "USD / scenario",
        ]);
      }
      tr.append(td);
    }
    body.append(tr);
  });
  t.append(body);
  wrap.append(t);
  card.append(
    wrap,
    el(
      "p",
      `Shared color scale: $0 to ${money(max)}. Values shown in thousands; source export keeps exact USD.`,
      "heatmap-key",
    ),
  );
  document.getElementById("charts").append(card);
  draw({
    ...DATA.charts[1],
    labels: ["Empirical", "LightGBM + opt.", "GLM + opt.", "Decision-focused"],
    max: 100,
    scope: "All 35 scenarios",
  });
}

function render() {
  instances.forEach((c) => c.destroy());
  instances = [];
  exportRows = [];
  document.getElementById("charts").replaceChildren();
  document.getElementById("metrics").replaceChildren();
  ({ bank, ncaa, baby, taxi })[DATA.kind]();
}

function csvDownload() {
  const safe = (v) => {
    const s = String(v ?? "");
    return '"' + (/^[=+@]/.test(s) ? "'" : "") + s.replaceAll('"', '""') + '"';
  };
  const csv = [
    ["View", "Scope", "Category", "Metric", "Value", "Unit"],
    ...exportRows,
  ]
    .map((r) => r.map(safe).join(","))
    .join("\r\n");
  const url = URL.createObjectURL(
    new Blob([csv], { type: "text/csv;charset=utf-8" }),
  );
  const a = el("a");
  a.href = url;
  a.download = `${DATA.kind}-analysis.csv`;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

fetch("data.json?v=20261102b")
  .then((r) => {
    if (!r.ok) throw Error(`HTTP ${r.status}`);
    return r.json();
  })
  .then((data) => {
    DATA = data;
    const populate = (id, values) => {
      const select = document.getElementById(id);
      values.forEach((v) => {
        const option = el("option", v || "Unspecified");
        option.value = v;
        select.append(option);
      });
    };
    if (DATA.kind === "ncaa") {
      populate(
        "season",
        [...new Set(DATA.records.map((r) => r.season))].sort(),
      );
      populate(
        "conference",
        [...new Set(DATA.records.map((r) => r.conference))].sort(),
      );
    }
    if (DATA.kind === "baby")
      populate("product", [...DATA.charts[1].labels].sort());
    document
      .querySelectorAll(".controls select")
      .forEach((s) => s.addEventListener("change", render));
    document.getElementById("reset").addEventListener("click", () => {
      document
        .querySelectorAll(".controls select")
        .forEach((s) => (s.selectedIndex = 0));
      render();
    });
    document.getElementById("export").textContent = "Download filtered data";
    document.getElementById("export").addEventListener("click", csvDownload);
    render();
  })
  .catch((error) => {
    document.getElementById("scope").textContent =
      "The data could not load. Reload the page or inspect the linked source.";
    console.error(error);
  });
