const colors = ["#4f46e5", "#0891b2", "#b45309"];
Chart.defaults.color = "#596277";
Chart.defaults.font.family = "Figtree, sans-serif";
Chart.defaults.borderColor = "#e6e9f2";
let charts = [];
function draw(spec, id) {
  const card = document.createElement("article");
  card.className = "chart-card" + (spec.labels.length > 8 ? " wide" : "");
  const h = document.createElement("h3");
  h.textContent = spec.title;
  const p = document.createElement("p");
  p.textContent = spec.caption;
  card.append(h, p);
  const box = document.createElement("div");
  box.className = "chart-box";
  if (spec.labels.length > 8)
    box.style.height = Math.max(330, spec.labels.length * 30) + "px";
  const canvas = document.createElement("canvas");
  canvas.id = id;
  canvas.setAttribute("role", "img");
  canvas.setAttribute(
    "aria-label",
    spec.title + ". Exact values are available in the table below.",
  );
  box.append(canvas);
  card.append(box);
  const details = document.createElement("details");
  const summary = document.createElement("summary");
  summary.textContent = "View exact values";
  details.append(summary);
  const wrap = document.createElement("div");
  wrap.className = "table-wrap";
  const table = document.createElement("table");
  const cap = document.createElement("caption");
  cap.textContent = spec.title + " (" + spec.unit + ")";
  table.append(cap);
  const head = document.createElement("thead"),
    hr = document.createElement("tr");
  ["Category", ...spec.series.map((s) => s.label)].forEach((v) => {
    const th = document.createElement("th");
    th.scope = "col";
    th.textContent = v;
    hr.append(th);
  });
  head.append(hr);
  table.append(head);
  const body = document.createElement("tbody");
  spec.labels.forEach((v, i) => {
    const tr = document.createElement("tr");
    [v, ...spec.series.map((s) => s.values[i])].forEach((x, j) => {
      const td = document.createElement(j === 0 ? "th" : "td");
      if (j === 0) td.scope = "row";
      td.textContent =
        typeof x === "number"
          ? x.toLocaleString(undefined, { maximumFractionDigits: 4 })
          : x;
      tr.append(td);
    });
    body.append(tr);
  });
  table.append(body);
  wrap.append(table);
  details.append(wrap);
  card.append(details);
  document.getElementById("charts").append(card);
  charts.push(
    new Chart(canvas, {
      type: "bar",
      data: {
        labels: spec.labels,
        datasets: spec.series.map((s, i) => ({
          label: s.label,
          data: s.values,
          backgroundColor: colors[i % colors.length],
          borderRadius: 3,
        })),
      },
      options: {
        animation: false,
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: spec.series.length > 1, position: "bottom" },
        },
        scales: {
          x: {
            ticks: { maxRotation: 65, autoSkip: false, font: { size: 11 } },
          },
          y: {
            beginAtZero: true,
            title: { display: true, text: spec.unit },
            ...(spec.unit === "ROC-AUC" ? { max: 1 } : {}),
            ...(spec.unit.startsWith("%") ? { max: 100 } : {}),
          },
        },
      },
    }),
  );
}
function render() {
  charts.forEach((c) => c.destroy());
  charts = [];
  document.getElementById("charts").replaceChildren();
  const specs = [...DATA.charts];
  if (DATA.slots) {
    const r = DATA.slots[document.getElementById("slot").value];
    specs.push({
      title: "Total modeled cost for selected slot",
      caption:
        "A: flat idle cost, fixed fleet. B: borough idle cost, fixed fleet. C: borough idle cost, adjustable fleet; total includes rental adjustment.",
      labels: ["A · baseline", "B · fixed fleet", "C · adjustable fleet"],
      series: [
        {
          label: "Total modeled cost",
          values: ["total_cost_A", "total_cost_B", "total_cost_C"].map((k) =>
            Number(r[k]),
          ),
        },
      ],
      unit: "USD / slot",
    });
    document.getElementById("slotStatus").textContent =
      "Model C selected fleet: " +
      Number(r.F_star).toLocaleString(undefined, { maximumFractionDigits: 0 }) +
      " vehicles. " +
      (r.direction.includes("contract") ? "Smaller fleet selected." : "Current fleet retained.");
  }
  const choice = document.getElementById("view").value;
  specs.forEach((s, i) => {
    if (choice === "all" || Number(choice) === i) draw(s, "chart" + i);
  });
}
fetch("data.json")
  .then((r) => {
    if (!r.ok) throw Error("HTTP " + r.status);
    return r.json();
  })
  .then((data) => {
    window.DATA = data;
    const view = document.getElementById("view");
    [...data.charts, ...(data.slots ? [{title: "Selected slot cost comparison"}] : [])].forEach((s, i) => {
      const o = document.createElement("option");
      o.value = i;
      o.textContent = s.title;
      view.append(o);
    });
    view.addEventListener("change", render);
    document.getElementById("slot")?.addEventListener("change", render);
    render();
  })
  .catch((e) => {
    document.getElementById("charts").textContent =
      "The analysis could not load. Please reload or inspect the source data using the link above.";
    console.error(e);
  });
