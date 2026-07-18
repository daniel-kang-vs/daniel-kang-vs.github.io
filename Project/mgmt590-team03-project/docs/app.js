Chart.defaults.color = "#4a5a6a";
Chart.defaults.borderColor = "rgba(15, 28, 42, 0.08)";
Chart.defaults.font.family = "Figtree, sans-serif";

const SLOTS = {
  "tuesday-midday": {
    label: "Tuesday Midday", time: "09:00–15:59",
    context: "Peak business travel — airports (zones 230, 138) and Midtown absorb the most cabs. Highest-revenue slot in the bake-off.",
    revenue: 425713, nvCost: 831002, fleetUsed: 12978, shadowPrice: null,
    zones: [["230",834],["231",545],["138",496],["244",496],["87",338],["132",336],["243",327],["13",297]],
    borough: { Manhattan: 8350, Queens: 3353, Brooklyn: 898, Bronx: 255, "Staten Island": 144 }
  },
  "friday-evening": {
    label: "Friday Evening", time: "19:00–23:59",
    context: "Nightlife rush — Queens share rises to 21%. Shadow price λ = $25: valuable to add cabs, but below the $45 rental break-even.",
    revenue: 382395, nvCost: 768730, fleetUsed: 13000, shadowPrice: 25.31,
    zones: null,
    borough: { Manhattan: 9160, Queens: 2708, Brooklyn: 860, Bronx: 162, "Staten Island": 110 }
  },
  "monday-morning": {
    label: "Monday Morning", time: "06:00–08:59",
    context: "Commute rush — 82% of cabs stack in Manhattan. λ = 0 means the fleet is oversized; elastic analysis recommends ~8,200 cabs for this slot.",
    revenue: 352052, nvCost: 55773, fleetUsed: 12978, shadowPrice: 0,
    zones: null,
    borough: { Manhattan: 10678, Queens: 1664, Brooklyn: 424, Bronx: 128, "Staten Island": 84 }
  }
};

const BOROUGH_COLORS = ["#0e7490", "#0b1f33", "#2a7a5c", "#c47a3a", "#5b7c99"];

function fmtMoney(n) {
  if (n >= 1e6) return `$${(n / 1e6).toFixed(1)}M`;
  if (n >= 1e3) return `$${Math.round(n / 1000)}K`;
  return `$${Math.round(n)}`;
}

let allocationChart, boroughChart;

function renderSlot(key) {
  const s = SLOTS[key];
  document.getElementById("slotSubtitle").textContent = `${s.label} · ${s.time}`;
  document.getElementById("slotContext").textContent = s.context;
  document.getElementById("slotRevenue").textContent = fmtMoney(s.revenue);
  document.getElementById("slotFleet").textContent = s.fleetUsed.toLocaleString();
  document.getElementById("slotNvCost").textContent = fmtMoney(s.nvCost);
  document.getElementById("slotLambda").textContent = s.shadowPrice === null ? "—" : `$${s.shadowPrice}`;

  const title = document.getElementById("zoneChartTitle");
  if (s.zones) {
    title.textContent = "Top Zone Allocations (q*)";
    allocationChart.data.labels = s.zones.map(z => `Zone ${z[0]}`);
    allocationChart.data.datasets[0].data = s.zones.map(z => z[1]);
  } else {
    title.textContent = "Cabs by Borough";
    const entries = Object.entries(s.borough).sort((a, b) => b[1] - a[1]);
    allocationChart.data.labels = entries.map(e => e[0]);
    allocationChart.data.datasets[0].data = entries.map(e => e[1]);
  }
  allocationChart.update();

  const boro = Object.entries(s.borough).sort((a, b) => b[1] - a[1]);
  const total = boro.reduce((sum, e) => sum + e[1], 0);
  boroughChart.data.labels = boro.map(e => e[0]);
  boroughChart.data.datasets[0].data = boro.map(e => Math.round(e[1] / total * 1000) / 10);
  boroughChart.update();
}

document.querySelectorAll(".slot-btn").forEach(btn => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".slot-btn").forEach(b => b.classList.remove("active"));
    btn.classList.add("active");
    renderSlot(btn.dataset.slot);
  });
});

new Chart(document.getElementById("censoringChart"), {
  type: "bar",
  data: {
    labels: ["Manhattan", "Queens", "Brooklyn", "Bronx", "Staten Is."],
    datasets: [
      { label: "Yellow pickups (observed)", data: [4200, 980, 720, 310, 95], backgroundColor: "#5b7c99" },
      { label: "Estimated demand (FHV proxy)", data: [5100, 2850, 1680, 890, 380], backgroundColor: "#0e7490" }
    ]
  },
  options: {
    responsive: true, maintainAspectRatio: false,
    plugins: { legend: { position: "bottom" } },
    scales: { y: { title: { display: true, text: "Daily pickups (000s, illustrative)" } } }
  }
});

new Chart(document.getElementById("costChart"), {
  type: "bar",
  data: {
    labels: ["Underage (missed fare)", "Overage (idle cab)"],
    datasets: [{ data: [17, 12], backgroundColor: ["#b85c5c", "#2a7a5c"], borderRadius: 8 }]
  },
  options: {
    responsive: true, maintainAspectRatio: false,
    plugins: { legend: { display: false } },
    scales: { y: { title: { display: true, text: "Cost per trip ($)" }, max: 22 } }
  }
});

new Chart(document.getElementById("impactChart"), {
  type: "bar",
  data: {
    labels: ["Per Tuesday", "Annualized (52 wks)"],
    datasets: [
      { label: "Status quo", data: [188, 9776], backgroundColor: "#5b7c99", borderRadius: 6 },
      { label: "Optimized", data: [426, 22152], backgroundColor: "#0e7490", borderRadius: 6 }
    ]
  },
  options: {
    responsive: true, maintainAspectRatio: false,
    plugins: { legend: { position: "top", labels: { color: "rgba(245,248,251,0.75)" } } },
    scales: {
      x: { ticks: { color: "rgba(245,248,251,0.65)" }, grid: { color: "rgba(255,255,255,0.06)" } },
      y: {
        title: { display: true, text: "Revenue ($K)", color: "rgba(245,248,251,0.65)" },
        ticks: { color: "rgba(245,248,251,0.65)" },
        grid: { color: "rgba(255,255,255,0.06)" }
      }
    }
  }
});

new Chart(document.getElementById("bakeoffChart"), {
  type: "bar",
  data: {
    labels: ["Empirical", "LightGBM", "Poisson GLM", "DFL/SPO+"],
    datasets: [{ data: [369468, 383747, 402046, 561423], backgroundColor: ["#2a7a5c", "#0e7490", "#c47a3a", "#b85c5c"], borderRadius: 6 }]
  },
  options: {
    responsive: true, maintainAspectRatio: false,
    plugins: { legend: { display: false } },
    scales: { y: { ticks: { callback: v => `$${(v/1000).toFixed(0)}k` } } }
  }
});

allocationChart = new Chart(document.getElementById("allocationChart"), {
  type: "bar",
  data: { labels: [], datasets: [{ data: [], backgroundColor: "#0e7490", borderRadius: 4 }] },
  options: { indexAxis: "y", responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false } } }
});

boroughChart = new Chart(document.getElementById("boroughChart"), {
  type: "doughnut",
  data: { labels: [], datasets: [{ data: [], backgroundColor: BOROUGH_COLORS, borderWidth: 0 }] },
  options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { position: "bottom" } } }
});

renderSlot("tuesday-midday");
