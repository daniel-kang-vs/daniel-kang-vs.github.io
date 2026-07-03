const strollers = [
  "Triv Next","Vista V2","Cruz V2","Fox 3","Fox 5","Fox 5 (2pc)",
  "Mixx Next","City Mini GT2","Single-to-Dbl","Portable Playard",
  "Close2Baby","Breeze Plus","Swivel Sleeper","Smart Sleeper",
  "Lullaby Playard","Travel Crib Light","Lotus Travel Crib"
];

const optimized = [62.208,256,82.944,62.208,256,62.208,62.72,124.416,124.416,43.2,62.208,43.2,124.416,144,82.944,62.72,62.208];
const perfect   = [53.76,256,69.12,62.208,256,62.208,62.72,93.312,124.416,43.2,62.208,43.2,124.416,144,82.944,48,25.6];
const penalty   = optimized.map((v,i) => +(v - perfect[i]).toFixed(2));

const boxAssignments = { "Box 6": 2, "Box 8": 2, "S-16765": 5, "S-4478": 3, "S-4659": 2, "S-4866": 2, "S-16766": 1 };
const dimWeights = optimized;
const actualWeights = [18.4,27,25.5,21.8,21.1,28,28.3,22.4,27,14,30,23,32,38,25,13,14];
const strollerDims = {
  L: [24,17.3,16.5,35,17.3,35,27,31,34,10.31,30,12,34,38,10.5,5,24],
  W: [22.5,25.7,22.8,21,23.6,20.9,23.6,25.5,24.5,10.31,18,12,34,22,19,23.5,12],
  H: [15,33,29.5,15,35.4,13.4,19,10.5,18,27.99,10,30,23,15.5,30,19,8]
};

Chart.defaults.color = "#94a3b8";
Chart.defaults.borderColor = "rgba(255,255,255,0.05)";
Chart.defaults.font.family = "Inter, sans-serif";

new Chart(document.getElementById("strollerDimChart"), {
  type: "bar",
  data: {
    labels: strollers,
    datasets: [
      { label: "L", data: strollerDims.L, backgroundColor: "#34d399" },
      { label: "W", data: strollerDims.W, backgroundColor: "#60a5fa" },
      { label: "H", data: strollerDims.H, backgroundColor: "#f59e0b" }
    ]
  },
  options: {
    responsive: true, maintainAspectRatio: false,
    plugins: { legend: { position: "top" } },
    scales: { x: { ticks: { maxRotation: 45, font: { size: 10 } } }, y: { title: { display: true, text: "Inches" } } }
  }
});

new Chart(document.getElementById("dwVsActualChart"), {
  type: "bar",
  data: {
    labels: strollers,
    datasets: [
      { label: "Billable (DIM)", data: dimWeights, backgroundColor: "rgba(96,165,250,0.8)" },
      { label: "Actual weight", data: actualWeights, backgroundColor: "rgba(148,163,184,0.5)" }
    ]
  },
  options: {
    responsive: true, maintainAspectRatio: false,
    plugins: { legend: { position: "top" } },
    scales: { x: { ticks: { maxRotation: 45, font: { size: 9 } } }, y: { title: { display: true, text: "lb" } } }
  }
});

new Chart(document.getElementById("skuChart"), {
  type: "bar",
  data: {
    labels: ["Box SKUs", "Warehouse complexity"],
    datasets: [
      { label: "Before (1:1)", data: [17, 100], backgroundColor: "#ef4444", borderRadius: 6 },
      { label: "After (optimized)", data: [7, 41], backgroundColor: "#34d399", borderRadius: 6 }
    ]
  },
  options: {
    responsive: true, maintainAspectRatio: false,
    plugins: { legend: { position: "top" } },
    scales: { y: { title: { display: true, text: "Count / Index" } } }
  }
});

new Chart(document.getElementById("utilizationChart"), {
  type: "doughnut",
  data: {
    labels: Object.keys(boxAssignments),
    datasets: [{ data: Object.values(boxAssignments),
      backgroundColor: ["#fb923c","#ef4444","#34d399","#f59e0b","#60a5fa","#a78bfa","#f472b6"], borderWidth: 0 }]
  },
  options: {
    responsive: true, maintainAspectRatio: false,
    plugins: { legend: { position: "right" } }
  }
});

new Chart(document.getElementById("comparisonChart"), {
  type: "bar",
  data: {
    labels: strollers,
    datasets: [
      { label: "7-box system", data: optimized, backgroundColor: "#34d399" },
      { label: "Perfect fit", data: perfect, backgroundColor: "#f59e0b" }
    ]
  },
  options: {
    responsive: true, maintainAspectRatio: false,
    plugins: { legend: { position: "top" } },
    scales: { x: { ticks: { maxRotation: 45, font: { size: 9 } } }, y: { title: { display: true, text: "Ship weight (lb)" } } }
  }
});

new Chart(document.getElementById("penaltyChart"), {
  type: "bar",
  data: {
    labels: strollers,
    datasets: [{ data: penalty, backgroundColor: penalty.map(v => v > 20 ? "#ef4444" : v > 0 ? "#f59e0b" : "#334155") }]
  },
  options: {
    responsive: true, maintainAspectRatio: false,
    plugins: { legend: { display: false } },
    scales: { x: { ticks: { maxRotation: 45, font: { size: 9 } } }, y: { title: { display: true, text: "Extra lb vs perfect" } } }
  }
});
