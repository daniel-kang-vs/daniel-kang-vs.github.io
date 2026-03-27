// ── Data ──
const strollers = [
  "Triv Next","Vista V2","Cruz V2","Fox 3","Fox 5","Fox 5 (2pc)",
  "Mixx Next","City Mini GT2","Single-to-Dbl","Portable Playard",
  "Close2Baby","Breeze Plus","Swivel Sleeper","Smart Sleeper",
  "Lullaby Playard","Travel Crib Light","Lotus Travel Crib"
];

const optimized = [62.208,256,82.944,62.208,256,62.208,62.72,124.416,124.416,43.2,62.208,43.2,124.416,144,82.944,62.72,62.208];
const perfect   = [53.76,256,69.12,62.208,256,62.208,62.72,93.312,124.416,43.2,62.208,43.2,124.416,144,82.944,48,25.6];
const penalty   = optimized.map((v,i) => +(v - perfect[i]).toFixed(2));

const boxAssignments = {
  "Box 6": 2, "Box 8": 2, "S-16765": 5, "S-4478": 3,
  "S-4659": 2, "S-4866": 2, "S-16766": 1
};

const dimWeights = [62.208,256,82.944,62.208,256,62.208,62.72,124.416,124.416,43.2,62.208,43.2,124.416,144,82.944,62.72,62.208];
const actualWeights = [18.4,27,25.5,21.8,21.1,28,28.3,22.4,27,14,30,23,32,38,25,13,14];

const strollerDims = {
  L: [24,17.3,16.5,35,17.3,35,27,31,34,10.31,30,12,34,38,10.5,5,24],
  W: [22.5,25.7,22.8,21,23.6,20.9,23.6,25.5,24.5,10.31,18,12,34,22,19,23.5,12],
  H: [15,33,29.5,15,35.4,13.4,19,10.5,18,27.99,10,30,23,15.5,30,19,8]
};

// ── Box data for table ──
const boxes = [
  ["Walmart Small",17,11,12],["Walmart Medium",22,13,15],["Walmart Large",27,15,17],["Walmart Ext. Large",26,18,18],
  ["1",10,10,10],["2",12,12,6],["3",12,12,12],["4",14,14,14],["5",15,12,10],["6",15,15,48],["7",16,16,4],
  ["8",40,40,40],["9",18,18,18],["10",20,12,12],["11",20,20,12],["12",20,20,20],["13",24,18,6],["14",24,18,18],
  ["15",24,24,16],["16",24,24,24],["17",30,24,6],["18",18,18,16],["19",24,18,18],["20",16,12,12],["21",22,22,21.5],
  ["S-16765",36,24,18],["S-4478",36,36,24],["S-4659",24,24,36],["S-4739",24,20,16],["S-4803",24,20,18],
  ["S-21038",24,24,30],["S-15043",25,16,16],["S-4963",26,26,20],["S-4190",26,26,26],["S-22660",28,24,12],
  ["S-16785",28,24,20],["S-4806",28,20,24],["S-4866",28,28,20],["S-4229",30,20,20],["S-4417",30,24,12],
  ["S-4509",30,24,24],["S-16766",40,30,30],["S-4457",36,36,18]
];

const boxBody = document.getElementById("box-table-body");
boxes.forEach(b => {
  const dw = ((b[1]*b[2]*b[3])/250).toFixed(1);
  const row = document.createElement("tr");
  row.innerHTML = `<td>${b[0]}</td><td>${b[1]}</td><td>${b[2]}</td><td>${b[3]}</td><td>${dw}</td>`;
  boxBody.appendChild(row);
});

// ── Tabs ──
document.querySelectorAll(".tab").forEach(tab => {
  tab.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach(t => t.classList.remove("active"));
    document.querySelectorAll(".tab-content").forEach(c => c.classList.remove("active"));
    tab.classList.add("active");
    document.getElementById(tab.dataset.tab).classList.add("active");
  });
});

// ── Chart defaults ──
Chart.defaults.color = "#94a3b8";
Chart.defaults.borderColor = "rgba(255,255,255,0.05)";
Chart.defaults.font.family = "Inter, sans-serif";

// ── Stroller Dimensions Chart ──
new Chart(document.getElementById("strollerDimChart"), {
  type: "bar",
  data: {
    labels: strollers,
    datasets: [
      { label: "Length", data: strollerDims.L, backgroundColor: "#34d399" },
      { label: "Width",  data: strollerDims.W, backgroundColor: "#60a5fa" },
      { label: "Height", data: strollerDims.H, backgroundColor: "#f59e0b" }
    ]
  },
  options: {
    responsive: true,
    plugins: { legend: { position: "top" } },
    scales: {
      x: { ticks: { maxRotation: 45, minRotation: 45, font: { size: 11 } } },
      y: { title: { display: true, text: "Inches" } }
    }
  }
});

// ── Comparison Chart ──
new Chart(document.getElementById("comparisonChart"), {
  type: "bar",
  data: {
    labels: strollers,
    datasets: [
      { label: "7-Box System", data: optimized, backgroundColor: "#34d399" },
      { label: "Perfect Box",  data: perfect,   backgroundColor: "#f59e0b", borderRadius: 3 }
    ]
  },
  options: {
    responsive: true, maintainAspectRatio: true,
    plugins: { legend: { position: "top" } },
    scales: {
      x: { ticks: { maxRotation: 50, minRotation: 50, font: { size: 10 } } },
      y: { title: { display: true, text: "Shipping Weight (lb)" } }
    }
  }
});

// ── Penalty Chart ──
new Chart(document.getElementById("penaltyChart"), {
  type: "bar",
  data: {
    labels: strollers,
    datasets: [{
      label: "Extra Weight",
      data: penalty,
      backgroundColor: penalty.map(v => v > 0 ? "#ef4444" : "#334155")
    }]
  },
  options: {
    responsive: true, maintainAspectRatio: true,
    plugins: { legend: { display: false } },
    scales: {
      x: { ticks: { maxRotation: 50, minRotation: 50, font: { size: 10 } } },
      y: { title: { display: true, text: "Extra Weight (lb)" } }
    }
  }
});

// ── Utilization Donut ──
const boxLabels = Object.keys(boxAssignments);
const boxCounts = Object.values(boxAssignments);
const boxColors = ["#fb923c","#ef4444","#34d399","#f59e0b","#60a5fa","#a78bfa","#f472b6"];

new Chart(document.getElementById("utilizationChart"), {
  type: "doughnut",
  data: {
    labels: boxLabels,
    datasets: [{ data: boxCounts, backgroundColor: boxColors, borderWidth: 0 }]
  },
  options: {
    responsive: true, maintainAspectRatio: false,
    plugins: {
      legend: { position: "right" },
      tooltip: { callbacks: { label: ctx => `${ctx.label}: ${ctx.raw} product${ctx.raw>1?'s':''}` } }
    }
  }
});

// ── DW vs Actual Weight ──
new Chart(document.getElementById("dwVsActualChart"), {
  type: "bar",
  data: {
    labels: strollers,
    datasets: [
      { label: "Dim. Weight (box)", data: dimWeights, backgroundColor: "rgba(96,165,250,0.7)" },
      { label: "Actual Weight",     data: actualWeights, backgroundColor: "rgba(239,68,68,0.7)" }
    ]
  },
  options: {
    responsive: true, maintainAspectRatio: false,
    plugins: {
      legend: { position: "top" },
      tooltip: {
        callbacks: {
          afterBody: (items) => {
            const idx = items[0].dataIndex;
            const dw = dimWeights[idx], aw = actualWeights[idx];
            return `Billed: ${Math.max(dw,aw).toFixed(1)} lb (${dw > aw ? 'dim weight' : 'actual weight'})`;
          }
        }
      }
    },
    scales: {
      x: { ticks: { maxRotation: 50, minRotation: 50, font: { size: 10 } } },
      y: { title: { display: true, text: "Weight (lb)" } }
    }
  }
});
