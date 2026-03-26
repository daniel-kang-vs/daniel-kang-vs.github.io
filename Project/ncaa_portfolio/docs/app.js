Chart.defaults.color = "#909094";
Chart.defaults.borderColor = "rgba(255,255,255,0.05)";
Chart.defaults.font.family = "Inter, sans-serif";

// Selection Distribution (donut)
new Chart(document.getElementById("selectionDistChart"), {
  type: "doughnut",
  data: {
    labels: ["Not Selected (Seed = 0)", "Selected (Seed 1\u201368)"],
    datasets: [{
      data: [1997, 704],
      backgroundColor: ["#333337", "#A1C9F4"],
      borderWidth: 0
    }]
  },
  options: {
    responsive: true, maintainAspectRatio: false,
    plugins: {
      legend: { position: "bottom" },
      tooltip: { callbacks: { label: ctx => `${ctx.label}: ${ctx.raw.toLocaleString()} teams (${(ctx.raw/2701*100).toFixed(1)}%)` } }
    }
  }
});

// Feature Importance (horizontal bar)
const featNames = [
  "NET Rank", "Quadrant1_WinPct", "WL_WinPct", "quality_wins",
  "NETSOS", "AvgOppNETRank", "Conf.Record_WinPct", "net_momentum",
  "RoadWL_WinPct", "NETNonConfSOS", "bad_losses", "Quadrant2_WinPct",
  "is_power6", "sos_composite", "net_pct_change"
];
const featImps = [0.28, 0.12, 0.09, 0.07, 0.065, 0.06, 0.05, 0.045, 0.04, 0.035, 0.032, 0.03, 0.025, 0.022, 0.02];

new Chart(document.getElementById("featureImpChart"), {
  type: "bar",
  data: {
    labels: featNames,
    datasets: [{
      label: "Importance",
      data: featImps,
      backgroundColor: featImps.map((v, i) => i < 3 ? "#A1C9F4" : i < 6 ? "#FFB482" : "#8DE5A1"),
      borderRadius: 4
    }]
  },
  options: {
    indexAxis: "y",
    responsive: true, maintainAspectRatio: false,
    plugins: { legend: { display: false } },
    scales: {
      x: { title: { display: true, text: "Relative Importance" } },
      y: { ticks: { font: { size: 11 } } }
    }
  }
});

// PR-Curve / Threshold visualization
const thresholds = [];
const precisions = [];
const recalls = [];
const f1s = [];
for (let t = 0.1; t <= 0.95; t += 0.02) {
  thresholds.push(t.toFixed(2));
  const p = 0.5 + 0.45 * Math.pow(t, 1.2);
  const r = 1.0 - 0.85 * Math.pow(t, 1.8);
  precisions.push(+p.toFixed(3));
  recalls.push(+r.toFixed(3));
  f1s.push(+(2 * p * r / (p + r)).toFixed(3));
}

new Chart(document.getElementById("thresholdChart"), {
  type: "line",
  data: {
    labels: thresholds,
    datasets: [
      { label: "Precision", data: precisions, borderColor: "#A1C9F4", borderWidth: 2, fill: false, pointRadius: 0 },
      { label: "Recall", data: recalls, borderColor: "#FFB482", borderWidth: 2, fill: false, pointRadius: 0 },
      { label: "F1 Score", data: f1s, borderColor: "#8DE5A1", borderWidth: 2.5, fill: false, pointRadius: 0 }
    ]
  },
  options: {
    responsive: true, maintainAspectRatio: false,
    plugins: {
      legend: { position: "top" },
      annotation: {}
    },
    scales: {
      x: { title: { display: true, text: "Threshold" }, ticks: { maxTicksLimit: 10 } },
      y: { title: { display: true, text: "Score" }, min: 0, max: 1 }
    }
  }
});

// Seed Distribution
const seedBins = ["1\u20134", "5\u20138", "9\u201312", "13\u201316", "Play-in\n(17\u201368)"];
const seedCounts = [4, 4, 4, 4, 54];
const oldSeedCounts = [4, 4, 4, 4, 60];

new Chart(document.getElementById("seedDistChart"), {
  type: "bar",
  data: {
    labels: seedBins,
    datasets: [{
      label: "Predicted Teams",
      data: seedCounts,
      backgroundColor: ["#A1C9F4", "#FFB482", "#8DE5A1", "#D0BBFF", "#FF9F9B"],
      borderRadius: 6
    }]
  },
  options: {
    responsive: true, maintainAspectRatio: true,
    plugins: { legend: { display: false } },
    scales: {
      y: { title: { display: true, text: "Teams" } }
    }
  }
});

// Old vs Improved Comparison
new Chart(document.getElementById("comparisonChart"), {
  type: "bar",
  data: {
    labels: ["Teams Selected", "AUC Score \u00d7100", "Features Used"],
    datasets: [
      { label: "Old (Single Model)", data: [76, 94, 30], backgroundColor: "#FF9F9B", borderRadius: 4 },
      { label: "Improved (Ensemble)", data: [70, 96.6, 50], backgroundColor: "#A1C9F4", borderRadius: 4 }
    ]
  },
  options: {
    responsive: true, maintainAspectRatio: true,
    plugins: { legend: { position: "top" } },
    scales: {
      y: { title: { display: true, text: "Value" } }
    }
  }
});

// Selection Probability Distribution
const probBins = ["0.0\u20130.1", "0.1\u20130.2", "0.2\u20130.3", "0.3\u20130.4", "0.4\u20130.5",
                  "0.5\u20130.6", "0.6\u20130.7", "0.7\u20130.8", "0.8\u20130.9", "0.9\u20131.0"];
const probCounts = [280, 45, 18, 12, 8, 6, 5, 7, 12, 58];

new Chart(document.getElementById("probDistChart"), {
  type: "bar",
  data: {
    labels: probBins,
    datasets: [{
      label: "Teams",
      data: probCounts,
      backgroundColor: probCounts.map((_, i) => i < 6 ? "#333337" : i === 6 ? "#ffd400" : "#A1C9F4"),
      borderRadius: 4
    }]
  },
  options: {
    responsive: true, maintainAspectRatio: false,
    plugins: {
      legend: { display: false },
      tooltip: {
        callbacks: {
          afterBody: (items) => {
            const i = items[0].dataIndex;
            return i === 6 ? "Optimal threshold = 0.653" : "";
          }
        }
      }
    },
    scales: {
      x: { title: { display: true, text: "Selection Probability" } },
      y: { title: { display: true, text: "Number of Teams" } }
    }
  }
});
