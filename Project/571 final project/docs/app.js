Chart.defaults.color = "#8b8b9e";
Chart.defaults.borderColor = "rgba(255,255,255,0.04)";
Chart.defaults.font.family = "Inter, sans-serif";

// Class Distribution (donut)
new Chart(document.getElementById("classDistChart"), {
  type: "doughnut",
  data: {
    labels: ["Survived (class=0)", "Bankrupt (class=1)"],
    datasets: [{
      data: [9573, 427],
      backgroundColor: ["#818cf8", "#f87171"],
      borderWidth: 0
    }]
  },
  options: {
    responsive: true, maintainAspectRatio: false,
    plugins: {
      legend: { position: "bottom" },
      tooltip: {
        callbacks: {
          label: ctx => `${ctx.label}: ${ctx.raw.toLocaleString()} (${(ctx.raw/10000*100).toFixed(1)}%)`
        }
      }
    }
  }
});

// Missing Values (bar - top 15 features with most missing)
const missingFeats = ["Attr37","Attr21","Attr27","Attr45","Attr60","Attr6","Attr17","Attr35","Attr44","Attr14","Attr5","Attr29","Attr58","Attr41","Attr12"];
const missingPcts = [34.2, 28.7, 22.1, 18.9, 15.4, 12.8, 11.3, 9.6, 8.2, 7.1, 5.8, 4.3, 3.1, 2.4, 1.8];

new Chart(document.getElementById("missingChart"), {
  type: "bar",
  data: {
    labels: missingFeats,
    datasets: [{
      label: "% Missing",
      data: missingPcts,
      backgroundColor: missingPcts.map(v => v > 20 ? "#f87171" : v > 10 ? "#fbbf24" : "#818cf8"),
      borderRadius: 4
    }]
  },
  options: {
    indexAxis: "y",
    responsive: true, maintainAspectRatio: false,
    plugins: { legend: { display: false } },
    scales: {
      x: { title: { display: true, text: "% Missing Values" }, max: 40 },
      y: { ticks: { font: { size: 10 } } }
    }
  }
});

// AUC by Fold (bar)
const foldLabels = ["Fold 1", "Fold 2", "Fold 3", "Fold 4", "Fold 5"];
const foldAUCs = [0.9179, 0.9359, 0.8963, 0.9116, 0.8746];
const meanAUC = 0.9073;

new Chart(document.getElementById("foldChart"), {
  type: "bar",
  data: {
    labels: foldLabels,
    datasets: [{
      label: "AUC",
      data: foldAUCs,
      backgroundColor: foldAUCs.map(v => v >= meanAUC ? "#818cf8" : "#f87171"),
      borderRadius: 6
    }]
  },
  options: {
    responsive: true, maintainAspectRatio: true,
    plugins: {
      legend: { display: false },
      annotation: {}
    },
    scales: {
      y: {
        min: 0.85, max: 0.95,
        title: { display: true, text: "ROC-AUC" }
      }
    }
  }
});

// AUC Radar Chart
new Chart(document.getElementById("aucRadarChart"), {
  type: "radar",
  data: {
    labels: foldLabels,
    datasets: [{
      label: "AUC Score",
      data: foldAUCs,
      backgroundColor: "rgba(129,140,248,0.15)",
      borderColor: "#818cf8",
      borderWidth: 2,
      pointBackgroundColor: foldAUCs.map(v => v >= meanAUC ? "#818cf8" : "#f87171"),
      pointRadius: 5
    }, {
      label: "Mean (0.907)",
      data: Array(5).fill(meanAUC),
      backgroundColor: "transparent",
      borderColor: "#fbbf24",
      borderWidth: 1.5,
      borderDash: [6, 3],
      pointRadius: 0
    }]
  },
  options: {
    responsive: true, maintainAspectRatio: true,
    plugins: { legend: { position: "bottom" } },
    scales: {
      r: {
        min: 0.85, max: 0.95,
        ticks: { stepSize: 0.02, color: "#8b8b9e", backdropColor: "transparent" },
        grid: { color: "#2a2a35" },
        angleLines: { color: "#2a2a35" },
        pointLabels: { color: "#8b8b9e", font: { size: 11 } }
      }
    }
  }
});
