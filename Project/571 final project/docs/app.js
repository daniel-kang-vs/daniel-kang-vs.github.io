Chart.defaults.color = "#8b8b9e";
Chart.defaults.borderColor = "rgba(255,255,255,0.04)";
Chart.defaults.font.family = "Inter, sans-serif";

new Chart(document.getElementById("classDistChart"), {
  type: "doughnut",
  data: {
    labels: ["Survived", "Bankrupt"],
    datasets: [{ data: [9573, 427], backgroundColor: ["#818cf8", "#f87171"], borderWidth: 0 }]
  },
  options: {
    responsive: true, maintainAspectRatio: false,
    plugins: { legend: { position: "bottom" } }
  }
});

new Chart(document.getElementById("missingChart"), {
  type: "bar",
  data: {
    labels: ["Attr37","Attr21","Attr27","Attr45","Attr60","Attr6","Attr17","Attr35"],
    datasets: [{ data: [34.2, 28.7, 22.1, 18.9, 15.4, 12.8, 11.3, 9.6],
      backgroundColor: "#f87171", borderRadius: 4 }]
  },
  options: {
    indexAxis: "y", responsive: true, maintainAspectRatio: false,
    plugins: { legend: { display: false } },
    scales: { x: { title: { display: true, text: "% Missing" }, max: 40 } }
  }
});

new Chart(document.getElementById("pipelineChart"), {
  type: "bar",
  data: {
    labels: ["Raw", "+ Winsorize", "+ KNN Impute", "+ XGBoost"],
    datasets: [{ label: "CV AUC", data: [0.872, 0.891, 0.902, 0.907],
      backgroundColor: ["#334155", "#6366f1", "#818cf8", "#a5b4fc"], borderRadius: 6 }]
  },
  options: {
    responsive: true, maintainAspectRatio: false,
    plugins: { legend: { display: false } },
    scales: { y: { min: 0.85, max: 0.92, title: { display: true, text: "ROC-AUC" } } }
  }
});

const foldAUCs = [0.9179, 0.9359, 0.8963, 0.9116, 0.8746];
const meanAUC = 0.9073;

new Chart(document.getElementById("foldChart"), {
  type: "bar",
  data: {
    labels: ["F1","F2","F3","F4","F5"],
    datasets: [{ data: foldAUCs,
      backgroundColor: foldAUCs.map(v => v >= meanAUC ? "#818cf8" : "#f87171"), borderRadius: 6 }]
  },
  options: {
    responsive: true, maintainAspectRatio: false,
    plugins: { legend: { display: false } },
    scales: { y: { min: 0.85, max: 0.95 } }
  }
});

new Chart(document.getElementById("leaderboardChart"), {
  type: "bar",
  data: {
    labels: ["1st","2nd","3rd (You)","4th","5th","6th","7th","8th"],
    datasets: [{ data: [0.918, 0.912, 0.907, 0.901, 0.898, 0.894, 0.889, 0.885],
      backgroundColor: ["#fbbf24","#94a3b8","#818cf8","#334155","#334155","#334155","#334155","#334155"],
      borderRadius: 4 }]
  },
  options: {
    indexAxis: "y", responsive: true, maintainAspectRatio: false,
    plugins: { legend: { display: false } },
    scales: { x: { min: 0.87, max: 0.93, title: { display: true, text: "Mean AUC" } } }
  }
});
