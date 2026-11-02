# Daniel Kang · Data Analyst Portfolio

Business questions, inspectable analysis, and practical reporting workflows.

**[Live portfolio](https://daniel-kang-vs.github.io/) · [Experience](https://daniel-kang-vs.github.io/journey.html) · [Projects](https://daniel-kang-vs.github.io/projects.html)**

## Focus

SQL and Python analysis · KPI design · Interactive dashboards · Workflow automation · Decision modeling

## Selected projects

- **[Seed Planning Dashboard](projects/seed-demand.html)** — A Python cleaning workflow and interactive dashboard turn inconsistent yearly exports into a view of product mix, stability, and trends. [GitHub](https://github.com/daniel-kang-vs/seed-demand-forecasting).
- **[Athlete Scoring — Sogility](projects/sogility.html)** — A dual scoring framework with public Python implementations, synthetic checks, and documented scoring limitations. [GitHub](https://github.com/daniel-kang-vs/sogility-athlete-scoring).
- **[Packaging Standardization — BabyShip](projects/babyship.html)** — A mixed-integer model chooses shared packaging and product assignments from 43 candidate boxes, making the shipping-weight tradeoff explicit. [GitHub](https://github.com/daniel-kang-vs/stroller-box-optimization).
- **[Bankruptcy Risk Scoring](projects/bankruptcy.html)** — An XGBoost classifier ranks firms using 64 financial ratios and evaluates discrimination with stratified cross-validation. [GitHub](https://github.com/daniel-kang-vs/bankruptcy-prediction).
- **[NCAA Selection & Seed Modeling](projects/ncaa.html)** — A two-stage modeling workflow separates tournament selection from overall seed prediction, with engineered team-performance features. [GitHub](https://github.com/daniel-kang-vs/ncaa-tournament-prediction).
- **[NYC Taxi Fleet Positioning](projects/nyc-taxi.html)** — A newsvendor allocation workflow compares demand models by decision cost and distributes a constrained fleet across NYC taxi zones. [GitHub](https://github.com/daniel-kang-vs/nyc-taxi-fleet-optimization).

## Evidence standards

Every case study identifies the question, data scope, method, results and limitations. Synthetic demos, recorded model outputs and historical reported metrics are labeled separately. Internal workplace data and private dashboards are not reconstructed as if they were original artifacts.

## Local preview

```bash
python3 -m http.server 8765
```

Open http://127.0.0.1:8765/index.html. The website is static HTML and CSS. Interactive project dashboards are maintained in their project repositories; the taxi dashboard is built from `Project/mgmt590-team03-project/docs/`.

## Repository guide

- `index.html`: introduction, capabilities and featured work.
- `journey.html`: experience and education.
- `projects.html` and `projects/`: project index and six case studies.
- `style.css`: shared bright visual theme and responsive layout.
- `pngs/`: portfolio preview images.
- `.github/workflows/deploy-pages.yml`: GitHub Pages deployment.
- `PROJECT_EVIDENCE.md`: source and interpretation notes for public projects.

## Publishing

The Pages workflow publishes the selected site files and taxi dashboard assets. A successful push is followed by a successful deployment and a live-page check. Local research folders are not all part of the published website.
