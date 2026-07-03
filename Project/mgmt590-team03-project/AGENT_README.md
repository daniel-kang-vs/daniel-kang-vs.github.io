# NYC Taxi Optimization Agent

Phase-2 agent layer: LangGraph orchestration + Streamlit UI over the optimization engine.

## What you need to provide

### 1. OpenAI API key

```bash
cp .env.example .env
# Edit .env and set:
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini
```

Without it, the agent runs in **heuristic mode** (keyword-based intent + config parsing, no executive summary).

**Never commit real API keys** — `.env` is gitignored.

### 2. Primary data file (required for Fast Path)

Place trip-level parquet at project root:

```
cleaned_trips_2025.parquet
```

The agent builds an engine-ready weekly aggregate on first run and caches it at:

```
agent_runs/cache/agg_from_cleaned_trips_2025.parquet
```

`updated_2025_agg.parquet` is **not** required for the agent path.

## Install

```bash
python3 -m pip install -r requirements.txt
```

## Run UI

```bash
./run_streamlit.sh
# or double-click: Launch NYC Taxi Agent.command
# or: bash scripts/launch_agent.sh
```

Uses Anaconda Python when available (required for LightGBM on macOS).

## Architecture

| Path | Trigger | Action | Report |
|------|---------|--------|--------|
| **Fast** | constraint change, what-if | `run_pipeline()` on cached/overlay agg | Single scenario |
| **Slow** | new data, full bake-off | `run_results.py` (Confirm required) | Full bake-off |

## Temporary overlays (no canonical file overwrites)

New constraints or uploaded data are stored under:

```
agent_runs/overlays/{session_id}/
```

- Config patches → `config.json`
- Uploads → `uploads/`
- Derived agg → `agg_overlay.parquet`

`cleaned_trips_2025.parquet` is **never** modified.

After a run with overlays, the UI asks:

- **Save permanently** → copies to `agent_runs/canonical/` (config + agg baseline for future sessions)
- **Discard overlay** → deletes session overlay, reverts to baseline

## Project layout

```
agent/
  graph.py         # LangGraph workflow
  tools.py         # run_pipeline, run_slow_bakeoff
  extract.py       # OpenAI SDK / heuristic NL parsing (+ document context)
  documents.py     # Multimodal ingest: PDF, DOCX, images, text
  llm.py           # OpenAI client (no LangChain)
  data_source.py   # cleaned_trips → engine agg (DuckDB cache)
  overlays.py      # Temp storage + permanent promote
  report.py        # Markdown reports
  catalog.py       # Data metadata only (no row reads)
  registry.py      # Run log (agent_runs/registry.jsonl)
app/
  streamlit_app.py
```

## Example prompts

- `Fleet size 11000 on Tuesday peak — run allocation`
- Upload `samples/management_scenario.md` + `Apply this scenario and run allocation`
- Attach a PDF memo with constraints + `Run what-if from the attached document`
- `Show last report`
- `Run full bake-off` → Confirm card appears
- Upload new trip parquet in Config panel → temp overlay → Save permanently or Discard

## Multimodal document support

The agent reads **text and images** from chat attachments:

| Format | Handling |
|--------|----------|
| PDF, DOCX, TXT, MD | Text extracted locally, sent to LLM |
| PNG, JPG, WEBP | Vision API (requires OpenAI key + gpt-4o-mini) |
| Parquet, CSV | Routed to trip-data overlay (same as Config panel) |
| XLSX | First 30 rows previewed as text |

Attach files in the chat uploader, then send a message. Brief prompts like
`Apply the attached document` work when the constraints are in the file.
