# Real-Time Manufacturing Defect Inspection System

Detects surface defects (scratches, dents, cracks) on manufactured parts from images,
served via a REST API with a live inspector dashboard. Built to mirror a real
production ML deployment, not just a notebook model.

📄 [Design doc](docs/design_doc.md)

## Status
🚧 In progress — Phase 0 (design) complete. Next: data pipeline.

## Quickstart (once built)
```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
docker-compose up --build
# API: http://localhost:8000/docs
# Dashboard: http://localhost:8501
```

## Architecture
See `docs/design_doc.md` section 6 for the diagram.

## Project structure
See root-level folders: `src/` (pipeline code), `app/` (demo UI),
`deployment/` (Docker + CI/CD), `tests/`, `monitoring/`.
