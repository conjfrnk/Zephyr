# Zephyr Webapp Improvement + AWS Deployment Design

## Bug Fixes
1. `_generate_two_leg_loops` scoping bug - only iterates last radius_range's intermediate nodes
2. `showLoading("Initializing...")` missing container ID
3. `enrich_elevations` passes global `G` instead of `loaded_graph`
4. Weather fetches on every map move with no debounce
5. Fragile selectedIndex calculation in route selection
6. Thread safety: global `G` accessed without lock in routing

## Modularization (app.py -> package)
```
zephyr/
  __init__.py          # Flask app factory
  models.py            # SQLAlchemy models (Run, DoneEdge, Pref)
  graph.py             # Graph loading, elevation, publishing
  routing.py           # Route generation strategies + scoring
  weather.py           # Weather API integration
  helpers.py           # Distance calc, utility functions
  routes/
    __init__.py
    api.py             # JSON API endpoints
    views.py           # HTML template routes
```

## Frontend Improvements
- Mobile-responsive layout with collapsible sidebar
- Better route option cards (ascent/descent, strategy type)
- Debounced weather updates
- Smooth loading states
- Viewport meta tag

## Backend Improvements
- Gunicorn for production
- Health check endpoint
- Pin dependency versions
- Remove unused deps (rasterio, scikit-learn)

## AWS Deployment (zephyrmap.com)
- Docker multi-stage build
- ECS Fargate (0.25 vCPU, 0.5GB, auto-scale 1-3)
- ALB with ACM certificate
- EFS for SQLite + graph cache
- Route 53 A record -> ALB
- CloudFormation IaC
