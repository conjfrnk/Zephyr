"""
Zephyr - Smart route planner for runners
Copyright (C) 2025 Connor Frank
License: GPLv3 (see LICENSE)
"""

import logging
import os
from functools import wraps

from flask import Flask, current_app, jsonify, request
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

from .models import db, Pref, ensure_done_edge_constraints

logger = logging.getLogger(__name__)

limiter = Limiter(key_func=get_remote_address, default_limits=["200/hour"])


def require_admin_token(view_func):
    """Gate a view behind an admin token when ADMIN_TOKEN is configured.

    Open mode (token unset) intentionally lets all callers through so existing
    deployments without secrets keep working.
    """

    @wraps(view_func)
    def wrapper(*args, **kwargs):
        token = current_app.config.get("ADMIN_TOKEN")
        if not token:
            return view_func(*args, **kwargs)
        provided = None
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            provided = auth_header[len("Bearer "):].strip()
        if not provided:
            provided = request.headers.get("X-Admin-Token", "").strip() or None
        if provided != token:
            return jsonify({"error": "unauthorized"}), 401
        return view_func(*args, **kwargs)

    return wrapper


def create_app():
    """Flask application factory."""
    base_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    data_dir = os.environ.get("DATA_DIR", base_dir)
    db_path = os.path.join(data_dir, "zephyr.db")

    app = Flask(
        __name__,
        template_folder=os.path.join(base_dir, "templates"),
        static_folder=os.path.join(base_dir, "static"),
    )
    app.config["SQLALCHEMY_DATABASE_URI"] = f"sqlite:///{db_path}"
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
    app.config["MAX_CONTENT_LENGTH"] = 1_000_000

    secret_env = os.environ.get("SECRET_KEY")
    if not secret_env:
        logger.warning(
            "SECRET_KEY env var unset; generating ephemeral key (sessions reset on restart)"
        )
    app.config["SECRET_KEY"] = secret_env or os.urandom(32).hex()

    admin_token = os.environ.get("ADMIN_TOKEN")
    if admin_token:
        app.config["ADMIN_TOKEN"] = admin_token
    else:
        logger.warning(
            "ADMIN_TOKEN env var unset; admin endpoints are open to all callers"
        )

    db.init_app(app)
    limiter.init_app(app)

    with app.app_context():
        db.create_all()
        if Pref.query.first() is None:
            db.session.add(Pref(zip_codes="10001"))
            db.session.commit()
        ensure_done_edge_constraints(db.session)

    @app.after_request
    def _hsts(response):
        response.headers["Strict-Transport-Security"] = (
            "max-age=63072000; includeSubDomains; preload"
        )
        return response

    from .routes.api import api
    from .routes.views import views
    app.register_blueprint(api)
    app.register_blueprint(views)

    return app
