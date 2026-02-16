"""
Zephyr - Smart route planner for runners
Copyright (C) 2025 Connor Frank
License: GPLv3 (see LICENSE)
"""

import os

from flask import Flask

from .models import db, Pref


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

    db.init_app(app)

    with app.app_context():
        db.create_all()
        if Pref.query.first() is None:
            db.session.add(Pref(zip_codes="10001"))
            db.session.commit()

    # Register blueprints
    from .routes.api import api
    from .routes.views import views
    app.register_blueprint(api)
    app.register_blueprint(views)

    return app
