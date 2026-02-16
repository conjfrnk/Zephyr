"""
Zephyr - Database models
Copyright (C) 2025 Connor Frank
License: GPLv3 (see LICENSE)
"""

from datetime import date
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()


class Run(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    date = db.Column(db.Date, default=date.today, nullable=False)
    distance_m = db.Column(db.Float, nullable=False)
    route_geojson = db.Column(db.Text, nullable=False)
    status = db.Column(db.String(10), default="planned")


class DoneEdge(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    u = db.Column(db.Integer, nullable=False)
    v = db.Column(db.Integer, nullable=False)
    key = db.Column(db.Integer, nullable=False)


class Pref(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    ideal_min_temp_f = db.Column(db.Float, default=50.0)
    ideal_max_temp_f = db.Column(db.Float, default=68.0)
    max_wind_mph = db.Column(db.Float, default=15.0)
    target_miles = db.Column(db.Float, default=5.0)
    zip_codes = db.Column(db.Text, default="")


def get_pref():
    return Pref.query.first()


def update_pref(**kw):
    p = get_pref()
    for k, v in kw.items():
        if hasattr(p, k) and v is not None:
            setattr(p, k, v)
    db.session.commit()
