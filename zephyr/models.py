"""
Zephyr - Database models
Copyright (C) 2025 Connor Frank
License: GPLv3 (see LICENSE)
"""

from datetime import date
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

db = SQLAlchemy()


class Run(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    date = db.Column(db.Date, default=date.today, nullable=False)
    distance_m = db.Column(db.Float, nullable=False)
    route_geojson = db.Column(db.Text, nullable=False)
    status = db.Column(db.String(10), default="planned")


class DoneEdge(db.Model):
    __table_args__ = (
        db.UniqueConstraint('u', 'v', 'key', name='uq_done_edge_uvk'),
    )

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
    """Return the singleton Pref row, creating one with defaults if absent.

    This guarantees callers (notably ``update_pref``) always receive a
    persistable Pref instance instead of None.
    """
    p = Pref.query.first()
    if p is None:
        p = Pref()
        db.session.add(p)
        db.session.commit()
    return p


def update_pref(**kw):
    p = get_pref()
    for k, v in kw.items():
        if hasattr(p, k) and v is not None:
            setattr(p, k, v)
    db.session.commit()


def ensure_done_edge_constraints(db_session):
    """Dedupe existing DoneEdge rows and create unique index if missing.

    Run at app startup AFTER db.create_all() to make existing databases
    safe even if their CREATE TABLE didn't include the constraint.
    """
    # Delete duplicates, keeping the row with the smallest id for each
    # (u, v, key) tuple.
    db_session.execute(text(
        """
        DELETE FROM done_edge
        WHERE id NOT IN (
            SELECT MIN(id) FROM done_edge GROUP BY u, v, key
        )
        """
    ))
    # Add the uniqueness guarantee for already-existing tables that were
    # created before the UniqueConstraint was added to the model.
    db_session.execute(text(
        "CREATE UNIQUE INDEX IF NOT EXISTS ix_done_edge_uvk "
        "ON done_edge(u, v, key)"
    ))
    db_session.commit()


def add_done_edge_unique(db_session, u, v, key):
    """Add a DoneEdge if not already present. Returns True if newly added.

    Uses SQLite's INSERT OR IGNORE semantics via a prefixed insert; falls
    back to a try/except IntegrityError + rollback path on other backends
    or if the prefixed insert silently fails to no-op.
    """
    try:
        result = db_session.execute(
            DoneEdge.__table__.insert()
            .prefix_with("OR IGNORE")
            .values(u=u, v=v, key=key)
        )
        db_session.commit()
        # rowcount is 0 when the OR IGNORE swallowed the conflict.
        return bool(result.rowcount)
    except IntegrityError:
        db_session.rollback()
        return False
