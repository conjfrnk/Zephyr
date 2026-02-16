"""
Zephyr - View routes (HTML templates)
Copyright (C) 2025 Connor Frank
License: GPLv3 (see LICENSE)
"""

import os
from flask import Blueprint, render_template, send_from_directory

views = Blueprint("views", __name__)
BASE_DIR = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


@views.route("/")
def index():
    return render_template("index.html")


@views.route("/static/<path:p>")
def staticfile(p):
    return send_from_directory(os.path.join(BASE_DIR, "static"), p)
