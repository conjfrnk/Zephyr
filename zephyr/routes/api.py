"""
Zephyr - API routes
Copyright (C) 2025 Connor Frank
License: GPLv3 (see LICENSE)
"""

import hashlib
import json
import logging
import math

import osmnx as ox
from flask import Blueprint, abort, jsonify, request, make_response

from .. import limiter, require_admin_token
from ..models import db, Run, DoneEdge, get_pref, update_pref, add_done_edge_unique
from ..weather import wx
# Import the module itself so we always access current global values
from .. import graph as graph_mod
from ..routing import auto_path, path_geojson

api = Blueprint("api", __name__)
logger = logging.getLogger(__name__)


def _done_edge_pct_for_graph(graph_obj):
    """Return percent of current-graph edges marked done (filtered, not all-time)."""
    if not graph_obj or graph_obj.number_of_edges() == 0:
        return 0
    current_edge_keys = {(u, v, k) for u, v, k in graph_obj.edges(keys=True)}
    done_in_graph = sum(
        1 for de in DoneEdge.query.all()
        if (de.u, de.v, de.key) in current_edge_keys
    )
    return int(100 * done_in_graph / max(graph_obj.number_of_edges(), 1))


@api.route("/health")
def health():
    resp = make_response(jsonify({"status": "healthy"}), 200)
    resp.headers["Cache-Control"] = "no-store"
    return resp


@api.route("/prefs", methods=["GET", "POST"])
def prefs():
    if request.method == "POST":
        return _prefs_post()
    p_obj = get_pref()
    return jsonify({
        "tmin": p_obj.ideal_min_temp_f,
        "tmax": p_obj.ideal_max_temp_f,
        "wmax": p_obj.max_wind_mph,
        "target": p_obj.target_miles,
    })


@require_admin_token
def _prefs_post():
    d = request.get_json(silent=True)
    if d is None:
        return jsonify({"error": "Invalid JSON or Content-Type"}), 400
    update_pref(
        ideal_min_temp_f=d.get("tmin"),
        ideal_max_temp_f=d.get("tmax"),
        max_wind_mph=d.get("wmax"),
        target_miles=d.get("target"),
    )
    return jsonify({"ok": True})


@api.route("/status")
def status():
    current_G = graph_mod.G
    global_done_pct = _done_edge_pct_for_graph(current_G) if current_G else 0
    total_graph_length_m = 0.0
    total_done_length_m = 0.0

    with graph_mod.LOCK:
        current_zip_status_copy = graph_mod.ZIP_STATUS.copy()
        is_graph_ready = graph_mod.GRAPH_READY
        graph_instance = graph_mod.G

    if is_graph_ready and graph_instance:
        for u, v, data in graph_instance.edges(data=True):
            total_graph_length_m += data.get('length', 0.0)

        done_edges_db = DoneEdge.query.all()
        for done_edge in done_edges_db:
            u, v, k = done_edge.u, done_edge.v, done_edge.key
            if graph_instance.has_edge(u, v, k):
                total_done_length_m += graph_instance.edges[u, v, k].get('length', 0.0)

    final_zip_status = {}
    for zc, data in current_zip_status_copy.items():
        current_elev_pct = data.get("elev", 0)
        if is_graph_ready and graph_instance and zc in graph_mod.CURRENT_ZIPS:
            if graph_instance.number_of_nodes() > 0:
                elev_count = sum(
                    1 for _, d_node in graph_instance.nodes(data=True)
                    if "elevation" in d_node and d_node['elevation'] is not None
                )
                current_elev_pct = int(
                    100 * elev_count / graph_instance.number_of_nodes()
                ) if graph_instance.number_of_nodes() > 0 else 100
            else:
                current_elev_pct = 100

        zip_entry = {
            "road": data.get(
                "road",
                100 if is_graph_ready and graph_instance and zc in graph_mod.CURRENT_ZIPS else 0,
            ),
            "elev": current_elev_pct,
            "done": global_done_pct,
        }
        if not (is_graph_ready and zc in graph_mod.CURRENT_ZIPS):
            zip_entry["road"] = data.get("road", 0)
            zip_entry["elev"] = data.get("elev", 0)
        if "note" in data:
            zip_entry["note"] = data["note"]
        final_zip_status[zc] = zip_entry

    resp = make_response(jsonify({
        "ready": is_graph_ready,
        "zips": final_zip_status,
        "done_edge_pct": global_done_pct,
        "total_graph_length_m": total_graph_length_m,
        "total_done_length_m": total_done_length_m,
    }))
    resp.headers["Cache-Control"] = "no-cache"
    return resp


@api.route("/set_zipcodes", methods=["POST"])
@limiter.limit("5/minute")
@require_admin_token
def set_zips():
    from flask import current_app

    d = request.get_json(silent=True)
    if d is None:
        return jsonify({"error": "Invalid JSON or Content-Type"}), 400
    z_raw = d.get("zips", "")
    z = [
        s.strip() for s in z_raw.replace(",", " ").split()
        if s.strip().isdigit() and len(s.strip()) == 5
    ]
    if not z:
        return jsonify({"error": "No valid zip codes provided or invalid format."}), 400
    unique_zips = sorted(list(set(z)))
    if len(unique_zips) > 5:
        abort(400, "max 5 zip codes")
    graph_mod.fetch_graph_async(unique_zips, current_app._get_current_object())
    return jsonify({"started": True, "zips_being_loaded": unique_zips})


@api.route("/graph")
def graph_endpoint():
    if not graph_mod.GRAPH_READY or not graph_mod.G:
        return jsonify({"type": "FeatureCollection", "features": []}), 503
    with graph_mod.LOCK:
        gj = graph_mod.ALL_EDGES_GJ

    body = json.dumps(gj, separators=(",", ":"))
    etag = hashlib.md5(body.encode()).hexdigest()

    if_none_match = request.headers.get("If-None-Match")
    if if_none_match and if_none_match.strip('"') == etag:
        resp = make_response("", 304)
        resp.headers["ETag"] = f'"{etag}"'
        return resp

    resp = make_response(body, 200)
    resp.headers["Content-Type"] = "application/json"
    resp.headers["Cache-Control"] = "public, max-age=60"
    resp.headers["ETag"] = f'"{etag}"'
    return resp


@api.route("/plan_auto")
@limiter.limit("30/minute")
def plan_auto():
    current_G = graph_mod.G
    if not current_G or not graph_mod.GRAPH_READY:
        return jsonify({"paths": []}), 503
    try:
        lat = float(request.args["lat"])
        lon = float(request.args["lon"])
        tmin = float(request.args.get("tmin", 50))
        tmax = float(request.args.get("tmax", 68))
        wmax = float(request.args.get("wmax", 15))
        target = float(request.args.get("target", 5.0))
        if not all(math.isfinite(v) for v in (lat, lon, tmin, tmax, wmax, target)):
            return jsonify({"error": "Non-finite numeric parameter"}), 400
    except (ValueError, TypeError, KeyError):
        return jsonify({"error": "Invalid lat/lon"}), 400

    avoid = request.args.get("avoid_hills", "false").lower() in ("true", "1", "yes", "on")

    target = max(0.5, min(30.0, target))
    if not math.isfinite(target):
        return jsonify({"error": "Non-finite target"}), 400

    w_data = wx(lat, lon)
    target_miles_effective = target

    weather_factor = 1.0
    if w_data.get("available"):
        temp_factor = 1.0
        wind_factor = 1.0
        temp_f = w_data["temp_f"]
        wind_mph = w_data["wind_mph"]
        if temp_f is not None:
            if temp_f < tmin:
                temp_factor = max(0.7, 1.0 - ((tmin - temp_f) * 0.02))
            elif temp_f > tmax:
                temp_factor = max(0.7, 1.0 - ((temp_f - tmax) * 0.02))
        if wind_mph is not None and wind_mph > wmax:
            wind_factor = max(0.6, 1.0 - ((wind_mph - wmax) * 0.03))
        weather_factor = min(temp_factor, wind_factor)

    target_miles_effective *= weather_factor
    target_miles_effective = max(target_miles_effective, 1.0)

    list_of_path_nodes = auto_path(lat, lon, target_miles_effective, avoid)

    if not list_of_path_nodes:
        return jsonify({"paths": []})

    geojson_paths = []
    for i, result in enumerate(list_of_path_nodes):
        nodes = result['path']
        strategy = result.get('strategy', '')
        gj = path_geojson(nodes, current_G)
        if gj and gj.get("geometry"):
            gj["properties"]["candidate_index"] = i
            gj["properties"]["strategy"] = strategy
            geojson_paths.append(gj)

    return jsonify({"paths": geojson_paths})


@api.route("/runs", methods=["GET", "POST", "PUT"])
def runs():
    if request.method == "POST":
        return _runs_post()
    if request.method == "PUT":
        return _runs_put()
    return _runs_get()


@require_admin_token
def _runs_post():
    d = request.get_json(silent=True)
    if not d or "distance_m" not in d or "geojson" not in d:
        return jsonify({"error": "Missing data for run"}), 400

    geojson_to_save = d["geojson"]
    if not isinstance(geojson_to_save, dict):
        return jsonify({"error": "GeoJSON must be a valid Feature object"}), 400

    try:
        distance_m = float(d["distance_m"])
        if not math.isfinite(distance_m):
            return jsonify({"error": "Invalid distance_m"}), 400
    except (ValueError, TypeError):
        return jsonify({"error": "Invalid distance_m"}), 400

    props = geojson_to_save.get("properties", {})
    props["distance_m"] = distance_m
    props["total_ascent_m"] = d["geojson"].get("properties", {}).get("total_ascent_m", 0)
    props["total_descent_m"] = d["geojson"].get("properties", {}).get("total_descent_m", 0)
    props["node_ids"] = d["geojson"].get("properties", {}).get("node_ids", [])
    props["new_distance_m"] = d["geojson"].get("properties", {}).get("new_distance_m", 0)
    props["percentage_new_distance"] = d["geojson"].get("properties", {}).get(
        "percentage_new_distance", 0
    )
    geojson_to_save["properties"] = props

    geojson_str = json.dumps(geojson_to_save)

    try:
        Run.query.filter_by(status="planned").delete()
        run_obj = Run(distance_m=distance_m, route_geojson=geojson_str, status="planned")
        db.session.add(run_obj)
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        logger.exception("Failed to persist planned run")
        return jsonify({"error": "Failed to persist run", "detail": str(e)}), 500
    return jsonify({"run_id": run_obj.id})


@require_admin_token
def _runs_put():
    current_G = graph_mod.G
    json_data = request.get_json(silent=True)
    if not json_data or "run_id" not in json_data:
        return jsonify({"error": "Missing run_id in request"}), 400

    try:
        rid = int(json_data["run_id"])
    except (ValueError, TypeError):
        return jsonify({"error": "Invalid run_id"}), 400

    run_obj = db.session.get(Run, rid)
    if not run_obj:
        return jsonify({"error": "Run not found"}), 404

    try:
        geojson_data = json.loads(run_obj.route_geojson)
        path_node_ids = geojson_data.get("properties", {}).get("node_ids")

        if not path_node_ids:
            path_coords = geojson_data.get("geometry", {}).get("coordinates", [])
            if current_G and path_coords and len(path_coords) >= 2:
                path_node_ids = ox.nearest_nodes(
                    current_G,
                    X=[c[0] for c in path_coords],
                    Y=[c[1] for c in path_coords],
                )
            else:
                path_node_ids = []

        if current_G and path_node_ids and len(path_node_ids) >= 2:
            for i in range(len(path_node_ids) - 1):
                u_node, v_node = path_node_ids[i], path_node_ids[i + 1]
                if u_node == v_node:
                    continue
                if current_G.has_edge(u_node, v_node):
                    for k_edge in current_G[u_node][v_node]:
                        add_done_edge_unique(db.session, u_node, v_node, k_edge)
                if current_G.has_edge(v_node, u_node):
                    for k_edge in current_G[v_node][u_node]:
                        add_done_edge_unique(db.session, v_node, u_node, k_edge)

        run_obj.status = "completed"
        db.session.commit()

        if current_G:
            graph_mod.update_all_edges_gj(current_G)

    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
        db.session.rollback()
        logger.exception("Error processing run completion for run_id %s", rid)
        return jsonify({"error": "Failed to complete run", "detail": str(e)}), 500
    except Exception as e:
        db.session.rollback()
        logger.exception("Unexpected error completing run_id %s", rid)
        return jsonify({"error": "Internal error", "detail": str(e)}), 500
    return jsonify({"ok": True})


def _runs_get():
    fil = request.args.get("status")
    try:
        limit = int(request.args.get("limit", 100))
        offset = int(request.args.get("offset", 0))
    except (ValueError, TypeError):
        return jsonify({"error": "Invalid limit/offset"}), 400
    limit = max(1, min(500, limit))
    offset = max(0, offset)

    q = Run.query
    if fil in ["planned", "completed"]:
        q = q.filter_by(status=fil)
    q = q.order_by(Run.id.desc()).limit(limit).offset(offset)
    feats = []
    for r_item in q.all():
        try:
            gj_feature = json.loads(r_item.route_geojson)
            props = gj_feature.get("properties", {})
            props["run_id"] = r_item.id
            props["status"] = r_item.status
            props["date"] = r_item.date.isoformat()
            props["distance_m"] = r_item.distance_m
            props["total_ascent_m"] = props.get("total_ascent_m", 0)
            props["total_descent_m"] = props.get("total_descent_m", 0)
            props["new_distance_m"] = props.get("new_distance_m", 0)
            props["percentage_new_distance"] = props.get("percentage_new_distance", 0)
            gj_feature["properties"] = props
            feats.append(gj_feature)
        except (json.JSONDecodeError, TypeError):
            continue
    return jsonify({"type": "FeatureCollection", "features": feats})


@api.route("/run_start_finish_points")
def run_start_finish_points():
    points = []
    runs_list = Run.query.order_by(Run.date.desc(), Run.id.desc()).all()
    for run_item in runs_list:
        try:
            geojson_data = json.loads(run_item.route_geojson)
            coords = geojson_data.get("geometry", {}).get("coordinates")
            if coords and isinstance(coords, list) and len(coords) > 0:
                start_coord = coords[0]
                finish_coord = coords[-1]
                if isinstance(start_coord, list) and len(start_coord) == 2:
                    points.append({
                        "type": "start", "run_id": run_item.id,
                        "status": run_item.status,
                        "lat": start_coord[1], "lng": start_coord[0],
                    })
                if (isinstance(finish_coord, list) and len(finish_coord) == 2
                        and len(coords) > 1):
                    points.append({
                        "type": "finish", "run_id": run_item.id,
                        "status": run_item.status,
                        "lat": finish_coord[1], "lng": finish_coord[0],
                    })
        except (json.JSONDecodeError, TypeError, KeyError, IndexError):
            continue
    return jsonify(points)


@api.route("/weather")
def weather():
    try:
        lat = float(request.args["lat"])
        lon = float(request.args["lon"])
        if not (math.isfinite(lat) and math.isfinite(lon)):
            return jsonify({"error": "Non-finite lat/lon"}), 400
    except (ValueError, TypeError, KeyError):
        return jsonify({"error": "Invalid lat/lon parameters"}), 400
    data = wx(lat, lon)
    resp = make_response(jsonify(data))
    resp.headers["Cache-Control"] = "public, max-age=300"
    return resp
