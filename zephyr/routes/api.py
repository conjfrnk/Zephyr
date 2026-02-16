"""
Zephyr - API routes
Copyright (C) 2025 Connor Frank
License: GPLv3 (see LICENSE)
"""

import json

import osmnx as ox
from flask import Blueprint, jsonify, request

from ..models import db, Run, DoneEdge, get_pref, update_pref
from ..weather import wx
# Import the module itself so we always access current global values
from .. import graph as graph_mod
from ..routing import auto_path, path_geojson

api = Blueprint("api", __name__)


@api.route("/health")
def health():
    return jsonify({"status": "healthy"}), 200


@api.route("/prefs", methods=["GET", "POST"])
def prefs():
    if request.method == "POST":
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
    p_obj = get_pref()
    return jsonify({
        "tmin": p_obj.ideal_min_temp_f,
        "tmax": p_obj.ideal_max_temp_f,
        "wmax": p_obj.max_wind_mph,
        "target": p_obj.target_miles,
        "zip_codes": p_obj.zip_codes,
    })


@api.route("/status")
def status():
    global_done_pct = graph_mod.done_pct(graph_mod.G) if graph_mod.G else 0
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

        final_zip_status[zc] = {
            "road": data.get(
                "road",
                100 if is_graph_ready and graph_instance and zc in graph_mod.CURRENT_ZIPS else 0,
            ),
            "elev": current_elev_pct,
            "done": global_done_pct,
        }
        if not (is_graph_ready and zc in graph_mod.CURRENT_ZIPS):
            final_zip_status[zc]["road"] = data.get("road", 0)
            final_zip_status[zc]["elev"] = data.get("elev", 0)

    return jsonify({
        "ready": is_graph_ready,
        "zips": final_zip_status,
        "done_edge_pct": global_done_pct,
        "total_graph_length_m": total_graph_length_m,
        "total_done_length_m": total_done_length_m,
    })


@api.route("/set_zipcodes", methods=["POST"])
def set_zips():
    from flask import current_app

    z_raw = request.json.get("zips", "")
    z = [
        s.strip() for s in z_raw.replace(",", " ").split()
        if s.strip().isdigit() and len(s.strip()) == 5
    ]
    if not z:
        return jsonify({"error": "No valid zip codes provided or invalid format."}), 400
    unique_zips = sorted(list(set(z)))
    update_pref(zip_codes=",".join(unique_zips))
    graph_mod.fetch_graph_async(unique_zips, current_app._get_current_object())
    return jsonify({"started": True, "zips_being_loaded": unique_zips})


@api.route("/graph")
def graph_endpoint():
    if not graph_mod.GRAPH_READY or not graph_mod.G:
        return jsonify({"type": "FeatureCollection", "features": []}), 503
    with graph_mod.LOCK:
        gj = graph_mod.ALL_EDGES_GJ
    return jsonify(gj)


@api.route("/plan_auto")
def plan_auto():
    current_G = graph_mod.G
    if not current_G or not graph_mod.GRAPH_READY:
        return jsonify({"paths": []}), 503
    try:
        lat = float(request.args["lat"])
        lon = float(request.args["lon"])
    except (ValueError, TypeError, KeyError):
        return jsonify({"error": "Invalid lat/lon"}), 400

    avoid = request.args.get("avoid_hills", "false").lower() == "true"
    current_prefs_obj = get_pref()
    w_data = wx(lat, lon)
    target_miles_effective = current_prefs_obj.target_miles

    temp_factor = 1.0
    wind_factor = 1.0

    if w_data["temp_f"] < current_prefs_obj.ideal_min_temp_f:
        temp_diff = current_prefs_obj.ideal_min_temp_f - w_data["temp_f"]
        temp_factor = max(0.7, 1.0 - (temp_diff * 0.02))
    elif w_data["temp_f"] > current_prefs_obj.ideal_max_temp_f:
        temp_diff = w_data["temp_f"] - current_prefs_obj.ideal_max_temp_f
        temp_factor = max(0.7, 1.0 - (temp_diff * 0.02))

    if w_data["wind_mph"] > current_prefs_obj.max_wind_mph:
        wind_diff = w_data["wind_mph"] - current_prefs_obj.max_wind_mph
        wind_factor = max(0.6, 1.0 - (wind_diff * 0.03))

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
    current_G = graph_mod.G
    fil = request.args.get("status")

    if request.method == "POST":
        Run.query.filter_by(status="planned").delete()
        db.session.commit()
        d = request.get_json(silent=True)
        if not d or "distance_m" not in d or "geojson" not in d:
            return jsonify({"error": "Missing data for run"}), 400

        geojson_to_save = d["geojson"]
        if not isinstance(geojson_to_save, dict):
            return jsonify({"error": "GeoJSON must be a valid Feature object"}), 400

        props = geojson_to_save.get("properties", {})
        props["distance_m"] = d["distance_m"]
        props["total_ascent_m"] = d["geojson"].get("properties", {}).get("total_ascent_m", 0)
        props["total_descent_m"] = d["geojson"].get("properties", {}).get("total_descent_m", 0)
        props["node_ids"] = d["geojson"].get("properties", {}).get("node_ids", [])
        props["new_distance_m"] = d["geojson"].get("properties", {}).get("new_distance_m", 0)
        props["percentage_new_distance"] = d["geojson"].get("properties", {}).get(
            "percentage_new_distance", 0
        )
        geojson_to_save["properties"] = props

        geojson_str = json.dumps(geojson_to_save)
        run_obj = Run(distance_m=float(d["distance_m"]), route_geojson=geojson_str, status="planned")
        db.session.add(run_obj)
        db.session.commit()
        return jsonify({"run_id": run_obj.id})

    if request.method == "PUT":
        json_data = request.get_json(silent=True)
        if not json_data or "run_id" not in json_data:
            return jsonify({"error": "Missing run_id in request"}), 400
        rid = int(json_data["run_id"])
        run_obj = db.session.get(Run, rid)
        if not run_obj:
            return jsonify({"error": "Run not found"}), 404
        run_obj.status = "completed"

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
                            if not DoneEdge.query.filter_by(
                                u=u_node, v=v_node, key=k_edge
                            ).first():
                                db.session.add(DoneEdge(u=u_node, v=v_node, key=k_edge))

            db.session.commit()

            if current_G:
                graph_mod.ALL_EDGES_GJ = graph_mod.edges_geojson(current_G)

        except (json.JSONDecodeError, KeyError, TypeError) as e:
            print(f"Error processing run completion for run_id {rid}: {e}")
            db.session.commit()
            return jsonify({
                "ok": True,
                "warning": "Could not mark all edges as done due to processing error.",
            })
        return jsonify({"ok": True})

    # GET
    q = Run.query
    if fil in ["planned", "completed"]:
        q = q.filter_by(status=fil)
    q = q.order_by(Run.date.desc(), Run.id.desc())
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
    except (ValueError, TypeError, KeyError):
        return jsonify({"error": "Invalid lat/lon parameters"}), 400
    return jsonify(wx(lat, lon))
