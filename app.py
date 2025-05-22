"""
Zephyr
Copyright (C) 2025 Connor Frank

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.

Contact: Connor Frank <conjfrnk@gmail.com>
"""

import os, json, threading, math, random
from datetime import date
from functools import lru_cache, partial

import requests, networkx as nx, osmnx as ox
from shapely.geometry import LineString, mapping, Point
from shapely.ops import unary_union
from geopy.geocoders import Nominatim
from flask import Flask, jsonify, request, render_template, send_from_directory, abort
from flask_sqlalchemy import SQLAlchemy

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DB_PATH = os.path.join(BASE_DIR, "zephyr.db")

app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = f"sqlite:///{DB_PATH}"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db = SQLAlchemy(app)


# ── models ───────────────────────────────────────────────────────────────
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


with app.app_context():
    db.create_all()
    if Pref.query.first() is None:
        db.session.add(Pref(zip_codes="15243"))
        db.session.commit()


# ── helpers -----------------------------------------------------------------
def get_pref():
    return Pref.query.first()


def update_pref(**kw):
    p = get_pref()
    for k, v in kw.items():
        if hasattr(p, k) and v is not None:
            setattr(p, k, v)
    db.session.commit()


@lru_cache(maxsize=128)
def wx(lat, lon):
    meta_url = f"https://api.weather.gov/points/{lat},{lon}"
    try:
        meta_resp = requests.get(meta_url, headers={"User-Agent": "Zephyr/0.1"}, timeout=8)
        meta_resp.raise_for_status()
        meta = meta_resp.json()
        
        hourly_url = meta["properties"]["forecastHourly"]
        hourly_resp = requests.get(hourly_url, headers={"User-Agent": "Zephyr/0.1"}, timeout=8)
        hourly_resp.raise_for_status()
        
        p = hourly_resp.json()["properties"]["periods"][0]
        return {
            "temp_f": p["temperature"],
            "short": p["shortForecast"],
            "wind_mph": float(p["windSpeed"].split()[0]),
        }
    except requests.exceptions.RequestException as e:
        print(f"Weather API error: {e}")
        return {"temp_f": 0, "short": "N/A", "wind_mph": 0} 


def dist_m(lat1, lon1, lat2, lon2):
    R = 6371000 
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    a = math.sin(delta_phi / 2)**2 + \
        math.cos(phi1) * math.cos(phi2) * \
        math.sin(delta_lambda / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


# ── graph globals -----------------------------------------------------------
G = None
GRAPH_READY = False
ZIP_STATUS = {}
ALL_EDGES_GJ = {}
CURRENT_ZIPS = []
LOCK = threading.Lock()
ROAD_FILTER = '["highway"]["highway"!~"footway|path|steps|pedestrian|cycleway|motorway|motorway_link|trunk|trunk_link"]'


def done_pct(graph_obj): # Global percentage done
    if not graph_obj or graph_obj.number_of_edges() == 0:
        return 0
    # This calculates based on number of *distinct done edges in DB* vs *total edges in current G*
    # This might not be a true "percentage of total miles" if edge lengths vary.
    # For a more accurate "miles done / total miles", see total_done_length_m / total_graph_length_m in /status
    with app.app_context():
        done_count = db.session.query(DoneEdge).count() 
    total_graph_edges = graph_obj.number_of_edges()
    if total_graph_edges == 0: return 0
    # A simple edge count percentage.
    return int(100 * done_count / total_graph_edges) if total_graph_edges > 0 else 0


def edges_geojson(graph_obj):
    with app.app_context():
        done_set = {(e.u, e.v, e.key) for e in DoneEdge.query.all()}
    feats = []
    if graph_obj:
        for u, v, k, d in graph_obj.edges(data=True, keys=True):
            geom = d.get("geometry")
            if not geom:
                 geom = LineString([
                    (graph_obj.nodes[u]["x"], graph_obj.nodes[u]["y"]),
                    (graph_obj.nodes[v]["x"], graph_obj.nodes[v]["y"]),
                ])
            is_done = (u, v, k) in done_set 
            feats.append({
                "type": "Feature",
                "geometry": mapping(geom),
                "properties": {"done": is_done, "plan": False, "u":u, "v":v, "key":k},
            })
    return {"type": "FeatureCollection", "features": feats}


def publish_graph(graph_obj, zips):
    global G, GRAPH_READY, CURRENT_ZIPS, ALL_EDGES_GJ
    with LOCK:
        if graph_obj: 
            for u_node, v_node, k_edge, data_edge in graph_obj.edges(data=True, keys=True):
                data_edge['_edge_key_'] = k_edge 

        G = graph_obj 
        CURRENT_ZIPS = zips
        ALL_EDGES_GJ = edges_geojson(G) 
        
        elev_nodes_count = 0
        if G and G.number_of_nodes() > 0: 
            elev_nodes_count = sum(1 for _, d_node in G.nodes(data=True) if "elevation" in d_node and d_node["elevation"] is not None)
            elev_pct = int(100 * elev_nodes_count / G.number_of_nodes()) if G.number_of_nodes() > 0 else 0
        else:
            elev_pct = 0

        # Note: current_done_pct is a global percentage of unique *edges run* vs *total edges*.
        # For per-zip completion, ZIP_STATUS["done"] would need zip-specific calculation.
        current_done_pct = done_pct(G) 
        for z in zips:
            zip_data = ZIP_STATUS.get(z, {"road": 0, "elev": 0})
            zip_data["done"] = current_done_pct # Assigning global done percentage here
            zip_data["elev"] = elev_pct
            if G: 
                zip_data["road"] = 100
            ZIP_STATUS[z] = zip_data
        GRAPH_READY = True


# ── elevation fetch ---------------------------------------------------------
def enrich_elevations(graph_obj, fname, zips): 
    session = requests.Session()
    nodes_to_fetch = [n for n, d in graph_obj.nodes(data=True) if d.get("elevation") is None]
    total_to_fetch = len(nodes_to_fetch)
    
    if total_to_fetch == 0:
        for z in zips:
            if z in ZIP_STATUS: ZIP_STATUS[z]["elev"] = 100
        publish_graph(graph_obj, zips) 
        return

    for i, n_id in enumerate(nodes_to_fetch, 1):
        node_data = graph_obj.nodes[n_id]
        lat, lon = node_data["y"], node_data["x"]
        try:
            elev_resp = session.get(
                f"https://epqs.nationalmap.gov/v1/json?x={lon}&y={lat}&units=Meters", timeout=8,
            )
            elev_resp.raise_for_status()
            elev = elev_resp.json().get("value")
            if elev is not None and isinstance(elev, (int, float)) and elev > -10000:
                 graph_obj.nodes[n_id]["elevation"] = float(elev)
            else:
                 graph_obj.nodes[n_id]["elevation"] = 0.0 
        except Exception as e:
            print(f"Elevation API error for node {n_id}: {e}")
            graph_obj.nodes[n_id]["elevation"] = 0.0 
        
        if i % 200 == 0 or i == total_to_fetch:
            pct = int(100 * i / total_to_fetch) if total_to_fetch > 0 else 100
            for z_code in zips:
                if z_code in ZIP_STATUS: ZIP_STATUS[z_code]["elev"] = pct
            if i == total_to_fetch: 
                 for z_code in zips: 
                     if z_code in ZIP_STATUS: ZIP_STATUS[z_code]["elev"] = 100
    
    if graph_obj.number_of_nodes() > 0:
        try:
            ox.add_edge_grades(graph_obj, add_absolute=True) 
        except Exception as e:
            print(f"Error adding edge grades: {e}")
        
    ox.save_graphml(graph_obj, fname) 
    publish_graph(graph_obj, zips) 


# ── graph loader ------------------------------------------------------------
def fetch_graph_async(zips_list):
    def worker():
        global GRAPH_READY, G 
        GRAPH_READY = False
        for z_code in zips_list:
            ZIP_STATUS[z_code] = {"road": 0, "elev": 0, "done": 0}
        
        fname = os.path.join(BASE_DIR, f"graph_{'_'.join(sorted(list(set(zips_list))))}.graphml")

        if os.path.isfile(fname):
            try:
                loaded_graph = ox.load_graphml(fname)
                publish_graph(loaded_graph, zips_list) 
                needs_elevation_enrich = any(data.get("elevation") is None for _, data in loaded_graph.nodes(data=True))
                if needs_elevation_enrich and loaded_graph.number_of_nodes() > 0 :
                    threading.Thread(target=enrich_elevations, args=(G, fname, zips_list), daemon=True).start()
                return
            except Exception as e:
                print(f"Error loading graph {fname}: {e}. Fetching new graph.")
                if os.path.exists(fname): os.remove(fname)

        geocoder = Nominatim(user_agent="Zephyr/0.1", timeout=10)
        polys = []
        valid_zips_for_graph = [] 

        for z_code in zips_list:
            geom_obj = None
            try:
                gdf = ox.geocode_to_gdf(f"{z_code}, USA")
                if not gdf.empty and 'geometry' in gdf.columns:
                    geom_candidate = gdf.iloc[0]["geometry"]
                    if geom_candidate.geom_type in ("Polygon", "MultiPolygon"):
                        geom_obj = geom_candidate
                    elif geom_candidate.geom_type == "Point":
                        geom_obj = Point(geom_candidate.x, geom_candidate.y).buffer(0.03)
                    else:
                        raise ValueError(f"Unexpected geometry type {geom_candidate.geom_type} from ox.geocode_to_gdf for {z_code}")
                else:
                    raise ValueError(f"Empty GeoDataFrame from ox.geocode_to_gdf for {z_code}")
            
            except Exception as e_ox:
                try:
                    loc = geocoder.geocode(f"{z_code}, USA", exactly_one=True, timeout=10)
                    if loc:
                        geom_obj = Point(loc.longitude, loc.latitude).buffer(0.03)
                    else:
                        ZIP_STATUS[z_code]["road"] = -1
                except Exception as e_nom:
                    ZIP_STATUS[z_code]["road"] = -1
            
            if geom_obj:
                polys.append(geom_obj)
                valid_zips_for_graph.append(z_code)
                ZIP_STATUS[z_code]["road"] = 20
            elif z_code in ZIP_STATUS and ZIP_STATUS[z_code].get("road", 0) != -1: 
                ZIP_STATUS[z_code]["road"] = -1
        
        if not polys:
            print("No valid geocoded areas found for any of the provided zips:", zips_list)
            GRAPH_READY = True
            for z_code_iter in zips_list:
                if z_code_iter in ZIP_STATUS and ZIP_STATUS[z_code_iter].get("road", 0) != -1 :
                    ZIP_STATUS[z_code_iter]["road"] = 100 
                    ZIP_STATUS[z_code_iter]["elev"] = 100
            return

        boundary = unary_union(polys)
        for z_code in valid_zips_for_graph: ZIP_STATUS[z_code]["road"] = 50

        temp_G_graph = None
        try:
            temp_G_graph = ox.graph_from_polygon(boundary, network_type="drive", custom_filter=ROAD_FILTER, simplify=True, retain_all=False)
        except Exception as e:
            print(f"Error creating graph from polygon: {e}")
            GRAPH_READY = True 
            for z_code in valid_zips_for_graph: ZIP_STATUS[z_code]["road"] = 100 
            return

        for z_code in valid_zips_for_graph: ZIP_STATUS[z_code]["road"] = 100
        publish_graph(temp_G_graph, valid_zips_for_graph) 
        
        if G.number_of_nodes() > 0: 
            threading.Thread(target=enrich_elevations, args=(G, fname, valid_zips_for_graph), daemon=True).start()
        else: 
            ox.save_graphml(G, fname) 
            publish_graph(G, valid_zips_for_graph)

    threading.Thread(target=worker, daemon=True).start()


# ── routing helpers ---------------------------------------------------------
def weight_factory(avoid_hills, prioritize_new_roads=True, temp_avoid_node_pairs=None):
    with app.app_context():
        globally_done_set = {(e.u, e.v, e.key) for e in DoneEdge.query.all()}

    current_path_avoid_node_pairs = temp_avoid_node_pairs if temp_avoid_node_pairs is not None else set()

    def cost_func(u, v, edge_data):
        k = edge_data.get('_edge_key_')
        weight = float(edge_data.get("length", 1.0))

        if tuple(sorted((u, v))) in current_path_avoid_node_pairs:
            weight *= 5000  

        if prioritize_new_roads:
            if k is not None and (u, v, k) in globally_done_set:
                weight *= 20000 # Increased penalty

        if avoid_hills:
            grade = abs(float(edge_data.get("grade_abs", edge_data.get("grade", 0.0))))
            weight *= (1 + grade * 10)

        return max(weight, 0.0001)
    return cost_func

def _calculate_path_details(path_nodes, graph_ref, globally_done_set):
    if not path_nodes or len(path_nodes) < 2:
        return {"distance_m": 0, "total_ascent_m": 0, "total_descent_m": 0, 
                "new_distance_m": 0, "percentage_new_distance": 0, "valid": False}
    
    total_distance_m = 0.0; new_distance_m = 0.0
    total_ascent_m = 0.0; total_descent_m = 0.0
    valid_path = True

    for i in range(len(path_nodes) - 1):
        u_node_id, v_node_id = path_nodes[i], path_nodes[i+1]
        if u_node_id == v_node_id: continue
        
        node_u_data = graph_ref.nodes.get(u_node_id)
        node_v_data = graph_ref.nodes.get(v_node_id)
        if not node_u_data or not node_v_data: valid_path = False; break

        best_k, min_len, chosen_edge_data = None, float('inf'), None
        if graph_ref.has_edge(u_node_id, v_node_id):
            for k_edge, data_edge in graph_ref[u_node_id][v_node_id].items():
                current_len = data_edge.get('length', float('inf'))
                if current_len < min_len:
                    min_len = current_len; best_k = k_edge; chosen_edge_data = data_edge
        
        if chosen_edge_data and 'length' in chosen_edge_data:
            segment_length = chosen_edge_data['length']
            total_distance_m += segment_length
            if (best_k is not None) and ((u_node_id, v_node_id, best_k) not in globally_done_set):
                new_distance_m += segment_length
            
            elev_u = node_u_data.get('elevation'); elev_v = node_v_data.get('elevation')
            if elev_u is not None and elev_v is not None:
                diff = elev_v - elev_u
                if diff > 0: total_ascent_m += diff
                else: total_descent_m += abs(diff)
        else: valid_path = False; break 
            
    if not valid_path: return {"distance_m": float('inf'), "valid": False}

    percentage_new_distance = round((new_distance_m / total_distance_m) * 100, 1) if total_distance_m > 0 else 0.0
    
    return {"distance_m": total_distance_m, "total_ascent_m": total_ascent_m, 
            "total_descent_m": total_descent_m, "new_distance_m": new_distance_m, 
            "percentage_new_distance": percentage_new_distance, "valid": True}


def path_geojson(path_nodes, graph_ref):
    with app.app_context(): 
        globally_done_set_for_geojson = {(e.u, e.v, e.key) for e in DoneEdge.query.all()}
    
    details = _calculate_path_details(path_nodes, graph_ref, globally_done_set_for_geojson)

    if not path_nodes or len(path_nodes) < 2 or not details.get("valid"):
        return {"type": "Feature", "geometry": None,
                "properties": {"distance_m": 0, "total_ascent_m": 0, "total_descent_m": 0, 
                               "node_ids": [], "new_distance_m": 0, "percentage_new_distance": 0, "plan": True}}
    
    coords = []
    for i in range(len(path_nodes) - 1):
        u_node_id, v_node_id = path_nodes[i], path_nodes[i+1]
        if u_node_id == v_node_id: continue
        
        node_u_data = graph_ref.nodes.get(u_node_id)
        node_v_data = graph_ref.nodes.get(v_node_id)

        best_k, min_len, chosen_edge_data = None, float('inf'), None
        if graph_ref.has_edge(u_node_id, v_node_id):
            for k_edge, data_edge in graph_ref[u_node_id][v_node_id].items():
                current_len = data_edge.get('length', float('inf'))
                if current_len < min_len: min_len = current_len; best_k = k_edge; chosen_edge_data = data_edge
        
        if chosen_edge_data:
            segment_coords_geom = chosen_edge_data.get("geometry")
            if segment_coords_geom: s_coords_list = list(segment_coords_geom.coords)
            else: s_coords_list = [(node_u_data['x'], node_u_data['y']), (node_v_data['x'], node_v_data['y'])]
            if coords and s_coords_list and coords[-1] == s_coords_list[0]: coords.extend(s_coords_list[1:])
            else: coords.extend(s_coords_list)
                
    return {
        "type": "Feature",
        "geometry": mapping(LineString(coords)) if coords else None,
        "properties": {
            "distance_m": details["distance_m"], "total_ascent_m": details["total_ascent_m"],
            "total_descent_m": details["total_descent_m"], "node_ids": path_nodes, 
            "new_distance_m": details["new_distance_m"], "percentage_new_distance": details["percentage_new_distance"],
            "plan": True
        },
    }

def auto_path(lat, lon, target_miles, avoid_hills, num_intermediate_samples=30, top_n_paths_to_return=5):
    if not G or G.number_of_nodes() == 0:
        print("auto_path (loop): Graph not ready or empty.")
        return None
    try:
        start_node = ox.nearest_nodes(G, X=lon, Y=lat)
    except Exception as e:
        print(f"auto_path (loop): Error finding nearest node for ({lat},{lon}): {e}")
        return None

    target_m = target_miles * 1609.34
    print(f"auto_path (loop): Start {start_node}, Target: {target_m:.0f}m")

    start_node_data = G.nodes[start_node]
    start_node_y, start_node_x = start_node_data['y'], start_node_data['x']

    with app.app_context(): 
        globally_done_set_for_loops = {(e.u, e.v, e.key) for e in DoneEdge.query.all()}
    
    leg1_weight_func = weight_factory(avoid_hills=avoid_hills, prioritize_new_roads=True, temp_avoid_node_pairs=None)
    
    half_target_m = target_m / 2.0
    min_intermediate_radius_m = max(250.0, half_target_m * 0.25) 
    max_intermediate_radius_m = max(750.0, half_target_m * 0.75)
    
    intermediate_nodes = []
    for node_id, node_data in G.nodes(data=True):
        if node_id == start_node: continue
        d = dist_m(start_node_y, start_node_x, node_data['y'], node_data['x'])
        if min_intermediate_radius_m <= d <= max_intermediate_radius_m:
            intermediate_nodes.append(node_id)
    
    if len(intermediate_nodes) > num_intermediate_samples:
        intermediate_nodes = random.sample(intermediate_nodes, num_intermediate_samples)
    
    if not intermediate_nodes:
        fallback_radius = max(500.0, half_target_m * 0.6) 
        for node_id, node_data in G.nodes(data=True):
            if node_id == start_node: continue
            if node_id in intermediate_nodes: continue 
            if dist_m(start_node_y, start_node_x, node_data['y'], node_data['x']) <= fallback_radius:
                intermediate_nodes.append(node_id)
        if len(intermediate_nodes) > int(num_intermediate_samples * 0.8) : 
             intermediate_nodes = random.sample(intermediate_nodes, int(num_intermediate_samples * 0.8))

    print(f"auto_path (loop): Testing {len(intermediate_nodes)} intermediate nodes.")
    if not intermediate_nodes: return None

    candidate_loop_paths_details = []

    for intermediate_node in intermediate_nodes:
        try:
            path1_nodes = nx.shortest_path(G, source=start_node, target=intermediate_node, weight=leg1_weight_func)
            if len(path1_nodes) < 2: continue

            path1_node_segments_to_avoid = {tuple(sorted((path1_nodes[i], path1_nodes[i+1]))) for i in range(len(path1_nodes)-1)}
            
            leg2_weight_func = weight_factory(avoid_hills=avoid_hills, prioritize_new_roads=True, temp_avoid_node_pairs=path1_node_segments_to_avoid)
            path2_nodes = nx.shortest_path(G, source=intermediate_node, target=start_node, weight=leg2_weight_func)
            if len(path2_nodes) < 2: continue
            
            loop_path_nodes = path1_nodes[:-1] + path2_nodes 
            if len(loop_path_nodes) < 3 : continue

            path_details = _calculate_path_details(loop_path_nodes, G, globally_done_set_for_loops)
            if not path_details.get("valid"): continue
            
            # Broad initial filter for loops
            if target_m * 0.5 <= path_details['distance_m'] <= target_m * 1.75: 
                candidate_loop_paths_details.append({
                    'path': loop_path_nodes, 
                    'distance': path_details['distance_m'],
                    'percentage_new': path_details['percentage_new_distance']
                })
        except (nx.NetworkXNoPath, nx.NodeNotFound): continue
        except Exception as e: 
            print(f"Error constructing loop via intermediate {intermediate_node}: {type(e).__name__} {e}") # Print type of error
            continue
    
    if not candidate_loop_paths_details:
        print("auto_path (loop): No candidate loops constructed after iterating intermediates.")
        return None

    # Sort candidates: primary by newness (desc), secondary by distance to target (asc)
    candidate_loop_paths_details.sort(key=lambda p: (-p['percentage_new'], abs(p['distance'] - target_m)))
    
    # Stricter final filter for distance for the "ideal" paths
    final_candidates_ideal_distance = [
        p for p in candidate_loop_paths_details
        if p['distance'] >= target_m * 0.75 and p['distance'] <= target_m * 1.25 
    ]
    
    top_n_to_return_list = []
    if final_candidates_ideal_distance:
        top_n_to_return_list = final_candidates_ideal_distance[:top_n_paths_to_return]
        # print(f"auto_path (loop): Selected {len(top_n_to_return_list)} paths from ideal distance range, prioritized by newness.")
    elif candidate_loop_paths_details: 
        top_n_to_return_list = candidate_loop_paths_details[:top_n_paths_to_return]
        # print(f"auto_path (loop): No loops in ideal distance range, returning best {len(top_n_to_return_list)} overall (newness prioritized).")
    
    if not top_n_to_return_list:
        print("auto_path (loop): No paths left after all filtering.")
        return None

    selected_paths_nodes = [p['path'] for p in top_n_to_return_list]
    
    if selected_paths_nodes:
        best_p_details = top_n_to_return_list[0]
        print(f"auto_path (loop): Returning {len(selected_paths_nodes)} candidate paths. Top choice: newness {best_p_details['percentage_new']:.1f}%, dist {best_p_details['distance']:.0f}m.")
    else: 
        print("auto_path (loop): No loop path selected (empty final_candidates).")
        
    return selected_paths_nodes if selected_paths_nodes else None


# ── routes ------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/prefs", methods=["GET", "POST"])
def prefs(): 
    if request.method == "POST":
        d = request.get_json(silent=True) 
        if d is None:
            return jsonify({"error": "Invalid JSON or Content-Type"}), 400
        update_pref(
            ideal_min_temp_f=d.get("tmin"), ideal_max_temp_f=d.get("tmax"),
            max_wind_mph=d.get("wmax"), target_miles=d.get("target"),
        )
        return jsonify({"ok": True})
    p_obj = get_pref()
    return jsonify({
        "tmin": p_obj.ideal_min_temp_f, "tmax": p_obj.ideal_max_temp_f,
        "wmax": p_obj.max_wind_mph, "target": p_obj.target_miles,
        "zip_codes": p_obj.zip_codes,
    })

# MODIFIED /status to include mileage stats
@app.route("/status")
def status():
    global_done_pct = done_pct(G) if G else 0
    total_graph_length_m = 0.0
    total_done_length_m = 0.0

    current_zip_status_copy = {} # To avoid holding lock for too long
    is_graph_ready_locked = False # Value from within lock

    with LOCK:
        current_zip_status_copy = ZIP_STATUS.copy()
        is_graph_ready_locked = GRAPH_READY
        graph_instance = G # Copy G reference under lock

    if is_graph_ready_locked and graph_instance:
        for u, v, data in graph_instance.edges(data=True):
            total_graph_length_m += data.get('length', 0.0)
        
        with app.app_context(): 
            done_edges_db = DoneEdge.query.all()
        for done_edge_db_entry in done_edges_db:
            u, v, k = done_edge_db_entry.u, done_edge_db_entry.v, done_edge_db_entry.key
            if graph_instance.has_edge(u, v, k): # Check specific key
                total_done_length_m += graph_instance.edges[u,v,k].get('length', 0.0)
            # else: edge from DB not in current graph G (e.g. different zips loaded) - do nothing

    final_zip_status = {}
    for zc, data in current_zip_status_copy.items():
        current_elev_pct = data.get("elev", 0) 
        if is_graph_ready_locked and graph_instance and zc in CURRENT_ZIPS : 
            if graph_instance.number_of_nodes() > 0: 
                elev_count = sum(1 for _, d_node in graph_instance.nodes(data=True) if "elevation" in d_node and d_node['elevation'] is not None)
                current_elev_pct = int(100 * elev_count / graph_instance.number_of_nodes()) if graph_instance.number_of_nodes() > 0 else 100
            else: 
                current_elev_pct = 100 
        
        final_zip_status[zc] = {
            "road": data.get("road", 100 if is_graph_ready_locked and graph_instance and zc in CURRENT_ZIPS else 0),
            "elev": current_elev_pct,
            "done": global_done_pct # This is overall edge count percentage
        }
        if not (is_graph_ready_locked and zc in CURRENT_ZIPS):
             final_zip_status[zc]["road"] = data.get("road", 0)
             final_zip_status[zc]["elev"] = data.get("elev",0)

    return jsonify({
        "ready": is_graph_ready_locked, 
        "zips": final_zip_status, 
        "done_edge_pct": global_done_pct, # Renamed for clarity
        "total_graph_length_m": total_graph_length_m,
        "total_done_length_m": total_done_length_m
    })

@app.route("/set_zipcodes", methods=["POST"])
def set_zips():
    z_raw = request.json.get("zips", "")
    z = [s.strip() for s in z_raw.replace(",", " ").split() if s.strip().isdigit() and len(s.strip()) == 5]
    if not z:
        return jsonify({"error": "No valid zip codes provided or invalid format."}), 400
    unique_zips = sorted(list(set(z)))
    update_pref(zip_codes=",".join(unique_zips))
    fetch_graph_async(unique_zips)
    return jsonify({"started": True, "zips_being_loaded": unique_zips})

@app.route("/graph")
def graph(): 
    if not GRAPH_READY or not G :
        return jsonify({"type": "FeatureCollection", "features": []}), 503
    return jsonify(ALL_EDGES_GJ)

@app.route("/plan_auto")
def plan_auto(): 
    if not G or not GRAPH_READY:
        return jsonify({"paths": []}), 503 
    try:
        lat = float(request.args["lat"])
        lon = float(request.args["lon"])
    except (ValueError, TypeError):
        return jsonify({"error": "Invalid lat/lon"}), 400
    avoid = request.args.get("avoid_hills", "false").lower() == "true"
    current_prefs_obj = get_pref() 
    w_data = wx(lat, lon)
    target_miles_effective = current_prefs_obj.target_miles
    if (w_data["temp_f"] < current_prefs_obj.ideal_min_temp_f or
        w_data["temp_f"] > current_prefs_obj.ideal_max_temp_f or
        w_data["wind_mph"] > current_prefs_obj.max_wind_mph):
        target_miles_effective *= 0.5
        if target_miles_effective < 0.5 : target_miles_effective = 0.5 
    
    list_of_path_nodes = auto_path(lat, lon, target_miles_effective, avoid)
    
    if not list_of_path_nodes: 
        print("/plan_auto: auto_path returned None or empty list. No suitable loop route found.")
        return jsonify({"paths": []}) 
        
    geojson_paths = []
    for i, nodes in enumerate(list_of_path_nodes):
        gj = path_geojson(nodes, G)
        if gj and gj.get("geometry"): 
            gj["properties"]["candidate_index"] = i 
            geojson_paths.append(gj)
            if i == 0: 
                print(f"  Best path (idx 0): dist={gj['properties']['distance_m']:.0f}m, new%={gj['properties']['percentage_new_distance']:.1f}%")
        else:
            print(f"  Warning: path_geojson returned invalid data for candidate path {i}")

    if not geojson_paths: 
        print("/plan_auto: No valid GeoJSON paths could be generated from auto_path results.")
        return jsonify({"paths": []})

    return jsonify({"paths": geojson_paths}) 


@app.route("/runs", methods=["GET", "POST", "PUT"])
def runs(): 
    global ALL_EDGES_GJ 
    fil = request.args.get("status")
    if request.method == "POST": 
        Run.query.filter_by(status="planned").delete()
        db.session.commit()
        d = request.get_json(silent=True)
        if not d or "distance_m" not in d or "geojson" not in d:
            return jsonify({"error": "Missing data for run"}), 400
        
        geojson_to_save = d["geojson"] 
        if not isinstance(geojson_to_save, dict) : 
             return jsonify({"error": "GeoJSON must be a valid Feature object"}), 400
        
        props = geojson_to_save.get("properties", {})
        props["distance_m"] = d["distance_m"] 
        props["total_ascent_m"] = d["geojson"].get("properties",{}).get("total_ascent_m", 0)
        props["total_descent_m"] = d["geojson"].get("properties",{}).get("total_descent_m", 0)
        props["node_ids"] = d["geojson"].get("properties",{}).get("node_ids", []) 
        props["new_distance_m"] = d["geojson"].get("properties",{}).get("new_distance_m", 0)
        props["percentage_new_distance"] = d["geojson"].get("properties",{}).get("percentage_new_distance", 0)
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
        if not run_obj: return jsonify({"error": "Run not found"}), 404
        run_obj.status = "completed"
        
        new_done_edges_added_in_tx = False
        try:
            geojson_data = json.loads(run_obj.route_geojson)
            path_node_ids = geojson_data.get("properties", {}).get("node_ids")
            
            if not path_node_ids: 
                print(f"Warning: node_ids not found in GeoJSON for completed run {rid}. Falling back to coordinate snapping.")
                path_coords = geojson_data.get("geometry", {}).get("coordinates", [])
                if G and path_coords and len(path_coords) >= 2:
                    path_node_ids = ox.nearest_nodes(G, X=[c[0] for c in path_coords], Y=[c[1] for c in path_coords])
                else: path_node_ids = []
            
            if G and path_node_ids and len(path_node_ids) >= 2:
                print(f"Marking {len(path_node_ids)-1} segments as done for run {rid} using {len(path_node_ids)} node_ids.")
                for i in range(len(path_node_ids) - 1):
                    u_node, v_node = path_node_ids[i], path_node_ids[i+1]
                    if u_node == v_node: continue
                    if G.has_edge(u_node, v_node):
                        # Mark all parallel edges for the (u,v) segment from the path
                        # This is a simple way; a more precise way would require storing the exact keys from path_geojson.
                        for k_edge in G[u_node][v_node]: 
                            if not DoneEdge.query.filter_by(u=u_node, v=v_node, key=k_edge).first():
                                db.session.add(DoneEdge(u=u_node, v=v_node, key=k_edge))
                                new_done_edges_added_in_tx = True 
            
            if new_done_edges_added_in_tx:
                db.session.commit() 
                print(f"Committed DoneEdges. Total in DB: {db.session.query(DoneEdge).count()}.")
            else:
                db.session.commit() 
                print(f"Run {rid} status updated. No new edges added to DoneEdge.")
            
            if G: ALL_EDGES_GJ = edges_geojson(G) # Always refresh after potential DB change
            print("Regenerated ALL_EDGES_GJ for map display after run completion.")

        except (json.JSONDecodeError, KeyError, TypeError) as e:
            print(f"Error processing run completion for run_id {rid}: {e}")
            db.session.commit() 
            return jsonify({"ok": True, "warning": "Could not mark all edges as done due to processing error."})
        return jsonify({"ok": True})

    # GET request
    q = Run.query
    if fil in ["planned", "completed"]: q = q.filter_by(status=fil)
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
            print(f"Skipping run {r_item.id} due to invalid GeoJSON.")
            continue
    return jsonify({"type": "FeatureCollection", "features": feats})

@app.route("/run_start_finish_points")
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
                    points.append({"type": "start", "run_id": run_item.id, "status": run_item.status, "lat": start_coord[1], "lng": start_coord[0]})
                if isinstance(finish_coord, list) and len(finish_coord) == 2 and len(coords) > 1 : 
                    points.append({"type": "finish", "run_id": run_item.id, "status": run_item.status, "lat": finish_coord[1], "lng": finish_coord[0]})
        except (json.JSONDecodeError, TypeError, KeyError, IndexError) as e:
            print(f"Error parsing GeoJSON for run {run_item.id} in run_start_finish_points: {e}")
            continue
    return jsonify(points)

@app.route("/weather")
def weather(): 
    try:
        lat = float(request.args["lat"])
        lon = float(request.args["lon"])
    except (ValueError, TypeError):
        return jsonify({"error":"Invalid lat/lon parameters"}), 400
    return jsonify(wx(lat, lon))

@app.route("/static/<path:p>")
def staticfile(p):
    return send_from_directory(os.path.join(BASE_DIR, "static"), p)

with app.app_context():
    initial_prefs = get_pref()
    if initial_prefs and initial_prefs.zip_codes:
        zs = [s.strip() for s in initial_prefs.zip_codes.split(',') if s.strip()]
        if zs:
            print(f"Autoloading graph for zips: {zs} (async from init)")
            fetch_graph_async(zs)

if __name__ == "__main__":
    app.run(debug=True, host='127.0.0.1', port=5000)
