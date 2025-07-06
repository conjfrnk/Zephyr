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
def weight_factory(avoid_hills, prioritize_new_roads=True, temp_avoid_node_pairs=None, route_variety_factor=1.0):
    with app.app_context():
        globally_done_set = {(e.u, e.v, e.key) for e in DoneEdge.query.all()}

    current_path_avoid_node_pairs = temp_avoid_node_pairs if temp_avoid_node_pairs is not None else set()

    def cost_func(u, v, edge_data):
        k = edge_data.get('_edge_key_')
        weight = float(edge_data.get("length", 1.0))

        # Avoid repeating segments from current path
        if tuple(sorted((u, v))) in current_path_avoid_node_pairs:
            weight *= 1000  # Reduced from 5000 for better route variety

        # Prioritize new roads with more nuanced weighting
        if prioritize_new_roads and k is not None:
            if (u, v, k) in globally_done_set:
                # More sophisticated penalty based on how recently the road was done
                weight *= 15  # Reduced from 20000 for better balance
            else:
                # Bonus for new roads, but not too aggressive
                weight *= 0.8

        # Enhanced hill avoidance with gradual penalty
        if avoid_hills:
            grade = abs(float(edge_data.get("grade_abs", edge_data.get("grade", 0.0))))
            # More gradual penalty curve
            grade_penalty = 1 + (grade * 5)  # Reduced from 10
            weight *= grade_penalty

        # Add variety factor to encourage different route types
        weight *= route_variety_factor

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

def auto_path(lat, lon, target_miles, avoid_hills, num_intermediate_samples=50, top_n_paths_to_return=8):
    """
    Enhanced route generation with multiple strategies and better optimization
    """
    if not G or G.number_of_nodes() == 0:
        print("auto_path: Graph not ready or empty.")
        return None
    
    try:
        start_node = ox.nearest_nodes(G, X=lon, Y=lat)
    except Exception as e:
        print(f"auto_path: Error finding nearest node for ({lat},{lon}): {e}")
        return None

    target_m = target_miles * 1609.34
    print(f"auto_path: Start {start_node}, Target: {target_m:.0f}m")

    start_node_data = G.nodes[start_node]
    start_node_y, start_node_x = start_node_data['y'], start_node_data['x']
    
    # Create a subgraph for faster processing
    radius_m = target_m * 0.75  # More generous radius
    subgraph_nodes = [n for n, d in G.nodes(data=True) if dist_m(start_node_y, start_node_x, d['y'], d['x']) < radius_m]
    if start_node not in subgraph_nodes:
        subgraph_nodes.append(start_node)
    
    if len(subgraph_nodes) < 30: # If area is sparse, use the whole graph
        H = G
        print(f"Using full graph ({G.number_of_nodes()} nodes) due to sparse area.")
    else:
        H = G.subgraph(subgraph_nodes).copy()
        print(f"Created subgraph with {H.number_of_nodes()} nodes and {H.number_of_edges()} edges.")


    with app.app_context(): 
        globally_done_set_for_loops = {(e.u, e.v, e.key) for e in DoneEdge.query.all()}
    
    # Multiple route generation strategies, now operating on subgraph H
    all_candidates = []
    
    # Strategy 1: Enhanced two-leg loops
    candidates_1 = _generate_two_leg_loops(H, start_node, target_m, avoid_hills, globally_done_set_for_loops, 
                                          num_intermediate_samples, start_node_y, start_node_x)
    all_candidates.extend(candidates_1)
    
    # Strategy 2: Multi-point routes
    candidates_2 = _generate_multi_point_routes(H, start_node, target_m, avoid_hills, globally_done_set_for_loops,
                                              start_node_y, start_node_x)
    all_candidates.extend(candidates_2)
    
    # Strategy 3: Out-and-back with different return paths
    candidates_3 = _generate_out_and_back_routes(H, start_node, target_m, avoid_hills, globally_done_set_for_loops,
                                                start_node_y, start_node_x)
    all_candidates.extend(candidates_3)
    
    # Strategy 4: Zigzag routes for efficient parallel street coverage
    candidates_4 = _generate_zigzag_routes(H, start_node, target_m, avoid_hills, globally_done_set_for_loops,
                                          start_node_y, start_node_x)
    all_candidates.extend(candidates_4)
    
    if not all_candidates:
        print("auto_path: No candidate routes found with any strategy.")
        return None

    # Enhanced scoring and filtering
    scored_candidates = _score_and_filter_candidates(all_candidates, target_m)
    
    if not scored_candidates:
        print("auto_path: No routes passed scoring criteria.")
        return None

    # Return top candidates
    selected_paths_nodes = [p['path'] for p in scored_candidates[:top_n_paths_to_return]]
    
    if selected_paths_nodes:
        best_p_details = scored_candidates[0]
        print(f"auto_path: Returning {len(selected_paths_nodes)} candidate paths. Top choice: newness {best_p_details['percentage_new']:.1f}%, dist {best_p_details['distance']:.0f}m.")
    else: 
        print("auto_path: No routes selected.")
        
    return selected_paths_nodes if selected_paths_nodes else None

def _generate_two_leg_loops(graph, start_node, target_m, avoid_hills, globally_done_set, num_samples, start_y, start_x):
    """Enhanced two-leg loop generation with better intermediate point selection"""
    candidates = []
    
    # Multiple radius ranges for better coverage
    radius_ranges = [
        (target_m * 0.3, target_m * 0.7),  # Close range
        (target_m * 0.4, target_m * 0.8),  # Medium range  
        (target_m * 0.5, target_m * 0.9),  # Far range
    ]
    
    for min_radius, max_radius in radius_ranges:
        intermediate_nodes = []
        for node_id, node_data in graph.nodes(data=True):
            if node_id == start_node: continue
            d = dist_m(start_y, start_x, node_data['y'], node_data['x'])
            if min_radius <= d <= max_radius:
                intermediate_nodes.append(node_id)
        
        if len(intermediate_nodes) > num_samples // len(radius_ranges):
            intermediate_nodes = random.sample(intermediate_nodes, num_samples // len(radius_ranges))

    for intermediate_node in intermediate_nodes:
        try:
            # First leg with standard weighting
            leg1_weight_func = weight_factory(avoid_hills=avoid_hills, prioritize_new_roads=True)
            path1_nodes = nx.shortest_path(graph, source=start_node, target=intermediate_node, weight=leg1_weight_func)
            if len(path1_nodes) < 2: continue

            # Second leg avoiding first leg segments
            path1_segments = {tuple(sorted((path1_nodes[i], path1_nodes[i+1]))) for i in range(len(path1_nodes)-1)}
            leg2_weight_func = weight_factory(avoid_hills=avoid_hills, prioritize_new_roads=True, temp_avoid_node_pairs=path1_segments)
            path2_nodes = nx.shortest_path(graph, source=intermediate_node, target=start_node, weight=leg2_weight_func)
            if len(path2_nodes) < 2: continue
            
            loop_path_nodes = path1_nodes[:-1] + path2_nodes 
            if len(loop_path_nodes) < 3: continue

            path_details = _calculate_path_details(loop_path_nodes, graph, globally_done_set)
            if not path_details.get("valid"): continue
            
            # More flexible distance filtering
            if target_m * 0.4 <= path_details['distance_m'] <= target_m * 1.8: 
                candidates.append({
                    'path': loop_path_nodes, 
                    'distance': path_details['distance_m'],
                    'percentage_new': path_details['percentage_new_distance'],
                    'strategy': 'two_leg'
                })
        except (nx.NetworkXNoPath, nx.NodeNotFound): continue
        except Exception as e: 
            print(f"Error in two-leg loop via {intermediate_node}: {e}")
            continue
    
    return candidates

def _generate_multi_point_routes(graph, start_node, target_m, avoid_hills, globally_done_set, start_y, start_x):
    """Generate routes with 3-4 points for more variety"""
    candidates = []
    
    if not graph:
        return candidates
    
    # Find potential waypoints at different distances
    waypoint_candidates = []
    for node_id, node_data in graph.nodes(data=True):
        if node_id == start_node: continue
        d = dist_m(start_y, start_x, node_data['y'], node_data['x'])
        if target_m * 0.2 <= d <= target_m * 0.6:
            waypoint_candidates.append(node_id)
    
    if len(waypoint_candidates) > 20:
        waypoint_candidates = random.sample(waypoint_candidates, 20)
    
    for waypoint in waypoint_candidates:
        try:
            # Create 3-point route: start -> waypoint -> different_point -> start
            leg1_weight = weight_factory(avoid_hills=avoid_hills, prioritize_new_roads=True)
            path1 = nx.shortest_path(graph, source=start_node, target=waypoint, weight=leg1_weight)
            if len(path1) < 2: continue
            
            # Find a different return point
            path1_segments = {tuple(sorted((path1[i], path1[i+1]))) for i in range(len(path1)-1)}
            for return_point in waypoint_candidates[:10]:  # Try first 10 as return points
                if return_point == waypoint: continue
                
                try:
                    leg2_weight = weight_factory(avoid_hills=avoid_hills, prioritize_new_roads=True, temp_avoid_node_pairs=path1_segments)
                    path2 = nx.shortest_path(graph, source=waypoint, target=return_point, weight=leg2_weight)
                    if len(path2) < 2: continue
                    
                    # Combine segments and avoid both previous paths
                    all_segments = path1_segments | {tuple(sorted((path2[i], path2[i+1]))) for i in range(len(path2)-1)}
                    leg3_weight = weight_factory(avoid_hills=avoid_hills, prioritize_new_roads=True, temp_avoid_node_pairs=all_segments)
                    path3 = nx.shortest_path(graph, source=return_point, target=start_node, weight=leg3_weight)
                    if len(path3) < 2: continue
                    
                    multi_path = path1[:-1] + path2[:-1] + path3
                    if len(multi_path) < 4: continue
                    
                    path_details = _calculate_path_details(multi_path, graph, globally_done_set)
                    if not path_details.get("valid"): continue
                    
                    if target_m * 0.5 <= path_details['distance_m'] <= target_m * 1.6:
                        candidates.append({
                            'path': multi_path,
                            'distance': path_details['distance_m'],
                            'percentage_new': path_details['percentage_new_distance'],
                            'strategy': 'multi_point'
                        })
                        break  # Found a good route for this waypoint
                        
                except (nx.NetworkXNoPath, nx.NodeNotFound): continue
                
        except (nx.NetworkXNoPath, nx.NodeNotFound): continue
        except Exception as e:
            print(f"Error in multi-point route via {waypoint}: {e}")
            continue
    
    return candidates

def _generate_out_and_back_routes(graph, start_node, target_m, avoid_hills, globally_done_set, start_y, start_x):
    """Generate out-and-back routes with different return paths"""
    candidates = []
    
    if not graph:
        return candidates
    
    # Find potential turnaround points
    turnaround_candidates = []
    for node_id, node_data in graph.nodes(data=True):
        if node_id == start_node: continue
        d = dist_m(start_y, start_x, node_data['y'], node_data['x'])
        if target_m * 0.3 <= d <= target_m * 0.7:
            turnaround_candidates.append(node_id)
    
    if len(turnaround_candidates) > 15:
        turnaround_candidates = random.sample(turnaround_candidates, 15)
    
    for turnaround in turnaround_candidates:
        try:
            # Outbound leg
            out_weight = weight_factory(avoid_hills=avoid_hills, prioritize_new_roads=True)
            out_path = nx.shortest_path(graph, source=start_node, target=turnaround, weight=out_weight)
            if len(out_path) < 2: continue
            
            # Return leg with different path
            out_segments = {tuple(sorted((out_path[i], out_path[i+1]))) for i in range(len(out_path)-1)}
            back_weight = weight_factory(avoid_hills=avoid_hills, prioritize_new_roads=True, temp_avoid_node_pairs=out_segments)
            back_path = nx.shortest_path(graph, source=turnaround, target=start_node, weight=back_weight)
            if len(back_path) < 2: continue
            
            out_and_back = out_path[:-1] + back_path
            if len(out_and_back) < 3: continue
            
            path_details = _calculate_path_details(out_and_back, graph, globally_done_set)
            if not path_details.get("valid"): continue
            
            if target_m * 0.4 <= path_details['distance_m'] <= target_m * 1.7:
                candidates.append({
                    'path': out_and_back,
                    'distance': path_details['distance_m'],
                    'percentage_new': path_details['percentage_new_distance'],
                    'strategy': 'out_and_back'
                })
                
        except (nx.NetworkXNoPath, nx.NodeNotFound): continue
        except Exception as e:
            print(f"Error in out-and-back route via {turnaround}: {e}")
            continue
    
    return candidates

def _generate_zigzag_routes(graph, start_node, target_m, avoid_hills, globally_done_set, start_y, start_x):
    """Generate zigzag routes that efficiently cover parallel streets"""
    candidates = []
    
    if not graph:
        return candidates
    
    # Find streets that run roughly parallel (similar orientation)
    def get_street_orientation(node1, node2):
        """Calculate the orientation angle of a street segment"""
        dx = node2['x'] - node1['x']
        dy = node2['y'] - node1['y']
        return math.atan2(dy, dx)
    
    # Group streets by orientation (within 15 degrees)
    orientation_groups = {}
    for u, v, data in graph.edges(data=True):
        if u in graph.nodes and v in graph.nodes:
            u_data = graph.nodes[u]
            v_data = graph.nodes[v]
            orientation = get_street_orientation(u_data, v_data)
            # Normalize to 0-180 degrees
            orientation = abs(orientation) % math.pi
            # Group into 15-degree bins
            bin_key = int(orientation * 180 / math.pi / 15)
            if bin_key not in orientation_groups:
                orientation_groups[bin_key] = []
            orientation_groups[bin_key].append((u, v, data))
    
    # Find the most common orientations (likely main streets)
    sorted_groups = sorted(orientation_groups.items(), key=lambda x: len(x[1]), reverse=True)
    
    for group_id, edges in sorted_groups[:3]:  # Top 3 orientations
        if len(edges) < 3:  # Need at least 3 parallel streets
            continue
            
        # Sort edges by distance from start
        edges_with_dist = []
        for u, v, data in edges:
            mid_lat = (graph.nodes[u]['y'] + graph.nodes[v]['y']) / 2
            mid_lon = (graph.nodes[u]['x'] + graph.nodes[v]['x']) / 2
            dist = dist_m(start_y, start_x, mid_lat, mid_lon)
            edges_with_dist.append((u, v, data, dist))
        
        edges_with_dist.sort(key=lambda x: x[3])
        
        # Create zigzag route using these parallel streets
        zigzag_path = _create_zigzag_from_parallel_streets(graph, start_node, edges_with_dist, target_m, avoid_hills, globally_done_set)
        
        if zigzag_path and len(zigzag_path) > 3:
            path_details = _calculate_path_details(zigzag_path, graph, globally_done_set)
            if path_details.get("valid") and target_m * 0.4 <= path_details['distance_m'] <= target_m * 1.8:
                candidates.append({
                    'path': zigzag_path,
                    'distance': path_details['distance_m'],
                    'percentage_new': path_details['percentage_new_distance'],
                    'strategy': 'zigzag'
                })
    
    return candidates

def _create_zigzag_from_parallel_streets(graph, start_node, parallel_edges, target_m, avoid_hills, globally_done_set):
    """Create a zigzag route using parallel streets"""
    if not parallel_edges:
        return None

    # Find the closest parallel street to start
    closest_edge = min(parallel_edges, key=lambda x: x[3])
    u, v, data, dist = closest_edge
    
    # Find path to the closest parallel street
    weight_func = weight_factory(avoid_hills=avoid_hills, prioritize_new_roads=True)
    try:
        path_to_parallel = nx.shortest_path(graph, source=start_node, target=u, weight=weight_func)
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        try:
            path_to_parallel = nx.shortest_path(graph, source=start_node, target=v, weight=weight_func)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None
    
    if len(path_to_parallel) < 2:
        return None
    
    # Start building zigzag path
    zigzag_path = path_to_parallel[:-1]  # Don't include the endpoint yet
    
    current_node = path_to_parallel[-1]
    current_distance = 0
    for i in range(len(zigzag_path)-1):
        if graph.has_edge(zigzag_path[i], zigzag_path[i+1]):
            # Get the edge data safely - handle multiple edges between same nodes
            edges = graph.get_edge_data(zigzag_path[i], zigzag_path[i+1])
            if edges:
                # Get the first edge (or any edge) - they should have similar lengths
                edge_key = list(edges.keys())[0]
                edge_data = edges[edge_key]
                if isinstance(edge_data, dict):
                    current_distance += edge_data.get('length', 0)
    
    # Sort parallel edges by distance from start
    sorted_edges = sorted(parallel_edges, key=lambda x: x[3])
    
    # Create zigzag pattern
    for i, (u, v, data, dist) in enumerate(sorted_edges[:min(5, len(sorted_edges))]):  # Use up to 5 parallel streets
        if current_distance > target_m * 0.8:  # Stop if we're getting close to target
            break
            
        # Find path to this parallel street
        try:
            path_to_next = nx.shortest_path(graph, source=current_node, target=u, weight=weight_func)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            try:
                path_to_next = nx.shortest_path(graph, source=current_node, target=v, weight=weight_func)
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue
        
        if len(path_to_next) < 2:
            continue
        
        # Add the connecting path (excluding the endpoint to avoid duplication)
        zigzag_path.extend(path_to_next[:-1])
        
        # Add the parallel street itself
        zigzag_path.append(u)
        zigzag_path.append(v)
        
        # Update current position and distance
        current_node = v
        # Add distance from connecting path
        for j in range(len(path_to_next)-1):
            if graph.has_edge(path_to_next[j], path_to_next[j+1]):
                edges = graph.get_edge_data(path_to_next[j], path_to_next[j+1])
                if edges:
                    edge_key = list(edges.keys())[0]
                    edge_data = edges[edge_key]
                    if isinstance(edge_data, dict):
                        current_distance += edge_data.get('length', 0)
        # Add distance from the parallel street itself
        current_distance += data.get('length', 0)
    
    # Find path back to start
    try:
        path_back = nx.shortest_path(graph, source=current_node, target=start_node, weight=weight_func)
        zigzag_path.extend(path_back)
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        # If we can't get back to start, just return what we have
        pass
    
    return zigzag_path if len(zigzag_path) > 3 else None

def _score_and_filter_candidates(candidates, target_m):
    """Enhanced scoring system for route selection"""
    if not candidates:
        return []
    
    # Calculate scores for each candidate
    scored_candidates = []
    for candidate in candidates:
        distance = candidate['distance']
        percentage_new = candidate['percentage_new']
        strategy = candidate.get('strategy', 'unknown')
        
        # Distance score (closer to target is better)
        distance_score = 1.0 / (1.0 + abs(distance - target_m) / target_m)
        
        # Newness score (higher is better)
        newness_score = percentage_new / 100.0
        
        # Strategy bonus (prefer variety and efficiency)
        strategy_bonus = {
            'two_leg': 1.0,
            'multi_point': 1.1,  # Slight preference for complexity
            'out_and_back': 1.05,
            'zigzag': 1.2  # Strong preference for zigzag routes (efficient exploration)
        }.get(strategy, 1.0)
        
        # Combined score
        total_score = (distance_score * 0.4 + newness_score * 0.6) * strategy_bonus
        
        scored_candidates.append({
            **candidate,
            'score': total_score,
            'distance_score': distance_score,
            'newness_score': newness_score
        })
    
    # Sort by total score
    scored_candidates.sort(key=lambda x: x['score'], reverse=True)
    
    # Filter to reasonable distance range
    final_candidates = [
        c for c in scored_candidates
        if target_m * 0.6 <= c['distance'] <= target_m * 1.4 # Loosened range
    ]
    
    if not final_candidates and scored_candidates:
        print("No routes in ideal distance range, returning best available options.")
        return scored_candidates

    return final_candidates


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
    
    # More nuanced weather-based distance adjustment
    weather_factor = 1.0
    temp_factor = 1.0
    wind_factor = 1.0
    
    # Temperature adjustment (gradual)
    if w_data["temp_f"] < current_prefs_obj.ideal_min_temp_f:
        temp_diff = current_prefs_obj.ideal_min_temp_f - w_data["temp_f"]
        temp_factor = max(0.7, 1.0 - (temp_diff * 0.02))  # Gradual reduction
    elif w_data["temp_f"] > current_prefs_obj.ideal_max_temp_f:
        temp_diff = w_data["temp_f"] - current_prefs_obj.ideal_max_temp_f
        temp_factor = max(0.7, 1.0 - (temp_diff * 0.02))  # Gradual reduction
    
    # Wind adjustment (gradual)
    if w_data["wind_mph"] > current_prefs_obj.max_wind_mph:
        wind_diff = w_data["wind_mph"] - current_prefs_obj.max_wind_mph
        wind_factor = max(0.6, 1.0 - (wind_diff * 0.03))  # Gradual reduction
    
    # Combined weather factor
    weather_factor = min(temp_factor, wind_factor)
    target_miles_effective *= weather_factor
    
    # Ensure minimum reasonable distance
    if target_miles_effective < 1.0:
        target_miles_effective = 1.0 
    
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
