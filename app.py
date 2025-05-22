import os, json, threading, math, random
from datetime import date
from functools import lru_cache

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
    route_geojson = db.Column(db.Text, nullable=False) # GeoJSON now stores node_ids in properties
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


def done_pct(graph_obj):
    if not graph_obj or graph_obj.number_of_edges() == 0:
        return 0
    with app.app_context():
        done_count = db.session.query(DoneEdge).count()
    total_graph_edges = graph_obj.number_of_edges()
    if total_graph_edges == 0: return 0
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

        current_done_pct = done_pct(G)
        for z in zips:
            zip_data = ZIP_STATUS.get(z, {"road": 0, "elev": 0})
            zip_data["done"] = current_done_pct
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
                        print(f"Zip {z_code} geocoded as {geom_candidate.geom_type} by ox.geocode_to_gdf.")
                    elif geom_candidate.geom_type == "Point":
                        geom_obj = Point(geom_candidate.x, geom_candidate.y).buffer(0.03)
                        print(f"Zip {z_code} geocoded as Point by ox.geocode_to_gdf and buffered.")
                    else:
                        raise ValueError(f"Unexpected geometry type {geom_candidate.geom_type} from ox.geocode_to_gdf for {z_code}")
                else:
                    raise ValueError(f"Empty GeoDataFrame from ox.geocode_to_gdf for {z_code}")

            except Exception as e_ox:
                print(f"OSMnx geocoding attempt for zip {z_code} failed: {e_ox}. Falling back to geopy.Nominatim direct call.")
                try:
                    loc = geocoder.geocode(f"{z_code}, USA", exactly_one=True, timeout=10)
                    if loc:
                        geom_obj = Point(loc.longitude, loc.latitude).buffer(0.03)
                        print(f"Zip {z_code} geocoded as Point by geopy.Nominatim and buffered.")
                    else:
                        print(f"geopy.Nominatim also failed to geocode zip {z_code}.")
                        ZIP_STATUS[z_code]["road"] = -1
                except Exception as e_nom:
                    print(f"Error during geopy.Nominatim fallback for {z_code}: {e_nom}")
                    ZIP_STATUS[z_code]["road"] = -1

            if geom_obj:
                polys.append(geom_obj)
                valid_zips_for_graph.append(z_code)
                ZIP_STATUS[z_code]["road"] = 20
            elif z_code in ZIP_STATUS and ZIP_STATUS[z_code].get("road", 0) != -1:
                print(f"Zip {z_code} could not be geocoded into a usable geometry after all attempts.")
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
def weight_factory(avoid_hills, prioritize_new_roads=True, temp_avoid_edges=None):
    with app.app_context():
        globally_done_set = {(e.u, e.v, e.key) for e in DoneEdge.query.all()}

    current_path_avoid_set = temp_avoid_edges if temp_avoid_edges is not None else set()

    def cost_func(u, v, edge_data):
        k = edge_data.get('_edge_key_')
        weight = float(edge_data.get("length", 1.0))

        if prioritize_new_roads:
            if k is not None and (u, v, k) in globally_done_set:
                weight *= 1000 
        
        # Penalize edges already used in the first leg of the current loop plan
        # Check both (u,v,k) and (v,u,k) if current_path_avoid_set stores undirected segments
        # For now, assuming current_path_avoid_set stores directed (u,v,k) from path1
        if k is not None and (u, v, k) in current_path_avoid_set:
            weight *= 50 # Heavy penalty for re-using path1 segment in path2

        if avoid_hills:
            grade = abs(float(edge_data.get("grade_abs", edge_data.get("grade", 0.0))))
            weight *= (1 + grade * 10)

        return max(weight, 0.0001)
    return cost_func


def path_geojson(path_nodes, graph_ref):
    if not path_nodes or len(path_nodes) < 2:
        return {"type": "Feature", "geometry": None,
                "properties": {"distance_m": 0, "total_ascent_m": 0, "total_descent_m": 0, "node_ids": [], "plan": True}}

    coords = []
    total_distance_m = 0.0
    total_ascent_m = 0.0
    total_descent_m = 0.0

    for i in range(len(path_nodes) - 1):
        u_node_id, v_node_id = path_nodes[i], path_nodes[i+1]
        if u_node_id == v_node_id: continue

        best_k = None
        min_len = float('inf')

        node_u_data = graph_ref.nodes.get(u_node_id)
        node_v_data = graph_ref.nodes.get(v_node_id)
        if not node_u_data or not node_v_data:
            print(f"Warning: Node data missing for {u_node_id} or {v_node_id} in path_geojson.")
            continue

        if graph_ref.has_edge(u_node_id, v_node_id):
            for k_edge, data_edge in graph_ref[u_node_id][v_node_id].items():
                current_len = data_edge.get('length', float('inf'))
                if current_len < min_len:
                    min_len = current_len
                    best_k = k_edge

        if best_k is not None:
            edge_data = graph_ref[u_node_id][v_node_id][best_k]
            total_distance_m += edge_data.get("length", 0.0)

            elev_u = node_u_data.get('elevation')
            elev_v = node_v_data.get('elevation')
            if elev_u is not None and elev_v is not None:
                diff = elev_v - elev_u
                if diff > 0:
                    total_ascent_m += diff
                else:
                    total_descent_m += abs(diff)

            segment_coords_geom = edge_data.get("geometry")
            if segment_coords_geom:
                s_coords_list = list(segment_coords_geom.coords)
            else:
                s_coords_list = [(node_u_data['x'], node_u_data['y']), (node_v_data['x'], node_v_data['y'])]

            if coords and s_coords_list and coords[-1] == s_coords_list[0]:
                coords.extend(s_coords_list[1:])
            else:
                coords.extend(s_coords_list)
        else:
            print(f"Warning: No edge found or usable between {u_node_id} and {v_node_id} in path_geojson calculation.")

    if not coords: # Should not happen if path_nodes is valid
        return {"type": "Feature", "geometry": None,
                "properties": {"distance_m": 0, "total_ascent_m": 0, "total_descent_m": 0, "node_ids": path_nodes if path_nodes else [], "plan": True}}
    return {
        "type": "Feature",
        "geometry": mapping(LineString(coords)),
        "properties": {
            "distance_m": total_distance_m,
            "total_ascent_m": total_ascent_m,
            "total_descent_m": total_descent_m,
            "node_ids": path_nodes, # Store the node IDs
            "plan": True
        },
    }

def auto_path(lat, lon, target_miles, avoid_hills): # Generates LOOP routes
    if not G or G.number_of_nodes() == 0:
        print("auto_path (loop): Graph not ready or empty.")
        return None
    try:
        start_node = ox.nearest_nodes(G, X=lon, Y=lat)
    except Exception as e:
        print(f"auto_path (loop): Error finding nearest node for ({lat},{lon}): {e}")
        return None

    target_m = target_miles * 1609.34
    print(f"auto_path (loop): Start node {start_node}, Target loop: {target_m:.0f}m")

    start_node_data = G.nodes[start_node]
    start_node_y, start_node_x = start_node_data['y'], start_node_data['x']

    leg1_weight_func = weight_factory(avoid_hills=avoid_hills, prioritize_new_roads=True, temp_avoid_edges=None)
    
    half_target_m = target_m / 2.0
    min_intermediate_radius_m = max(300.0, half_target_m * 0.4) 
    max_intermediate_radius_m = max(750.0, half_target_m * 0.8) 
    
    print(f"auto_path (loop): Searching for intermediate nodes between {min_intermediate_radius_m:.0f}m and {max_intermediate_radius_m:.0f}m straight-line radius.")

    intermediate_nodes = []
    for node_id, node_data in G.nodes(data=True):
        if node_id == start_node: continue
        d = dist_m(start_node_y, start_node_x, node_data['y'], node_data['x'])
        if min_intermediate_radius_m <= d <= max_intermediate_radius_m:
            intermediate_nodes.append(node_id)
    
    # Limit intermediate nodes for performance
    if len(intermediate_nodes) > 50: # Reduced sample size further
        intermediate_nodes = random.sample(intermediate_nodes, 50)
        print(f"auto_path (loop): Sampled to 50 intermediate nodes.")
    
    if not intermediate_nodes:
        fallback_radius = max(500.0, half_target_m * 0.5) # Reduced fallback radius
        print(f"auto_path (loop): No intermediate nodes in band, trying fallback radius {fallback_radius:.0f}m.")
        for node_id, node_data in G.nodes(data=True):
            if node_id == start_node: continue
            if node_id in intermediate_nodes: continue 
            d = dist_m(start_node_y, start_node_x, node_data['y'], node_data['x'])
            if d <= fallback_radius: intermediate_nodes.append(node_id)
        if len(intermediate_nodes) > 25: # Smaller sample for fallback
             intermediate_nodes = random.sample(intermediate_nodes, 25)

    print(f"auto_path (loop): Testing {len(intermediate_nodes)} intermediate nodes.")
    if not intermediate_nodes:
        print("auto_path (loop): No suitable intermediate nodes found. Cannot generate loop.")
        return None

    candidate_loop_paths = []

    for intermediate_node in intermediate_nodes:
        try:
            path1_nodes = nx.shortest_path(G, source=start_node, target=intermediate_node, weight=leg1_weight_func)
            if len(path1_nodes) < 2: continue

            path1_edges_set = set()
            for i in range(len(path1_nodes) - 1):
                u1, v1 = path1_nodes[i], path1_nodes[i+1]
                if G.has_edge(u1,v1):
                    for k1_actual in G[u1][v1]: # Get actual keys of edges between u1,v1
                        # We need to use the key that shortest_path would have chosen for path1.
                        # This is the key that corresponds to the edge data used by leg1_weight_func.
                        # This detail is complex. A simpler heuristic is to penalize any edge (u1,v1,k) or (v1,u1,k).
                        # For now, let's add all (u1,v1,k_actual) to the temp_avoid_set.
                        # This might be overly restrictive if parallel edges are very different.
                        path1_edges_set.add((u1,v1,k1_actual))
                        # To discourage exact retracing, also add reverse if it exists (more robust for two-way streets)
                        # This requires knowing the reverse edge's key.
                        # Simpler: penalize node pairs (see previous thought block if this is too slow/complex)
                        # For now, using directed edges from path1.
            
            leg2_weight_func = weight_factory(avoid_hills=avoid_hills, 
                                              prioritize_new_roads=True, 
                                              temp_avoid_edges=path1_edges_set)

            path2_nodes = nx.shortest_path(G, source=intermediate_node, target=start_node, weight=leg2_weight_func)
            if len(path2_nodes) < 2: continue
            
            loop_path_nodes = path1_nodes[:-1] + path2_nodes 
            if len(loop_path_nodes) < 3 : continue

            loop_distance_m = 0.0; valid_loop_path = True
            for i in range(len(loop_path_nodes) - 1):
                u, v = loop_path_nodes[i], loop_path_nodes[i+1]
                if not G.has_edge(u,v): valid_loop_path = False; break
                edge_options = G[u][v]
                chosen_edge_data = None; min_l = float('inf')
                for k,d_edge in edge_options.items():
                    l = d_edge.get('length', float('inf')); 
                    if l < min_l: min_l = l; chosen_edge_data = d_edge
                if chosen_edge_data and 'length' in chosen_edge_data: loop_distance_m += chosen_edge_data['length']
                else: valid_loop_path = False; break
            
            if not valid_loop_path: continue

            if target_m * 0.4 <= loop_distance_m <= target_m * 2.5: # Wider initial filter for loops
                candidate_loop_paths.append({'path': loop_path_nodes, 'distance': loop_distance_m})
                # print(f"  Found candidate loop via {intermediate_node}, dist: {loop_distance_m:.0f}m")

        except (nx.NetworkXNoPath, nx.NodeNotFound): continue
        except Exception as e: 
            print(f"Error constructing loop via intermediate {intermediate_node}: {e}")
            continue
    
    if not candidate_loop_paths:
        print("auto_path (loop): No candidate loops constructed.")
        return None

    paths_meeting_criteria = [
        p for p in candidate_loop_paths 
        if p['distance'] <= target_m * 1.05 
    ]

    best_overall_path_details = None
    if paths_meeting_criteria:
        paths_meeting_criteria.sort(key=lambda p_item: abs(p_item['distance'] - target_m)) 
        best_overall_path_details = paths_meeting_criteria[0]
        print(f"auto_path (loop): Selected from {len(paths_meeting_criteria)} paths <= target*1.05. Chosen dist: {best_overall_path_details['distance']:.0f}m.")
    elif candidate_loop_paths: 
        print(f"auto_path (loop): No paths <= target*1.05. Picking shortest from {len(candidate_loop_paths)} available loops.")
        candidate_loop_paths.sort(key=lambda p_item: p_item['distance']) 
        best_overall_path_details = candidate_loop_paths[0]
        print(f"auto_path (loop): Selected shortest fallback. Dist: {best_overall_path_details['distance']:.0f}m.")
        
    if best_overall_path_details:
        return best_overall_path_details['path']
    else:
        print("auto_path (loop): No loop path selected after all filtering.")
        return None


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

@app.route("/status")
def status():
    global_done_pct = done_pct(G) if G else 0
    current_zip_status_copy = {}
    with LOCK:
        current_zip_status_copy = ZIP_STATUS.copy()
        is_graph_ready = GRAPH_READY
    final_zip_status = {}
    for zc, data in current_zip_status_copy.items():
        current_elev_pct = data.get("elev", 0) 
        if is_graph_ready and G and zc in CURRENT_ZIPS : 
            if G.number_of_nodes() > 0: 
                elev_count = sum(1 for _, d_node in G.nodes(data=True) if "elevation" in d_node and d_node['elevation'] is not None)
                current_elev_pct = int(100 * elev_count / G.number_of_nodes())
            else: 
                current_elev_pct = 100 
        
        final_zip_status[zc] = {
            "road": data.get("road", 100 if is_graph_ready and G and zc in CURRENT_ZIPS else 0),
            "elev": current_elev_pct,
            "done": global_done_pct
        }
        if not (is_graph_ready and zc in CURRENT_ZIPS):
             final_zip_status[zc]["road"] = data.get("road", 0)
             final_zip_status[zc]["elev"] = data.get("elev",0)


    return jsonify({"ready": is_graph_ready, "zips": final_zip_status, "done": global_done_pct})

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
        return jsonify({"error": "Graph not ready"}), 503
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
    
    print(f"/plan_auto: Requesting loop route: lat={lat}, lon={lon}, target_eff_miles={target_miles_effective}, avoid_hills={avoid}")
    path_nodes = auto_path(lat, lon, target_miles_effective, avoid)
    
    if path_nodes is None: 
        print("/plan_auto: auto_path returned None. No suitable loop route found.")
        return jsonify({}) 
        
    print(f"/plan_auto: auto_path successful for loop, {len(path_nodes)} nodes. Generating GeoJSON.")
    return jsonify(path_geojson(path_nodes, G))

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
        
        # Ensure GeoJSON properties are included when saving
        geojson_data_to_save = d["geojson"]
        if not isinstance(geojson_data_to_save, dict) : # Should be a dict (Feature obj)
             return jsonify({"error": "GeoJSON must be a valid Feature object"}), 400
        if "properties" not in geojson_data_to_save:
            geojson_data_to_save["properties"] = {}
        
        # Ensure all required properties from path_geojson are present
        geojson_data_to_save["properties"]["distance_m"] = d["distance_m"]
        geojson_data_to_save["properties"]["total_ascent_m"] = d["geojson"].get("properties",{}).get("total_ascent_m", 0)
        geojson_data_to_save["properties"]["total_descent_m"] = d["geojson"].get("properties",{}).get("total_descent_m", 0)
        geojson_data_to_save["properties"]["node_ids"] = d["geojson"].get("properties",{}).get("node_ids", [])
        
        geojson_str = json.dumps(geojson_data_to_save)

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
            # **MODIFICATION**: Use stored node_ids if available for accuracy
            path_node_ids = geojson_data.get("properties", {}).get("node_ids")
            
            if not path_node_ids: # Fallback to snapping coordinates if node_ids weren't stored
                print(f"Warning: node_ids not found in GeoJSON for run {rid}. Falling back to coordinate snapping.")
                path_coords = geojson_data.get("geometry", {}).get("coordinates", [])
                if G and path_coords and len(path_coords) >= 2:
                    path_node_ids = ox.nearest_nodes(G, X=[c[0] for c in path_coords], Y=[c[1] for c in path_coords])
                else:
                    path_node_ids = []
            
            if G and path_node_ids and len(path_node_ids) >= 2:
                print(f"Marking {len(path_node_ids)-1} segments as done for run {rid} using stored/derived node_ids.")
                for i in range(len(path_node_ids) - 1):
                    u_node, v_node = path_node_ids[i], path_node_ids[i+1]
                    if u_node == v_node: continue
                    if G.has_edge(u_node, v_node):
                        # Iterate all parallel edges and mark them based on the (u,v) from path_node_ids
                        # This assumes the specific key used in the path is not critical for "done" status,
                        # or that all parallel ways are considered "done" if one is run.
                        # For more precision, the exact (u,v,k) list should be stored.
                        # The current path_geojson stores node_ids, not specific (u,v,k) edges.
                        for k_edge in G[u_node][v_node]: 
                            if not DoneEdge.query.filter_by(u=u_node, v=v_node, key=k_edge).first():
                                db.session.add(DoneEdge(u=u_node, v=v_node, key=k_edge))
                                new_done_edges_added_in_tx = True
                                print(f"  Added DoneEdge: ({u_node}, {v_node}, {k_edge})")
            
            if new_done_edges_added_in_tx:
                db.session.commit() 
                print(f"Committed {db.session.query(DoneEdge).count()} total done edges to DB.")
                ALL_EDGES_GJ = edges_geojson(G) 
                print("Regenerated ALL_EDGES_GJ for map.")
            else:
                db.session.commit() # Commit run status change
        
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
            gj_feature["properties"] = gj_feature.get("properties", {}) 
            gj_feature["properties"]["run_id"] = r_item.id
            gj_feature["properties"]["status"] = r_item.status
            gj_feature["properties"]["date"] = r_item.date.isoformat()
            gj_feature["properties"]["distance_m"] = r_item.distance_m 
            gj_feature["properties"]["total_ascent_m"] = gj_feature["properties"].get("total_ascent_m", 0)
            gj_feature["properties"]["total_descent_m"] = gj_feature["properties"].get("total_descent_m", 0)
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

# ── autoload -----------------------------------------------------------------
with app.app_context():
    initial_prefs = get_pref()
    if initial_prefs and initial_prefs.zip_codes:
        zs = [s.strip() for s in initial_prefs.zip_codes.split(',') if s.strip()]
        if zs:
            print(f"Autoloading graph for zips: {zs} (async from init)")
            fetch_graph_async(zs)

if __name__ == "__main__":
    app.run(debug=True, host='127.0.0.1', port=5000)
