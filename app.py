import os, json, threading, math
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
    route_geojson = db.Column(db.Text, nullable=False)
    status = db.Column(db.String(10), default="planned")  # planned|completed


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
    target_miles = db.Column(db.Float, default=3.0)
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
    meta = requests.get(
        f"https://api.weather.gov/points/{lat},{lon}",
        headers={"User-Agent": "Zephyr/0.1"},
        timeout=8,
    ).json()
    hourly = meta["properties"]["forecastHourly"]
    p = requests.get(hourly, headers={"User-Agent": "Zephyr/0.1"}, timeout=8).json()[
        "properties"
    ]["periods"][0]
    return {
        "temp_f": p["temperature"],
        "short": p["shortForecast"],
        "wind_mph": float(p["windSpeed"].split()[0]),
    }


def dist_m(lat1, lon1, lat2, lon2):
    dy = (lat2 - lat1) * 111132
    dx = (lon2 - lon1) * 111320 * math.cos(math.radians((lat1 + lat2) / 2))
    return math.hypot(dx, dy)


# ── graph globals -----------------------------------------------------------
G = None
GRAPH_READY = False
ZIP_STATUS = {}
ALL_EDGES_GJ = {}
CURRENT_ZIPS = []
LOCK = threading.Lock()
ROAD_FILTER = '["highway"]["highway"!~"footway|path|steps|pedestrian|cycleway"]'


def done_pct(graph):
    if not graph:
        return 0
    total = graph.number_of_edges()
    if total == 0:
        return 0
    with app.app_context():
        done = db.session.query(DoneEdge).count()
    return int(100 * done / total)


def edges_geojson(graph):
    with app.app_context():
        done_set = {(e.u, e.v, e.key) for e in DoneEdge.query.all()}
    feats = []
    for u, v, k, d in graph.edges(data=True, keys=True):
        geom = d.get("geometry") or LineString(
            [
                (graph.nodes[u]["x"], graph.nodes[u]["y"]),
                (graph.nodes[v]["x"], graph.nodes[v]["y"]),
            ]
        )
        feats.append(
            {
                "type": "Feature",
                "geometry": mapping(geom),
                "properties": {"done": (u, v, k) in done_set, "plan": False},
            }
        )
    return {"type": "FeatureCollection", "features": feats}


def publish_graph(graph, zips):
    global G, GRAPH_READY, CURRENT_ZIPS, ALL_EDGES_GJ
    with LOCK:
        G = graph
        CURRENT_ZIPS = zips
        ALL_EDGES_GJ = edges_geojson(graph)
        elev = int(
            100
            * sum("elevation" in d for _, d in graph.nodes(data=True))
            / graph.number_of_nodes()
        )
        done = done_pct(graph)
        for z in zips:
            ZIP_STATUS[z] = {"road": 100, "elev": elev, "done": done}
        GRAPH_READY = True


# ── elevation fetch ---------------------------------------------------------
def enrich_elevations(graph, fname, zips):
    session = requests.Session()
    nodes = list(graph.nodes)
    total = len(nodes)
    for i, n in enumerate(nodes, 1):
        if "elevation" in graph.nodes[n]:
            continue
        lat, lon = graph.nodes[n]["y"], graph.nodes[n]["x"]
        try:
            elev = (
                session.get(
                    f"https://epqs.nationalmap.gov/v1/json?x={lon}&y={lat}&units=Meters",
                    timeout=8,
                )
                .json()
                .get("value", 0)
            )
        except Exception:
            elev = 0
        graph.nodes[n]["elevation"] = elev
        if i % 2000 == 0:
            pct = int(100 * i / total)
            for z in zips:
                ZIP_STATUS[z]["elev"] = pct
    ox.add_edge_grades(graph)
    ox.save_graphml(graph, fname)
    for z in zips:
        ZIP_STATUS[z]["elev"] = 100


# ── graph loader ------------------------------------------------------------
def fetch_graph_async(zips):
    def worker():
        global GRAPH_READY
        GRAPH_READY = False
        for z in zips:
            ZIP_STATUS[z] = {"road": 0, "elev": 0, "done": 0}
        fname = os.path.join(BASE_DIR, f"graph_{'_'.join(zips)}.graphml")
        if os.path.isfile(fname):
            publish_graph(ox.load_graphml(fname), zips)
            return
        geocoder = Nominatim(user_agent="Zephyr")
        polys = []
        for z in zips:
            try:
                geom = ox.geocode_to_gdf(f"{z}, USA").loc[0, "geometry"]
                if geom.geom_type == "Point":
                    raise ValueError
            except Exception:
                loc = geocoder.geocode(f"{z} USA", exactly_one=True, timeout=8)
                if not loc:
                    continue
                geom = Point(loc.longitude, loc.latitude).buffer(0.02)
            polys.append(geom)
        boundary = unary_union(polys)
        graph = ox.graph_from_polygon(
            boundary, custom_filter=ROAD_FILTER, simplify=True
        )
        for z in zips:
            ZIP_STATUS[z]["road"] = 100
        publish_graph(graph, zips)
        threading.Thread(
            target=enrich_elevations, args=(graph, fname, zips), daemon=True
        ).start()

    threading.Thread(target=worker, daemon=True).start()


# ── routing helpers ---------------------------------------------------------
def weight_factory(avoid):
    with app.app_context():
        done_set = {(e.u, e.v, e.key) for e in DoneEdge.query.all()}

    def cost(*args):
        if len(args) == 4:
            u, v, k, d = args
        else:
            u, v, d = args
            k = 0
        w = d.get("length", 1.0)
        if avoid:
            w *= 1 + abs(d.get("grade", 0))
        if (u, v, k) in done_set:
            w *= 5
        return w

    return cost


def edge_coords(u, v):
    data = G[u][v][min(G[u][v].keys())]
    geom = data.get("geometry")
    return (
        list(geom.coords)
        if geom
        else [(G.nodes[u]["x"], G.nodes[u]["y"]), (G.nodes[v]["x"], G.nodes[v]["y"])]
    )


def path_geojson(path):
    coords = []
    dist = 0.0
    for u, v in zip(path, path[1:]):
        seg = edge_coords(u, v)
        if coords:
            seg = seg[1:]
        coords.extend(seg)
        dist += G[u][v][min(G[u][v].keys())].get("length", 1.0)
    return {
        "type": "Feature",
        "geometry": mapping(LineString(coords)),
        "properties": {"distance_m": dist, "plan": True},
    }


def auto_path(lat, lon, target, avoid):
    start = ox.distance.nearest_nodes(G, lon, lat)
    radius = 183
    near = [
        n
        for n, d in G.nodes(data=True)
        if dist_m(lat, lon, d["y"], d["x"]) <= radius and n != start
    ] or [start]
    best = None
    err_best = math.inf
    for n in near:
        try:
            p = nx.shortest_path(G, start, n, weight=weight_factory(avoid))
            dist = sum(
                G[u][v][min(G[u][v].keys())].get("length", 1.0)
                for u, v in zip(p, p[1:])
            )
            if dist <= target:
                err = target - dist
                if err < err_best:
                    err_best, best = err, p
        except nx.NetworkXNoPath:
            continue
    # if all paths longer than target, choose shortest
    if not best:
        best = min(
            (nx.shortest_path(G, start, n, weight=weight_factory(avoid)) for n in near),
            key=lambda path: sum(
                G[u][v][min(G[u][v].keys())].get("length", 1.0)
                for u, v in zip(path, path[1:])
            ),
        )
    return best


# ── routes ------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/prefs", methods=["GET", "POST"])
def prefs():
    if request.method == "POST":
        d = request.get_json()
        update_pref(
            ideal_min_temp_f=d.get("tmin"),
            ideal_max_temp_f=d.get("tmax"),
            max_wind_mph=d.get("wmax"),
            target_miles=d.get("target"),
        )
        return {"ok": True}
    p = get_pref()
    return {
        "tmin": p.ideal_min_temp_f,
        "tmax": p.ideal_max_temp_f,
        "wmax": p.max_wind_mph,
        "target": p.target_miles,
        "zip_codes": p.zip_codes,
    }


@app.route("/status")
def status():
    return {"ready": GRAPH_READY, "zips": ZIP_STATUS, "done": done_pct(G)}


@app.route("/set_zipcodes", methods=["POST"])
def set_zips():
    z = [s.strip() for s in request.json.get("zips", []) if s.strip()]
    if not z:
        return {"error": "No zips"}, 400
    update_pref(zip_codes=",".join(z))
    fetch_graph_async(z)
    return {"started": True}


@app.route("/graph")
def graph():
    return jsonify(ALL_EDGES_GJ) if GRAPH_READY else abort(503)


@app.route("/plan_auto")
def plan_auto():
    lat = float(request.args["lat"])
    lon = float(request.args["lon"])
    avoid = request.args.get("avoid") == "true"
    pref = get_pref()
    w = wx(lat, lon)
    target = pref.target_miles * 1609.34
    if (
        w["temp_f"] < pref.ideal_min_temp_f
        or w["temp_f"] > pref.ideal_max_temp_f
        or w["wind_mph"] > pref.max_wind_mph
    ):
        target *= 0.5
    path = auto_path(lat, lon, target, avoid)
    return jsonify(path_geojson(path))


@app.route("/runs", methods=["GET", "POST", "PUT"])
def runs():
    fil = request.args.get("status")
    if request.method == "POST":
        Run.query.filter_by(status="planned").delete()
        d = request.get_json()
        run = Run(
            distance_m=d["distance_m"],
            route_geojson=json.dumps(d["geojson"]),
            status="planned",
        )
        db.session.add(run)
        db.session.commit()
        return {"run_id": run.id}
    if request.method == "PUT":
        rid = int(request.json["run_id"])
        run = Run.query.get_or_404(rid)
        run.status = "completed"
        db.session.commit()
        coords = json.loads(run.route_geojson)["geometry"]["coordinates"]
        nodes = [ox.distance.nearest_nodes(G, x, y) for x, y in coords]
        with app.app_context():
            for u, v in zip(nodes, nodes[1:]):
                if u == v:
                    continue
                if G.has_edge(u, v):
                    k = min(G[u][v].keys())
                    uu, vv = u, v
                elif G.has_edge(v, u):
                    k = min(G[v][u].keys())
                    uu, vv = v, u
                else:
                    continue
                if not DoneEdge.query.filter_by(u=uu, v=vv, key=k).first():
                    db.session.add(DoneEdge(u=uu, v=vv, key=k))
            db.session.commit()
        publish_graph(G, CURRENT_ZIPS)
        return {"ok": True}
    q = Run.query.filter_by(status=fil) if fil else Run.query
    feats = [
        json.loads(r.route_geojson)
        | {"properties": {"run_id": r.id, "status": r.status}}
        for r in q.all()
    ]
    return jsonify({"type": "FeatureCollection", "features": feats})


@app.route("/weather")
def weather():
    return jsonify(wx(float(request.args["lat"]), float(request.args["lon"])))


@app.route("/static/<path:p>")
def staticfile(p):
    return send_from_directory(os.path.join(BASE_DIR, "static"), p)


# ── autoload -----------------------------------------------------------------
with app.app_context():
    pref = get_pref()
    zs = pref.zip_codes.split(",") if pref.zip_codes else []
    cache = os.path.join(BASE_DIR, f"graph_{'_'.join(zs)}.graphml")
    if zs and os.path.isfile(cache):
        publish_graph(ox.load_graphml(cache), zs)
    elif zs:
        fetch_graph_async(zs)

if __name__ == "__main__":
    app.run(debug=True)
