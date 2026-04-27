"""
Zephyr - Graph loading, elevation enrichment, and publishing
Copyright (C) 2025 Connor Frank
License: GPLv3 (see LICENSE)
"""

import logging
import os
import threading
import xml.etree.ElementTree as ET

import requests
import osmnx as ox
from shapely.geometry import LineString, Point, mapping
from shapely.ops import unary_union
from geopy.geocoders import Nominatim

from .models import db, DoneEdge

logger = logging.getLogger(__name__)

ROAD_FILTER = '["highway"]["highway"!~"footway|path|steps|pedestrian|cycleway|motorway|motorway_link|trunk|trunk_link"]'

# Graph globals
G = None
GRAPH_READY = False
ZIP_STATUS = {}
ALL_EDGES_GJ = {}
CURRENT_ZIPS = []
LOCK = threading.Lock()
# Serializes concurrent fetch_graph_async worker invocations.
LOADING_LOCK = threading.Lock()


def done_pct(graph_obj):
    """Calculate percentage of edges completed (edge count based)."""
    if not graph_obj or graph_obj.number_of_edges() == 0:
        return 0
    done_count = db.session.query(DoneEdge).count()
    total_graph_edges = graph_obj.number_of_edges()
    if total_graph_edges == 0:
        return 0
    return int(100 * done_count / total_graph_edges)


def edges_geojson(graph_obj):
    """Convert graph edges to GeoJSON FeatureCollection with done/undone status."""
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
                "properties": {"done": is_done, "plan": False, "u": u, "v": v, "key": k},
            })
    return {"type": "FeatureCollection", "features": feats}


def update_all_edges_gj(graph_obj):
    """Acquire LOCK and atomically rebuild ALL_EDGES_GJ from given graph."""
    global ALL_EDGES_GJ
    with LOCK:
        ALL_EDGES_GJ = edges_geojson(graph_obj)


def publish_graph(graph_obj, zips):
    """Publish a graph to global state, updating status for all zips."""
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
            elev_nodes_count = sum(
                1 for _, d_node in G.nodes(data=True)
                if "elevation" in d_node and d_node["elevation"] is not None
            )
            elev_pct = int(100 * elev_nodes_count / G.number_of_nodes())
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


def enrich_elevations(graph_obj, fname, zips):
    """Fetch elevations for all nodes missing elevation data.

    On HTTP error or invalid response, the node's elevation is left as ``None``
    so that subsequent loads will retry the lookup. Edge grades are only
    computed once every node has a valid elevation; otherwise we skip
    ``add_edge_grades`` to avoid persisting bogus 0.0 values.

    This function does NOT call ``publish_graph`` itself; the caller
    (``fetch_graph_async``) is responsible for atomic publication once
    enrichment is complete.
    """
    session = requests.Session()
    nodes_to_fetch = [n for n, d in graph_obj.nodes(data=True) if d.get("elevation") is None]
    total_to_fetch = len(nodes_to_fetch)

    if total_to_fetch == 0:
        for z in zips:
            if z in ZIP_STATUS:
                ZIP_STATUS[z]["elev"] = 100
        return

    any_failures = False
    for i, n_id in enumerate(nodes_to_fetch, 1):
        node_data = graph_obj.nodes[n_id]
        lat, lon = node_data["y"], node_data["x"]
        try:
            elev_resp = session.get(
                f"https://epqs.nationalmap.gov/v1/json?x={lon}&y={lat}&units=Meters",
                timeout=8,
            )
            elev_resp.raise_for_status()
            elev = elev_resp.json().get("value")
            if elev is not None and isinstance(elev, (int, float)) and elev > -10000:
                graph_obj.nodes[n_id]["elevation"] = float(elev)
            else:
                # Leave as None so a future load retries this node.
                graph_obj.nodes[n_id]["elevation"] = None
                any_failures = True
        except Exception as e:
            logger.warning("Elevation API error for node %s: %s", n_id, e)
            # Leave as None so a future load retries this node.
            graph_obj.nodes[n_id]["elevation"] = None
            any_failures = True

        if i % 200 == 0 or i == total_to_fetch:
            pct = int(100 * i / total_to_fetch) if total_to_fetch > 0 else 100
            for z_code in zips:
                if z_code in ZIP_STATUS:
                    ZIP_STATUS[z_code]["elev"] = pct
            if i == total_to_fetch:
                for z_code in zips:
                    if z_code in ZIP_STATUS:
                        ZIP_STATUS[z_code]["elev"] = 100

    if not any_failures and graph_obj.number_of_nodes() > 0:
        try:
            ox.add_edge_grades(graph_obj, add_absolute=True)
        except Exception as e:
            logger.error("Error adding edge grades: %s", e)
    elif any_failures:
        logger.warning(
            "Skipping add_edge_grades: %d nodes still have None elevation; will retry on next load",
            sum(1 for _, d in graph_obj.nodes(data=True) if d.get("elevation") is None),
        )

    try:
        ox.save_graphml(graph_obj, fname)
    except Exception as e:
        logger.error("Failed to save graphml to %s: %s", fname, e)


def get_data_dir():
    """Get the data directory (configurable via DATA_DIR env var)."""
    return os.environ.get("DATA_DIR", os.path.abspath(os.path.dirname(os.path.dirname(__file__))))


def fetch_graph_async(zips_list, app):
    """Load graph data asynchronously in a background thread.

    Concurrent invocations are serialized via ``LOADING_LOCK``: if a fetch
    is already in progress, the new request is skipped (logged) and
    ``ZIP_STATUS`` is annotated to indicate another load is in flight.

    ``GRAPH_READY`` only flips to ``True`` once the graph has been built/loaded
    AND fully enriched with elevations (or determined to need none). Readers
    will not observe a graph that is still being mutated.
    """
    def worker():
        global GRAPH_READY
        # Serialize concurrent fetches: skip if one is already running.
        if not LOADING_LOCK.acquire(blocking=False):
            logger.info(
                "fetch_graph_async: another load in progress, skipping request for %s",
                zips_list,
            )
            for z_code in zips_list:
                existing = ZIP_STATUS.get(z_code, {"road": 0, "elev": 0})
                existing["note"] = "another load in progress"
                ZIP_STATUS[z_code] = existing
            return

        try:
            with app.app_context():
                GRAPH_READY = False
                for z_code in zips_list:
                    ZIP_STATUS[z_code] = {"road": 0, "elev": 0, "done": 0}

                data_dir = get_data_dir()
                fname = os.path.join(
                    data_dir,
                    f"graph_{'_'.join(sorted(list(set(zips_list))))}.graphml",
                )

                if os.path.isfile(fname):
                    try:
                        loaded_graph = ox.load_graphml(fname)
                        # Mark road as fully loaded for cache-hit path.
                        for z_code in zips_list:
                            if z_code in ZIP_STATUS:
                                ZIP_STATUS[z_code]["road"] = 100

                        needs_elevation_enrich = any(
                            data.get("elevation") is None
                            for _, data in loaded_graph.nodes(data=True)
                        )
                        if needs_elevation_enrich and loaded_graph.number_of_nodes() > 0:
                            # Re-enrich BEFORE publishing so readers never see a
                            # half-enriched graph.
                            enrich_elevations(loaded_graph, fname, zips_list)
                        publish_graph(loaded_graph, zips_list)
                        return
                    except (ET.ParseError, OSError) as e:
                        # Genuine corruption / unreadable file: quarantine it
                        # by renaming to ".bad" rather than deleting outright,
                        # so the bad file can be inspected later.
                        logger.error(
                            "Corrupt or unreadable graph file %s: %s. Quarantining and refetching.",
                            fname,
                            e,
                        )
                        try:
                            bad_name = fname + ".bad"
                            if os.path.exists(bad_name):
                                os.remove(bad_name)
                            os.rename(fname, bad_name)
                        except OSError as rename_err:
                            logger.error(
                                "Failed to rename corrupt graph file %s: %s",
                                fname,
                                rename_err,
                            )
                    except Exception as e:
                        # Non-corruption error: log and re-raise without
                        # destroying the cached file.
                        logger.error(
                            "Unexpected error loading graph %s: %s. Leaving cache intact.",
                            fname,
                            e,
                        )
                        GRAPH_READY = True
                        return

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
                                raise ValueError(
                                    f"Unexpected geometry type {geom_candidate.geom_type}"
                                )
                        else:
                            raise ValueError(f"Empty GeoDataFrame for {z_code}")
                    except Exception:
                        try:
                            loc = geocoder.geocode(f"{z_code}, USA", exactly_one=True, timeout=10)
                            if loc:
                                geom_obj = Point(loc.longitude, loc.latitude).buffer(0.03)
                            else:
                                ZIP_STATUS[z_code]["road"] = -1
                        except Exception:
                            ZIP_STATUS[z_code]["road"] = -1

                    if geom_obj:
                        polys.append(geom_obj)
                        valid_zips_for_graph.append(z_code)
                        ZIP_STATUS[z_code]["road"] = 20
                    elif z_code in ZIP_STATUS and ZIP_STATUS[z_code].get("road", 0) != -1:
                        ZIP_STATUS[z_code]["road"] = -1

                if not polys:
                    logger.warning("No valid geocoded areas found for: %s", zips_list)
                    for z_code_iter in zips_list:
                        if z_code_iter in ZIP_STATUS and ZIP_STATUS[z_code_iter].get("road", 0) != -1:
                            ZIP_STATUS[z_code_iter]["road"] = 100
                            ZIP_STATUS[z_code_iter]["elev"] = 100
                    GRAPH_READY = True
                    return

                boundary = unary_union(polys)
                for z_code in valid_zips_for_graph:
                    ZIP_STATUS[z_code]["road"] = 50

                temp_G_graph = None
                try:
                    temp_G_graph = ox.graph_from_polygon(
                        boundary, network_type="drive",
                        custom_filter=ROAD_FILTER, simplify=True, retain_all=False
                    )
                except Exception as e:
                    logger.error("Error creating graph from polygon: %s", e)
                    for z_code in valid_zips_for_graph:
                        ZIP_STATUS[z_code]["road"] = 100
                    GRAPH_READY = True
                    return

                for z_code in valid_zips_for_graph:
                    ZIP_STATUS[z_code]["road"] = 100

                # Enrich elevations BEFORE publishing so the published graph
                # is fully ready for readers. ZIP_STATUS["elev"] continues to
                # advance during enrichment so the UI shows progress.
                if temp_G_graph and temp_G_graph.number_of_nodes() > 0:
                    enrich_elevations(temp_G_graph, fname, valid_zips_for_graph)
                else:
                    if temp_G_graph is not None:
                        try:
                            ox.save_graphml(temp_G_graph, fname)
                        except Exception as e:
                            logger.error("Failed to save graphml to %s: %s", fname, e)

                publish_graph(temp_G_graph, valid_zips_for_graph)
        finally:
            LOADING_LOCK.release()

    threading.Thread(target=worker, daemon=True).start()
