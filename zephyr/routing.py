"""
Zephyr - Route generation strategies and scoring
Copyright (C) 2025 Connor Frank
License: GPLv3 (see LICENSE)
"""

import math
import random

import networkx as nx
import osmnx as ox
from shapely.geometry import LineString, mapping

from .models import db, DoneEdge
from .helpers import dist_m
from . import graph as graph_mod


def weight_factory(avoid_hills, prioritize_new_roads=True,
                   temp_avoid_node_pairs=None, route_variety_factor=1.0):
    globally_done_set = {(e.u, e.v, e.key) for e in DoneEdge.query.all()}
    current_path_avoid_node_pairs = temp_avoid_node_pairs or set()

    def cost_func(u, v, edge_data):
        k = edge_data.get('_edge_key_')
        weight = float(edge_data.get("length", 1.0))

        if tuple(sorted((u, v))) in current_path_avoid_node_pairs:
            weight *= 1000

        if prioritize_new_roads and k is not None:
            if (u, v, k) in globally_done_set:
                weight *= 15
            else:
                weight *= 0.8

        if avoid_hills:
            grade = abs(float(edge_data.get("grade_abs", edge_data.get("grade", 0.0))))
            grade_penalty = 1 + (grade * 5)
            weight *= grade_penalty

        weight *= route_variety_factor
        return max(weight, 0.0001)

    return cost_func


def _calculate_path_details(path_nodes, graph_ref, globally_done_set):
    if not path_nodes or len(path_nodes) < 2:
        return {"distance_m": 0, "total_ascent_m": 0, "total_descent_m": 0,
                "new_distance_m": 0, "percentage_new_distance": 0, "valid": False}

    total_distance_m = 0.0
    new_distance_m = 0.0
    total_ascent_m = 0.0
    total_descent_m = 0.0
    valid_path = True

    for i in range(len(path_nodes) - 1):
        u_node_id, v_node_id = path_nodes[i], path_nodes[i + 1]
        if u_node_id == v_node_id:
            continue

        node_u_data = graph_ref.nodes.get(u_node_id)
        node_v_data = graph_ref.nodes.get(v_node_id)
        if not node_u_data or not node_v_data:
            valid_path = False
            break

        best_k, min_len, chosen_edge_data = None, float('inf'), None
        if graph_ref.has_edge(u_node_id, v_node_id):
            for k_edge, data_edge in graph_ref[u_node_id][v_node_id].items():
                current_len = data_edge.get('length', float('inf'))
                if current_len < min_len:
                    min_len = current_len
                    best_k = k_edge
                    chosen_edge_data = data_edge

        if chosen_edge_data and 'length' in chosen_edge_data:
            segment_length = chosen_edge_data['length']
            total_distance_m += segment_length
            if best_k is not None and (u_node_id, v_node_id, best_k) not in globally_done_set:
                new_distance_m += segment_length

            elev_u = node_u_data.get('elevation')
            elev_v = node_v_data.get('elevation')
            if elev_u is not None and elev_v is not None:
                diff = elev_v - elev_u
                if diff > 0:
                    total_ascent_m += diff
                else:
                    total_descent_m += abs(diff)
        else:
            valid_path = False
            break

    if not valid_path:
        return {"distance_m": float('inf'), "valid": False}

    percentage_new_distance = round(
        (new_distance_m / total_distance_m) * 100, 1
    ) if total_distance_m > 0 else 0.0

    return {
        "distance_m": total_distance_m,
        "total_ascent_m": total_ascent_m,
        "total_descent_m": total_descent_m,
        "new_distance_m": new_distance_m,
        "percentage_new_distance": percentage_new_distance,
        "valid": True,
    }


def path_geojson(path_nodes, graph_ref):
    globally_done_set = {(e.u, e.v, e.key) for e in DoneEdge.query.all()}
    details = _calculate_path_details(path_nodes, graph_ref, globally_done_set)

    if not path_nodes or len(path_nodes) < 2 or not details.get("valid"):
        return {
            "type": "Feature", "geometry": None,
            "properties": {
                "distance_m": 0, "total_ascent_m": 0, "total_descent_m": 0,
                "node_ids": [], "new_distance_m": 0, "percentage_new_distance": 0,
                "plan": True,
            },
        }

    coords = []
    for i in range(len(path_nodes) - 1):
        u_node_id, v_node_id = path_nodes[i], path_nodes[i + 1]
        if u_node_id == v_node_id:
            continue

        node_u_data = graph_ref.nodes.get(u_node_id)
        node_v_data = graph_ref.nodes.get(v_node_id)

        best_k, min_len, chosen_edge_data = None, float('inf'), None
        if graph_ref.has_edge(u_node_id, v_node_id):
            for k_edge, data_edge in graph_ref[u_node_id][v_node_id].items():
                current_len = data_edge.get('length', float('inf'))
                if current_len < min_len:
                    min_len = current_len
                    best_k = k_edge
                    chosen_edge_data = data_edge

        if chosen_edge_data:
            segment_coords_geom = chosen_edge_data.get("geometry")
            if segment_coords_geom:
                s_coords_list = list(segment_coords_geom.coords)
            else:
                s_coords_list = [
                    (node_u_data['x'], node_u_data['y']),
                    (node_v_data['x'], node_v_data['y']),
                ]
            if coords and s_coords_list and coords[-1] == s_coords_list[0]:
                coords.extend(s_coords_list[1:])
            else:
                coords.extend(s_coords_list)

    return {
        "type": "Feature",
        "geometry": mapping(LineString(coords)) if coords else None,
        "properties": {
            "distance_m": details["distance_m"],
            "total_ascent_m": details["total_ascent_m"],
            "total_descent_m": details["total_descent_m"],
            "node_ids": path_nodes,
            "new_distance_m": details["new_distance_m"],
            "percentage_new_distance": details["percentage_new_distance"],
            "plan": True,
        },
    }


def auto_path(lat, lon, target_miles, avoid_hills,
              num_intermediate_samples=50, top_n_paths_to_return=8):
    """Generate candidate routes from a starting point."""
    # Thread-safe access to global graph
    with graph_mod.LOCK:
        graph_ref = graph_mod.G

    if not graph_ref or graph_ref.number_of_nodes() == 0:
        print("auto_path: Graph not ready or empty.")
        return None

    try:
        start_node = ox.nearest_nodes(graph_ref, X=lon, Y=lat)
    except Exception as e:
        print(f"auto_path: Error finding nearest node for ({lat},{lon}): {e}")
        return None

    target_m = target_miles * 1609.34
    print(f"auto_path: Start {start_node}, Target: {target_m:.0f}m")

    start_node_data = graph_ref.nodes[start_node]
    start_node_y, start_node_x = start_node_data['y'], start_node_data['x']

    # Create a subgraph for faster processing
    radius_m = target_m * 0.75
    subgraph_nodes = [
        n for n, d in graph_ref.nodes(data=True)
        if dist_m(start_node_y, start_node_x, d['y'], d['x']) < radius_m
    ]
    if start_node not in subgraph_nodes:
        subgraph_nodes.append(start_node)

    if len(subgraph_nodes) < 30:
        H = graph_ref
        print(f"Using full graph ({graph_ref.number_of_nodes()} nodes) due to sparse area.")
    else:
        H = graph_ref.subgraph(subgraph_nodes).copy()
        print(f"Created subgraph with {H.number_of_nodes()} nodes and {H.number_of_edges()} edges.")

    globally_done_set = {(e.u, e.v, e.key) for e in DoneEdge.query.all()}

    all_candidates = []

    candidates_1 = _generate_two_leg_loops(
        H, start_node, target_m, avoid_hills, globally_done_set,
        num_intermediate_samples, start_node_y, start_node_x
    )
    all_candidates.extend(candidates_1)

    candidates_2 = _generate_multi_point_routes(
        H, start_node, target_m, avoid_hills, globally_done_set,
        start_node_y, start_node_x
    )
    all_candidates.extend(candidates_2)

    candidates_3 = _generate_out_and_back_routes(
        H, start_node, target_m, avoid_hills, globally_done_set,
        start_node_y, start_node_x
    )
    all_candidates.extend(candidates_3)

    candidates_4 = _generate_zigzag_routes(
        H, start_node, target_m, avoid_hills, globally_done_set,
        start_node_y, start_node_x
    )
    all_candidates.extend(candidates_4)

    if not all_candidates:
        print("auto_path: No candidate routes found with any strategy.")
        return None

    scored_candidates = _score_and_filter_candidates(all_candidates, target_m)

    if not scored_candidates:
        print("auto_path: No routes passed scoring criteria.")
        return None

    selected = scored_candidates[:top_n_paths_to_return]
    selected_results = [
        {'path': p['path'], 'strategy': p.get('strategy', '')}
        for p in selected
    ]

    if selected_results:
        best = selected[0]
        print(
            f"auto_path: Returning {len(selected_results)} paths. "
            f"Top: newness {best['percentage_new']:.1f}%, dist {best['distance']:.0f}m."
        )

    return selected_results if selected_results else None


def _generate_two_leg_loops(graph, start_node, target_m, avoid_hills,
                            globally_done_set, num_samples, start_y, start_x):
    """Enhanced two-leg loop generation with multiple radius ranges.

    BUG FIX: Previously, the iteration over intermediate_nodes was outside the
    radius_ranges loop, so only the last range's nodes were used. Now all ranges
    contribute candidates.
    """
    candidates = []

    radius_ranges = [
        (target_m * 0.3, target_m * 0.7),
        (target_m * 0.4, target_m * 0.8),
        (target_m * 0.5, target_m * 0.9),
    ]

    for min_radius, max_radius in radius_ranges:
        intermediate_nodes = []
        for node_id, node_data in graph.nodes(data=True):
            if node_id == start_node:
                continue
            d = dist_m(start_y, start_x, node_data['y'], node_data['x'])
            if min_radius <= d <= max_radius:
                intermediate_nodes.append(node_id)

        samples_per_range = num_samples // len(radius_ranges)
        if len(intermediate_nodes) > samples_per_range:
            intermediate_nodes = random.sample(intermediate_nodes, samples_per_range)

        # BUG FIX: This loop was previously OUTSIDE the radius_ranges for loop
        for intermediate_node in intermediate_nodes:
            try:
                leg1_weight_func = weight_factory(
                    avoid_hills=avoid_hills, prioritize_new_roads=True
                )
                path1_nodes = nx.shortest_path(
                    graph, source=start_node, target=intermediate_node,
                    weight=leg1_weight_func
                )
                if len(path1_nodes) < 2:
                    continue

                path1_segments = {
                    tuple(sorted((path1_nodes[i], path1_nodes[i + 1])))
                    for i in range(len(path1_nodes) - 1)
                }
                leg2_weight_func = weight_factory(
                    avoid_hills=avoid_hills, prioritize_new_roads=True,
                    temp_avoid_node_pairs=path1_segments
                )
                path2_nodes = nx.shortest_path(
                    graph, source=intermediate_node, target=start_node,
                    weight=leg2_weight_func
                )
                if len(path2_nodes) < 2:
                    continue

                loop_path_nodes = path1_nodes[:-1] + path2_nodes
                if len(loop_path_nodes) < 3:
                    continue

                path_details = _calculate_path_details(
                    loop_path_nodes, graph, globally_done_set
                )
                if not path_details.get("valid"):
                    continue

                if target_m * 0.4 <= path_details['distance_m'] <= target_m * 1.8:
                    candidates.append({
                        'path': loop_path_nodes,
                        'distance': path_details['distance_m'],
                        'percentage_new': path_details['percentage_new_distance'],
                        'strategy': 'two_leg',
                    })
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue
            except Exception as e:
                print(f"Error in two-leg loop via {intermediate_node}: {e}")
                continue

    return candidates


def _generate_multi_point_routes(graph, start_node, target_m, avoid_hills,
                                 globally_done_set, start_y, start_x):
    candidates = []
    if not graph:
        return candidates

    waypoint_candidates = []
    for node_id, node_data in graph.nodes(data=True):
        if node_id == start_node:
            continue
        d = dist_m(start_y, start_x, node_data['y'], node_data['x'])
        if target_m * 0.2 <= d <= target_m * 0.6:
            waypoint_candidates.append(node_id)

    if len(waypoint_candidates) > 20:
        waypoint_candidates = random.sample(waypoint_candidates, 20)

    for waypoint in waypoint_candidates:
        try:
            leg1_weight = weight_factory(avoid_hills=avoid_hills, prioritize_new_roads=True)
            path1 = nx.shortest_path(graph, source=start_node, target=waypoint, weight=leg1_weight)
            if len(path1) < 2:
                continue

            path1_segments = {
                tuple(sorted((path1[i], path1[i + 1])))
                for i in range(len(path1) - 1)
            }
            for return_point in waypoint_candidates[:10]:
                if return_point == waypoint:
                    continue
                try:
                    leg2_weight = weight_factory(
                        avoid_hills=avoid_hills, prioritize_new_roads=True,
                        temp_avoid_node_pairs=path1_segments
                    )
                    path2 = nx.shortest_path(
                        graph, source=waypoint, target=return_point, weight=leg2_weight
                    )
                    if len(path2) < 2:
                        continue

                    all_segments = path1_segments | {
                        tuple(sorted((path2[i], path2[i + 1])))
                        for i in range(len(path2) - 1)
                    }
                    leg3_weight = weight_factory(
                        avoid_hills=avoid_hills, prioritize_new_roads=True,
                        temp_avoid_node_pairs=all_segments
                    )
                    path3 = nx.shortest_path(
                        graph, source=return_point, target=start_node, weight=leg3_weight
                    )
                    if len(path3) < 2:
                        continue

                    multi_path = path1[:-1] + path2[:-1] + path3
                    if len(multi_path) < 4:
                        continue

                    path_details = _calculate_path_details(multi_path, graph, globally_done_set)
                    if not path_details.get("valid"):
                        continue

                    if target_m * 0.5 <= path_details['distance_m'] <= target_m * 1.6:
                        candidates.append({
                            'path': multi_path,
                            'distance': path_details['distance_m'],
                            'percentage_new': path_details['percentage_new_distance'],
                            'strategy': 'multi_point',
                        })
                        break
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    continue
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            continue
        except Exception as e:
            print(f"Error in multi-point route via {waypoint}: {e}")
            continue

    return candidates


def _generate_out_and_back_routes(graph, start_node, target_m, avoid_hills,
                                  globally_done_set, start_y, start_x):
    candidates = []
    if not graph:
        return candidates

    turnaround_candidates = []
    for node_id, node_data in graph.nodes(data=True):
        if node_id == start_node:
            continue
        d = dist_m(start_y, start_x, node_data['y'], node_data['x'])
        if target_m * 0.3 <= d <= target_m * 0.7:
            turnaround_candidates.append(node_id)

    if len(turnaround_candidates) > 15:
        turnaround_candidates = random.sample(turnaround_candidates, 15)

    for turnaround in turnaround_candidates:
        try:
            out_weight = weight_factory(avoid_hills=avoid_hills, prioritize_new_roads=True)
            out_path = nx.shortest_path(
                graph, source=start_node, target=turnaround, weight=out_weight
            )
            if len(out_path) < 2:
                continue

            out_segments = {
                tuple(sorted((out_path[i], out_path[i + 1])))
                for i in range(len(out_path) - 1)
            }
            back_weight = weight_factory(
                avoid_hills=avoid_hills, prioritize_new_roads=True,
                temp_avoid_node_pairs=out_segments
            )
            back_path = nx.shortest_path(
                graph, source=turnaround, target=start_node, weight=back_weight
            )
            if len(back_path) < 2:
                continue

            out_and_back = out_path[:-1] + back_path
            if len(out_and_back) < 3:
                continue

            path_details = _calculate_path_details(out_and_back, graph, globally_done_set)
            if not path_details.get("valid"):
                continue

            if target_m * 0.4 <= path_details['distance_m'] <= target_m * 1.7:
                candidates.append({
                    'path': out_and_back,
                    'distance': path_details['distance_m'],
                    'percentage_new': path_details['percentage_new_distance'],
                    'strategy': 'out_and_back',
                })
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            continue
        except Exception as e:
            print(f"Error in out-and-back route via {turnaround}: {e}")
            continue

    return candidates


def _generate_zigzag_routes(graph, start_node, target_m, avoid_hills,
                            globally_done_set, start_y, start_x):
    candidates = []
    if not graph:
        return candidates

    def get_street_orientation(node1, node2):
        dx = node2['x'] - node1['x']
        dy = node2['y'] - node1['y']
        return math.atan2(dy, dx)

    orientation_groups = {}
    for u, v, data in graph.edges(data=True):
        if u in graph.nodes and v in graph.nodes:
            u_data = graph.nodes[u]
            v_data = graph.nodes[v]
            orientation = get_street_orientation(u_data, v_data)
            orientation = abs(orientation) % math.pi
            bin_key = int(orientation * 180 / math.pi / 15)
            if bin_key not in orientation_groups:
                orientation_groups[bin_key] = []
            orientation_groups[bin_key].append((u, v, data))

    sorted_groups = sorted(
        orientation_groups.items(), key=lambda x: len(x[1]), reverse=True
    )

    for group_id, edges in sorted_groups[:3]:
        if len(edges) < 3:
            continue

        edges_with_dist = []
        for u, v, data in edges:
            mid_lat = (graph.nodes[u]['y'] + graph.nodes[v]['y']) / 2
            mid_lon = (graph.nodes[u]['x'] + graph.nodes[v]['x']) / 2
            d = dist_m(start_y, start_x, mid_lat, mid_lon)
            edges_with_dist.append((u, v, data, d))

        edges_with_dist.sort(key=lambda x: x[3])

        zigzag_path = _create_zigzag_from_parallel_streets(
            graph, start_node, edges_with_dist, target_m, avoid_hills, globally_done_set
        )

        if zigzag_path and len(zigzag_path) > 3:
            path_details = _calculate_path_details(zigzag_path, graph, globally_done_set)
            if (path_details.get("valid") and
                    target_m * 0.4 <= path_details['distance_m'] <= target_m * 1.8):
                candidates.append({
                    'path': zigzag_path,
                    'distance': path_details['distance_m'],
                    'percentage_new': path_details['percentage_new_distance'],
                    'strategy': 'zigzag',
                })

    return candidates


def _create_zigzag_from_parallel_streets(graph, start_node, parallel_edges,
                                         target_m, avoid_hills, globally_done_set):
    if not parallel_edges:
        return None

    closest_edge = min(parallel_edges, key=lambda x: x[3])
    u, v, data, dist_val = closest_edge

    weight_func = weight_factory(avoid_hills=avoid_hills, prioritize_new_roads=True)
    try:
        path_to_parallel = nx.shortest_path(graph, source=start_node, target=u, weight=weight_func)
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        try:
            path_to_parallel = nx.shortest_path(
                graph, source=start_node, target=v, weight=weight_func
            )
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None

    if len(path_to_parallel) < 2:
        return None

    zigzag_path = path_to_parallel[:-1]

    current_node = path_to_parallel[-1]
    current_distance = 0
    for i in range(len(zigzag_path) - 1):
        if graph.has_edge(zigzag_path[i], zigzag_path[i + 1]):
            edges = graph.get_edge_data(zigzag_path[i], zigzag_path[i + 1])
            if edges:
                edge_key = list(edges.keys())[0]
                edge_data = edges[edge_key]
                if isinstance(edge_data, dict):
                    current_distance += edge_data.get('length', 0)

    sorted_edges = sorted(parallel_edges, key=lambda x: x[3])

    for i, (u, v, data, dist_val) in enumerate(sorted_edges[:5]):
        if current_distance > target_m * 0.8:
            break

        try:
            path_to_next = nx.shortest_path(
                graph, source=current_node, target=u, weight=weight_func
            )
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            try:
                path_to_next = nx.shortest_path(
                    graph, source=current_node, target=v, weight=weight_func
                )
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue

        if len(path_to_next) < 2:
            continue

        zigzag_path.extend(path_to_next[:-1])
        zigzag_path.append(u)
        zigzag_path.append(v)

        current_node = v
        for j in range(len(path_to_next) - 1):
            if graph.has_edge(path_to_next[j], path_to_next[j + 1]):
                edges = graph.get_edge_data(path_to_next[j], path_to_next[j + 1])
                if edges:
                    edge_key = list(edges.keys())[0]
                    edge_data = edges[edge_key]
                    if isinstance(edge_data, dict):
                        current_distance += edge_data.get('length', 0)
        current_distance += data.get('length', 0)

    try:
        path_back = nx.shortest_path(
            graph, source=current_node, target=start_node, weight=weight_func
        )
        zigzag_path.extend(path_back)
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        pass

    return zigzag_path if len(zigzag_path) > 3 else None


def _score_and_filter_candidates(candidates, target_m):
    if not candidates:
        return []

    scored_candidates = []
    for candidate in candidates:
        distance = candidate['distance']
        percentage_new = candidate['percentage_new']
        strategy = candidate.get('strategy', 'unknown')

        distance_score = 1.0 / (1.0 + abs(distance - target_m) / target_m)
        newness_score = percentage_new / 100.0

        strategy_bonus = {
            'two_leg': 1.0,
            'multi_point': 1.1,
            'out_and_back': 1.05,
            'zigzag': 1.2,
        }.get(strategy, 1.0)

        total_score = (distance_score * 0.4 + newness_score * 0.6) * strategy_bonus

        scored_candidates.append({
            **candidate,
            'score': total_score,
            'distance_score': distance_score,
            'newness_score': newness_score,
        })

    scored_candidates.sort(key=lambda x: x['score'], reverse=True)

    final_candidates = [
        c for c in scored_candidates
        if target_m * 0.6 <= c['distance'] <= target_m * 1.4
    ]

    if not final_candidates and scored_candidates:
        print("No routes in ideal distance range, returning best available options.")
        return scored_candidates

    return final_candidates
