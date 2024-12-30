import re
import warnings
import numpy as np
import osmnx as ox
from copy import deepcopy
from shapely.geometry import (
    box,
    Point,
    Polygon,
    MultiPolygon,
    LineString,
    MultiLineString,
)
from geopandas import GeoDataFrame
from shapely.affinity import rotate, scale
from shapely.ops import unary_union
from shapely.errors import ShapelyDeprecationWarning

from IPython.display import display


def parse_query(query):
    """
    Detect whether `query` is an OSM address, lat/long tuple, or a GeoDataFrame boundary.
    """
    if isinstance(query, GeoDataFrame):
        return "polygon"
    elif isinstance(query, tuple):
        return "coordinates"
    elif re.match(r"[A-Z][0-9]+", str(query)):
        return "osmid"
    else:
        return "address"


def get_boundary(query, radius, circle=False, rotation=0):
    """
    Create a circle or rotated-square boundary around a lat/long point, or
    fetch a boundary from OSM if the query is an address.
    """
    # geocode if needed
    if parse_query(query) == "coordinates":
        lat, lon = query
    else:
        lat, lon = ox.geocode(query)

    # project to local UTM
    point_gdf = GeoDataFrame(geometry=[Point(lon, lat)], crs="EPSG:4326")
    point_gdf = ox.project_gdf(point_gdf)

    if circle:
        # circular boundary
        point_gdf.geometry = point_gdf.geometry.buffer(radius)
    else:
        # square boundary
        x, y = np.concatenate(point_gdf.geometry[0].xy)
        r = radius
        # rotate uses degrees by default
        # create square, rotate around center
        boundary_poly = rotate(
            Polygon([(x - r, y - r), (x + r, y - r), (x + r, y + r), (x - r, y + r)]),
            rotation,
            use_radians=False,
        )
        point_gdf.geometry = [boundary_poly]

    # unproject back to lat/long
    return point_gdf.to_crs("EPSG:4326")


def get_perimeter(query, radius=None, by_osmid=False, circle=False, dilate=None, rotation=0, **kwargs):
    """
    Get a perimeter geometry from a user-provided GeoDataFrame, OSM's polygon boundary,
    or a circle/square boundary around a lat/long query.
    """
    pq = parse_query(query)

    if radius and pq != "polygon":
        # user wants a circle/square boundary around a point
        perimeter = get_boundary(query, radius, circle=circle, rotation=rotation)
    else:
        if pq == "polygon":
            perimeter = query
        else:
            # attempt to fetch from OSM (geocode_to_gdf)
            perimeter = ox.geocode_to_gdf(query, by_osmid=by_osmid, **kwargs)

    # apply dilation if any
    perimeter = ox.project_gdf(perimeter)
    if dilate is not None:
        perimeter.geometry = perimeter.geometry.buffer(dilate)
    return perimeter.to_crs("EPSG:4326")


def get_gdf(layer, perimeter, perimeter_tolerance=0, tags=None, osmid=None, custom_filter=None, **kwargs):
    """
    For a bounding shape `perimeter`, fetch features from OSM using
    either `graph_from_polygon` or `features_from_polygon` depending on the layer.
    """
    # perimeter + tolerance
    perimeter_proj = ox.project_gdf(perimeter)
    perimeter_proj_tolerant = perimeter_proj.buffer(perimeter_tolerance)
    perimeter_union = unary_union(perimeter_proj_tolerant.geometry)
    bbox = box(*perimeter_union.bounds).buffer(0)

    # fetch logic
    if layer in ["streets", "railway", "waterway"]:
        try:
            graph = ox.graph_from_polygon(bbox, custom_filter=custom_filter, truncate_by_edge=True)
            gdf = ox.graph_to_gdfs(graph, nodes=False)
        except Exception:
            gdf = GeoDataFrame(geometry=[])
    else:
        # building/landuse/boundaries/etc.
        try:
            if osmid is None:
                gdf = ox.features_from_polygon(bbox, tags={tags: True} if isinstance(tags, str) else tags)
            else:
                gdf = ox.geocode_to_gdf(osmid, by_osmid=True)
        except Exception:
            gdf = GeoDataFrame(geometry=[])

    # intersection
    gdf.geometry = gdf.geometry.intersection(perimeter_union)
    gdf = gdf[~gdf.geometry.is_empty]
    return gdf


def get_gdfs(query, layers_dict, radius, dilate, rotation=0):
    """
    High-level aggregator: get a dictionary of GeoDataFrames, one per layer in `layers_dict`.
    """
    # get perimeter
    perimeter_opts = deepcopy(layers_dict.get("perimeter", {}))
    perimeter_opts.pop("dilate", None)  # we handle top-level dilate below
    perimeter = get_perimeter(query, radius=radius, circle=perimeter_opts.get("circle", False),
                              dilate=dilate, rotation=rotation, **perimeter_opts)

    gdfs = {"perimeter": perimeter}
    # fetch the other layers
    for layer_name, layer_opts in layers_dict.items():
        if layer_name == "perimeter":
            continue
        gdf = get_gdf(layer_name, perimeter, **layer_opts)
        gdfs[layer_name] = gdf

    return gdfs