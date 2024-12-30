"""
Prettymaps - A minimal Python library to draw pretty maps from OpenStreetMap Data
Copyright (C) 2021 Marcelo Prates
Modified 2024 by Frederic Rohrer

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published
by the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
import re
import os
import json
import pathlib
import warnings
import matplotlib
import numpy as np
import osmnx as ox
import shapely.ops
import pandas as pd
import geopandas as gp
import shapely.affinity
from copy import deepcopy
from dataclasses import dataclass
from matplotlib import pyplot as plt
from matplotlib.colors import hex2color
from matplotlib.patches import Path, PathPatch
from shapely.geometry.base import BaseGeometry
from typing import Optional, Union, Tuple, List, Dict, Any, Iterable
from shapely.geometry import (
    Point,
    LineString,
    MultiLineString,
    Polygon,
    MultiPolygon,
    GeometryCollection,
    box,
)

try:
    import vsketch
except ImportError:
    vsketch = None
    warnings.warn(
        'vsketch is not installed. Please install it if you plan to use pen plotter mode (mode="plotter").'
    )


# -------------------------- Data structures -------------------------- #
class Subplot:
    """
    Class implementing a prettymaps Subplot. Attributes:
    - query: prettymaps.plot() query
    - kwargs: dictionary of prettymaps.plot() parameters
    """

    def __init__(self, query, **kwargs):
        self.query = query
        self.kwargs = kwargs


@dataclass
class Plot:
    """
    Dataclass implementing a prettymaps Plot object. Attributes:
    - geodataframes: A dictionary of GeoDataFrames (one for each plot layer)
    - fig: A matplotlib figure
    - ax: A matplotlib axis object
    - background: Background layer (shapely object)
    """

    geodataframes: Dict[str, gp.GeoDataFrame]
    fig: matplotlib.figure.Figure
    ax: matplotlib.axes.Axes
    background: BaseGeometry


@dataclass
class Preset:
    """
    Dataclass implementing a prettymaps Preset object. Attributes:
    - params: dictionary of prettymaps.plot() parameters
    """

    params: dict
    # If you want to customize how your Preset is displayed in notebooks, you can implement
    # `_ipython_display_` or other IPython integration methods here.


# -------------------------- Geometric transforms -------------------------- #
def transform_gdfs(
    gdfs: Dict[str, gp.GeoDataFrame],
    x: float = 0,
    y: float = 0,
    scale_x: float = 1,
    scale_y: float = 1,
    rotation: float = 0,
) -> Dict[str, gp.GeoDataFrame]:
    """
    Apply geometric transformations to a dictionary of GeoDataFrames.

    Args:
        gdfs (Dict[str, gp.GeoDataFrame]): Dictionary of GeoDataFrames
        x (float, optional): x-axis translation. Defaults to 0.
        y (float, optional): y-axis translation. Defaults to 0.
        scale_x (float, optional): x-axis scale factor. Defaults to 1.
        scale_y (float, optional): y-axis scale factor. Defaults to 1.
        rotation (float, optional): rotation angle in DEGREES. Defaults to 0.

    Returns:
        Dict[str, gp.GeoDataFrame]: dictionary of transformed GeoDataFrames
    """
    # Project geometries before transformations
    gdfs = {
        name: ox.project_gdf(gdf) if len(gdf) > 0 else gdf for name, gdf in gdfs.items()
    }

    # Create geometry collection from gdfs' geometries
    collection = GeometryCollection(
        [GeometryCollection(list(gdf.geometry)) for gdf in gdfs.values()]
    )

    # Apply transformations in projected coordinate space
    collection = shapely.affinity.translate(collection, x, y)
    collection = shapely.affinity.scale(collection, scale_x, scale_y)
    # Shapely's default is use_radians=False, so 'rotation' is interpreted as degrees
    collection = shapely.affinity.rotate(collection, rotation, use_radians=False)

    # Update geometries in each gdf
    for i, layer in enumerate(gdfs):
        if len(gdfs[layer]) > 0:
            gdfs[layer].geometry = list(collection.geoms[i].geoms)
            # Reproject back to EPSG:4326
            gdfs[layer] = ox.project_gdf(gdfs[layer], to_crs="EPSG:4326")

    return gdfs


# -------------------------- Matplotlib patches -------------------------- #
def PolygonPatch(shape: BaseGeometry, **kwargs) -> PathPatch:
    """
    Create a matplotlib PathPatch from a (Multi)Polygon geometry.

    Args:
        shape (BaseGeometry): Shapely geometry
        kwargs: parameters for matplotlib's PathPatch constructor

    Returns:
        PathPatch: matplotlib PathPatch created from input shapely geometry
    """
    vertices, codes = [], []
    for geom in shape.geoms if hasattr(shape, "geoms") else [shape]:
        for poly in geom.geoms if hasattr(geom, "geoms") else [geom]:
            if not isinstance(poly, Polygon):
                continue
            # Exterior
            exterior = np.array(poly.exterior.xy)
            # Interiors
            interiors = [np.array(interior.xy) for interior in poly.interiors]
            # Build lists
            vertices += [exterior] + interiors
            codes += [
                [Path.MOVETO]
                + [Path.LINETO] * (p.shape[1] - 2)
                + [Path.CLOSEPOLY]
                for p in [exterior] + interiors
            ]
    # Create path
    return PathPatch(Path(np.concatenate(vertices, 1).T, np.concatenate(codes)), **kwargs)


# -------------------------- Plotting helpers -------------------------- #
def plot_gdf(
    layer: str,
    gdf: gp.GeoDataFrame,
    ax: matplotlib.axes.Axes,
    mode: str = "matplotlib",
    vsk=None,
    palette: Optional[List[str]] = None,
    width: Optional[Union[dict, float]] = None,
    union: bool = False,
    dilate_points: Optional[float] = None,
    dilate_lines: Optional[float] = None,
    **kwargs,
) -> None:
    """
    Plot a layer onto a matplotlib Axes or onto a vsketch in "plotter" mode.

    Args:
        layer (str): layer name
        gdf (gp.GeoDataFrame): GeoDataFrame
        ax (matplotlib.axes.Axes): matplotlib axis object (required for 'matplotlib' mode)
        mode (str, optional): drawing mode. 'matplotlib' or 'plotter'. Defaults to 'matplotlib'.
        vsk (vsketch.Vsketch, optional): Vsketch object (required if mode='plotter'). Defaults to None.
        palette (Optional[List[str]], optional): A color palette (list of hex codes). Defaults to None.
        width (Optional[Union[dict, float]], optional): For street-based layers, can be a dictionary of widths by highway or a float. Defaults to None.
        union (bool, optional): Whether to union all geometries in the layer. Defaults to False.
        dilate_points (Optional[float], optional): If provided, buffer around point geometries. Defaults to None.
        dilate_lines (Optional[float], optional): If provided, buffer around line geometries. Defaults to None.

    Raises:
        ValueError: If mode='plotter' but vsk is not available or not installed.
        Exception: If unknown mode is given.
    """
    # If hatch color is provided separately
    hatch_c = kwargs.pop("hatch_c", None)

    # Convert GDF to shapely geometry
    geometries = gdf_to_shapely(layer, gdf, width, point_size=dilate_points, line_width=dilate_lines)

    # Union geometries if requested
    if union:
        geometries = shapely.ops.unary_union(GeometryCollection([geometries]))

    # If 'fc' is a list, treat it as a palette
    if (palette is None) and ("fc" in kwargs) and (isinstance(kwargs["fc"], list)):
        palette = kwargs.pop("fc")

    # Plot each geometry
    if mode == "matplotlib":
        for shape in geometries.geoms if hasattr(geometries, "geoms") else [geometries]:
            if isinstance(shape, (Polygon, MultiPolygon)):
                # Patch fill
                ax.add_patch(
                    PolygonPatch(
                        shape,
                        lw=0,
                        ec=hatch_c if hatch_c else kwargs.get("ec", None),
                        fc=kwargs.get("fc", np.random.choice(palette) if palette else None),
                        **{k: v for k, v in kwargs.items() if k not in ["lw", "ec", "fc"]}
                    )
                )
                # Outline
                ax.add_patch(
                    PolygonPatch(
                        shape,
                        fill=False,
                        **{k: v for k, v in kwargs.items() if k not in ["hatch", "fill"]}
                    )
                )
            elif isinstance(shape, LineString):
                ax.plot(
                    *shape.xy,
                    c=kwargs.get("ec", None),
                    **{k: v for k, v in kwargs.items() if k in ["lw", "ls", "dashes", "zorder"]}
                )
            elif isinstance(shape, MultiLineString):
                for c in shape.geoms:
                    ax.plot(
                        *c.xy,
                        c=kwargs.get("ec", None),
                        **{k: v for k, v in kwargs.items() if k in ["lw", "ls", "dashes", "zorder"]}
                    )

    elif mode == "plotter":
        if vsk is None or (vsketch is None):
            raise ValueError(
                "Plotter mode requires a valid vsketch object and the vsketch package installed."
            )
        # Draw geometry with vsketch
        for shape in geometries.geoms if hasattr(geometries, "geoms") else [geometries]:
            # Apply pen / stroke settings
            if kwargs.get("draw", True):
                stroke = kwargs.get("stroke", 1)
                pen_width = kwargs.get("penWidth", 0.3)
                fill_ = kwargs.get("fill", None)

                vsk.stroke(stroke)
                vsk.penWidth(pen_width)
                if fill_ is not None:
                    vsk.fill(fill_)
                else:
                    vsk.noFill()

                vsk.geometry(shape)

    else:
        raise Exception(f"Unknown mode {mode}")


# For demonstration or labeling, currently unused in the main pipeline
def plot_legends(gdf, ax):
    """
    For debugging/demonstration only: plot text labels at geometry centroids.
    """
    for _, row in gdf.iterrows():
        name = row.name
        x, y = np.concatenate(row.geometry.centroid.xy)
        ax.text(x, y, name)


# -------------------------- Conversions -------------------------- #
def graph_to_shapely(gdf: gp.GeoDataFrame, width: float = 1.0) -> BaseGeometry:
    """
    Given a GeoDataFrame containing a street (or similar) network, convert
    line geometries into buffered polygons by 'width'.

    If 'width' is a dictionary, attempt to buffer each geometry by a type-dependent value.

    Args:
        gdf (gp.GeoDataFrame): input GeoDataFrame containing a line-based network
        width (float or dict): The buffer distance. Could be a float or dict by highway type.

    Returns:
        BaseGeometry: A shapely geometry (often MultiPolygon) of the union of all buffered lines.
    """

    def highway_to_width(highway):
        if isinstance(highway, str) and (highway in width):
            return width[highway]
        elif isinstance(highway, Iterable):
            for h in highway:
                if h in width:
                    return width[h]
            return np.nan
        else:
            return np.nan

    # If width is dict, map each geometry’s “highway” value
    if isinstance(width, dict):
        gdf["width"] = gdf.highway.map(highway_to_width)
        gdf.dropna(subset=["width"], inplace=True)
    else:
        # single float for all
        gdf["width"] = width

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=shapely.errors.ShapelyDeprecationWarning)
        # Buffer lines
        gdf.geometry = gdf.apply(lambda row: row.geometry.buffer(row.width), axis=1)

    return shapely.ops.unary_union(gdf.geometry)


def geometries_to_shapely(
    gdf: gp.GeoDataFrame,
    point_size: Optional[float] = None,
    line_width: Optional[float] = None,
) -> GeometryCollection:
    """
    Convert geometry in a GeoDataFrame into (Multi)Polygon(s), optionally buffering points or lines.

    Args:
        gdf (gp.GeoDataFrame): Input geodata
        point_size (Optional[float]): Buffer points by this amount (for circle-like representation).
        line_width (Optional[float]): Buffer lines by this amount.

    Returns:
        GeometryCollection: The combined geometry.
    """
    geoms = gdf.geometry.tolist()
    collections = [x for x in geoms if isinstance(x, GeometryCollection)]

    points = [
        x for x in geoms if isinstance(x, Point)
    ] + [y for x in collections for y in x.geoms if isinstance(y, Point)]
    lines = [
        x for x in geoms if isinstance(x, (LineString, MultiLineString))
    ] + [y for x in collections for y in x.geoms if isinstance(y, (LineString, MultiLineString))]
    polys = [
        x for x in geoms if isinstance(x, (Polygon, MultiPolygon))
    ] + [y for x in collections for y in x.geoms if isinstance(y, (Polygon, MultiPolygon))]

    if point_size and point_size > 0:
        points = [p.buffer(point_size) for p in points]
    if line_width and line_width > 0:
        lines = [l.buffer(line_width) for l in lines]

    return GeometryCollection(list(points) + list(lines) + list(polys))


def gdf_to_shapely(
    layer: str,
    gdf: gp.GeoDataFrame,
    width: Optional[Union[dict, float]] = None,
    point_size: Optional[float] = None,
    line_width: Optional[float] = None,
    **kwargs,
) -> GeometryCollection:
    """
    Convert a single layer's GeoDataFrame to shapely geometry (MultiPolygon, etc.),
    buffering points or lines if requested.

    Args:
        layer (str): The layer name
        gdf (gp.GeoDataFrame): Input geodata
        width (Optional[Union[dict, float]]): For street-like layers, buffer lines by 'width'.
        point_size (Optional[float]): Buffer around points.
        line_width (Optional[float]): Buffer around lines.

    Returns:
        GeometryCollection: A shapely (multi)geometry representing the entire layer.
    """
    # Ensure projection
    if not gdf.empty:
        try:
            gdf = ox.project_gdf(gdf)
        except Exception:
            pass

    if layer in ["streets", "railway", "waterway"]:
        geometries = graph_to_shapely(gdf, width if width else 1.0)
    else:
        geometries = geometries_to_shapely(gdf, point_size=point_size, line_width=line_width)

    return geometries


# -------------------------- Overrides -------------------------- #
def override_args(
    layers: dict, circle: Optional[bool], dilate: Optional[Union[float, bool]]
) -> dict:
    """
    Attach 'circle' and 'dilate' keys to each layer if not present.
    """
    for layer in layers:
        for arg in ["circle", "dilate"]:
            if arg not in layers[layer]:
                layers[layer][arg] = locals()[arg]
    return layers


def override_params(default_dict: dict, new_dict: dict) -> dict:
    """
    Recursively override parameters in 'default_dict' with those in 'new_dict'.
    """
    final_dict = deepcopy(default_dict)
    for key, val in new_dict.items():
        if isinstance(val, dict) and key in final_dict:
            final_dict[key] = override_params(final_dict[key], val)
        else:
            final_dict[key] = val
    return final_dict


# -------------------------- Background creation -------------------------- #
def create_background(
    gdfs: Dict[str, gp.GeoDataFrame], style: Dict[str, dict]
) -> Tuple[BaseGeometry, float, float, float, float, float, float]:
    """
    Create a background geometry slightly larger than the perimeter, using style['background'].get('pad') if available.
    """
    background_pad = 1.1
    if "background" in style and "pad" in style["background"]:
        background_pad = style["background"].pop("pad")

    perimeter_gdf = gdfs.get("perimeter", None)
    if perimeter_gdf is None or perimeter_gdf.empty:
        # Create an empty bounding box if no perimeter
        background = box(0, 0, 1, 1)
    else:
        # Combine perimeter
        perimeter_union = shapely.ops.unary_union(ox.project_gdf(perimeter_gdf).geometry)
        background = shapely.affinity.scale(box(*perimeter_union.bounds), background_pad, background_pad)

    if "background" in style and "dilate" in style["background"]:
        dil = style["background"].pop("dilate")
        background = background.buffer(dil)

    xmin, ymin, xmax, ymax = background.bounds
    dx, dy = xmax - xmin, ymax - ymin
    return background, xmin, ymin, xmax, ymax, dx, dy


def draw_text(params: Dict[str, dict], background: BaseGeometry) -> None:
    """
    Draw text (like OSM credit or a caption) on the map, using relative coordinates in [0..1].
    """
    # Defaults
    default_params = dict(
        text="",
        x=0,
        y=1,
        horizontalalignment="left",
        verticalalignment="top",
        bbox=dict(boxstyle="square", fc="#fff", ec="#000"),
        fontfamily="Ubuntu Mono",
    )
    params = override_params(default_params, params)
    x, y, text = params.pop("x"), params.pop("y"), params.pop("text")

    xmin, ymin, xmax, ymax = background.bounds
    X = np.interp([x], [0, 1], [xmin, xmax])[0]
    Y = np.interp([y], [0, 1], [ymin, ymax])[0]

    plt.text(X, Y, text, **params)


# -------------------------- Preset management -------------------------- #
def presets_directory():
    return os.path.join(pathlib.Path(__file__).resolve().parent, "presets")


def create_preset(
    name: str,
    layers: Optional[Dict[str, dict]] = None,
    style: Optional[Dict[str, dict]] = None,
    circle: Optional[bool] = None,
    radius: Optional[Union[float, bool]] = None,
    dilate: Optional[Union[float, bool]] = None,
) -> None:
    """
    Create a preset file and save it on the presets folder under 'name.json'.
    """
    path = os.path.join(presets_directory(), f"{name}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "layers": layers,
                "style": style,
                "circle": circle,
                "radius": radius,
                "dilate": dilate,
            },
            f,
            ensure_ascii=False,
        )


def read_preset(name: str) -> Dict[str, dict]:
    """
    Read a preset from the 'presets' folder.
    """
    path = os.path.join(presets_directory(), f"{name}.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def delete_preset(name: str) -> None:
    """
    Delete a preset from the 'presets' folder.
    """
    path = os.path.join(presets_directory(), f"{name}.json")
    if os.path.exists(path):
        os.remove(path)


def override_preset(
    name: str,
    layers: Dict[str, dict] = {},
    style: Dict[str, dict] = {},
    circle: Optional[bool] = None,
    radius: Optional[Union[float, bool]] = None,
    dilate: Optional[Union[float, bool]] = None,
) -> Tuple[dict, dict, Optional[bool], Optional[Union[float, bool]], Optional[Union[float, bool]]]:
    """
    Load a preset from file, then override it with additional parameters from the caller.
    """
    params = read_preset(name)

    # Override with user-provided parameters
    if "layers" in params:
        layers = override_params(params["layers"], layers)
    if "style" in params:
        style = override_params(params["style"], style)
    if circle is None and "circle" in params:
        circle = params["circle"]
    if radius is None and "radius" in params:
        radius = params["radius"]
    if dilate is None and "dilate" in params:
        dilate = params["dilate"]

    # Remove any layers marked as False
    for lyr in [key for key in layers.keys() if layers[key] is False]:
        del layers[lyr]

    return layers, style, circle, radius, dilate


def manage_presets(
    load_preset: Optional[str],
    save_preset: Optional[str],
    update_preset: Optional[str],
    layers: Dict[str, dict],
    style: Dict[str, dict],
    circle: Optional[bool],
    radius: Optional[Union[float, bool]],
    dilate: Optional[Union[float, bool]],
) -> Tuple[
    dict, dict, Optional[bool], Optional[Union[float, bool]], Optional[Union[float, bool]]
]:
    """
    Handle preset loading, updating, and saving.
    - If update_preset is provided, we load that preset, apply user overrides, and re-save.
    - If load_preset is provided, we simply load it and override with user data.
    - If save_preset is provided, we save current parameters to a new preset file.
    """
    # update_preset => load + re-save
    if update_preset is not None:
        load_preset = update_preset
        save_preset = update_preset

    if load_preset is not None:
        layers, style, circle, radius, dilate = override_preset(
            load_preset, layers, style, circle, radius, dilate
        )

    if save_preset is not None:
        create_preset(
            save_preset, layers=layers, style=style, circle=circle, radius=radius, dilate=dilate
        )

    return layers, style, circle, radius, dilate


def presets():
    """
    Return a DataFrame listing all available presets in the 'presets' folder.
    """
    files = [
        file.split(".")[0]
        for file in os.listdir(presets_directory())
        if file.endswith(".json")
    ]
    files = sorted(files)
    df = pd.DataFrame({"preset": files, "params": list(map(read_preset, files))})
    return df


def preset(name):
    """
    Create a Preset object from a stored preset file in 'presets' folder.
    """
    with open(os.path.join(presets_directory(), f"{name}.json"), "r", encoding="utf-8") as f:
        params = json.load(f)
    return Preset(params)


# -------------------------- Main plotting function -------------------------- #
def plot(
    query: Union[str, Tuple[float, float], gp.GeoDataFrame],
    backup=None,
    layers={},
    style={},
    preset="default",
    save_preset=None,
    update_preset=None,
    postprocessing=None,
    circle=None,
    radius=None,
    dilate=None,
    save_as=None,
    fig=None,
    ax=None,
    title=None,
    figsize=(12, 12),
    constrained_layout=True,
    credit={},
    mode="matplotlib",
    multiplot=False,
    show=True,
    x=0,
    y=0,
    scale_x=1,
    scale_y=1,
    rotation=0,
    vsk=None,
):
    """
    Draw a map from OpenStreetMap data or a custom user-provided GeoDataFrame boundary.

    Parameters
    ----------
    query : str | Tuple[float, float] | gp.GeoDataFrame
        The address or coordinate pair (lat, long) or a custom polygon GDF.
    backup : Plot (optional)
        A previously returned Plot object, containing geodataframes to reuse.
    layers : dict
        Key-value pairs specifying OSM tags & queries for each layer.
    style : dict
        Matplotlib style parameters for each layer (e.g. fc, ec, lw).
    preset : str
        Name of a preset to load. Default is 'default'.
    save_preset : str
        Name of a new preset file to create.
    update_preset : str
        Name of a preset to load, override, and re-save (if provided).
    postprocessing : callable
        Custom function that takes a dict of layer => GDF and returns a modified dict of GDFs.
    circle : bool
        If True, use a circular boundary instead of square.
    radius : float or bool
        The radius (in meters) for the boundary if the query is a point. If None, uses OSM polygon boundary.
    dilate : float or bool
        If not None, buffer the boundary by this amount (in projected coordinate system).
    save_as : str
        If provided, output file name to save the resulting figure (matplotlib) or vsketch.
    fig, ax : optional
        Matplotlib figure or axes to draw on. If not provided, a new figure/axes is created.
    title : str
        Optional figure title.
    figsize : tuple
        Figure size (inches) if creating a new figure. Default (12, 12).
    credit : dict
        Dictionary specifying text to place as OSM credit or other annotation. E.g. {'text': '© OSM'}
    mode : {'matplotlib', 'plotter'}
        Drawing mode. 'matplotlib' uses a standard Matplotlib Axes. 'plotter' requires vsketch installed.
    multiplot : bool
        If True, signals that we might be overlaying multiple subplots on the same figure.
    show : bool
        If False, does not display the figure in Matplotlib.
    x, y : float
        Translate the geometry by x, y (in projected coordinates).
    scale_x, scale_y : float
        Scale the geometry by these factors (in projected coordinates).
    rotation : float
        Rotate the geometry by this angle (in DEGREES).

    Returns
    -------
    Plot
        A Plot dataclass with references to (fig, ax, background, geodataframes).
    """
    # 1. Manage presets
    layers, style, circle, radius, dilate = manage_presets(
        load_preset=preset,
        save_preset=save_preset,
        update_preset=update_preset,
        layers=layers,
        style=style,
        circle=circle,
        radius=radius,
        dilate=dilate,
    )

    # 2. Init matplotlib figure/ax if needed (matplotlib mode)
    if mode == "matplotlib":
        if fig is None:
            fig = plt.figure(figsize=figsize, dpi=300, constrained_layout=constrained_layout)
        if ax is None:
            ax = fig.add_subplot(111, aspect="equal")

    # 3. Override arguments in layers
    layers = override_args(layers, circle, dilate)

    # 4. Get or load geodataframes
    if backup:
        gdfs = backup.geodataframes
    else:
        from .fetch import get_gdfs  # (Ensure your local "fetch.py" is updated accordingly)
        gdfs = get_gdfs(query, layers, radius, dilate, -rotation)

        # 5. Apply transformations (in projected space)
        gdfs = transform_gdfs(gdfs, x, y, scale_x, scale_y, rotation)

    # 6. Optional post-processing
    if postprocessing is not None:
        gdfs = postprocessing(gdfs)

    # 7. Create background & get bounding box
    background, xmin, ymin, xmax, ymax, dx, dy = create_background(gdfs, style)

    # 8. Plot layers
    if mode == "plotter":
        # Check if vsketch or vsk is missing
        if (vsketch is None) or (vsk is None):
            raise ValueError("Plotter mode requires vsketch and a valid 'vsk' object.")

        class Sketch(vsketch.SketchClass):
            def draw(self, _vsk: vsketch.Vsketch):
                _vsk.size("a4", landscape=True)
                for layer_name, layer_params in layers.items():
                    if layer_name in gdfs:
                        plot_gdf(
                            layer_name,
                            gdfs[layer_name],
                            ax,  # Not used in plotter mode
                            width=layer_params.get("width"),
                            mode="plotter",
                            vsk=_vsk,  # pass the local _vsk or the outer vsk
                            **(style.get(layer_name, {}))
                        )
                if save_as:
                    _vsk.save(save_as)

            def finalize(self, _vsk: vsketch.Vsketch):
                _vsk.vpype("linemerge linesimplify reloop linesort")

        sketch = Sketch()
        sketch.display()
    else:
        # Matplotlib mode
        for layer_name, layer_params in layers.items():
            if layer_name in gdfs:
                plot_gdf(
                    layer_name,
                    gdfs[layer_name],
                    ax,
                    width=layer_params.get("width", None),
                    mode="matplotlib",
                    **(style.get(layer_name, {}))
                )

    # 9. Draw background on top or bottom (depends on zorder)
    if mode == "matplotlib" and "background" in style:
        bg_zorder = style["background"].pop("zorder", -1)
        ax.add_patch(
            PolygonPatch(
                background,
                **{k: v for k, v in style["background"].items() if k != "dilate"},
                zorder=bg_zorder,
            )
        )

    # 10. Credit text
    if mode == "matplotlib" and (credit != False) and (not multiplot):
        draw_text(credit, background)

    # 11. Final adjustments
    if mode == "matplotlib":
        ax.axis("off")
        ax.axis("equal")
        ax.autoscale()
        plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
        if title:
            plt.title(title)
        if save_as:
            plt.savefig(save_as)
        if not show:
            plt.close(fig)

    return Plot(gdfs, fig, ax, background)


# -------------------------- Multi-plot utility -------------------------- #
def multiplot(*subplots, figsize=None, credit={}, **kwargs):
    """
    Draw multiple subplots (multiple queries/layers) into the same figure & axes.

    Example usage:
    >>> s1 = Subplot(query="New York")
    >>> s2 = Subplot(query="Paris")
    >>> multiplot(s1, s2, figsize=(15, 10))

    Each Subplot can provide own 'layers', 'style', etc. which get merged with any kwargs provided to this function.
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, aspect="equal")

    subplots_results = []
    for subplot in subplots:
        # override subplot kwargs with any global kwargs not explicitly set
        combined_kwargs = override_params(subplot.kwargs, kwargs)
        # ensure we do not override the 'ax' or 'fig' from outside
        combined_kwargs["fig"] = fig
        combined_kwargs["ax"] = ax
        # set multiplot to True so we skip credit text duplication
        combined_kwargs["multiplot"] = True

        res = plot(subplot.query, **combined_kwargs)
        subplots_results.append(res)

    ax.axis("off")
    ax.axis("equal")
    ax.autoscale()

    # If we want a global credit across all subplots:
    if credit != False:
        backgrounds = [r.background for r in subplots_results]
        global_bounds = shapely.ops.unary_union(backgrounds).bounds
        global_background = box(*global_bounds)
        draw_text(credit, global_background)
    return subplots_results