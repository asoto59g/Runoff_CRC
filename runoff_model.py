from __future__ import annotations

import heapq
import importlib.util
import json
import math
import os
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

def _configure_bundled_proj_data() -> None:
    spec = importlib.util.find_spec("rasterio")
    if spec is None or spec.origin is None:
        return
    proj_dir = Path(spec.origin).resolve().parent / "proj_data"
    if (proj_dir / "proj.db").exists():
        os.environ["PROJ_DATA"] = str(proj_dir)
        os.environ["PROJ_LIB"] = str(proj_dir)


_configure_bundled_proj_data()

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from scipy import ndimage
from affine import Affine
from pyproj import CRS, Geod, Transformer
from rasterio.enums import Resampling
from rasterio.features import geometry_mask, geometry_window
from rasterio.io import MemoryFile
from rasterio.windows import Window
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon, box, mapping, shape
from shapely.ops import transform as shapely_transform
from shapely.ops import unary_union
from remote_raster import DEFAULT_DRIVE_DEM_URL, is_http_url, open_raster_source, read_remote_tiff_info


DEM_CRS = CRS.from_epsg(5367)  # CR05 / CRTM05, meters. The source TIFF is tagged as LOCAL_CS["CRTM05"].
WGS84 = CRS.from_epsg(4326)
GEOD = Geod(ellps="WGS84")
DEFAULT_DEM_PATH = "MDE_5K.tif"
REMOTE_DEM_DRIVE_URL = DEFAULT_DRIVE_DEM_URL
RUNOFF_MODEL_VERSION = "2026-09-06-global-crs-v2"


class RunoffModelError(ValueError):
    pass


@dataclass(frozen=True)
class SimulationConfig:
    dem_path: str = DEFAULT_DEM_PATH
    rainfall_mm: float = 75.0
    duration_min: float = 60.0
    infiltration_rate_mm_h: float = 10.0
    initial_loss_mm: float = 2.0
    runoff_coefficient: float = 0.65
    max_cells: int = 700_000
    condition_dem: bool = True
    fill_epsilon_m: float = 0.001
    concentration_percentile: float = 98.0
    flood_percentile: float = 97.0
    stream_percentile: float = 99.0
    channel_base_half_width_m: float = 5.0
    channel_spill_threshold_mm: float = 50.0
    channel_max_overflow_width_m: float = 80.0
    large_channel_depth_threshold_m: float = 1.5
    large_channel_search_radius_m: float = 60.0
    large_channel_bank_percentile: float = 75.0
    input_crs: str = "auto"


@dataclass
class RasterPreview:
    crs_label: str
    raster_crs_wkt: str
    bounds_dem: tuple[float, float, float, float]
    bounds_wgs84: tuple[float, float, float, float]
    width: int
    height: int
    resolution: tuple[float, float]


@dataclass
class RunoffResult:
    dem: np.ndarray
    raster_crs_wkt: str
    routing_dem: np.ndarray
    fill_depth_m: np.ndarray
    valid_mask: np.ndarray
    transform: Affine
    geometry_dem: Polygon | MultiPolygon
    geometry_wgs84: Polygon | MultiPolygon
    flow_accumulation_m2: np.ndarray
    runoff_accumulation_m3: np.ndarray
    slope_percent: np.ndarray
    flood_index: np.ndarray
    concentration_mask: np.ndarray
    flood_mask: np.ndarray
    relative_flood_mask: np.ndarray
    stream_center_mask: np.ndarray
    channel_base_mask: np.ndarray
    large_channel_mask: np.ndarray
    overbank_mask: np.ndarray
    distance_to_channel_m: np.ndarray
    channel_depth_m: np.ndarray
    sink_mask: np.ndarray
    outlet_mask: np.ndarray
    summary: dict[str, Any]


@dataclass
class ClipDemResult:
    data: bytes
    summary: dict[str, Any]
    geometry_wgs84: Polygon | MultiPolygon


def raster_crs_for_source(dem_path: str | Path = DEFAULT_DEM_PATH) -> CRS:
    try:
        with open_raster_source(dem_path) as src:
            return _analysis_crs_from_rasterio(src.crs)
    except Exception as exc:
        if not is_http_url(dem_path):
            raise
        try:
            read_remote_tiff_info(dem_path)
        except Exception:
            raise exc
        # The bundled Google Drive DEM is tagged LOCAL_CS["CRTM05"].
        return DEM_CRS


def read_raster_preview(dem_path: str | Path = DEFAULT_DEM_PATH) -> RasterPreview:
    try:
        with open_raster_source(dem_path) as src:
            raster_crs = _analysis_crs_from_rasterio(src.crs)
            bounds_dem = tuple(float(v) for v in src.bounds)
            bounds_wgs84 = transform_bounds_dem_to_wgs84(bounds_dem, raster_crs)
            res = (abs(float(src.transform.a)), abs(float(src.transform.e)))
            return RasterPreview(
                crs_label=_crs_label(raster_crs),
                raster_crs_wkt=raster_crs.to_wkt(),
                bounds_dem=bounds_dem,
                bounds_wgs84=bounds_wgs84,
                width=src.width,
                height=src.height,
                resolution=res,
            )
    except Exception as exc:
        if not is_http_url(dem_path):
            raise
        try:
            info = read_remote_tiff_info(dem_path)
        except Exception:
            raise exc
        raster_crs = DEM_CRS
        bounds_dem = tuple(float(v) for v in info.bounds)
        return RasterPreview(
            crs_label=_crs_label(raster_crs),
            raster_crs_wkt=raster_crs.to_wkt(),
            bounds_dem=bounds_dem,
            bounds_wgs84=transform_bounds_dem_to_wgs84(bounds_dem, raster_crs),
            width=info.width,
            height=info.height,
            resolution=info.resolution,
        )


def load_geojson_geometry(data: str | bytes | dict[str, Any]) -> Polygon | MultiPolygon:
    if isinstance(data, bytes):
        data = data.decode("utf-8")
    if isinstance(data, str):
        data = json.loads(data)

    geom_type = data.get("type")
    geometries = []

    if geom_type == "FeatureCollection":
        for feature in data.get("features", []):
            geom = feature.get("geometry")
            if geom:
                geometries.append(shape(geom))
    elif geom_type == "Feature":
        geometries.append(shape(data["geometry"]))
    elif geom_type in {"Polygon", "MultiPolygon", "GeometryCollection"}:
        geometries.append(shape(data))
    else:
        raise RunoffModelError("El GeoJSON debe contener Polygon, MultiPolygon o FeatureCollection.")

    if not geometries:
        raise RunoffModelError("No se encontraron geometrias validas en el GeoJSON.")

    geom = unary_union([g for g in geometries if not g.is_empty])
    geom = _polygonal_geometry(geom)
    if geom.is_empty:
        raise RunoffModelError("La geometria no contiene area poligonal.")
    return geom


def normalize_geometry(
    geometry: Polygon | MultiPolygon,
    input_crs: str = "auto",
    raster_crs: CRS | str | None = None,
) -> tuple[Polygon | MultiPolygon, Polygon | MultiPolygon, str]:
    geometry = _repair_geometry(_polygonal_geometry(geometry))
    target_crs = _coerce_analysis_crs(raster_crs) if raster_crs is not None else DEM_CRS
    detected = detect_input_crs(geometry, target_crs) if input_crs == "auto" else input_crs

    if detected == "EPSG:4326":
        geometry_wgs84 = geometry
        geometry_dem = transform_geometry(geometry, WGS84, target_crs)
    elif detected == "EPSG:5367":
        geometry_dem = transform_geometry(geometry, DEM_CRS, target_crs)
        geometry_wgs84 = transform_geometry(geometry, DEM_CRS, WGS84)
    elif detected == "raster":
        geometry_dem = geometry
        geometry_wgs84 = transform_geometry(geometry, target_crs, WGS84)
    else:
        raise RunoffModelError(f"CRS de entrada no soportado: {detected}")

    return _repair_geometry(geometry_dem), _repair_geometry(geometry_wgs84), detected


def detect_input_crs(geometry: Polygon | MultiPolygon, raster_crs: CRS | str | None = None) -> str:
    minx, miny, maxx, maxy = geometry.bounds

    if -180.0 <= minx <= 180.0 and -180.0 <= maxx <= 180.0 and -90.0 <= miny <= 90.0 and -90.0 <= maxy <= 90.0:
        return "EPSG:4326"

    if 200_000 <= minx <= 750_000 and 200_000 <= maxx <= 750_000 and 800_000 <= miny <= 1_350_000 and 800_000 <= maxy <= 1_350_000:
        return "EPSG:5367"

    if raster_crs is not None:
        return "raster"

    raise RunoffModelError(
        "No pude detectar el CRS del poligono. Selecciona WGS84, CRTM05 o CRS del raster manualmente."
    )


def _analysis_crs_from_rasterio(source_crs: Any) -> CRS:
    if source_crs is None:
        raise RunoffModelError("El GeoTIFF debe tener CRS definido para ubicarlo en el mapa.")
    crs_text = source_crs.to_wkt() if hasattr(source_crs, "to_wkt") else str(source_crs)
    if "CRTM05" in crs_text.upper() or "CR05" in crs_text.upper():
        return DEM_CRS
    if hasattr(source_crs, "to_wkt"):
        return _canonical_crs(CRS.from_wkt(crs_text))
    return _canonical_crs(CRS.from_user_input(source_crs))


def _coerce_analysis_crs(source_crs: CRS | str) -> CRS:
    if isinstance(source_crs, CRS):
        return _canonical_crs(source_crs)
    crs_text = str(source_crs)
    if "CRTM05" in crs_text.upper() or "CR05" in crs_text.upper():
        return DEM_CRS
    return _canonical_crs(CRS.from_user_input(source_crs))


def _canonical_crs(crs: CRS) -> CRS:
    epsg = crs.to_epsg()
    if epsg:
        return CRS.from_epsg(epsg)
    return crs


def _crs_label(crs: CRS) -> str:
    epsg = crs.to_epsg()
    if epsg:
        return f"{crs.name} / EPSG:{epsg}"
    return crs.name or crs.to_string()


def transform_geometry(
    geometry: Polygon | MultiPolygon,
    source_crs: CRS,
    target_crs: CRS,
) -> Polygon | MultiPolygon:
    transformer = Transformer.from_crs(source_crs, target_crs, always_xy=True)
    return _polygonal_geometry(shapely_transform(transformer.transform, geometry))


def transform_bounds_dem_to_wgs84(
    bounds: tuple[float, float, float, float],
    source_crs: CRS | str | None = None,
) -> tuple[float, float, float, float]:
    minx, miny, maxx, maxy = bounds
    crs = _coerce_analysis_crs(source_crs) if source_crs is not None else DEM_CRS
    transformer = Transformer.from_crs(crs, WGS84, always_xy=True)
    west, south, east, north = transformer.transform_bounds(minx, miny, maxx, maxy, densify_pts=21)
    return float(west), float(south), float(east), float(north)

def _pixel_size_m(
    transform: Affine,
    raster_crs: CRS,
    geometry_dem: Polygon | MultiPolygon,
) -> tuple[float, float]:
    xres = abs(float(transform.a))
    yres = abs(float(transform.e))
    if raster_crs.is_projected:
        return xres * _axis_to_meter_factor(raster_crs, 0), yres * _axis_to_meter_factor(raster_crs, 1)
    if raster_crs.is_geographic:
        centroid = geometry_dem.centroid
        transformer = Transformer.from_crs(raster_crs, WGS84, always_xy=True)
        cx, cy = float(centroid.x), float(centroid.y)
        lon_w, lat_w = transformer.transform(cx - xres / 2.0, cy)
        lon_e, lat_e = transformer.transform(cx + xres / 2.0, cy)
        lon_s, lat_s = transformer.transform(cx, cy - yres / 2.0)
        lon_n, lat_n = transformer.transform(cx, cy + yres / 2.0)
        _, _, dx = GEOD.inv(lon_w, lat_w, lon_e, lat_e)
        _, _, dy = GEOD.inv(lon_s, lat_s, lon_n, lat_n)
        if np.isfinite(dx) and np.isfinite(dy) and dx > 0 and dy > 0:
            return float(dx), float(dy)
    return xres, yres


def _axis_to_meter_factor(crs: CRS, axis_index: int) -> float:
    try:
        factor = float(crs.axis_info[axis_index].unit_conversion_factor)
    except (AttributeError, IndexError, TypeError, ValueError):
        return 1.0
    if not math.isfinite(factor) or factor <= 0:
        return 1.0
    return factor

def simulate_runoff(
    geometry: Polygon | MultiPolygon,
    config: SimulationConfig,
) -> RunoffResult:
    raster_crs = raster_crs_for_source(config.dem_path)
    geometry_dem, geometry_wgs84, detected_crs = normalize_geometry(geometry, config.input_crs, raster_crs)

    rainfall_mm = max(0.0, float(config.rainfall_mm))
    duration_h = max(float(config.duration_min) / 60.0, 1.0 / 60.0)
    infiltration_loss_mm = max(0.0, float(config.infiltration_rate_mm_h)) * duration_h
    initial_loss_mm = max(0.0, float(config.initial_loss_mm))
    runoff_coefficient = float(np.clip(config.runoff_coefficient, 0.0, 1.0))
    effective_rainfall_mm = max(0.0, rainfall_mm - infiltration_loss_mm - initial_loss_mm) * runoff_coefficient
    effective_depth_m = effective_rainfall_mm / 1000.0

    dem, transform, valid_mask, geometry_dem = _read_dem_clip(
        config.dem_path,
        geometry_dem,
        max_cells=max(10_000, int(config.max_cells)),
    )

    if np.count_nonzero(valid_mask) < 9:
        raise RunoffModelError("El poligono seleccionado no contiene suficientes celdas validas del MDE.")

    xres = abs(float(transform.a))
    yres = abs(float(transform.e))
    xres_m, yres_m = _pixel_size_m(transform, raster_crs, geometry_dem)
    cell_area_m2 = xres_m * yres_m
    area_m2 = float(np.count_nonzero(valid_mask) * cell_area_m2)

    if config.condition_dem:
        routing_dem, fill_depth_m = _condition_dem_for_routing(dem, valid_mask, xres_m, yres_m, config.fill_epsilon_m)
    else:
        routing_dem = dem.astype("float32", copy=True)
        fill_depth_m = np.zeros_like(dem, dtype="float32")
        fill_depth_m[~valid_mask] = np.nan

    down_flat, best_slope, sink_mask, outlet_mask = _d8_flow_direction(routing_dem, valid_mask, xres_m, yres_m)
    flow_accumulation_m2 = _flow_accumulation_area(routing_dem, valid_mask, down_flat, cell_area_m2)
    runoff_accumulation_m3 = flow_accumulation_m2 * effective_depth_m
    slope_percent = _slope_percent(dem, valid_mask, xres_m, yres_m)

    flood_index = _relative_flood_index(flow_accumulation_m2, slope_percent, sink_mask, valid_mask)
    concentration_mask = _percentile_mask(
        flow_accumulation_m2,
        valid_mask,
        float(config.concentration_percentile),
    )
    (
        stream_center_mask,
        channel_base_mask,
        large_channel_mask,
        overbank_mask,
        distance_to_channel_m,
        channel_depth_m,
    ) = _channel_flood_masks(
        dem,
        flow_accumulation_m2,
        slope_percent,
        flood_index,
        valid_mask,
        effective_rainfall_mm,
        config,
        xres_m,
        yres_m,
    )
    relative_flood_mask = _lowland_flood_mask(
        flow_accumulation_m2,
        slope_percent,
        flood_index,
        valid_mask,
        channel_base_mask | overbank_mask,
        effective_rainfall_mm,
        config,
    )
    flood_mask = relative_flood_mask | channel_base_mask | overbank_mask

    flow_values = flow_accumulation_m2[valid_mask & np.isfinite(flow_accumulation_m2)]
    runoff_values = runoff_accumulation_m3[valid_mask & np.isfinite(runoff_accumulation_m3)]
    flood_values = flood_index[valid_mask & np.isfinite(flood_index)]
    depth_values = channel_depth_m[large_channel_mask & np.isfinite(channel_depth_m)]
    fill_values = fill_depth_m[valid_mask & np.isfinite(fill_depth_m) & (fill_depth_m > 0)]

    summary = {
        "model_version": RUNOFF_MODEL_VERSION,
        "raster_crs": _crs_label(raster_crs),
        "input_crs_detected": detected_crs,
        "cells": int(np.count_nonzero(valid_mask)),
        "rows": int(dem.shape[0]),
        "cols": int(dem.shape[1]),
        "pixel_size_m": float((xres_m + yres_m) / 2.0),
        "area_ha": area_m2 / 10_000.0,
        "rainfall_mm": rainfall_mm,
        "duration_min": float(config.duration_min),
        "rainfall_intensity_mm_h": rainfall_mm / duration_h,
        "infiltration_loss_mm": infiltration_loss_mm,
        "initial_loss_mm": initial_loss_mm,
        "runoff_coefficient": runoff_coefficient,
        "condition_dem": bool(config.condition_dem),
        "fill_epsilon_m": float(config.fill_epsilon_m),
        "depression_fill_cells": int(np.count_nonzero(valid_mask & np.isfinite(fill_depth_m) & (fill_depth_m > 0))),
        "max_depression_fill_m": _safe_max(fill_values),
        "p95_depression_fill_m": _safe_percentile(fill_values, 95),
        "effective_rainfall_mm": effective_rainfall_mm,
        "direct_runoff_volume_m3": area_m2 * effective_depth_m,
        "max_accumulated_area_ha": _safe_max(flow_values) / 10_000.0,
        "p95_accumulated_area_ha": _safe_percentile(flow_values, 95) / 10_000.0,
        "max_accumulated_runoff_m3": _safe_max(runoff_values),
        "concentration_area_ha": float(np.count_nonzero(concentration_mask) * cell_area_m2 / 10_000.0),
        "relative_flood_area_ha": float(np.count_nonzero(relative_flood_mask) * cell_area_m2 / 10_000.0),
        "flood_susceptible_area_ha": float(np.count_nonzero(flood_mask) * cell_area_m2 / 10_000.0),
        "stream_center_area_ha": float(np.count_nonzero(stream_center_mask) * cell_area_m2 / 10_000.0),
        "channel_occupied_area_ha": float(np.count_nonzero(channel_base_mask) * cell_area_m2 / 10_000.0),
        "large_channel_area_ha": float(np.count_nonzero(large_channel_mask) * cell_area_m2 / 10_000.0),
        "overbank_area_ha": float(np.count_nonzero(overbank_mask) * cell_area_m2 / 10_000.0),
        "stream_percentile": float(config.stream_percentile),
        "channel_base_half_width_m": float(config.channel_base_half_width_m),
        "effective_channel_half_width_m": float(max(float(config.channel_base_half_width_m), min(xres_m, yres_m) / 2.0)),
        "channel_spill_threshold_mm": float(config.channel_spill_threshold_mm),
        "channel_overflow_excess_mm": max(0.0, effective_rainfall_mm - float(config.channel_spill_threshold_mm)),
        "channel_overflow_width_m": _channel_overflow_width_m(effective_rainfall_mm, config),
        "channel_max_overflow_width_m": float(config.channel_max_overflow_width_m),
        "large_channel_depth_threshold_m": float(config.large_channel_depth_threshold_m),
        "large_channel_search_radius_m": float(config.large_channel_search_radius_m),
        "large_channel_bank_percentile": float(config.large_channel_bank_percentile),
        "max_large_channel_depth_m": _safe_max(depth_values),
        "p95_large_channel_depth_m": _safe_percentile(depth_values, 95),
        "sink_cells": int(np.count_nonzero(sink_mask & valid_mask)),
        "outlet_cells": int(np.count_nonzero(outlet_mask & valid_mask)),
        "max_flood_index": _safe_max(flood_values),
    }

    return RunoffResult(
        dem=dem,
        routing_dem=routing_dem,
        fill_depth_m=fill_depth_m,
        valid_mask=valid_mask,
        transform=transform,
        raster_crs_wkt=raster_crs.to_wkt(),
        geometry_dem=geometry_dem,
        geometry_wgs84=geometry_wgs84,
        flow_accumulation_m2=flow_accumulation_m2,
        runoff_accumulation_m3=runoff_accumulation_m3,
        slope_percent=slope_percent,
        flood_index=flood_index,
        concentration_mask=concentration_mask,
        flood_mask=flood_mask,
        relative_flood_mask=relative_flood_mask,
        stream_center_mask=stream_center_mask,
        channel_base_mask=channel_base_mask,
        large_channel_mask=large_channel_mask,
        overbank_mask=overbank_mask,
        distance_to_channel_m=distance_to_channel_m,
        channel_depth_m=channel_depth_m,
        sink_mask=sink_mask,
        outlet_mask=outlet_mask,
        summary=summary,
    )


def clip_dem_to_geotiff_bytes(
    geometry: Polygon | MultiPolygon,
    dem_path: str | Path = REMOTE_DEM_DRIVE_URL,
    input_crs: str = "auto",
    buffer_m: float = 0.0,
    max_cells: int | None = None,
) -> ClipDemResult:
    raster_crs = raster_crs_for_source(dem_path)
    geometry_dem, geometry_wgs84, detected_crs = normalize_geometry(geometry, input_crs, raster_crs)
    if buffer_m:
        geometry_dem = _repair_geometry(_polygonal_geometry(geometry_dem.buffer(float(buffer_m))))

    dem, transform, valid_mask, geometry_dem = _read_dem_clip(dem_path, geometry_dem, max_cells=max_cells)
    if np.count_nonzero(valid_mask) < 1:
        raise RunoffModelError("El recorte no contiene celdas validas del MDE.")

    nodata = -9999.0
    write_arr = np.where(valid_mask & np.isfinite(dem), dem, nodata).astype("float32")
    with MemoryFile() as memfile:
        with memfile.open(
            driver="GTiff",
            width=write_arr.shape[1],
            height=write_arr.shape[0],
            count=1,
            dtype="float32",
            crs=raster_crs.to_wkt(),
            transform=transform,
            nodata=nodata,
            compress="deflate",
            tiled=True,
            blockxsize=128,
            blockysize=128,
        ) as dst:
            dst.write(write_arr, 1)
            dst.update_tags(AREA_OR_POINT="Area")
        data = memfile.read()

    xres_m, yres_m = _pixel_size_m(transform, raster_crs, geometry_dem)
    summary = {
        "source": str(dem_path),
        "model_version": RUNOFF_MODEL_VERSION,
        "raster_crs": _crs_label(raster_crs),
        "input_crs_detected": detected_crs,
        "rows": int(write_arr.shape[0]),
        "cols": int(write_arr.shape[1]),
        "cells": int(np.count_nonzero(valid_mask)),
        "pixel_size_m": float((xres_m + yres_m) / 2.0),
        "area_ha": float(np.count_nonzero(valid_mask) * xres_m * yres_m / 10000.0),
        "output_bytes": len(data),
    }
    return ClipDemResult(data=data, summary=summary, geometry_wgs84=geometry_wgs84)


def _read_dem_clip(
    dem_path: str | Path,
    geometry_dem: Polygon | MultiPolygon,
    max_cells: int | None,
) -> tuple[np.ndarray, Affine, np.ndarray, Polygon | MultiPolygon]:
    with open_raster_source(dem_path, subset_bounds=geometry_dem.bounds) as src:
        raster_bounds = box(*src.bounds)
        if not raster_bounds.intersects(geometry_dem):
            raise RunoffModelError("El poligono no intersecta el MDE.")
        geometry_dem = _polygonal_geometry(geometry_dem.intersection(raster_bounds))

        try:
            window = geometry_window(src, [mapping(geometry_dem)], pad_x=1, pad_y=1)
        except ValueError as exc:
            raise RunoffModelError("No se pudo calcular la ventana del raster para el poligono.") from exc

        window = _clamp_window(window, src.width, src.height)
        if window.width < 2 or window.height < 2:
            raise RunoffModelError("La ventana del poligono es demasiado pequena.")

        total_cells = float(window.width * window.height)
        if max_cells is None:
            scale = 1
        else:
            scale = max(1, int(math.ceil(math.sqrt(total_cells / max(1, max_cells)))))
        out_width = max(2, int(math.ceil(window.width / scale)))
        out_height = max(2, int(math.ceil(window.height / scale)))

        dem = src.read(
            1,
            window=window,
            out_shape=(out_height, out_width),
            resampling=Resampling.bilinear,
            masked=True,
        )
        base_transform = src.window_transform(window)
        transform = base_transform * Affine.scale(window.width / out_width, window.height / out_height)

        arr = dem.filled(np.nan).astype("float32")
        if src.nodata is not None:
            arr[np.isclose(arr, src.nodata)] = np.nan

        valid_geom = geometry_mask(
            [mapping(geometry_dem)],
            out_shape=arr.shape,
            transform=transform,
            invert=True,
            all_touched=False,
        )
        valid_mask = valid_geom & np.isfinite(arr)
        arr[~valid_mask] = np.nan

    return arr, transform, valid_mask, geometry_dem


def _clamp_window(window: Window, width: int, height: int) -> Window:
    col_off = max(0, int(math.floor(window.col_off)))
    row_off = max(0, int(math.floor(window.row_off)))
    col_stop = min(width, int(math.ceil(window.col_off + window.width)))
    row_stop = min(height, int(math.ceil(window.row_off + window.height)))
    return Window(col_off, row_off, col_stop - col_off, row_stop - row_off)


def _condition_dem_for_routing(
    dem: np.ndarray,
    valid_mask: np.ndarray,
    xres: float,
    yres: float,
    epsilon_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    filled = dem.astype("float64", copy=True)
    filled[~valid_mask] = np.nan
    rows, cols = dem.shape
    visited = np.zeros((rows, cols), dtype=bool)
    seeds = _routing_outlet_seed_mask(valid_mask)
    heap: list[tuple[float, int, int]] = []

    seed_rows, seed_cols = np.nonzero(seeds)
    for r, c in zip(seed_rows, seed_cols, strict=False):
        visited[r, c] = True
        heapq.heappush(heap, (float(filled[r, c]), int(r), int(c)))

    if not heap:
        fill_depth = np.zeros_like(dem, dtype="float32")
        fill_depth[~valid_mask] = np.nan
        return dem.astype("float32", copy=True), fill_depth

    epsilon = max(float(epsilon_m), 0.0)
    neighbors = (
        (-1, 0),
        (1, 0),
        (0, -1),
        (0, 1),
        (-1, -1),
        (-1, 1),
        (1, -1),
        (1, 1),
    )

    while heap:
        elev, r, c = heapq.heappop(heap)
        for dr, dc in neighbors:
            nr = r + dr
            nc = c + dc
            if nr < 0 or nr >= rows or nc < 0 or nc >= cols:
                continue
            if visited[nr, nc] or not valid_mask[nr, nc]:
                continue
            visited[nr, nc] = True
            neighbor_elev = float(filled[nr, nc])
            if neighbor_elev <= elev:
                neighbor_elev = elev + epsilon
                filled[nr, nc] = neighbor_elev
            heapq.heappush(heap, (neighbor_elev, nr, nc))

    fill_depth = np.maximum(filled - dem, 0.0).astype("float32")
    fill_depth[~valid_mask] = np.nan
    routing_dem = filled.astype("float32")
    routing_dem[~valid_mask] = np.nan
    return routing_dem, fill_depth


def _routing_outlet_seed_mask(valid_mask: np.ndarray) -> np.ndarray:
    rows, cols = valid_mask.shape
    seeds = np.zeros_like(valid_mask, dtype=bool)
    seeds[0, :] = valid_mask[0, :]
    seeds[-1, :] = valid_mask[-1, :]
    seeds[:, 0] = valid_mask[:, 0]
    seeds[:, -1] = valid_mask[:, -1]

    neighbors = (
        (-1, 0),
        (1, 0),
        (0, -1),
        (0, 1),
        (-1, -1),
        (-1, 1),
        (1, -1),
        (1, 1),
    )
    for dr, dc in neighbors:
        src_r, dst_r = _paired_slices(rows, dr)
        src_c, dst_c = _paired_slices(cols, dc)
        outside_neighbor = valid_mask[src_r, src_c] & ~valid_mask[dst_r, dst_c]
        if np.any(outside_neighbor):
            local_seeds = seeds[src_r, src_c]
            local_seeds[outside_neighbor] = True
            seeds[src_r, src_c] = local_seeds
    return seeds

def _d8_flow_direction(
    dem: np.ndarray,
    valid_mask: np.ndarray,
    xres: float,
    yres: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rows, cols = dem.shape
    flat_index = np.arange(rows * cols, dtype=np.int64).reshape(rows, cols)
    down = np.full((rows, cols), -1, dtype=np.int64)
    best_slope = np.zeros((rows, cols), dtype="float32")

    directions = (
        (-1, 0, yres),
        (1, 0, yres),
        (0, -1, xres),
        (0, 1, xres),
        (-1, -1, math.hypot(xres, yres)),
        (-1, 1, math.hypot(xres, yres)),
        (1, -1, math.hypot(xres, yres)),
        (1, 1, math.hypot(xres, yres)),
    )

    for dr, dc, distance in directions:
        src_r, dst_r = _paired_slices(rows, dr)
        src_c, dst_c = _paired_slices(cols, dc)

        valid_pair = valid_mask[src_r, src_c] & valid_mask[dst_r, dst_c]
        drop = dem[src_r, src_c] - dem[dst_r, dst_c]
        slope = drop / distance
        update = valid_pair & np.isfinite(slope) & (slope > best_slope[src_r, src_c]) & (slope > 0)

        if np.any(update):
            local_best = best_slope[src_r, src_c]
            local_down = down[src_r, src_c]
            local_best[update] = slope[update]
            local_down[update] = flat_index[dst_r, dst_c][update]
            best_slope[src_r, src_c] = local_best
            down[src_r, src_c] = local_down

    sink_mask = valid_mask & (down < 0)
    edge_mask = np.zeros_like(valid_mask, dtype=bool)
    edge_mask[0, :] = valid_mask[0, :]
    edge_mask[-1, :] = valid_mask[-1, :]
    edge_mask[:, 0] = valid_mask[:, 0]
    edge_mask[:, -1] = valid_mask[:, -1]

    adjacent_outside = edge_mask.copy()
    for dr, dc, _ in directions:
        src_r, dst_r = _paired_slices(rows, dr)
        src_c, dst_c = _paired_slices(cols, dc)
        outside_neighbor = valid_mask[src_r, src_c] & ~valid_mask[dst_r, dst_c]
        if np.any(outside_neighbor):
            local_adjacent = adjacent_outside[src_r, src_c]
            local_adjacent[outside_neighbor] = True
            adjacent_outside[src_r, src_c] = local_adjacent

    outlet_mask = sink_mask & adjacent_outside
    return down.reshape(-1), best_slope, sink_mask, outlet_mask


def _paired_slices(size: int, delta: int) -> tuple[slice, slice]:
    if delta < 0:
        return slice(-delta, size), slice(0, size + delta)
    if delta > 0:
        return slice(0, size - delta), slice(delta, size)
    return slice(0, size), slice(0, size)


def _flow_accumulation_area(
    dem: np.ndarray,
    valid_mask: np.ndarray,
    down_flat: np.ndarray,
    cell_area_m2: float,
) -> np.ndarray:
    flat_dem = dem.reshape(-1)
    flat_valid = valid_mask.reshape(-1)
    valid_idx = np.flatnonzero(flat_valid & np.isfinite(flat_dem))
    order = valid_idx[np.argsort(flat_dem[valid_idx])[::-1]]

    accumulation = np.zeros(flat_dem.shape[0], dtype="float64")
    accumulation[valid_idx] = cell_area_m2

    for idx in order:
        downstream = down_flat[idx]
        if downstream >= 0:
            accumulation[downstream] += accumulation[idx]

    out = accumulation.reshape(dem.shape)
    out[~valid_mask] = np.nan
    return out


def _slope_percent(dem: np.ndarray, valid_mask: np.ndarray, xres: float, yres: float) -> np.ndarray:
    filled = _fill_invalid_with_nearest_mean(dem, valid_mask)
    gy, gx = np.gradient(filled, yres, xres)
    slope = np.sqrt(gx * gx + gy * gy) * 100.0
    slope = slope.astype("float32")
    slope[~valid_mask] = np.nan
    return slope


def _fill_invalid_with_nearest_mean(dem: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    filled = dem.astype("float32", copy=True)
    mean_value = float(np.nanmean(filled[valid_mask]))
    filled[~np.isfinite(filled)] = mean_value

    # A few passes are enough to soften polygon borders for slope rendering.
    for _ in range(3):
        invalid = ~valid_mask
        if not np.any(invalid):
            break
        padded = np.pad(filled, 1, mode="edge")
        neighbor_sum = np.zeros_like(filled, dtype="float32")
        neighbor_count = np.zeros_like(filled, dtype="float32")
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                neigh = padded[1 + dr : 1 + dr + filled.shape[0], 1 + dc : 1 + dc + filled.shape[1]]
                finite = np.isfinite(neigh)
                neighbor_sum += np.where(finite, neigh, 0)
                neighbor_count += finite
        replacement = np.divide(neighbor_sum, neighbor_count, out=np.full_like(filled, mean_value), where=neighbor_count > 0)
        filled[invalid] = replacement[invalid]
    return filled


def _relative_flood_index(
    flow_accumulation_m2: np.ndarray,
    slope_percent: np.ndarray,
    sink_mask: np.ndarray,
    valid_mask: np.ndarray,
) -> np.ndarray:
    slope_fraction = np.maximum(slope_percent / 100.0, 0.0005)
    twi_like = np.log1p(flow_accumulation_m2 / np.maximum(slope_fraction, 0.0005))
    acc_score = _robust_unit_scale(np.log1p(flow_accumulation_m2), valid_mask)
    flat_score = 1.0 - _robust_unit_scale(np.log1p(slope_percent + 0.01), valid_mask)
    wetness_score = _robust_unit_scale(twi_like, valid_mask)
    sink_score = sink_mask.astype("float32")
    index = 100.0 * (0.45 * wetness_score + 0.30 * acc_score + 0.20 * flat_score + 0.05 * sink_score)
    index = index.astype("float32")
    index[~valid_mask] = np.nan
    return index


def _lowland_flood_mask(
    flow_accumulation_m2: np.ndarray,
    slope_percent: np.ndarray,
    flood_index: np.ndarray,
    valid_mask: np.ndarray,
    excluded_mask: np.ndarray,
    effective_rainfall_mm: float,
    config: SimulationConfig,
) -> np.ndarray:
    empty = np.zeros_like(valid_mask, dtype=bool)
    if effective_rainfall_mm <= 0.0:
        return empty

    analysis_mask = (
        valid_mask
        & ~excluded_mask
        & np.isfinite(flow_accumulation_m2)
        & np.isfinite(slope_percent)
        & np.isfinite(flood_index)
    )
    if not np.any(analysis_mask):
        return empty

    flow_values = flow_accumulation_m2[analysis_mask]
    slope_values = slope_percent[analysis_mask]
    index_values = flood_index[analysis_mask]
    if flow_values.size == 0 or slope_values.size == 0 or index_values.size == 0:
        return empty

    flood_percentile = min(max(float(config.flood_percentile), 50.0), 99.9)
    rain_reference_mm = max(50.0, max(0.0, float(config.channel_spill_threshold_mm)) * 3.0)
    rain_pressure = min(1.0, max(0.0, float(effective_rainfall_mm)) / rain_reference_mm)
    dynamic_percentile = max(85.0, flood_percentile - 10.0 * rain_pressure)

    flow_threshold = float(np.nanpercentile(flow_values, dynamic_percentile))
    index_threshold = float(np.nanpercentile(index_values, dynamic_percentile))
    flat_slope_threshold = float(min(max(np.nanpercentile(slope_values, 45), 0.5), 5.0))
    envelope_slope_threshold = float(min(max(np.nanpercentile(slope_values, 65), 1.0), 8.0))

    accumulated_plain = analysis_mask & (flow_accumulation_m2 >= flow_threshold)
    wet_or_flat = (slope_percent <= flat_slope_threshold) | (flood_index >= index_threshold)
    lowland_mask = accumulated_plain & wet_or_flat

    if rain_pressure >= 0.35 and np.any(lowland_mask):
        envelope = analysis_mask & (slope_percent <= envelope_slope_threshold)
        lowland_mask = ndimage.binary_dilation(lowland_mask, iterations=1) & envelope

    return lowland_mask


def _channel_overflow_width_m(effective_rainfall_mm: float, config: SimulationConfig) -> float:
    spill_threshold_mm = max(0.0, float(config.channel_spill_threshold_mm))
    excess_mm = max(0.0, float(effective_rainfall_mm) - spill_threshold_mm)
    max_overflow_width_m = max(0.0, float(config.channel_max_overflow_width_m))
    if excess_mm <= 0.0 or max_overflow_width_m <= 0.0:
        return 0.0
    full_spill_mm = max(spill_threshold_mm, 25.0)
    rain_pressure = min(1.0, excess_mm / full_spill_mm)
    return max_overflow_width_m * rain_pressure


def _channel_flood_masks(
    dem: np.ndarray,
    flow_accumulation_m2: np.ndarray,
    slope_percent: np.ndarray,
    flood_index: np.ndarray,
    valid_mask: np.ndarray,
    effective_rainfall_mm: float,
    config: SimulationConfig,
    xres: float,
    yres: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    stream_center_mask = _percentile_mask(
        flow_accumulation_m2,
        valid_mask,
        float(config.stream_percentile),
    )
    distance_to_channel_m = np.full(flow_accumulation_m2.shape, np.nan, dtype="float32")
    channel_depth_m = np.full(flow_accumulation_m2.shape, np.nan, dtype="float32")
    if not np.any(stream_center_mask):
        empty = np.zeros_like(valid_mask, dtype=bool)
        return empty, empty, empty, empty, distance_to_channel_m, channel_depth_m

    distance = ndimage.distance_transform_edt(~stream_center_mask, sampling=(yres, xres)).astype("float32")
    distance[~valid_mask] = np.nan
    distance_to_channel_m = distance

    requested_half_width = max(0.0, float(config.channel_base_half_width_m))
    pixel_half_width = min(xres, yres) / 2.0
    effective_half_width = max(requested_half_width, pixel_half_width)
    fixed_channel_mask = valid_mask & (stream_center_mask | (distance_to_channel_m <= effective_half_width))

    large_channel_mask, channel_depth_m = _large_channel_mask(
        dem,
        valid_mask,
        stream_center_mask,
        distance_to_channel_m,
        config,
        xres,
        yres,
    )
    channel_base_mask = fixed_channel_mask | large_channel_mask

    overflow_width_m = _channel_overflow_width_m(effective_rainfall_mm, config)
    if overflow_width_m <= 0.0:
        empty = np.zeros_like(valid_mask, dtype=bool)
        return stream_center_mask, channel_base_mask, large_channel_mask, empty, distance_to_channel_m, channel_depth_m

    distance_from_channel_edge_m = ndimage.distance_transform_edt(~channel_base_mask, sampling=(yres, xres)).astype("float32")
    distance_from_channel_edge_m[~valid_mask] = np.nan
    overbank_mask = (
        valid_mask
        & ~channel_base_mask
        & np.isfinite(distance_from_channel_edge_m)
        & (distance_from_channel_edge_m <= overflow_width_m)
    )
    return stream_center_mask, channel_base_mask, large_channel_mask, overbank_mask, distance_to_channel_m, channel_depth_m


def _large_channel_mask(
    dem: np.ndarray,
    valid_mask: np.ndarray,
    stream_center_mask: np.ndarray,
    distance_to_channel_m: np.ndarray,
    config: SimulationConfig,
    xres: float,
    yres: float,
) -> tuple[np.ndarray, np.ndarray]:
    depth_threshold = max(0.0, float(config.large_channel_depth_threshold_m))
    search_radius_m = max(0.0, float(config.large_channel_search_radius_m))
    bank_percentile = float(np.clip(config.large_channel_bank_percentile, 50.0, 95.0))
    channel_depth_m = np.full(dem.shape, np.nan, dtype="float32")

    if depth_threshold <= 0.0 or search_radius_m <= 0.0:
        return np.zeros_like(valid_mask, dtype=bool), channel_depth_m

    radius_y = max(1, int(math.ceil(search_radius_m / max(yres, 0.001))))
    radius_x = max(1, int(math.ceil(search_radius_m / max(xres, 0.001))))
    filled = _fill_invalid_with_nearest_mean(dem, valid_mask)
    local_bank_elev = ndimage.percentile_filter(
        filled,
        percentile=bank_percentile,
        size=(radius_y * 2 + 1, radius_x * 2 + 1),
        mode="nearest",
    )
    depth = np.maximum(local_bank_elev - dem, 0.0).astype("float32")
    depth[~valid_mask] = np.nan
    channel_depth_m = depth

    near_stream = valid_mask & np.isfinite(distance_to_channel_m) & (distance_to_channel_m <= search_radius_m)
    large_channel = near_stream & (channel_depth_m >= depth_threshold)
    return large_channel, channel_depth_m

def _robust_unit_scale(values: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    valid_values = values[valid_mask & np.isfinite(values)]
    out = np.zeros(values.shape, dtype="float32")
    if valid_values.size == 0:
        out[~valid_mask] = np.nan
        return out
    lo = np.nanpercentile(valid_values, 5)
    hi = np.nanpercentile(valid_values, 99)
    if not np.isfinite(hi - lo) or hi <= lo:
        hi = float(np.nanmax(valid_values))
        lo = float(np.nanmin(valid_values))
    if hi <= lo:
        out[valid_mask] = 0.0
    else:
        out = np.clip((values - lo) / (hi - lo), 0, 1).astype("float32")
    out[~valid_mask] = np.nan
    return out


def _percentile_mask(values: np.ndarray, valid_mask: np.ndarray, percentile: float) -> np.ndarray:
    percentile = float(np.clip(percentile, 50.0, 99.9))
    valid_values = values[valid_mask & np.isfinite(values)]
    if valid_values.size == 0:
        return np.zeros_like(valid_mask, dtype=bool)
    threshold = float(np.nanpercentile(valid_values, percentile))
    return valid_mask & np.isfinite(values) & (values >= threshold)


def _safe_percentile(values: np.ndarray, percentile: float) -> float:
    if values.size == 0:
        return 0.0
    return float(np.nanpercentile(values, percentile))


def _safe_max(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    return float(np.nanmax(values))


def _repair_geometry(geometry: Polygon | MultiPolygon) -> Polygon | MultiPolygon:
    if geometry.is_valid:
        return geometry
    repaired = geometry.buffer(0)
    return _polygonal_geometry(repaired)


def _polygonal_geometry(geometry: Any) -> Polygon | MultiPolygon:
    if isinstance(geometry, (Polygon, MultiPolygon)):
        return geometry
    if isinstance(geometry, GeometryCollection):
        polys = [g for g in geometry.geoms if isinstance(g, (Polygon, MultiPolygon)) and not g.is_empty]
        if not polys:
            return Polygon()
        return _polygonal_geometry(unary_union(polys))
    if hasattr(geometry, "geoms"):
        polys = [g for g in geometry.geoms if isinstance(g, (Polygon, MultiPolygon)) and not g.is_empty]
        if polys:
            return _polygonal_geometry(unary_union(polys))
    raise RunoffModelError("La geometria debe ser poligonal.")


def result_array(result: RunoffResult, layer: str) -> np.ndarray:
    layers = {
        "dem": result.dem,
        "routing_dem": result.routing_dem,
        "fill_depth_m": result.fill_depth_m,
        "flow_accumulation_m2": result.flow_accumulation_m2,
        "runoff_accumulation_m3": result.runoff_accumulation_m3,
        "slope_percent": result.slope_percent,
        "flood_index": result.flood_index,
        "concentration_mask": result.concentration_mask.astype("float32"),
        "flood_mask": result.flood_mask.astype("float32"),
        "relative_flood_mask": result.relative_flood_mask.astype("float32"),
        "stream_center_mask": result.stream_center_mask.astype("float32"),
        "channel_base_mask": result.channel_base_mask.astype("float32"),
        "large_channel_mask": result.large_channel_mask.astype("float32"),
        "overbank_mask": result.overbank_mask.astype("float32"),
        "distance_to_channel_m": result.distance_to_channel_m,
        "channel_depth_m": result.channel_depth_m,
        "sink_mask": result.sink_mask.astype("float32"),
        "outlet_mask": result.outlet_mask.astype("float32"),
    }
    if layer not in layers:
        raise RunoffModelError(f"Capa no soportada: {layer}")
    arr = layers[layer].astype("float32", copy=True)
    arr[~result.valid_mask] = np.nan
    return arr


def geotiff_bytes(result: RunoffResult, layer: str) -> bytes:
    arr = result_array(result, layer)
    nodata = -9999.0
    write_arr = np.where(np.isfinite(arr), arr, nodata).astype("float32")

    with MemoryFile() as memfile:
        with memfile.open(
            driver="GTiff",
            width=write_arr.shape[1],
            height=write_arr.shape[0],
            count=1,
            dtype="float32",
            crs=result.raster_crs_wkt,
            transform=result.transform,
            nodata=nodata,
            compress="deflate",
        ) as dst:
            dst.write(write_arr, 1)
            dst.update_tags(AREA_OR_POINT="Area")
        return memfile.read()


def geojson_geometry_bytes(result: RunoffResult) -> bytes:
    feature = {
        "type": "Feature",
        "properties": {"crs": "EPSG:4326"},
        "geometry": mapping(result.geometry_wgs84),
    }
    return json.dumps(feature, ensure_ascii=False, indent=2).encode("utf-8")


def render_map_png(result: RunoffResult, mode: str = "overview", dpi: int = 150) -> bytes:
    dem = result.dem
    mask = result.valid_mask
    extent = _raster_extent(result.transform, dem.shape)

    fig, ax = plt.subplots(figsize=(9, 7), dpi=dpi)
    hillshade = _hillshade(dem, mask, result.transform)
    ax.imshow(hillshade, cmap="gray", extent=extent, origin="upper")

    if mode == "overview":
        _overlay_log_accumulation(ax, result, extent)
        _overlay_mask(ax, result.concentration_mask, extent, color=(0.0, 0.55, 1.0, 0.55))
        _overlay_mask(ax, result.channel_base_mask, extent, color=(0.0, 0.25, 1.0, 0.75))
        _overlay_mask(ax, result.overbank_mask, extent, color=(1.0, 0.24, 0.02, 0.65))
        _overlay_mask(ax, result.relative_flood_mask, extent, color=(1.0, 0.75, 0.0, 0.45))
        title = "Acumulacion de flujo y zonas criticas relativas"
    elif mode == "accumulation":
        _overlay_log_accumulation(ax, result, extent)
        title = "Acumulacion D8 de flujo"
    elif mode == "flood":
        flood = np.ma.masked_invalid(result.flood_index)
        im = ax.imshow(flood, cmap="inferno", extent=extent, origin="upper", alpha=0.72, vmin=0, vmax=100)
        fig.colorbar(im, ax=ax, shrink=0.78, label="Indice relativo")
        _overlay_mask(ax, result.flood_mask, extent, color=(0.0, 0.8, 1.0, 0.55))
        title = "Susceptibilidad relativa a inundacion"
    elif mode == "slope":
        slope = np.ma.masked_invalid(result.slope_percent)
        im = ax.imshow(slope, cmap="viridis", extent=extent, origin="upper", alpha=0.78)
        fig.colorbar(im, ax=ax, shrink=0.78, label="Pendiente (%)")
        title = "Pendiente del terreno"
    else:
        raise RunoffModelError(f"Mapa no soportado: {mode}")

    _plot_geometry_boundary(ax, result.geometry_dem)
    ax.set_title(title)
    ax.set_xlabel("Este CRTM05 (m)")
    ax.set_ylabel("Norte CRTM05 (m)")
    ax.set_aspect("equal")
    fig.tight_layout()

    output = BytesIO()
    fig.savefig(output, format="png", bbox_inches="tight")
    plt.close(fig)
    return output.getvalue()


def _overlay_log_accumulation(ax: plt.Axes, result: RunoffResult, extent: tuple[float, float, float, float]) -> None:
    acc = np.log10(np.maximum(result.flow_accumulation_m2, 1.0))
    acc_scaled = _robust_unit_scale(acc, result.valid_mask)
    acc_masked = np.ma.masked_invalid(acc_scaled)
    alpha = np.nan_to_num(np.clip(acc_scaled, 0, 0.82), nan=0.0)
    im = ax.imshow(acc_masked, cmap="Blues", extent=extent, origin="upper", alpha=alpha)
    ax.figure.colorbar(im, ax=ax, shrink=0.78, label="Acumulacion relativa")


def _overlay_mask(ax: plt.Axes, mask: np.ndarray, extent: tuple[float, float, float, float], color: tuple[float, float, float, float]) -> None:
    rgba = np.zeros((*mask.shape, 4), dtype="float32")
    rgba[mask] = color
    ax.imshow(rgba, extent=extent, origin="upper")


def _plot_geometry_boundary(ax: plt.Axes, geometry: Polygon | MultiPolygon) -> None:
    geoms = geometry.geoms if isinstance(geometry, MultiPolygon) else [geometry]
    for geom in geoms:
        x, y = geom.exterior.xy
        ax.plot(x, y, color="black", linewidth=1.2)
        ax.plot(x, y, color="white", linewidth=0.45)


def _raster_extent(transform: Affine, shape: tuple[int, int]) -> tuple[float, float, float, float]:
    rows, cols = shape
    left = transform.c
    top = transform.f
    right = left + cols * transform.a
    bottom = top + rows * transform.e
    return (left, right, bottom, top)


def _hillshade(dem: np.ndarray, valid_mask: np.ndarray, transform: Affine) -> np.ndarray:
    filled = _fill_invalid_with_nearest_mean(dem, valid_mask)
    xres = abs(float(transform.a))
    yres = abs(float(transform.e))
    gy, gx = np.gradient(filled, yres, xres)
    slope = np.pi / 2.0 - np.arctan(np.sqrt(gx * gx + gy * gy))
    aspect = np.arctan2(-gx, gy)
    azimuth = np.deg2rad(315.0)
    altitude = np.deg2rad(45.0)
    shaded = np.sin(altitude) * np.sin(slope) + np.cos(altitude) * np.cos(slope) * np.cos(azimuth - aspect)
    shaded = np.clip((shaded + 1.0) / 2.0, 0, 1)
    shaded[~valid_mask] = 1.0
    return shaded
