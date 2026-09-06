from __future__ import annotations

import argparse
import json
from pathlib import Path

from remote_raster import DEFAULT_DRIVE_DEM_URL, is_http_url, probe_http_range, resolve_raster_url
from runoff_model import RunoffModelError, clip_dem_to_geotiff_bytes, load_geojson_geometry


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Recorta un MDE remoto o local usando un poligono GeoJSON sin descargar el raster completo."
    )
    parser.add_argument("--geojson", required=True, help="Ruta al GeoJSON con Polygon/MultiPolygon/FeatureCollection.")
    parser.add_argument("--output", default="mde_recortado.tif", help="GeoTIFF de salida.")
    parser.add_argument("--dem-url", default=DEFAULT_DRIVE_DEM_URL, help="URL remoto o ruta local del MDE.")
    parser.add_argument(
        "--input-crs",
        default="auto",
        choices=["auto", "EPSG:4326", "EPSG:5367", "raster"],
        help="CRS del poligono. Auto detecta WGS84/CRTM05; raster asume coordenadas del GeoTIFF.",
    )
    parser.add_argument("--buffer-m", type=float, default=0.0, help="Buffer opcional alrededor del poligono, en metros.")
    parser.add_argument(
        "--max-cells",
        type=int,
        default=None,
        help="Limite opcional de celdas de salida. Si se omite, exporta a resolucion nativa.",
    )
    parser.add_argument(
        "--skip-range-check",
        action="store_true",
        help="Omite la validacion HTTP Range. No recomendado para fuentes remotas nuevas.",
    )
    args = parser.parse_args()

    dem_source = args.dem_url
    if is_http_url(dem_source) and not args.skip_range_check:
        resolved_url = resolve_raster_url(dem_source)
        probe = probe_http_range(resolved_url)
        if not probe.supports_range:
            raise SystemExit(
                "La fuente remota no respondio con HTTP 206 Partial Content. "
                "No se hara una descarga parcial para evitar bajar el raster completo."
            )
        print(f"Range OK: {probe.content_range}")

    geometry = load_geojson_geometry(Path(args.geojson).read_bytes())
    try:
        clip = clip_dem_to_geotiff_bytes(
            geometry,
            dem_path=dem_source,
            input_crs=args.input_crs,
            buffer_m=args.buffer_m,
            max_cells=args.max_cells,
        )
    except RunoffModelError as exc:
        raise SystemExit(str(exc)) from exc

    output = Path(args.output)
    output.write_bytes(clip.data)
    print(json.dumps(clip.summary, indent=2, ensure_ascii=False))
    print(f"Escrito: {output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())