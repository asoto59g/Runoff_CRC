from __future__ import annotations

import base64
import hashlib
import json
import tempfile
from io import BytesIO
from pathlib import Path

import folium
import numpy as np
import streamlit as st
from PIL import Image
from folium.plugins import Draw
from streamlit_folium import st_folium

from remote_raster import RemoteRasterError
from runoff_model import (
    DEFAULT_DEM_PATH,
    REMOTE_DEM_DRIVE_URL,
    RUNOFF_MODEL_VERSION,
    RunoffModelError,
    SimulationConfig,
    clip_dem_to_geotiff_bytes,
    geojson_geometry_bytes,
    geotiff_bytes,
    load_geojson_geometry,
    read_raster_preview,
    simulate_runoff,
    transform_bounds_dem_to_wgs84,
)


st.set_page_config(
    page_title="Escorrentia DEM",
    page_icon="",
    layout="wide",
)


def main() -> None:
    st.title("Modelo de escorrentia por DEM")

    dem_path = sidebar_dem_path()
    if not dem_path:
        st.info("Selecciona una fuente MDE para iniciar el analisis.")
        return
    preview = load_preview(dem_path)
    if preview is None:
        return

    geometry, geometry_source, input_crs = geometry_panel(preview)
    config = simulation_controls(dem_path, input_crs)

    col_action, col_status = st.columns([0.23, 0.77], vertical_alignment="center")
    with col_action:
        run_clicked = st.button("Ejecutar simulacion", type="primary", use_container_width=True)
    with col_status:
        if geometry is None:
            st.info("Dibuja un poligono o carga un GeoJSON para habilitar la simulacion.")
        else:
            st.caption(f"Poligono activo: {geometry_source}")

    if geometry is not None:
        clip_panel(geometry, dem_path, input_crs)

    if run_clicked:
        if geometry is None:
            st.warning("No hay poligono de analisis.")
        else:
            with st.spinner("Recortando MDE y calculando escorrentia..."):
                try:
                    st.session_state["result"] = simulate_runoff(geometry, config)
                except (RunoffModelError, RemoteRasterError) as exc:
                    st.error(str(exc))
                except Exception as exc:  # Keep unexpected GIS errors visible in the prototype.
                    st.exception(exc)

    result = st.session_state.get("result")
    if result is not None and (
        not hasattr(result, "fill_depth_m")
        or result.summary.get("model_version") != RUNOFF_MODEL_VERSION
    ):
        st.session_state.pop("result", None)
        result = None
        st.info("El modelo cambio. Ejecuta nuevamente la simulacion para generar cauce acondicionado, desborde y relleno DEM.")
    if result is not None:
        result_panel(result)


def sidebar_dem_path() -> str | None:
    with st.sidebar:
        st.header("Insumos")
        source_mode = st.radio(
            "Fuente MDE",
            ["Google Drive publico", "Subir GeoTIFF", "Ruta del servidor (avanzado)"],
            index=0,
            horizontal=False,
        )
        if source_mode == "Google Drive publico":
            dem_path = st.text_input("Enlace Drive", value=REMOTE_DEM_DRIVE_URL)
            st.caption("Fuente remota por defecto. La lectura usa rangos HTTP para no bajar el raster completo.")
            return dem_path
        if source_mode == "Subir GeoTIFF":
            return uploaded_dem_selector()

        st.caption(
            "Uso avanzado: navega carpetas del servidor donde corre la app. "
            "En Streamlit Cloud esas rutas son Linux; para elegir un archivo de tu Windows usa Subir GeoTIFF."
        )
        return local_dem_selector()


def uploaded_dem_selector() -> str | None:
    uploaded = st.file_uploader(
        "Seleccionar GeoTIFF desde este equipo",
        type=["tif", "tiff"],
        accept_multiple_files=False,
        help="Este selector abre las carpetas del equipo del usuario. En Streamlit Cloud no permite navegar el sistema de archivos Linux del servidor.",
    )
    if uploaded is None:
        st.info("Selecciona un archivo .tif o .tiff desde tu equipo para continuar.")
        return None

    suffix = Path(uploaded.name).suffix.lower()
    if suffix not in {".tif", ".tiff"}:
        st.error("El archivo debe ser .tif o .tiff.")
        return None

    data = uploaded.getbuffer()
    sample_size = min(len(data), 1_048_576)
    digest_source = hashlib.sha256()
    digest_source.update(uploaded.name.encode("utf-8", errors="ignore"))
    digest_source.update(str(len(data)).encode("ascii"))
    digest_source.update(data[:sample_size])
    if len(data) > sample_size:
        digest_source.update(data[-sample_size:])
    digest = digest_source.hexdigest()[:16]
    stem = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in Path(uploaded.name).stem)
    stem = (stem[:80] or "uploaded_dem").strip("._-") or "uploaded_dem"
    upload_dir = Path(tempfile.gettempdir()) / "runoff_crc_uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    dem_path = upload_dir / f"{stem}_{digest}{suffix}"

    if not dem_path.exists() or dem_path.stat().st_size != len(data):
        with dem_path.open("wb") as output:
            output.write(data)

    st.caption("GeoTIFF cargado en la sesion")
    st.code(uploaded.name, language="text")
    return str(dem_path)


def local_dem_selector() -> str:
    default_file = Path(DEFAULT_DEM_PATH)
    default_dir = default_file.parent if default_file.parent.exists() else Path.home()

    if "local_dem_dir" not in st.session_state:
        st.session_state["local_dem_dir"] = str(default_dir)

    current_dir = Path(st.session_state["local_dem_dir"]).expanduser()
    if not current_dir.exists() or not current_dir.is_dir():
        current_dir = default_dir
        st.session_state["local_dem_dir"] = str(current_dir)

    start_points = local_start_points(default_dir)
    selected_start = st.selectbox(
        "Ubicacion rapida",
        list(start_points.keys()),
        index=0,
        key="local_start_point",
    )
    if st.button("Abrir ubicacion", use_container_width=True):
        st.session_state["local_dem_dir"] = str(start_points[selected_start])
        st.rerun()

    typed_dir = st.text_input("Carpeta actual", value=str(current_dir))
    typed_path = Path(typed_dir).expanduser()
    if typed_dir != str(current_dir) and typed_path.exists() and typed_path.is_dir():
        st.session_state["local_dem_dir"] = str(typed_path)
        st.rerun()

    nav_cols = st.columns([0.32, 0.68])
    if nav_cols[0].button("Subir", use_container_width=True, disabled=current_dir.parent == current_dir):
        st.session_state["local_dem_dir"] = str(current_dir.parent)
        st.rerun()

    dirs, rasters = list_local_entries(current_dir)
    selected_dir = nav_cols[1].selectbox("Subcarpetas", [""] + dirs, index=0, key="local_subdir")
    if selected_dir:
        st.session_state["local_dem_dir"] = str(current_dir / selected_dir)
        st.session_state["local_subdir"] = ""
        st.rerun()

    if not rasters:
        st.warning("No hay archivos .tif o .tiff en esta carpeta.")
        manual_path = st.text_input("Ruta manual del MDE", value=str(default_file), key="manual_local_dem")
        return manual_path

    default_name = default_file.name if current_dir == default_file.parent and default_file.name in rasters else rasters[0]
    selected_file = st.selectbox("Raster local", rasters, index=rasters.index(default_name), key="local_dem_file")
    selected_path = str(current_dir / selected_file)
    st.caption("Ruta seleccionada")
    st.code(selected_path, language="text")
    manual_path = st.text_input("Ruta manual opcional", value="")
    return manual_path.strip() or selected_path


def local_start_points(default_dir: Path) -> dict[str, Path]:
    points = {
        "Carpeta inicial": default_dir if default_dir.exists() else Path.cwd(),
        "Proyecto Runoff": Path.cwd(),
        "Usuario": Path.home(),
    }
    documents_dir = Path.home() / "Documents"
    if documents_dir.exists():
        points["Documentos"] = documents_dir
    one_drive_dir = Path.home() / "OneDrive"
    if one_drive_dir.exists():
        points["OneDrive"] = one_drive_dir
    for drive in [Path("C:/"), Path("D:/")]:
        if drive.exists():
            points[str(drive)] = drive
    return points


def list_local_entries(folder: Path) -> tuple[list[str], list[str]]:
    try:
        entries = list(folder.iterdir())
    except OSError as exc:
        st.error(f"No se pudo leer la carpeta: {exc}")
        return [], []

    dirs = sorted([entry.name for entry in entries if entry.is_dir() and not entry.name.startswith(".")], key=str.lower)
    rasters = sorted(
        [entry.name for entry in entries if entry.is_file() and entry.suffix.lower() in {".tif", ".tiff"}],
        key=str.lower,
    )
    return dirs, rasters


@st.cache_data(show_spinner=False)
def load_preview(dem_path: str):
    try:
        return read_raster_preview(dem_path)
    except Exception as exc:
        st.error(f"No se pudo leer el MDE: {exc}")
        return None


def geometry_panel(preview):
    mode = st.radio(
        "Poligono de analisis",
        ["Dibujar", "Cargar GeoJSON", "Pegar GeoJSON"],
        horizontal=True,
    )

    if mode == "Dibujar":
        geometry = draw_geometry(preview)
        return geometry, "dibujo en mapa", "EPSG:4326"

    with st.sidebar:
        st.header("Poligono")
        input_crs_label = st.selectbox(
            "CRS del GeoJSON",
            ["Auto", "WGS84 / EPSG:4326", "CRTM05 / EPSG:5367"],
            index=0,
        )
        input_crs = {
            "Auto": "auto",
            "WGS84 / EPSG:4326": "EPSG:4326",
            "CRTM05 / EPSG:5367": "EPSG:5367",
        }[input_crs_label]

    if mode == "Cargar GeoJSON":
        uploaded = st.file_uploader("GeoJSON", type=["geojson", "json"])
        if not uploaded:
            return None, "archivo GeoJSON", input_crs
        try:
            return load_geojson_geometry(uploaded.getvalue()), uploaded.name, input_crs
        except Exception as exc:
            st.error(f"GeoJSON no valido: {exc}")
            return None, uploaded.name, input_crs

    text = st.text_area("GeoJSON", height=180)
    if not text.strip():
        return None, "texto GeoJSON", input_crs
    try:
        return load_geojson_geometry(text), "texto GeoJSON", input_crs
    except Exception as exc:
        st.error(f"GeoJSON no valido: {exc}")
        return None, "texto GeoJSON", input_crs


def draw_geometry(preview):
    min_lon, min_lat, max_lon, max_lat = preview.bounds_wgs84
    center = [(min_lat + max_lat) / 2.0, (min_lon + max_lon) / 2.0]
    m = folium.Map(location=center, zoom_start=8, tiles=None, control_scale=True)
    add_base_layers(m)
    folium.Rectangle(
        bounds=[[min_lat, min_lon], [max_lat, max_lon]],
        color="#2b6cb0",
        fill=False,
        weight=1,
        tooltip="Extension MDE",
    ).add_to(m)

    Draw(
        export=False,
        position="topleft",
        draw_options={
            "polyline": False,
            "rectangle": True,
            "polygon": True,
            "circle": False,
            "circlemarker": False,
            "marker": False,
        },
        edit_options={"edit": True, "remove": True},
    ).add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)

    state = st_folium(
        m,
        height=520,
        use_container_width=True,
        returned_objects=["all_drawings", "last_active_drawing"],
        key="draw_map",
    )
    drawings = state.get("all_drawings") if isinstance(state, dict) else None
    if not drawings:
        drawing = state.get("last_active_drawing") if isinstance(state, dict) else None
        drawings = [drawing] if drawing else []
    if not drawings:
        return None
    return load_geojson_geometry(_drawings_to_feature_collection(drawings))


def _drawings_to_feature_collection(drawings):
    features = []
    for drawing in drawings:
        if not drawing:
            continue
        if drawing.get("type") == "Feature":
            features.append(drawing)
        elif drawing.get("type") in {"Polygon", "MultiPolygon"}:
            features.append({"type": "Feature", "properties": {}, "geometry": drawing})
        elif "geometry" in drawing:
            features.append({"type": "Feature", "properties": drawing.get("properties", {}), "geometry": drawing["geometry"]})

    return {"type": "FeatureCollection", "features": features}


def clip_panel(geometry, dem_path: str, input_crs: str) -> None:
    with st.expander("Recortar MDE por poligono", expanded=False):
        cols = st.columns(3)
        buffer_m = cols[0].number_input("Buffer recorte (m)", min_value=0.0, max_value=5000.0, value=0.0, step=10.0)
        cell_limit_label = cols[1].selectbox(
            "Limite de celdas",
            ["1,000,000", "500,000", "250,000", "Nativa"],
            index=0,
        )
        max_cells = None if cell_limit_label == "Nativa" else int(cell_limit_label.replace(",", ""))
        file_name = cols[2].text_input("Archivo", value="mde_recortado.tif")

        if st.button("Generar recorte MDE", use_container_width=True):
            with st.spinner("Leyendo solo la ventana necesaria del MDE..."):
                try:
                    st.session_state["dem_clip"] = clip_dem_to_geotiff_bytes(
                        geometry,
                        dem_path=dem_path,
                        input_crs=input_crs,
                        buffer_m=buffer_m,
                        max_cells=max_cells,
                    )
                except (RunoffModelError, RemoteRasterError) as exc:
                    st.error(str(exc))
                except Exception as exc:
                    st.exception(exc)

        clip = st.session_state.get("dem_clip")
        if clip is not None:
            summary = clip.summary
            st.caption(
                f"Recorte listo: {summary['rows']} x {summary['cols']} celdas, "
                f"{summary['area_ha']:,.2f} ha, {summary['output_bytes'] / 1024:,.1f} KB."
            )
            st.download_button(
                "Descargar MDE recortado",
                data=clip.data,
                file_name=file_name or "mde_recortado.tif",
                mime="image/tiff",
                use_container_width=True,
            )

def simulation_controls(dem_path: str, input_crs: str) -> SimulationConfig:
    with st.sidebar:
        st.header("Lluvia")
        rainfall_mm = st.number_input("Lluvia total (mm)", min_value=0.0, max_value=1000.0, value=75.0, step=5.0)
        duration_min = st.number_input("Duracion (min)", min_value=1.0, max_value=1440.0, value=60.0, step=10.0)
        infiltration_rate_mm_h = st.number_input("Perdida por infiltracion (mm/h)", min_value=0.0, max_value=300.0, value=10.0, step=1.0)
        initial_loss_mm = st.number_input("Abstraccion inicial (mm)", min_value=0.0, max_value=100.0, value=2.0, step=1.0)
        runoff_coefficient = st.slider("Coeficiente de escorrentia", min_value=0.0, max_value=1.0, value=0.65, step=0.05)

        st.header("Analisis")
        max_cells = st.select_slider(
            "Celdas maximas",
            options=[100_000, 250_000, 500_000, 700_000, 1_000_000, 1_500_000],
            value=700_000,
        )
        concentration_percentile = st.slider("Umbral concentracion (%)", 90.0, 99.9, 98.0, 0.1)
        flood_percentile = st.slider("Umbral inundabilidad (%)", 90.0, 99.9, 97.0, 0.1)
        condition_dem = st.checkbox("Acondicionar depresiones del DEM", value=True)
        fill_epsilon_m = st.number_input(
            "Micro pendiente relleno (m/celda)",
            min_value=0.0,
            max_value=0.05,
            value=0.001,
            step=0.001,
            format="%.3f",
        )

        st.header("Cauces")
        stream_percentile = st.slider("Umbral red cauces (%)", 95.0, 99.9, 99.0, 0.1)
        channel_base_half_width_m = st.number_input(
            "Caudal existente: semi-ancho (m)",
            min_value=0.0,
            max_value=100.0,
            value=5.0,
            step=1.0,
        )
        channel_spill_threshold_mm = st.number_input(
            "Lluvia efectiva antes de desborde (mm)",
            min_value=0.0,
            max_value=500.0,
            value=50.0,
            step=5.0,
        )
        channel_max_overflow_width_m = st.number_input(
            "Ancho maximo desborde lateral (m)",
            min_value=0.0,
            max_value=1000.0,
            value=80.0,
            step=5.0,
        )
        large_channel_depth_threshold_m = st.number_input(
            "Diferencia cauce-planicie (m)",
            min_value=0.0,
            max_value=20.0,
            value=1.5,
            step=0.1,
        )
        large_channel_search_radius_m = st.number_input(
            "Radio planicie vecina (m)",
            min_value=5.0,
            max_value=500.0,
            value=60.0,
            step=5.0,
        )
        large_channel_bank_percentile = st.slider("Percentil planicie vecina (%)", 50.0, 95.0, 75.0, 1.0)

    return SimulationConfig(
        dem_path=dem_path,
        rainfall_mm=rainfall_mm,
        duration_min=duration_min,
        infiltration_rate_mm_h=infiltration_rate_mm_h,
        initial_loss_mm=initial_loss_mm,
        runoff_coefficient=runoff_coefficient,
        max_cells=max_cells,
        condition_dem=condition_dem,
        fill_epsilon_m=fill_epsilon_m,
        concentration_percentile=concentration_percentile,
        flood_percentile=flood_percentile,
        stream_percentile=stream_percentile,
        channel_base_half_width_m=channel_base_half_width_m,
        channel_spill_threshold_mm=channel_spill_threshold_mm,
        channel_max_overflow_width_m=channel_max_overflow_width_m,
        large_channel_depth_threshold_m=large_channel_depth_threshold_m,
        large_channel_search_radius_m=large_channel_search_radius_m,
        large_channel_bank_percentile=large_channel_bank_percentile,
        input_crs=input_crs,
    )


def add_base_layers(m: folium.Map, default_base: str = "OSM") -> None:
    folium.TileLayer(
        tiles="OpenStreetMap",
        name="OSM",
        control=True,
        show=default_base == "OSM",
    ).add_to(m)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri, Maxar, Earthstar Geographics, and the GIS User Community",
        name="Satelite",
        control=True,
        show=default_base == "Satelite",
    ).add_to(m)


def result_panel(result) -> None:
    summary = result.summary
    st.divider()
    st.subheader("Resultados")

    metric_cols = st.columns(6)
    metric_cols[0].metric("Area", f"{summary['area_ha']:,.1f} ha")
    metric_cols[1].metric("Lluvia efectiva", f"{summary['effective_rainfall_mm']:,.1f} mm")
    metric_cols[2].metric("Volumen directo", f"{summary['direct_runoff_volume_m3']:,.0f} m3")
    metric_cols[3].metric("Cauce ocupado", f"{summary['channel_occupied_area_ha']:,.1f} ha")
    metric_cols[4].metric("Desborde", f"{summary['overbank_area_ha']:,.1f} ha")
    metric_cols[5].metric("Zona inundable", f"{summary['flood_susceptible_area_ha']:,.1f} ha")

    tabs = st.tabs(["Mapa", "Descargas", "Criterio"])

    with tabs[0]:
        result_interactive_map(result)

    with tabs[1]:
        st.dataframe(summary_table(summary), use_container_width=True, hide_index=True)
        download_cols = st.columns(4)
        download_cols[0].download_button(
            "Poligono WGS84",
            data=geojson_geometry_bytes(result),
            file_name="poligono_analisis_wgs84.geojson",
            mime="application/geo+json",
            use_container_width=True,
        )
        add_download(download_cols[1], result, "flow_accumulation_m2", "acumulacion_flujo_m2.tif")
        add_download(download_cols[2], result, "flood_index", "indice_inundabilidad_relativa.tif")
        add_download(download_cols[3], result, "flood_mask", "zonas_inundables.tif")

        download_cols_2 = st.columns(3)
        add_download(download_cols_2[0], result, "concentration_mask", "zonas_concentracion.tif")
        add_download(download_cols_2[1], result, "channel_base_mask", "cauce_ocupado.tif")
        add_download(download_cols_2[2], result, "overbank_mask", "desborde_cauce.tif")

        download_cols_3 = st.columns(3)
        add_download(download_cols_3[0], result, "large_channel_mask", "cauce_grande_inciso.tif")
        add_download(download_cols_3[1], result, "channel_depth_m", "profundidad_relativa_cauce_m.tif")
        add_download(download_cols_3[2], result, "fill_depth_m", "relleno_acondicionamiento_dem_m.tif")

    with tabs[2]:
        st.write(
            "El calculo usa direccion de flujo D8 sobre un MDE acondicionado hidrologicamente "
            "para enrutar depresiones internas hacia salidas del poligono. "
            "El MDE original se conserva para pendientes y profundidad relativa de cauces. "
            "La lluvia efectiva se estima como lluvia total menos infiltracion por duracion y abstraccion inicial, "
            "multiplicada por el coeficiente de escorrentia. "
            "El cauce derivado de mayor acumulacion se considera ocupado por un caudal existente, "
            "con una franja base configurable desde el centro de la linea de cauce. "
            "Ademas detecta cauces grandes o incisos comparando la elevacion del cauce contra la planicie vecina; "
            "si la diferencia supera el umbral configurado, esa depresion se incluye como cauce ocupado. "
            "Cuando la lluvia efectiva supera la capacidad adicional definida, el modelo expande el desborde lateral "
            "como una franja continua desde el cauce ocupado. Los bajos/depresiones de llanura se calculan aparte usando acumulacion fuera del cauce. "
            "No sustituye un modelo hidraulico 1D/2D calibrado."
        )


def result_interactive_map(result) -> None:
    control_cols = st.columns([0.32, 0.68])
    base_map = control_cols[0].radio("Mapa base", ["OSM", "Satelite"], horizontal=True, key="result_base_map")
    layers = control_cols[1].multiselect(
        "Capas",
        ["Cauce ocupado", "Desborde de rios", "Bajos/depresiones", "Concentracion de flujo"],
        default=["Cauce ocupado", "Desborde de rios", "Bajos/depresiones", "Concentracion de flujo"],
        key=f"result_overlay_layers_{RUNOFF_MODEL_VERSION}",
    )

    m = build_result_map(result, base_map, layers)
    st_folium(
        m,
        height=650,
        use_container_width=True,
        returned_objects=[],
        key=f"result_map_{result.summary['rows']}_{result.summary['cols']}_{base_map}_{'_'.join(layers)}",
    )
    st.caption("Las zonas modeladas se muestran como capas semitransparentes sobre el mapa base y el poligono de analisis.")


def build_result_map(result, base_map: str, layers: list[str]) -> folium.Map:
    bounds = result_overlay_bounds(result)
    south, west = bounds[0]
    north, east = bounds[1]
    center = [(south + north) / 2.0, (west + east) / 2.0]

    m = folium.Map(location=center, zoom_start=14, tiles=None, control_scale=True)
    add_base_layers(m, default_base=base_map)

    if "Concentracion de flujo" in layers:
        folium.raster_layers.ImageOverlay(
            image=overlay_data_url(result, "concentration"),
            bounds=bounds,
            origin="upper",
            name="Concentracion de flujo",
            opacity=1.0,
            pixelated=True,
            show=True,
        ).add_to(m)

    if "Cauce ocupado" in layers:
        folium.raster_layers.ImageOverlay(
            image=overlay_data_url(result, "channel"),
            bounds=bounds,
            origin="upper",
            name="Cauce ocupado",
            opacity=1.0,
            pixelated=True,
            show=True,
        ).add_to(m)

    if "Desborde de rios" in layers:
        folium.raster_layers.ImageOverlay(
            image=overlay_data_url(result, "overbank"),
            bounds=bounds,
            origin="upper",
            name="Desborde de rios",
            opacity=1.0,
            pixelated=True,
            show=True,
        ).add_to(m)

    if "Bajos/depresiones" in layers:
        folium.raster_layers.ImageOverlay(
            image=overlay_data_url(result, "depressions"),
            bounds=bounds,
            origin="upper",
            name="Bajos/depresiones",
            opacity=1.0,
            pixelated=True,
            show=True,
        ).add_to(m)

    folium.GeoJson(
        data=json.loads(geojson_geometry_bytes(result).decode("utf-8")),
        name="Poligono de analisis",
        style_function=lambda _feature: {
            "color": "#101820",
            "weight": 3,
            "fillColor": "#ffffff",
            "fillOpacity": 0.04,
        },
    ).add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    m.fit_bounds(bounds)
    return m


def result_overlay_bounds(result) -> list[list[float]]:
    rows, cols = result.dem.shape
    left = result.transform.c
    top = result.transform.f
    right = left + cols * result.transform.a
    bottom = top + rows * result.transform.e
    west, south, east, north = transform_bounds_dem_to_wgs84((left, bottom, right, top))
    return [[south, west], [north, east]]


def overlay_data_url(result, layer: str) -> str:
    rgba = np.zeros((*result.valid_mask.shape, 4), dtype=np.uint8)

    if layer == "flood":
        overbank = result.overbank_mask & result.valid_mask
        relative = result.relative_flood_mask & result.valid_mask & ~result.channel_base_mask & ~overbank
        intensity = np.nan_to_num(result.flood_index / 100.0, nan=0.0)

        rgba[..., 0] = np.where(relative, 255, rgba[..., 0])
        rgba[..., 1] = np.where(relative, 195, rgba[..., 1])
        rgba[..., 2] = np.where(relative, 20, rgba[..., 2])
        rgba[..., 3] = np.where(relative, 95, rgba[..., 3])

        rgba[..., 0] = np.where(overbank, 255, rgba[..., 0])
        rgba[..., 1] = np.where(overbank, (135 - 65 * intensity).clip(45, 135), rgba[..., 1]).astype(np.uint8)
        rgba[..., 2] = np.where(overbank, 20, rgba[..., 2])
        rgba[..., 3] = np.where(overbank, (125 + 70 * intensity).clip(125, 195), rgba[..., 3]).astype(np.uint8)
    elif layer == "overbank":
        overbank = result.overbank_mask & result.valid_mask
        intensity = np.nan_to_num(result.flood_index / 100.0, nan=0.0)
        rgba[..., 0] = np.where(overbank, 255, rgba[..., 0])
        rgba[..., 1] = np.where(overbank, (135 - 65 * intensity).clip(45, 135), rgba[..., 1]).astype(np.uint8)
        rgba[..., 2] = np.where(overbank, 20, rgba[..., 2])
        rgba[..., 3] = np.where(overbank, (125 + 70 * intensity).clip(125, 195), rgba[..., 3]).astype(np.uint8)
    elif layer == "depressions":
        mask = result.relative_flood_mask & result.valid_mask & ~result.channel_base_mask & ~result.overbank_mask
        rgba[..., 0] = np.where(mask, 255, rgba[..., 0])
        rgba[..., 1] = np.where(mask, 195, rgba[..., 1])
        rgba[..., 2] = np.where(mask, 20, rgba[..., 2])
        rgba[..., 3] = np.where(mask, 135, rgba[..., 3])
    elif layer == "channel":
        fixed = result.channel_base_mask & result.valid_mask & ~result.large_channel_mask
        large = result.large_channel_mask & result.valid_mask
        rgba[..., 0] = np.where(fixed, 0, rgba[..., 0])
        rgba[..., 1] = np.where(fixed, 82, rgba[..., 1])
        rgba[..., 2] = np.where(fixed, 255, rgba[..., 2])
        rgba[..., 3] = np.where(fixed, 150, rgba[..., 3])
        rgba[..., 0] = np.where(large, 0, rgba[..., 0])
        rgba[..., 1] = np.where(large, 28, rgba[..., 1])
        rgba[..., 2] = np.where(large, 190, rgba[..., 2])
        rgba[..., 3] = np.where(large, 205, rgba[..., 3])
    elif layer == "concentration":
        mask = result.concentration_mask & result.valid_mask
        rgba[..., 0] = np.where(mask, 0, 0)
        rgba[..., 1] = np.where(mask, 190, 0)
        rgba[..., 2] = np.where(mask, 255, 0)
        rgba[..., 3] = np.where(mask, 105, 0)
    else:
        raise ValueError(f"Capa no soportada: {layer}")

    image = Image.fromarray(rgba)
    output = BytesIO()
    image.save(output, format="PNG")
    encoded = base64.b64encode(output.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"

def add_download(column, result, layer: str, file_name: str) -> None:
    column.download_button(
        Path(file_name).stem.replace("_", " ").title(),
        data=geotiff_bytes(result, layer),
        file_name=file_name,
        mime="image/tiff",
        use_container_width=True,
    )


def summary_table(summary: dict) -> list[dict[str, str]]:
    labels = {
        "model_version": "Version del modelo",
        "input_crs_detected": "CRS detectado",
        "cells": "Celdas procesadas",
        "rows": "Filas",
        "cols": "Columnas",
        "pixel_size_m": "Tamano de pixel (m)",
        "area_ha": "Area (ha)",
        "rainfall_mm": "Lluvia total (mm)",
        "duration_min": "Duracion (min)",
        "rainfall_intensity_mm_h": "Intensidad (mm/h)",
        "infiltration_loss_mm": "Perdida por infiltracion (mm)",
        "initial_loss_mm": "Abstraccion inicial (mm)",
        "runoff_coefficient": "Coeficiente de escorrentia",
        "condition_dem": "Acondicionamiento hidrologico DEM",
        "fill_epsilon_m": "Micro pendiente relleno (m/celda)",
        "depression_fill_cells": "Celdas rellenadas para enrutamiento",
        "max_depression_fill_m": "Relleno maximo DEM (m)",
        "p95_depression_fill_m": "P95 relleno DEM (m)",
        "effective_rainfall_mm": "Lluvia efectiva (mm)",
        "direct_runoff_volume_m3": "Volumen directo (m3)",
        "max_accumulated_area_ha": "Area acumulada maxima (ha)",
        "p95_accumulated_area_ha": "P95 area acumulada (ha)",
        "max_accumulated_runoff_m3": "Escorrentia acumulada maxima (m3)",
        "concentration_area_ha": "Area de concentracion (ha)",
        "relative_flood_area_ha": "Bajos/acumulacion en llanura (ha)",
        "flood_susceptible_area_ha": "Area inundable total (ha)",
        "stream_center_area_ha": "Linea de cauce derivada (ha)",
        "channel_occupied_area_ha": "Cauce ocupado por caudal existente (ha)",
        "large_channel_area_ha": "Cauce grande/inciso detectado (ha)",
        "overbank_area_ha": "Desborde lateral por lluvia (ha)",
        "stream_percentile": "Umbral red cauces (%)",
        "channel_base_half_width_m": "Semi-ancho cauce configurado (m)",
        "effective_channel_half_width_m": "Semi-ancho cauce efectivo raster (m)",
        "channel_spill_threshold_mm": "Lluvia efectiva antes de desborde (mm)",
        "channel_overflow_excess_mm": "Exceso efectivo para desborde (mm)",
        "channel_overflow_width_m": "Ancho calculado de desborde (m)",
        "channel_max_overflow_width_m": "Ancho maximo desborde lateral (m)",
        "large_channel_depth_threshold_m": "Diferencia cauce-planicie (m)",
        "large_channel_search_radius_m": "Radio planicie vecina (m)",
        "large_channel_bank_percentile": "Percentil planicie vecina (%)",
        "max_large_channel_depth_m": "Profundidad relativa maxima cauce (m)",
        "p95_large_channel_depth_m": "P95 profundidad relativa cauce (m)",
        "sink_cells": "Celdas sin salida",
        "outlet_cells": "Celdas de salida/borde",
        "max_flood_index": "Indice maximo",
    }
    rows = []
    for key, label in labels.items():
        value = summary.get(key)
        if isinstance(value, float):
            rendered = f"{value:,.3f}" if abs(value) < 100 else f"{value:,.1f}"
        else:
            rendered = str(value)
        rows.append({"Metrica": label, "Valor": rendered})
    return rows


if __name__ == "__main__":
    main()
