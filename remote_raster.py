from __future__ import annotations

import html
import importlib.util
import math
import os
import re
import struct
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator
from urllib.parse import parse_qs, quote, urlencode, urljoin, urlparse


def _configure_bundled_proj_data() -> None:
    spec = importlib.util.find_spec("rasterio")
    if spec is None or spec.origin is None:
        return
    proj_dir = Path(spec.origin).resolve().parent / "proj_data"
    if (proj_dir / "proj.db").exists():
        os.environ["PROJ_DATA"] = str(proj_dir)
        os.environ["PROJ_LIB"] = str(proj_dir)


_configure_bundled_proj_data()

import rasterio
from affine import Affine
import requests


DEFAULT_DRIVE_DEM_URL = "https://drive.google.com/file/d/1LfaVUfK6g1O3vFprYtRpzs_rqIGNvsu6/view?usp=sharing"
REMOTE_SUBSET_MAX_TILES = 6000


class RemoteRasterError(ValueError):
    pass


@dataclass(frozen=True)
class RangeProbe:
    url: str
    supports_range: bool
    status_code: int
    content_length: int | None
    content_range: str | None
    content_type: str | None


@dataclass(frozen=True)
class RemoteTiffInfo:
    url: str
    width: int
    height: int
    transform: Affine
    bounds: tuple[float, float, float, float]
    resolution: tuple[float, float]
    nodata: float | None
    tile_width: int
    tile_height: int


@dataclass(frozen=True)
class _BigTiffDirectory:
    url: str
    tags: dict[int, tuple[int, int, int]]


_TIFF_TYPE_SIZE = {
    1: 1,   # BYTE
    2: 1,   # ASCII
    3: 2,   # SHORT
    4: 4,   # LONG
    5: 8,   # RATIONAL
    11: 4,  # FLOAT
    12: 8,  # DOUBLE
    16: 8,  # LONG8
    17: 8,  # SLONG8
    18: 8,  # IFD8
}


_CLASSIC_TIFF_TYPE_SIZE = {
    1: 1,
    2: 1,
    3: 2,
    4: 4,
    5: 8,
    11: 4,
    12: 8,
}


def is_http_url(value: str | Path) -> bool:
    parsed = urlparse(str(value))
    return parsed.scheme in {"http", "https"}


def google_drive_file_id(url: str) -> str | None:
    patterns = (
        r"/file/d/([^/]+)",
        r"[?&]id=([^&]+)",
        r"/open\?id=([^&]+)",
    )
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None


def is_google_drive_url(url: str) -> bool:
    host = urlparse(url).netloc.lower()
    return "drive.google.com" in host or "drive.usercontent.google.com" in host


def resolve_raster_url(source: str, timeout: int = 30) -> str:
    if not is_http_url(source):
        return source
    if is_google_drive_url(source):
        return resolve_google_drive_download_url(source, timeout=timeout)
    return source


def resolve_google_drive_download_url(url: str, timeout: int = 30) -> str:
    file_id = google_drive_file_id(url)
    if not file_id:
        raise RemoteRasterError("No pude extraer el id del enlace de Google Drive.")

    direct_url = f"https://drive.google.com/uc?export=download&id={file_id}"
    session = requests.Session()
    response = session.get(direct_url, stream=True, allow_redirects=True, timeout=timeout)
    content_type = response.headers.get("Content-Type", "")
    disposition = response.headers.get("Content-Disposition", "")

    if "text/html" not in content_type.lower() and disposition:
        resolved = response.url
        response.close()
        return resolved

    text = response.text
    base_url = response.url
    response.close()

    action = _download_form_action(text)
    params = _hidden_form_inputs(text)
    if not action or not params:
        raise RemoteRasterError(
            "Google Drive no entrego un enlace descargable confirmado. "
            "Verifica que el archivo sea publico con permiso de lectura."
        )

    params.setdefault("id", file_id)
    params.setdefault("export", "download")
    params.setdefault("confirm", "t")
    return urljoin(base_url, action) + "?" + urlencode(params)


def _download_form_action(text: str) -> str | None:
    form_match = re.search(r"<form[^>]+id=[\"']download-form[\"'][^>]*>", text, re.IGNORECASE)
    if not form_match:
        return None
    attrs = _tag_attrs(form_match.group(0))
    action = attrs.get("action")
    return html.unescape(action) if action else None


def _hidden_form_inputs(text: str) -> dict[str, str]:
    params: dict[str, str] = {}
    for tag in re.findall(r"<input[^>]+>", text, flags=re.IGNORECASE):
        attrs = _tag_attrs(tag)
        if attrs.get("type", "").lower() == "hidden" and "name" in attrs:
            params[html.unescape(attrs["name"])] = html.unescape(attrs.get("value", ""))
    return params


def _tag_attrs(tag: str) -> dict[str, str]:
    return {
        name.lower(): value
        for name, value in re.findall(r"([a-zA-Z_:][-a-zA-Z0-9_:.]*)=[\"']([^\"']*)[\"']", tag)
    }


def probe_http_range(url: str, timeout: int = 30) -> RangeProbe:
    response = requests.get(
        url,
        headers={"Range": "bytes=0-0", "Accept-Encoding": "identity"},
        stream=True,
        allow_redirects=True,
        timeout=timeout,
    )
    try:
        first = next(response.iter_content(chunk_size=1), b"")
        status_code = response.status_code
        content_range = response.headers.get("Content-Range")
        content_length = response.headers.get("Content-Length")
        content_type = response.headers.get("Content-Type")
        supports_range = status_code == 206 and bool(content_range) and len(first) <= 1
        return RangeProbe(
            url=response.url,
            supports_range=supports_range,
            status_code=status_code,
            content_length=int(content_length) if content_length and content_length.isdigit() else None,
            content_range=content_range,
            content_type=content_type,
        )
    finally:
        response.close()


def rasterio_source(source: str | Path) -> str:
    source_str = str(source)
    if is_http_url(source_str):
        resolved = resolve_raster_url(source_str)
        return "/vsicurl?list_dir=no&use_head=no&url=" + quote(resolved, safe="")

    path = Path(source_str)
    if not path.exists():
        raise RemoteRasterError(f"No existe el raster: {path}")
    return str(path)


@contextmanager
def open_raster_source(
    source: str | Path,
    subset_bounds: tuple[float, float, float, float] | None = None,
) -> Iterator[rasterio.io.DatasetReader]:
    source_str = str(source)
    src_path = rasterio_source(source)
    options = _vsicurl_options() if src_path.startswith("/vsicurl") else {}
    open_error: Exception | None = None

    try:
        with rasterio.Env(**options):
            with rasterio.open(src_path) as src:
                yield src
                return
    except rasterio.errors.RasterioIOError as exc:
        open_error = exc

    if is_http_url(source_str) and subset_bounds is not None:
        with materialized_remote_bigtiff_subset(source_str, subset_bounds) as subset_path:
            with rasterio.open(subset_path) as src:
                yield src
                return

    if is_http_url(source_str):
        raise RemoteRasterError(
            "No pude abrir el MDE remoto como raster HTTP. "
            "Si es un GeoTIFF grande en Google Drive, debe ser COG o se debe leer desde un recorte por poligono."
        ) from open_error
    raise open_error


def _vsicurl_options() -> dict[str, str]:
    return {
        "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
        "CPL_VSIL_CURL_USE_HEAD": "NO",
        "GDAL_HTTP_MULTIRANGE": "NO",
        "GDAL_HTTP_HEADERS": "Accept-Encoding: identity",
        "VSI_CACHE": "TRUE",
        "VSI_CACHE_SIZE": "50000000",
    }


def read_remote_tiff_info(source: str | Path) -> RemoteTiffInfo:
    source_str = str(source)
    if not is_http_url(source_str):
        raise RemoteRasterError("La fuente no es una URL remota.")
    resolved = resolve_raster_url(source_str)
    with requests.Session() as session:
        directory = _read_bigtiff_directory(resolved, session)
        return _remote_tiff_info_from_directory(directory, session)


@contextmanager
def materialized_remote_bigtiff_subset(
    source: str | Path,
    bounds: tuple[float, float, float, float],
) -> Iterator[Path]:
    source_str = str(source)
    if not is_http_url(source_str):
        raise RemoteRasterError("La materializacion por rangos solo aplica a URLs remotas.")

    resolved = resolve_raster_url(source_str)
    tmp = tempfile.NamedTemporaryFile(prefix="runoff_remote_subset_", suffix=".tif", delete=False)
    tmp_path = Path(tmp.name)
    tmp.close()

    try:
        with requests.Session() as session:
            directory = _read_bigtiff_directory(resolved, session)
            _write_bigtiff_subset(directory, session, bounds, tmp_path)
        yield tmp_path
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass


def _fetch_range(url: str, start: int, size: int, session: requests.Session, timeout: int = 60) -> bytes:
    if size <= 0:
        return b""
    end = start + size - 1
    response = session.get(
        url,
        headers={"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity"},
        stream=False,
        allow_redirects=True,
        timeout=timeout,
    )
    try:
        if response.status_code != 206:
            raise RemoteRasterError(
                f"El servidor remoto no entrego el rango solicitado ({response.status_code})."
            )
        data = response.content
        if len(data) != size:
            raise RemoteRasterError(
                f"El rango remoto regreso {len(data)} bytes, se esperaban {size}."
            )
        return data
    finally:
        response.close()


def _read_bigtiff_directory(url: str, session: requests.Session) -> _BigTiffDirectory:
    header = _fetch_range(url, 0, 16, session)
    if header[:2] != b"II" or header[2:4] != b"+\x00":
        raise RemoteRasterError(
            "El MDE remoto no es BigTIFF little-endian. "
            "Usa un COG o un GeoTIFF local para esta fuente."
        )
    offset_size = struct.unpack_from("<H", header, 4)[0]
    if offset_size != 8:
        raise RemoteRasterError("BigTIFF remoto con tamano de offset no soportado.")
    first_ifd = struct.unpack_from("<Q", header, 8)[0]
    count_data = _fetch_range(url, first_ifd, 8, session)
    tag_count = struct.unpack_from("<Q", count_data, 0)[0]
    if tag_count <= 0 or tag_count > 1000:
        raise RemoteRasterError("Directorio BigTIFF remoto no valido.")

    directory_data = _fetch_range(url, first_ifd, 8 + tag_count * 20 + 8, session)
    tags: dict[int, tuple[int, int, int]] = {}
    for idx in range(tag_count):
        pos = 8 + idx * 20
        tag, tag_type = struct.unpack_from("<HH", directory_data, pos)
        count = struct.unpack_from("<Q", directory_data, pos + 4)[0]
        value_or_offset = struct.unpack_from("<Q", directory_data, pos + 12)[0]
        tags[int(tag)] = (int(tag_type), int(count), int(value_or_offset))
    return _BigTiffDirectory(url=url, tags=tags)


def _remote_tiff_info_from_directory(directory: _BigTiffDirectory, session: requests.Session) -> RemoteTiffInfo:
    width = int(_tag_values(directory, session, 256)[0])
    height = int(_tag_values(directory, session, 257)[0])
    tile_width = int(_tag_values(directory, session, 322)[0])
    tile_height = int(_tag_values(directory, session, 323)[0])
    scale = _tag_values(directory, session, 33550)
    tiepoint = _tag_values(directory, session, 33922)
    if len(scale) < 2 or len(tiepoint) < 6:
        raise RemoteRasterError("El BigTIFF remoto no tiene georreferenciacion suficiente.")
    xres = abs(float(scale[0]))
    yres = abs(float(scale[1]))
    origin_x = float(tiepoint[3]) - float(tiepoint[0]) * xres
    origin_y = float(tiepoint[4]) + float(tiepoint[1]) * yres
    transform = Affine.translation(origin_x, origin_y) * Affine.scale(xres, -yres)
    bounds = (origin_x, origin_y - height * yres, origin_x + width * xres, origin_y)
    nodata = _tag_ascii(directory, session, 42113)
    nodata_value = None
    if nodata:
        try:
            nodata_value = float(nodata.strip("\x00"))
        except ValueError:
            nodata_value = None
    return RemoteTiffInfo(
        url=directory.url,
        width=width,
        height=height,
        transform=transform,
        bounds=bounds,
        resolution=(xres, yres),
        nodata=nodata_value,
        tile_width=tile_width,
        tile_height=tile_height,
    )


def _tag_bytes(directory: _BigTiffDirectory, session: requests.Session, tag: int) -> bytes:
    try:
        tag_type, count, value_or_offset = directory.tags[tag]
    except KeyError as exc:
        raise RemoteRasterError(f"El BigTIFF remoto no contiene el tag requerido {tag}.") from exc
    try:
        type_size = _TIFF_TYPE_SIZE[tag_type]
    except KeyError as exc:
        raise RemoteRasterError(f"Tipo TIFF no soportado en tag {tag}: {tag_type}.") from exc
    total_size = type_size * count
    if total_size <= 8:
        return struct.pack("<Q", value_or_offset)[:total_size]
    return _fetch_range(directory.url, value_or_offset, total_size, session)


def _tag_values(directory: _BigTiffDirectory, session: requests.Session, tag: int) -> tuple[int | float, ...]:
    tag_type, count, _value_or_offset = directory.tags[tag]
    data = _tag_bytes(directory, session, tag)
    if tag_type == 3:
        return struct.unpack("<" + "H" * count, data)
    if tag_type == 4:
        return struct.unpack("<" + "I" * count, data)
    if tag_type == 12:
        return struct.unpack("<" + "d" * count, data)
    if tag_type == 16:
        return struct.unpack("<" + "Q" * count, data)
    raise RemoteRasterError(f"Tipo TIFF no soportado para valores numericos: {tag_type}")


def _tag_ascii(directory: _BigTiffDirectory, session: requests.Session, tag: int) -> str:
    tag_type, _count, _value_or_offset = directory.tags[tag]
    if tag_type != 2:
        raise RemoteRasterError(f"El tag {tag} no es ASCII.")
    return _tag_bytes(directory, session, tag).decode("ascii", errors="replace").rstrip("\x00")


def _write_bigtiff_subset(
    directory: _BigTiffDirectory,
    session: requests.Session,
    bounds: tuple[float, float, float, float],
    output_path: Path,
) -> None:
    info = _remote_tiff_info_from_directory(directory, session)
    minx, miny, maxx, maxy = bounds
    origin_x = info.transform.c
    origin_y = info.transform.f
    xres, yres = info.resolution

    col_start = max(0, math.floor((minx - origin_x) / xres) - 1)
    col_stop = min(info.width, math.ceil((maxx - origin_x) / xres) + 1)
    row_start = max(0, math.floor((origin_y - maxy) / yres) - 1)
    row_stop = min(info.height, math.ceil((origin_y - miny) / yres) + 1)
    if col_stop <= col_start or row_stop <= row_start:
        raise RemoteRasterError("El poligono no intersecta el MDE remoto.")

    tile_col_start = col_start // info.tile_width
    tile_col_stop = math.ceil(col_stop / info.tile_width)
    tile_row_start = row_start // info.tile_height
    tile_row_stop = math.ceil(row_stop / info.tile_height)
    source_tile_cols = math.ceil(info.width / info.tile_width)
    tile_count = (tile_col_stop - tile_col_start) * (tile_row_stop - tile_row_start)
    if tile_count <= 0:
        raise RemoteRasterError("No se identificaron teselas del MDE remoto para el poligono.")
    if tile_count > REMOTE_SUBSET_MAX_TILES:
        raise RemoteRasterError(
            "El poligono cubre demasiadas teselas del MDE remoto "
            f"({tile_count:,}). Reduce el area o usa un COG/local."
        )

    subset_col = tile_col_start * info.tile_width
    subset_row = tile_row_start * info.tile_height
    subset_width = min(info.width - subset_col, (tile_col_stop - tile_col_start) * info.tile_width)
    subset_height = min(info.height - subset_row, (tile_row_stop - tile_row_start) * info.tile_height)

    source_offsets = _tag_values(directory, session, 324)
    source_counts = _tag_values(directory, session, 325)
    tile_blobs: list[bytes] = []
    for tile_row in range(tile_row_start, tile_row_stop):
        for tile_col in range(tile_col_start, tile_col_stop):
            source_index = tile_row * source_tile_cols + tile_col
            tile_blobs.append(
                _fetch_range(directory.url, int(source_offsets[source_index]), int(source_counts[source_index]), session)
            )

    subset_origin_x = origin_x + subset_col * xres
    subset_origin_y = origin_y - subset_row * yres
    _write_classic_tiled_geotiff(directory, session, output_path, info, subset_width, subset_height, subset_origin_x, subset_origin_y, tile_blobs)


def _write_classic_tiled_geotiff(
    directory: _BigTiffDirectory,
    session: requests.Session,
    output_path: Path,
    info: RemoteTiffInfo,
    width: int,
    height: int,
    origin_x: float,
    origin_y: float,
    tile_blobs: list[bytes],
) -> None:
    entries: list[tuple[int, int, int, bytes]] = []

    def add(tag: int, tag_type: int, count: int, data: bytes) -> None:
        entries.append((tag, tag_type, count, data))

    add(256, 4, 1, _pack_classic_values(4, [width]))
    add(257, 4, 1, _pack_classic_values(4, [height]))
    add(258, 3, 1, _pack_classic_values(3, [32]))
    add(259, 3, 1, _pack_classic_values(3, [5]))
    add(262, 3, 1, _pack_classic_values(3, [1]))
    add(277, 3, 1, _pack_classic_values(3, [1]))
    add(284, 3, 1, _pack_classic_values(3, [1]))
    add(317, 3, 1, _pack_classic_values(3, [1]))
    add(322, 4, 1, _pack_classic_values(4, [info.tile_width]))
    add(323, 4, 1, _pack_classic_values(4, [info.tile_height]))
    add(324, 4, len(tile_blobs), b"\x00" * (4 * len(tile_blobs)))
    add(325, 4, len(tile_blobs), _pack_classic_values(4, [len(blob) for blob in tile_blobs]))
    add(339, 3, 1, _pack_classic_values(3, [3]))
    add(33550, 12, 3, struct.pack("<ddd", info.resolution[0], info.resolution[1], 0.0))
    add(33922, 12, 6, struct.pack("<dddddd", 0.0, 0.0, 0.0, origin_x, origin_y, 0.0))

    for tag in (34735, 34736, 34737, 42112, 42113):
        if tag in directory.tags:
            tag_type, _count, _value = directory.tags[tag]
            data = _tag_bytes(directory, session, tag)
            count = len(data) // _CLASSIC_TIFF_TYPE_SIZE.get(tag_type, 1)
            add(tag, tag_type, count, data)

    entries.sort(key=lambda item: item[0])
    entry_count = len(entries)
    if entry_count > 65535:
        raise RemoteRasterError("Demasiadas entradas TIFF para el subconjunto remoto.")

    ifd_offset = 8
    ifd_size = 2 + entry_count * 12 + 4
    data_offset = ifd_offset + ifd_size
    extra_data = bytearray()
    packed_entries: list[list[int | bytes]] = []

    for tag, tag_type, count, data in entries:
        type_size = _CLASSIC_TIFF_TYPE_SIZE.get(tag_type)
        if type_size is None:
            raise RemoteRasterError(f"Tipo TIFF no soportado para salida parcial: {tag_type}")
        expected_size = type_size * count
        if expected_size != len(data):
            raise RemoteRasterError(f"Tamano de datos inconsistente para tag {tag}.")
        if len(data) <= 4:
            value = data + b"\x00" * (4 - len(data))
        else:
            if data_offset % 2:
                extra_data.extend(b"\x00")
                data_offset += 1
            value = struct.pack("<I", data_offset)
            extra_data.extend(data)
            data_offset += len(data)
        packed_entries.append([tag, tag_type, count, value])

    if data_offset % 2:
        extra_data.extend(b"\x00")
        data_offset += 1

    tile_offsets: list[int] = []
    tile_data = bytearray()
    current_offset = data_offset
    for blob in tile_blobs:
        tile_offsets.append(current_offset)
        tile_data.extend(blob)
        current_offset += len(blob)
    if current_offset > 0xFFFFFFFF:
        raise RemoteRasterError("El subconjunto remoto excede el limite TIFF clasico de 4 GB.")

    tile_offsets_data = _pack_classic_values(4, tile_offsets)
    for packed in packed_entries:
        if packed[0] == 324:
            if int(packed[2]) == 1:
                packed[3] = tile_offsets_data + b"\x00" * (4 - len(tile_offsets_data))
            else:
                offsets_array_offset = struct.unpack("<I", packed[3])[0]
                start = offsets_array_offset - (ifd_offset + ifd_size)
                extra_data[start : start + len(tile_offsets_data)] = tile_offsets_data
            break

    with output_path.open("wb") as dst:
        dst.write(b"II*\x00")
        dst.write(struct.pack("<I", ifd_offset))
        dst.write(struct.pack("<H", entry_count))
        for tag, tag_type, count, value in packed_entries:
            dst.write(struct.pack("<HHI", int(tag), int(tag_type), int(count)))
            dst.write(value)
        dst.write(struct.pack("<I", 0))
        dst.write(extra_data)
        dst.write(tile_data)


def _pack_classic_values(tag_type: int, values: list[int | float] | tuple[int | float, ...]) -> bytes:
    if tag_type == 3:
        return struct.pack("<" + "H" * len(values), *[int(value) for value in values])
    if tag_type == 4:
        return struct.pack("<" + "I" * len(values), *[int(value) for value in values])
    if tag_type == 12:
        return struct.pack("<" + "d" * len(values), *[float(value) for value in values])
    raise RemoteRasterError(f"Tipo TIFF no soportado para empaquetar: {tag_type}")