from __future__ import annotations

import html
import re
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator
from urllib.parse import parse_qs, urlencode, urljoin, urlparse

import rasterio
import requests


DEFAULT_DRIVE_DEM_URL = "https://drive.google.com/file/d/1LfaVUfK6g1O3vFprYtRpzs_rqIGNvsu6/view?usp=sharing"


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
        headers={"Range": "bytes=0-0"},
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
        return "/vsicurl/" + resolve_raster_url(source_str)

    path = Path(source_str)
    if not path.exists():
        raise RemoteRasterError(f"No existe el raster: {path}")
    return str(path)


@contextmanager
def open_raster_source(source: str | Path) -> Iterator[rasterio.io.DatasetReader]:
    src_path = rasterio_source(source)
    if src_path.startswith("/vsicurl/"):
        options = {
            "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
            "CPL_VSIL_CURL_USE_HEAD": "NO",
            "GDAL_HTTP_MULTIRANGE": "YES",
            "VSI_CACHE": "TRUE",
            "VSI_CACHE_SIZE": "50000000",
        }
    else:
        options = {}

    with rasterio.Env(**options):
        with rasterio.open(src_path) as src:
            yield src