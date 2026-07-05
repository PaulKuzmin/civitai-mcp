"""Civitai MCP server — выбор и скачивание моделей.

Инструменты:
  - search_models         — поиск моделей (query/type/baseModel/sort)
  - get_model             — карточка модели: версии и файлы
  - get_model_version     — детали конкретной версии (+ downloadUrl)
  - get_model_images      — галерея примеров генерации
  - get_download_url      — прямая ссылка на файл (для больших моделей)
  - download_model        — скачать файл версии в указанную папку
  - read_image_params     — параметры генерации из локального файла (офлайн)
  - get_image_meta        — параметры генерации картинки с Civitai по id
  - get_buzz_balance      — баланс Buzz (⚠️ неофициальный эндпоинт)
  - estimate_generation   — оценка стоимости генерации (whatif)
  - generate_image        — генерация (Orchestration API, тратит Buzz, confirm=true)
  - get_workflow          — статус/результат генерации по workflowId

Ключ берётся из env CIVITAI_API_KEY, с fallback на apikey.txt рядом со скриптом.
"""

from __future__ import annotations

import json
import os
import re
import struct
import zlib
from pathlib import Path
from typing import Any, Optional

import httpx
from mcp.server.fastmcp import FastMCP

API_BASE = "https://civitai.com/api/v1"
DOWNLOAD_BASE = "https://civitai.com/api/download/models"
TRPC_BASE = "https://civitai.com/api/trpc"  # внутренний, неофициальный
ORCHESTRATION = "https://orchestration.civitai.com/v2/consumer/workflows"

mcp = FastMCP("civitai")


def _api_key() -> Optional[str]:
    key = os.environ.get("CIVITAI_API_KEY")
    if key:
        return key.strip()
    # fallback: apikey.txt рядом со скриптом
    f = Path(__file__).with_name("apikey.txt")
    if f.exists():
        return f.read_text(encoding="utf-8").strip()
    return None


def _headers() -> dict[str, str]:
    h = {"User-Agent": "civitai-mcp/1.0"}
    key = _api_key()
    if key:
        h["Authorization"] = f"Bearer {key}"
    return h


def _client(timeout: float = 60.0) -> httpx.Client:
    return httpx.Client(headers=_headers(), timeout=timeout, follow_redirects=True)


# ---- helpers для компактного вывода -------------------------------------

def _thumb(url: Optional[str], width: int = 450) -> Optional[str]:
    """Превью-версия картинки через CDN-трансформ (original=true -> width=N)."""
    if not url:
        return url
    return re.sub(r"/original=true/", f"/width={width}/", url)


def _slim_image(i: dict[str, Any]) -> dict[str, Any]:
    url = i.get("url")
    meta = i.get("meta") or {}
    return {
        "url": url,
        "thumb": _thumb(url),
        "nsfwLevel": i.get("nsfwLevel"),
        "width": i.get("width"),
        "height": i.get("height"),
        "type": i.get("type"),
        "prompt": meta.get("prompt"),
    }


def _slim_file(f: dict[str, Any]) -> dict[str, Any]:
    meta = f.get("metadata") or {}
    return {
        "id": f.get("id"),
        "name": f.get("name"),
        "type": f.get("type"),
        "sizeKB": f.get("sizeKB"),
        "format": meta.get("format"),
        "fp": meta.get("fp"),
        "size": meta.get("size"),
        "primary": f.get("primary", False),
        "sha256": (f.get("hashes") or {}).get("SHA256"),
        "downloadUrl": f.get("downloadUrl"),
    }


def _slim_version(v: dict[str, Any]) -> dict[str, Any]:
    images = v.get("images") or []
    return {
        "id": v.get("id"),
        "name": v.get("name"),
        "baseModel": v.get("baseModel"),
        "downloadUrl": v.get("downloadUrl"),
        "files": [_slim_file(f) for f in (v.get("files") or [])],
        "images": [_slim_image(i) for i in images[:6]],
    }


def _first_preview(m: dict[str, Any]) -> Optional[dict[str, Any]]:
    for v in m.get("modelVersions") or []:
        for i in v.get("images") or []:
            img = _slim_image(i)
            return {"url": img["url"], "thumb": img["thumb"],
                    "nsfwLevel": img["nsfwLevel"], "type": img["type"]}
    return None


def _slim_model(m: dict[str, Any]) -> dict[str, Any]:
    versions = m.get("modelVersions") or []
    return {
        "id": m.get("id"),
        "name": m.get("name"),
        "type": m.get("type"),
        "nsfw": m.get("nsfw"),
        "creator": (m.get("creator") or {}).get("username"),
        "stats": m.get("stats"),
        "preview": _first_preview(m),
        "versions": [
            {"id": v.get("id"), "name": v.get("name"), "baseModel": v.get("baseModel")}
            for v in versions
        ],
    }


# ---- tools ---------------------------------------------------------------

@mcp.tool()
def search_models(
    query: Optional[str] = None,
    types: Optional[str] = None,
    base_models: Optional[str] = None,
    sort: Optional[str] = None,
    limit: int = 10,
    page: int = 1,
    tag: Optional[str] = None,
    username: Optional[str] = None,
    nsfw: Optional[bool] = None,
) -> dict[str, Any]:
    """Поиск моделей на Civitai.

    Args:
        query: полнотекстовый поиск по названию.
        types: тип модели — Checkpoint, LORA, VAE, TextualInversion, Hypernetwork,
            LoCon, Controlnet и т.п. (можно через запятую).
        base_models: базовая модель — 'SDXL 1.0', 'Pony', 'Flux.1 D', 'Illustrious',
            'SD 1.5' (можно через запятую).
        sort: 'Highest Rated' | 'Most Downloaded' | 'Newest'.
        limit: 1..100 результатов на страницу.
        page: номер страницы (несовместимо с курсорным поиском по query).
        tag: фильтр по тегу.
        username: модели конкретного автора.
        nsfw: включать ли зрелый контент (True/False; по умолчанию — как у API).

    Returns:
        {items: [...краткие модели...], metadata: {...пагинация...}}.
    """
    params: dict[str, Any] = {"limit": max(1, min(limit, 100)), "page": page}
    if query:
        params["query"] = query
        params.pop("page", None)  # query несовместим с page
    if types:
        params["types"] = types
    if base_models:
        params["baseModels"] = base_models
    if sort:
        params["sort"] = sort
    if tag:
        params["tag"] = tag
    if username:
        params["username"] = username
    if nsfw is not None:
        params["nsfw"] = str(nsfw).lower()

    with _client() as c:
        r = c.get(f"{API_BASE}/models", params=params)
        r.raise_for_status()
        data = r.json()

    return {
        "items": [_slim_model(m) for m in data.get("items", [])],
        "metadata": data.get("metadata", {}),
    }


@mcp.tool()
def get_model(model_id: int) -> dict[str, Any]:
    """Карточка модели: все версии и файлы (с downloadUrl и SHA256).

    Args:
        model_id: числовой id модели (из search_models).
    """
    with _client() as c:
        r = c.get(f"{API_BASE}/models/{model_id}")
    if r.status_code == 404:
        return {"status": "not_found", "note": f"модель {model_id} не найдена."}
    if r.status_code >= 400:
        return {"status": "error", "error": f"{r.status_code}: {r.text[:200]}"}
    m = r.json()

    return {
        "id": m.get("id"),
        "name": m.get("name"),
        "type": m.get("type"),
        "nsfw": m.get("nsfw"),
        "description": m.get("description"),
        "creator": (m.get("creator") or {}).get("username"),
        "tags": m.get("tags"),
        "stats": m.get("stats"),
        "versions": [_slim_version(v) for v in (m.get("modelVersions") or [])],
    }


@mcp.tool()
def get_model_version(version_id: int) -> dict[str, Any]:
    """Детали конкретной версии модели: файлы, хэши, downloadUrl.

    Args:
        version_id: числовой id версии (modelVersions[].id).
    """
    with _client() as c:
        r = c.get(f"{API_BASE}/model-versions/{version_id}")
    if r.status_code == 404:
        return {"status": "not_found", "note": f"версия {version_id} не найдена."}
    if r.status_code >= 400:
        return {"status": "error", "error": f"{r.status_code}: {r.text[:200]}"}
    v = r.json()

    return {
        "id": v.get("id"),
        "modelId": v.get("modelId"),
        "name": v.get("name"),
        "baseModel": v.get("baseModel"),
        "air": v.get("air"),  # для generate_image / estimate_generation
        "downloadUrl": v.get("downloadUrl"),
        "files": [_slim_file(f) for f in (v.get("files") or [])],
        "images": [_slim_image(i) for i in (v.get("images") or [])],
    }


@mcp.tool()
def get_model_images(
    model_id: Optional[int] = None,
    version_id: Optional[int] = None,
    limit: int = 20,
    nsfw: Optional[str] = None,
    sort: Optional[str] = None,
) -> dict[str, Any]:
    """Галерея примеров генерации для модели/версии — для визуального выбора.

    Отдаёт полноразмерный url + лёгкий thumb + prompt каждой картинки.

    Args:
        model_id: id модели (взаимоисключимо с version_id, но можно оба).
        version_id: id конкретной версии.
        limit: сколько картинок (1..200).
        nsfw: 'None' | 'Soft' | 'Mature' | 'X' — порог зрелости (или не задавать).
        sort: 'Most Reactions' | 'Most Comments' | 'Newest'.

    Returns:
        {items: [{url, thumb, nsfwLevel, width, height, type, prompt}], metadata}.
    """
    if not model_id and not version_id:
        return {"status": "error", "error": "нужен model_id или version_id."}
    params: dict[str, Any] = {"limit": max(1, min(limit, 200))}
    if model_id:
        params["modelId"] = model_id
    if version_id:
        params["modelVersionId"] = version_id
    if nsfw:
        params["nsfw"] = nsfw
    if sort:
        params["sort"] = sort

    with _client() as c:
        r = c.get(f"{API_BASE}/images", params=params)
        r.raise_for_status()
        data = r.json()

    return {
        "items": [_slim_image(i) for i in data.get("items", [])],
        "metadata": data.get("metadata", {}),
    }


@mcp.tool()
def get_download_url(
    version_id: int,
    file_type: Optional[str] = None,
    file_format: Optional[str] = None,
    size: Optional[str] = None,
    fp: Optional[str] = None,
) -> dict[str, Any]:
    """Прямая ссылка на скачивание версии — отдать пользователю (без сохранения на диск).

    Возвращает подписанный CDN-URL (работает в браузере/менеджере загрузок,
    живёт ~1 час, ключ в нём не светится), плюс страницу модели на Civitai.
    Удобно для больших моделей вместо download_model.

    Args:
        version_id: id версии.
        file_type/file_format/size/fp: опц. выбор конкретного файла (см. download_model).

    Returns:
        {direct_url, page_url, filename, sizeKB, expires_note}.
    """
    if not _api_key():
        return {"status": "error", "error": "CIVITAI_API_KEY не задан — ссылка требует ключ."}

    # метаданные файла для имени/размера и страницы модели
    with _client() as c:
        vr = c.get(f"{API_BASE}/model-versions/{version_id}")
        vr.raise_for_status()
        v = vr.json()
    files = v.get("files") or []
    primary = next((f for f in files if f.get("primary")), files[0] if files else {})
    model_id = v.get("modelId")

    params: dict[str, Any] = {}
    if file_type:
        params["type"] = file_type
    if file_format:
        params["format"] = file_format
    if size:
        params["size"] = size
    if fp:
        params["fp"] = fp

    # резолвим 307 в подписанный CDN-URL, тело не качаем
    with httpx.Client(headers=_headers(), timeout=60.0, follow_redirects=False) as c:
        resp = c.get(f"{DOWNLOAD_BASE}/{version_id}", params=params)
        if resp.status_code in (301, 302, 303, 307, 308):
            direct = resp.headers.get("location")
        elif resp.status_code == 200:
            direct = str(resp.request.url)  # уже прямой
        else:
            return {"status": "error", "error": f"{resp.status_code} при получении ссылки."}

    return {
        "direct_url": direct,
        "page_url": f"https://civitai.com/models/{model_id}?modelVersionId={version_id}"
        if model_id else None,
        "filename": primary.get("name"),
        "sizeKB": primary.get("sizeKB"),
        "expires_note": "direct_url подписан и действует ~1 час; page_url — постоянная.",
    }


_CD_FILENAME = re.compile(r'filename\*?=(?:UTF-8\'\')?"?([^";]+)"?', re.IGNORECASE)


def _filename_from_response(resp: httpx.Response, fallback: str) -> str:
    cd = resp.headers.get("content-disposition", "")
    m = _CD_FILENAME.search(cd)
    name = m.group(1) if m else fallback
    # немного санитайза
    name = os.path.basename(name.strip())
    return name or fallback


@mcp.tool()
def download_model(
    version_id: int,
    dest_dir: str,
    file_type: Optional[str] = None,
    file_format: Optional[str] = None,
    size: Optional[str] = None,
    fp: Optional[str] = None,
    overwrite: bool = False,
    max_mb: Optional[float] = None,
) -> dict[str, Any]:
    """Скачать файл версии модели в указанную папку.

    Требует валидный CIVITAI_API_KEY. Следует CDN-редиректу и стримит файл на диск.

    Args:
        version_id: id версии (get_model / get_model_version).
        dest_dir: абсолютный путь папки назначения (будет создана при отсутствии).
        file_type: опц. 'Model' | 'Pruned Model' | 'VAE' | 'Training Data'...
        file_format: опц. 'SafeTensor' | 'PickleTensor' | 'Diffusers'.
        size: опц. 'full' | 'pruned'.
        fp: опц. 'fp16' | 'fp32' | 'bf16'.
        overwrite: перезаписать, если файл уже есть (иначе — вернуть 'skipped').
        max_mb: если задан и файл больше — НЕ качать, вернуть status='too_large'
            с прямой ссылкой (direct_url) для ручного скачивания.

    Returns:
        {status, path, filename, bytes} | {status:'too_large', direct_url, sizeMB} | ошибку.
    """
    if not _api_key():
        return {"status": "error", "error": "CIVITAI_API_KEY не задан — скачивание требует ключ."}

    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)

    params: dict[str, Any] = {}
    if file_type:
        params["type"] = file_type
    if file_format:
        params["format"] = file_format
    if size:
        params["size"] = size
    if fp:
        params["fp"] = fp

    url = f"{DOWNLOAD_BASE}/{version_id}"

    with _client(timeout=None) as c:
        with c.stream("GET", url, params=params) as resp:
            if resp.status_code == 401:
                return {"status": "error", "error": "401 — неверный/просроченный API-ключ."}
            if resp.status_code == 403:
                return {"status": "error", "error": "403 — нет доступа (early access / ограничение автора)."}
            if resp.status_code == 429:
                return {"status": "error", "error": "429 — превышен лимит скачиваний (24ч)."}
            resp.raise_for_status()

            filename = _filename_from_response(resp, fallback=f"model-version-{version_id}.bin")
            path = dest / filename

            # порог размера — вернуть ссылку вместо скачивания
            clen = resp.headers.get("content-length")
            if max_mb is not None and clen and int(clen) > max_mb * 1024 * 1024:
                size_mb = round(int(clen) / (1024 * 1024), 1)
                return {
                    "status": "too_large",
                    "filename": filename,
                    "sizeMB": size_mb,
                    "direct_url": str(resp.url),
                    "page_url": f"https://civitai.com/models?modelVersionId={version_id}",
                    "note": f"файл {size_mb} МБ > лимита {max_mb} МБ; "
                            "direct_url подписан и действует ~1 час.",
                }

            if path.exists() and not overwrite:
                return {
                    "status": "skipped",
                    "path": str(path),
                    "filename": filename,
                    "reason": "файл уже существует (overwrite=false)",
                }

            tmp = path.with_suffix(path.suffix + ".part")
            total = 0
            with open(tmp, "wb") as fh:
                for chunk in resp.iter_bytes(chunk_size=1 << 20):
                    fh.write(chunk)
                    total += len(chunk)
            tmp.replace(path)

    return {
        "status": "downloaded",
        "path": str(path),
        "filename": filename,
        "bytes": total,
    }


# ---- параметры генерации из изображения ---------------------------------

def _png_text_chunks(data: bytes) -> dict[str, str]:
    """Достаёт текстовые чанки PNG: tEXt, zTXt, iTXt (ключ -> значение)."""
    out: dict[str, str] = {}
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        return out
    pos = 8
    n = len(data)
    while pos + 8 <= n:
        length = struct.unpack(">I", data[pos:pos + 4])[0]
        ctype = data[pos + 4:pos + 8]
        body = data[pos + 8:pos + 8 + length]
        pos += 12 + length  # 4 len + 4 type + body + 4 crc
        try:
            if ctype == b"tEXt":
                k, _, v = body.partition(b"\x00")
                out[k.decode("latin1")] = v.decode("latin1", "replace")
            elif ctype == b"zTXt":
                k, _, rest = body.partition(b"\x00")
                # rest[0] = compression method, далее zlib-данные
                out[k.decode("latin1")] = zlib.decompress(rest[1:]).decode("latin1", "replace")
            elif ctype == b"iTXt":
                k, _, rest = body.partition(b"\x00")
                comp_flag = rest[0]
                # comp_method, lang\0, translated\0, text
                rest = rest[2:]
                _, _, rest = rest.partition(b"\x00")  # lang
                _, _, rest = rest.partition(b"\x00")  # translated keyword
                out[k.decode("latin1")] = (
                    zlib.decompress(rest).decode("utf-8", "replace")
                    if comp_flag == 1 else rest.decode("utf-8", "replace")
                )
            elif ctype == b"IEND":
                break
        except Exception:
            continue
    return out


def _parse_a1111(text: str) -> dict[str, Any]:
    """Разбирает строку параметров Automatic1111 в структуру."""
    neg_marker = "Negative prompt:"
    prompt, negative, tail = text, None, ""
    if neg_marker in text:
        prompt, _, rest = text.partition(neg_marker)
        # последняя строка с key: value — это параметры
        lines = rest.strip().splitlines()
        negative = "\n".join(lines[:-1]).strip() if len(lines) > 1 else ""
        tail = lines[-1] if lines else ""
    else:
        lines = text.strip().splitlines()
        if lines and re.search(r"\b(Steps|Sampler|Seed|CFG scale):", lines[-1]):
            tail = lines[-1]
            prompt = "\n".join(lines[:-1])
    params: dict[str, Any] = {}
    # разбор "Key: value, Key: value" с учётом значений в кавычках
    for m in re.finditer(r'(\w[\w ]*?):\s*("(?:[^"]*)"|[^,]+)', tail):
        params[m.group(1).strip()] = m.group(2).strip().strip('"')
    return {
        "prompt": prompt.strip(),
        "negativePrompt": (negative or "").strip() or None,
        "params": params,
        "raw": text,
    }


def _decode_usercomment(raw: bytes) -> Optional[str]:
    """EXIF UserComment: первые 8 байт — код кодировки (ASCII/UNICODE)."""
    if not raw:
        return None
    if isinstance(raw, str):
        return raw
    head, body = raw[:8], raw[8:]
    try:
        if head.startswith(b"UNICODE"):
            # чаще UTF-16BE, иногда LE
            for enc in ("utf-16-be", "utf-16-le"):
                s = body.decode(enc, "replace")
                if "�" not in s[:20]:
                    return s.split("\x00")[0]
            return body.decode("utf-16-be", "replace").split("\x00")[0]
        if head.startswith(b"ASCII"):
            return body.decode("ascii", "replace").split("\x00")[0]
        return body.decode("utf-8", "replace").split("\x00")[0]
    except Exception:
        return None


def _params_via_pillow(p: Path) -> Optional[dict[str, Any]]:
    """Извлечь параметры генерации через Pillow: PNG-text, EXIF UserComment, XMP."""
    try:
        from PIL import Image
        from PIL.ExifTags import TAGS
    except Exception:
        return None
    try:
        with Image.open(p) as im:
            # 1) PNG/текстовые поля (info): parameters / prompt / workflow / Comment
            info = getattr(im, "info", {}) or {}
            for key in ("parameters", "Comment"):
                val = info.get(key)
                if isinstance(val, str) and ("Steps:" in val or "Negative prompt:" in val):
                    res = _parse_a1111(val)
                    res["source"] = f"pillow:{key} (A1111)"
                    return res
            if "prompt" in info or "workflow" in info:
                def _tj(s):
                    try:
                        return json.loads(s) if isinstance(s, str) else s
                    except Exception:
                        return s
                return {"source": "pillow:ComfyUI",
                        "comfy_prompt": _tj(info.get("prompt")),
                        "comfy_workflow": _tj(info.get("workflow"))}
            # 2) EXIF UserComment (0x9286)
            exif = None
            try:
                exif = im.getexif()
            except Exception:
                exif = None
            if exif:
                uc = exif.get(0x9286)
                text = _decode_usercomment(uc.encode("latin1") if isinstance(uc, str) else uc) \
                    if uc else None
                if text and ("Steps:" in text or "Negative prompt:" in text):
                    res = _parse_a1111(text)
                    res["source"] = "pillow:exif UserComment (A1111)"
                    return res
                # ImageDescription иногда содержит параметры
                desc = exif.get(0x010E)
                if isinstance(desc, str) and ("Steps:" in desc or "Negative prompt:" in desc):
                    res = _parse_a1111(desc)
                    res["source"] = "pillow:exif ImageDescription (A1111)"
                    return res
            # 3) XMP (WebP/JPEG) — иногда параметры лежат там как текст
            xmp = info.get("XML:com.adobe.xmp") or info.get("xmp")
            if isinstance(xmp, bytes):
                xmp = xmp.decode("utf-8", "replace")
            if isinstance(xmp, str) and ("Steps:" in xmp or "Negative prompt:" in xmp):
                m = re.search(r"(?s)(.*?Steps:.*?)(?:</|$)", xmp)
                if m:
                    res = _parse_a1111(m.group(1).strip())
                    res["source"] = "pillow:xmp (A1111)"
                    return res
    except Exception:
        return None
    return None


@mcp.tool()
def read_image_params(path: str) -> dict[str, Any]:
    """Прочитать параметры генерации, зашитые в локальный файл изображения.

    Работает офлайн (без API). Понимает PNG-метаданные Automatic1111
    (чанк 'parameters') и ComfyUI ('prompt'/'workflow' JSON).

    Args:
        path: путь к файлу (.png в первую очередь; jpg/webp — если есть EXIF-UserComment).

    Returns:
        {source, prompt, negativePrompt, params, ...} либо {status:'not_found'} если
        метаданных нет.
    """
    p = Path(path)
    if not p.exists():
        return {"status": "error", "error": f"файл не найден: {path}"}
    data = p.read_bytes()

    if data[:8] == b"\x89PNG\r\n\x1a\n":
        chunks = _png_text_chunks(data)
        if "parameters" in chunks:  # Automatic1111 / Forge
            res = _parse_a1111(chunks["parameters"])
            res["source"] = "png:parameters (A1111)"
            return res
        if "prompt" in chunks or "workflow" in chunks:  # ComfyUI
            def _tryjson(s: Optional[str]):
                try:
                    return json.loads(s) if s else None
                except Exception:
                    return s
            return {
                "source": "png:ComfyUI",
                "comfy_prompt": _tryjson(chunks.get("prompt")),
                "comfy_workflow": _tryjson(chunks.get("workflow")),
                "other_chunks": {k: v for k, v in chunks.items()
                                 if k not in ("prompt", "workflow")},
            }
        if chunks:
            # текстовые чанки есть, но не распознаны — пробуем Pillow, иначе отдаём как есть
            via = _params_via_pillow(p)
            return via or {"source": "png:text", "chunks": chunks}
        via = _params_via_pillow(p)
        return via or {"status": "not_found", "note": "в PNG нет текстовых метаданных."}

    # JPEG/WebP/прочее: приоритет — Pillow (EXIF UserComment / XMP / ImageDescription)
    via = _params_via_pillow(p)
    if via:
        return via
    # fallback без Pillow: ручной поиск UserComment в UTF-16
    if data[:2] == b"\xff\xd8" or data[:4] == b"RIFF":
        idx = data.find(b"UNICODE\x00\x00")
        if idx != -1:
            text = _decode_usercomment(data[idx:].split(b"\xff\xd9")[0])
            if text and ("Steps:" in text or "Negative prompt:" in text):
                res = _parse_a1111(text)
                res["source"] = "exif:UserComment (A1111, manual)"
                return res
    return {"status": "not_found", "note": "метаданные генерации не найдены."}


@mcp.tool()
def get_image_meta(image_id: int, nsfw: Optional[str] = None) -> dict[str, Any]:
    """Параметры генерации картинки, размещённой на Civitai, по её id.

    ВНИМАНИЕ: у многих картинок публичной ленты Civitai поле meta = null (скрыто).
    У примеров версии модели (get_model_version.images) meta присутствует всегда.

    Args:
        image_id: числовой id картинки на Civitai.
        nsfw: порог зрелости, если нужен доступ к зрелым картинкам ('None'..'X').

    Returns:
        {url, width, height, nsfwLevel, meta:{...prompt/seed/steps/...}} либо not_found.
    """
    params: dict[str, Any] = {"imageId": image_id}
    if nsfw:
        params["nsfw"] = nsfw
    with _client() as c:
        r = c.get(f"{API_BASE}/images", params=params)
        r.raise_for_status()
        items = r.json().get("items", [])
    if not items:
        return {"status": "not_found", "note": f"картинка {image_id} не найдена/скрыта."}
    i = items[0]
    return {
        "id": i.get("id"),
        "url": i.get("url"),
        "width": i.get("width"),
        "height": i.get("height"),
        "nsfwLevel": i.get("nsfwLevel"),
        "meta": i.get("meta"),
        "meta_available": bool(i.get("meta")),
    }


# ---- генерация изображений (Civitai Orchestration API) -------------------

def _resolve_air(model_version_id: Optional[int], model_air: Optional[str]) -> tuple[Optional[str], Optional[str]]:
    """Вернуть (air, error). air из готового поля версии или переданной строки."""
    if model_air:
        return model_air, None
    if model_version_id:
        with _client() as c:
            r = c.get(f"{API_BASE}/model-versions/{model_version_id}")
            if r.status_code != 200:
                return None, f"версия {model_version_id} не найдена ({r.status_code})."
            air = r.json().get("air")
            if not air:
                return None, f"у версии {model_version_id} нет поля air (генерация недоступна)."
            return air, None
    return None, "нужен model_version_id или model_air."


def _ecosystem_from_air(air: str) -> Optional[str]:
    parts = air.split(":")
    return parts[2] if len(parts) > 2 else None


def _build_workflow(air: str, ecosystem: str, engine: str, prompt: str,
                    negative_prompt: Optional[str], width: int, height: int,
                    steps: int, cfg_scale: float, seed: Optional[int],
                    quantity: int, clip_skip: Optional[int],
                    sampler: Optional[str]) -> dict[str, Any]:
    inp: dict[str, Any] = {
        "engine": engine,
        "ecosystem": ecosystem,
        "operation": "createImage",
        "model": air,
        "prompt": prompt,
        "width": width,
        "height": height,
        "steps": steps,
        "cfgScale": cfg_scale,
        "quantity": quantity,
    }
    if negative_prompt:
        inp["negativePrompt"] = negative_prompt
    if seed is not None:
        inp["seed"] = seed
    if clip_skip is not None:
        inp["clipSkip"] = clip_skip
    if sampler:
        inp["sampleMethod"] = sampler
    return {"steps": [{"$type": "imageGen", "input": inp}]}


def _post_workflow(body: dict[str, Any], query: dict[str, Any]) -> httpx.Response:
    with httpx.Client(headers=_headers(), timeout=180.0, follow_redirects=True) as c:
        return c.post(ORCHESTRATION, params=query, json=body)


def _cost_from(resp_json: dict[str, Any]) -> dict[str, Any]:
    tx = (resp_json.get("transactions") or {})
    debits = [t for t in tx.get("list", []) if t.get("type") == "debit"]
    return {
        "buzz": sum(t.get("amount", 0) for t in debits),
        "breakdown": debits,
        "insufficientBuzz": tx.get("insufficientBuzz", False),
    }


def _images_from(resp_json: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for st in resp_json.get("steps", []):
        for img in ((st.get("output") or {}).get("images") or []):
            out.append({"id": img.get("id"), "url": img.get("url"),
                        "available": img.get("available"),
                        "width": img.get("width"), "height": img.get("height")})
    return out


@mcp.tool()
def estimate_generation(
    prompt: str,
    model_version_id: Optional[int] = None,
    model_air: Optional[str] = None,
    negative_prompt: Optional[str] = None,
    width: int = 1024,
    height: int = 1024,
    steps: int = 25,
    cfg_scale: float = 6.0,
    seed: Optional[int] = None,
    quantity: int = 1,
    clip_skip: Optional[int] = None,
    sampler: Optional[str] = None,
    engine: str = "sdcpp",
    ecosystem: Optional[str] = None,
) -> dict[str, Any]:
    """Оценить стоимость генерации в Buzz (whatif) — БЕЗ списания и без картинки.

    Модель задаётся `model_version_id` (AIR берётся из API) или готовым `model_air`.
    ecosystem определяется из AIR автоматически (sd1/sdxl/flux1...).

    Returns:
        {air, ecosystem, cost:{buzz, insufficientBuzz, breakdown}}.
    """
    air, err = _resolve_air(model_version_id, model_air)
    if err:
        return {"status": "error", "error": err}
    eco = ecosystem or _ecosystem_from_air(air)
    body = _build_workflow(air, eco, engine, prompt, negative_prompt, width, height,
                           steps, cfg_scale, seed, quantity, clip_skip, sampler)
    r = _post_workflow(body, {"whatif": "true"})
    if r.status_code >= 400:
        return {"status": "error", "error": f"{r.status_code}: {r.text[:300]}"}
    j = r.json()
    return {"air": air, "ecosystem": eco, "cost": _cost_from(j)}


@mcp.tool()
def generate_image(
    prompt: str,
    model_version_id: Optional[int] = None,
    model_air: Optional[str] = None,
    negative_prompt: Optional[str] = None,
    width: int = 1024,
    height: int = 1024,
    steps: int = 25,
    cfg_scale: float = 6.0,
    seed: Optional[int] = None,
    quantity: int = 1,
    clip_skip: Optional[int] = None,
    sampler: Optional[str] = None,
    engine: str = "sdcpp",
    ecosystem: Optional[str] = None,
    save_dir: Optional[str] = None,
    confirm: bool = False,
    wait: int = 60,
) -> dict[str, Any]:
    """Сгенерировать изображение через Civitai Orchestration API. ТРАТИТ BUZZ.

    ЗАЩИТА: без confirm=true инструмент НЕ генерирует — только возвращает
    предполагаемую стоимость (whatif). Чтобы реально запустить и списать Buzz,
    вызовите повторно с confirm=true.

    Модель: `model_version_id` (AIR из API) или готовый `model_air`.
    Если задан `save_dir` — готовые картинки скачиваются туда; в ответе и пути, и URL.

    Args:
        prompt: позитивный промпт.
        model_version_id / model_air: какая модель (одно из).
        negative_prompt, width, height, steps, cfg_scale, seed, quantity: параметры.
        clip_skip, sampler, engine, ecosystem: доп. настройки (ecosystem авто из AIR).
        save_dir: папка для скачивания результатов (опц.).
        confirm: ОБЯЗАТЕЛЬНО true для реального запуска (иначе только оценка стоимости).
        wait: сколько секунд держать соединение до поллинга (по умолчанию 60).

    Returns:
        confirm=false → {status:'preview', cost, hint};
        confirm=true  → {status, workflowId, cost, images:[{id,url,path?}]}.
    """
    if not _api_key():
        return {"status": "error", "error": "CIVITAI_API_KEY не задан."}
    air, err = _resolve_air(model_version_id, model_air)
    if err:
        return {"status": "error", "error": err}
    eco = ecosystem or _ecosystem_from_air(air)
    body = _build_workflow(air, eco, engine, prompt, negative_prompt, width, height,
                           steps, cfg_scale, seed, quantity, clip_skip, sampler)

    # всегда сначала whatif — узнать цену и хватает ли Buzz
    wr = _post_workflow(body, {"whatif": "true"})
    if wr.status_code >= 400:
        return {"status": "error", "error": f"whatif {wr.status_code}: {wr.text[:300]}"}
    cost = _cost_from(wr.json())

    if not confirm:
        return {
            "status": "preview",
            "air": air, "ecosystem": eco, "cost": cost,
            "hint": f"генерация спишет ~{cost['buzz']} Buzz; "
                    "вызовите снова с confirm=true для запуска.",
        }
    if cost.get("insufficientBuzz"):
        return {"status": "error", "error": "недостаточно Buzz.", "cost": cost}

    # реальный запуск
    rr = _post_workflow(body, {"wait": wait})
    if rr.status_code >= 400:
        return {"status": "error", "error": f"generate {rr.status_code}: {rr.text[:300]}"}
    j = rr.json()
    result: dict[str, Any] = {
        "status": j.get("status"),
        "workflowId": j.get("id"),
        "cost": _cost_from(j) or cost,
        "images": _images_from(j),
    }

    if save_dir and result["images"]:
        dest = Path(save_dir)
        dest.mkdir(parents=True, exist_ok=True)
        with httpx.Client(timeout=None, follow_redirects=True) as c:
            for img in result["images"]:
                url = img.get("url")
                if not url or img.get("available") is False:
                    continue
                name = os.path.basename((img.get("id") or "image") + ".jpg") \
                    if not (img.get("id") or "").endswith((".jpg", ".png", ".jpeg")) \
                    else os.path.basename(img["id"])
                path = dest / name
                try:
                    resp = c.get(url)
                    resp.raise_for_status()
                    path.write_bytes(resp.content)
                    img["path"] = str(path)
                except Exception as e:
                    img["download_error"] = str(e)

    if result["status"] not in ("succeeded", None) and not result["images"]:
        result["hint"] = ("не готово за wait сек — опросите get_workflow(workflowId).")
    return result


@mcp.tool()
def get_workflow(workflow_id: str) -> dict[str, Any]:
    """Статус и результат ранее запущенной генерации по workflowId (поллинг).

    Args:
        workflow_id: id из ответа generate_image.

    Returns:
        {status, images:[{id,url}]}.
    """
    with httpx.Client(headers=_headers(), timeout=60.0, follow_redirects=True) as c:
        r = c.get(f"{ORCHESTRATION}/{workflow_id}")
    if r.status_code >= 400:
        return {"status": "error", "error": f"{r.status_code}: {r.text[:200]}"}
    j = r.json()
    return {"status": j.get("status"), "workflowId": j.get("id"),
            "images": _images_from(j)}


@mcp.tool()
def get_buzz_balance() -> dict[str, Any]:
    """Баланс Buzz текущего аккаунта (по ключу CIVITAI_API_KEY).

    ⚠️ НЕОФИЦИАЛЬНО: использует ВНУТРЕННИЙ tRPC-эндпоинт сайта civitai.com
    (buzz.getBuzzAccount), а НЕ публичный /api/v1. Он не документирован и может
    измениться/перестать работать в любой момент без предупреждения.

    Returns:
        {userId, username, balance:{yellow,blue,green}, total, note} либо ошибку.
    """
    if not _api_key():
        return {"status": "error", "error": "CIVITAI_API_KEY не задан."}

    with _client() as c:
        me = c.get(f"{API_BASE}/me")
        me.raise_for_status()
        me_data = me.json()
        user_id = me_data.get("id")
        if not user_id:
            return {"status": "error", "error": "не удалось определить userId из /me."}

        inp = json.dumps({"json": {"accountId": user_id, "accountType": "user"}},
                         separators=(",", ":"))
        r = c.get(f"{TRPC_BASE}/buzz.getBuzzAccount", params={"input": inp})
        if r.status_code != 200:
            return {"status": "error",
                    "error": f"{r.status_code} от внутреннего tRPC — эндпоинт мог измениться."}
        data = (((r.json() or {}).get("result") or {}).get("data") or {}).get("json") or {}

    yellow = data.get("yellow", 0)
    blue = data.get("blue", 0)
    green = data.get("green", 0)
    return {
        "userId": user_id,
        "username": me_data.get("username"),
        "balance": {"yellow": yellow, "blue": blue, "green": green},
        "total": yellow + blue + green,
        "note": "неофициальный внутренний tRPC-эндпоинт; может перестать работать.",
    }


if __name__ == "__main__":
    mcp.run()
