# Civitai MCP

[![Built with Claude Code](https://img.shields.io/badge/Built%20with-Claude%20Code-D97757?logo=claude&logoColor=fff)](https://claude.com/claude-code)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=fff)](https://www.python.org/)
[![MCP](https://img.shields.io/badge/MCP-Model%20Context%20Protocol-000?logo=modelcontextprotocol&logoColor=fff)](https://modelcontextprotocol.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**MCP-сервер для работы с [Civitai](https://civitai.com) прямо из ассистента.**
Ищите модели по названию/типу/базовой модели, смотрите превью и примеры генерации,
берите параметры генерации из любой картинки (локальной или с Civitai), скачивайте
файлы на диск или получайте прямую ссылку для больших моделей.

Написан на Python ([FastMCP](https://github.com/modelcontextprotocol/python-sdk)) +
[httpx](https://www.python-httpx.org/) + [Pillow](https://python-pillow.org/).

## Возможности

- 🔎 **Поиск моделей** — фильтры по типу, базовой модели, тегу, автору, сортировка.
- 🖼️ **Визуальный выбор** — превью в результатах поиска и галерея примеров генерации
  (готовые URL + лёгкие превьюшки).
- 🧬 **Параметры генерации** — читает prompt/seed/steps/sampler/… из локального
  PNG (A1111/ComfyUI) или jpg/webp (EXIF/XMP), а также с картинок Civitai.
- ⬇️ **Скачивание** — файл версии на диск (стриминг, атомарная запись, докачка) или
  прямая подписанная ссылка для больших моделей.

## Инструменты

| Tool | Что делает |
|---|---|
| `search_models` | поиск моделей: `query`, `types`, `base_models`, `sort`, `tag`, `username`, `nsfw`, `limit`, `page`. В каждом результате — `preview` (картинка-превью) |
| `get_model` | карточка модели по id: версии, файлы (downloadUrl, SHA256) и примеры-картинки |
| `get_model_version` | детали версии по id: файлы + `images` (примеры генерации) |
| `get_model_images` | галерея примеров по `model_id`/`version_id` для визуального выбора (url + `thumb` + `prompt`) |
| `get_download_url` | прямая ссылка на файл (для больших моделей) — подписанный CDN-URL + страница модели, без сохранения на диск |
| `download_model` | скачать файл версии в `dest_dir`; опц. `file_type`/`file_format`/`size`/`fp`, `overwrite`, `max_mb` |
| `read_image_params` | параметры генерации из **локального** файла (PNG A1111/ComfyUI, EXIF), офлайн |
| `get_image_meta` | параметры генерации картинки, размещённой **на Civitai**, по её id |
| `get_buzz_balance` | баланс Buzz аккаунта (yellow/blue/green) — ⚠️ неофициальный эндпоинт |
| `estimate_generation` | оценить стоимость генерации в Buzz (whatif), без списания |
| `generate_image` | сгенерировать картинку (Orchestration API); **тратит Buzz**, требует `confirm=true` |
| `get_workflow` | статус/результат генерации по `workflowId` (поллинг) |

Типовой сценарий: `search_models` → посмотреть `preview`/`get_model_images` →
выбрать `version.id` → `download_model(version_id, dest_dir)` **или** `get_download_url(version_id)`.

### Картинки

Превью и галерея приходят готовыми URL-ами CDN. У каждой картинки есть:
- `url` — оригинал;
- `thumb` — лёгкая версия (`width=450`, строится трансформом CDN);
- `nsfwLevel`, `width`, `height`, `type` (`image`/`video`), `prompt`.

### Параметры генерации изображения

Два источника, потому что это разные вещи:

- `read_image_params(path)` — **локальный файл**. Читает офлайн зашитые метаданные:
  PNG Automatic1111/Forge (чанк `parameters`), ComfyUI (`prompt`/`workflow`),
  а для jpg/webp — EXIF `UserComment`/`ImageDescription` и XMP через Pillow.
  Возвращает `prompt`, `negativePrompt`, `params` (steps/sampler/cfg/seed/size/model…),
  `source` (откуда взято) и сырую строку. Работает для любой сгенерированной
  картинки, даже не с Civitai. PNG разбирается и без Pillow; jpg/webp — с Pillow
  (входит в requirements).
- `get_image_meta(image_id)` — **картинка на Civitai** по её id. Важно: у публичной
  ленты `meta` часто `null` (Civitai его скрывает). У примеров версии модели
  (`get_model_version.images`) `meta` есть всегда, поле `meta_available` это показывает.

### Большие модели → ссылка вместо скачивания

- `get_download_url(version_id)` — всегда возвращает `direct_url` (подписан, ~1 час)
  и `page_url` (постоянная страница модели).
- `download_model(..., max_mb=200)` — если файл больше лимита, вернёт
  `status: "too_large"` с `direct_url` вместо загрузки на диск.

### Генерация изображений

Официальный [Civitai Orchestration API](https://developer.civitai.com/orchestration/).
Модель задаётся `model_version_id` (AIR берётся из API автоматически) или готовым
`model_air`; `ecosystem` (sd1/sdxl/flux1…) определяется из AIR. Готовый AIR версии
также виден в поле `air` ответа `get_model_version`.

> ⚠️ **Генерация тратит реальный Buzz.** `generate_image` без `confirm=true`
> **не запускает** генерацию — только возвращает предполагаемую стоимость (whatif).
> Для реального запуска — повторный вызов с `confirm=true`. Проверка баланса
> и `insufficientBuzz` выполняется до списания.

```
estimate_generation(prompt, model_version_id=…)      → {cost:{buzz, insufficientBuzz}}
generate_image(prompt, model_version_id=…)           → preview со стоимостью (confirm=false)
generate_image(prompt, model_version_id=…, save_dir=…, confirm=true)
                                                     → {status, workflowId, images:[{url, path}]}
get_workflow(workflow_id)                            → статус/картинки для долгих задач
```

Ориентир по цене: SD1.5 512×512 / 20 шагов ≈ 4 Buzz, SDXL 1024×1024 / 25 шагов ≈ 10 Buzz.

### Баланс Buzz

`get_buzz_balance()` возвращает балансы кошельков (`yellow`/`blue`/`green`) текущего
аккаунта по ключу.

> ⚠️ **Неофициально.** Использует внутренний tRPC-эндпоинт сайта (`buzz.getBuzzAccount`),
> а не публичный `/api/v1`. Он не документирован и может перестать работать без
> предупреждения — в отличие от остальных инструментов.

## Установка

```bash
py -m venv .venv
./.venv/Scripts/python.exe -m pip install -r requirements.txt
```

## Ключ API

Ключ берётся из переменной окружения `CIVITAI_API_KEY`
(fallback — файл `apikey.txt` рядом с `server.py`).
Скачивание файлов требует валидный ключ; поиск/метаданные работают и без него.

Получить ключ: civitai.com → Account settings → API Keys.

## Подключение к Claude Code

```bash
claude mcp add civitai -e CIVITAI_API_KEY=<ваш-ключ> -- H:/CivitaiMcp/.venv/Scripts/python.exe H:/CivitaiMcp/server.py
```

Или вручную в `.mcp.json` (см. `mcp.example.json`):

```json
{
  "mcpServers": {
    "civitai": {
      "command": "H:/CivitaiMcp/.venv/Scripts/python.exe",
      "args": ["H:/CivitaiMcp/server.py"],
      "env": { "CIVITAI_API_KEY": "<ваш-ключ>" }
    }
  }
}
```

## Примеры значений

- `types`: `Checkpoint`, `LORA`, `LoCon`, `VAE`, `TextualInversion`, `Hypernetwork`, `Controlnet`
- `base_models`: `SD 1.5`, `SDXL 1.0`, `Pony`, `Illustrious`, `Flux.1 D`
- `sort`: `Highest Rated`, `Most Downloaded`, `Newest`
- `file_format`: `SafeTensor`, `PickleTensor`; `size`: `full`/`pruned`; `fp`: `fp16`/`fp32`/`bf16`

## Ошибки скачивания

`401` — неверный ключ · `403` — early-access/ограничение автора · `429` — лимит 24ч.

## Лицензия

[MIT](LICENSE).

---

<sub>🤖 Built with [Claude Code](https://claude.com/claude-code).</sub>
