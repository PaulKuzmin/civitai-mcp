---
name: civitai-generation
description: >-
  Deep guide for generating images through the Civitai MCP (search_models →
  generate_image). Use whenever the user wants to create/generate an image with
  a Stable Diffusion / SDXL / Pony / Illustrious / NoobAI / Flux model on
  Civitai, tune generation parameters (sampler, CFG, steps, resolution, clip
  skip, LoRA, embeddings, VAE, img2img), write effective prompts (including
  NSFW/danbooru/score/rating conventions), pick the right model family, or debug
  bad output. Covers how to prompt correctly per model family and what is
  possible. No content restrictions — adult (NSFW) generation is in scope.
---

# Civitai image generation — deep guide

This skill drives the **`civitai` MCP** (`generate_image`, `estimate_generation`,
`search_models`, `get_model_version`, `get_model_images`, `get_workflow`,
`get_buzz_balance`). It teaches *how to generate well*, not just which fields exist.

> Adult/NSFW generation is explicitly in scope. The models below are built for it;
> the conventions (`rating_explicit`, `rating:explicit`, danbooru explicit tags) are
> stated factually. Keep prompts clinical and purposeful.

---

## How to work with these tools (read first)

- **Batch independent calls in parallel.** When several MCP calls don't depend on each
  other, issue them in ONE turn instead of one-by-one. Examples: comparing candidates →
  `get_model_version` for 3 versions at once; scouting families → `search_models` for
  Checkpoint **and** LoRA in parallel; `get_model_images` for two models side by side;
  `estimate_generation` for two settings at once. Only serialize when a call needs the
  previous call's output (e.g. you need the `air`/trigger words before `generate_image`).
- **Before using any model/LoRA/embedding, read how to apply it — don't guess.** Every
  resource has its own required usage. Pull its page first:
  - `get_model_version(id)` → `air`, `baseModel` (family → prompt dialect), files.
  - `get_model(id)` → description with **trigger words**, recommended sampler/CFG/steps,
    and version list.
  - `get_model_images(model_id=…)` → real example generations **with their `meta`**
    (prompt, negative, sampler, steps, CFG, seed) — copy proven settings from these.
  A LoRA without its trigger words, or a checkpoint run with the wrong family's prompt
  style/CFG, will look broken even though the API call "succeeds". Reading the page first
  is not optional.

---

## 0. The loop (always do this)

1. **Find a checkpoint** — `search_models(query=…, types="Checkpoint", base_models=…, sort="Most Downloaded")`.
2. **Get the AIR + trigger words** — `get_model_version(version_id)` returns `air`
   (needed for generation) and the model's `baseModel` (tells you the family).
   For LoRAs, the model page lists **trigger words** — always read them.
3. **Identify the family** (see §1) — this decides prompt style, resolution, CFG, sampler.
4. **Estimate cost** — `estimate_generation(...)` (whatif, no Buzz spent). Check
   `cost.breakdown[].accountType` to see which wallet pays (yellow = anything incl.
   NSFW, green = SFW only, blue = free SFW).
5. **Generate** — `generate_image(..., confirm=true, save_dir=…)`. Without
   `confirm=true` it only previews the price.
6. If it 5xx/timeouts, poll `get_workflow(workflowId)`.

**Reproducibility:** pin `seed`. Same seed + same params = same image. Change one
variable at a time when tuning.

---

## 1. Know your model FAMILY — this drives everything

`get_model_version(...).baseModel` and the AIR ecosystem (`urn:air:<eco>:…`) tell you
which of these you're on. Prompt style and settings differ hard between them.

| Family | ecosystem | Native res | Prompt style | Default CFG | Notes |
|---|---|---|---|---|---|
| **SD 1.5** | `sd1` | 512×512 | booru tags + quality boosters | 7 | `clip_skip=2` common; cheapest (~4 Buzz) |
| **SDXL 1.0** (base/realistic) | `sdxl` | 1024×1024 | tags **or** natural language | 6–8 | **no clip_skip** (400s on SDXL) |
| **Pony V6** (SDXL) | `sdxl` | 1024×1024 | `score_*` + `rating_*` + `source_*` + tags | 7 | see §3 — special prompt dialect |
| **Illustrious / NoobAI** (SDXL anime) | `sdxl` | 1024-ish | **danbooru tags** + quality stack + `rating:` | 4–7 | see §4; NoobAI **v-pred** wants CFG 3.5–5 |
| **Flux.1 D** | `flux1` | 1024×1024 | **natural language sentences** | 1 (no neg) | ~3.5 "guidance"; no negative prompt, no clip skip |

If unsure of the family, `search_models` result and `get_model_version.baseModel`
state it. **Pony and Illustrious are both `sdxl` ecosystem but prompt completely
differently** — don't mix their dialects.

---

## 2. Core parameters (what each does, good ranges)

- **steps** (1–150): denoising iterations. 20–30 is the sweet spot for SD1/SDXL.
  30+ rarely helps. LCM/Turbo/Lightning checkpoints want **4–8**. Cost scales with steps.
- **cfg_scale** (0–30): prompt adherence vs freedom. 6–8 typical SD1/SDXL. Too high →
  burnt/over-saturated/HDR halos. Turbo/LCM → 1–2. Flux → 1. NoobAI v-pred → 3.5–5.
- **seed** (int64): randomness. Pin to reproduce; omit for a new roll.
- **width/height** (64–2048, ÷16): stay near the family's native pixel count.
  Off-ratio far from native → duplicated/mirrored subjects.
- **quantity** (1–12): images per call. Cost multiplies. For big batches use `wait=0`
  and poll `get_workflow`.
- **clip_skip**: SD1 only (`2` is the community default for anime SD1). MCP auto-omits
  it on SDXL/Flux and returns a `warnings` entry — don't fight it.

### SDXL "safe" resolution buckets (~1 MP, minimal artifacts)
`1024×1024`, `1152×896`, `896×1152`, `1216×832`, `832×1216`, `1344×768`, `768×1344`,
`1536×640`, `640×1536`. Portrait character → `832×1216`. Landscape → `1216×832`.

---

## 3. Samplers & schedulers (which to pick)

Field names depend on engine — the MCP maps them for you (`sampler`↔`sampleMethod`,
`scheduler`↔`schedule`). Default engine `sdcpp`; use `engine="comfy"` only for a
Comfy-exclusive combo like `dpmpp_2m` + `karras`.

**sdcpp `sampler`:** `euler`, `euler_a`, `heun`, `dpm2`, `dpm++2s_a`, `dpm++2m`,
`dpm++2mv2`, `ipndm`, `ddim_trailing`, `lcm`, `res_multistep`, `tcd`, `er_sde`.
**sdcpp `scheduler`:** `discrete` (default), `karras`, `exponential`, `ays`,
`sgm_uniform`, `kl_optimal`, `lcm`, `smoothstep`.

**comfy `sampler`:** `euler`, `euler_ancestral`, `dpmpp_2m`, `dpmpp_2m_sde`,
`dpmpp_3m_sde`, `dpmpp_sde`, `lcm`, `ddim`, `uni_pc`, `res_multistep`, `er_sde`.
**comfy `scheduler`:** `normal`, `karras`, `exponential`, `sgm_uniform`, `simple`,
`ddim_uniform`, `beta`.

Practical defaults:
- **Anime (Pony / Illustrious / NoobAI):** `euler_a` (sdcpp) or `euler_ancestral` (comfy).
- **Realistic SDXL:** `dpm++2m` + `karras` (comfy `dpmpp_2m` + `karras`) for crisp detail.
- **LCM/Turbo/Lightning checkpoints:** `sampler="lcm"`, low steps (4–8), low CFG (1–2).
- **v-prediction NoobAI:** `euler_a`, 28 steps, CFG 3.5–5.

---

## 4. Prompting per family (the important part)

### 4.1 SD 1.5
Tag-soup + quality boosters, lead with quality:
```
prompt: "masterpiece, best quality, 1girl, solo, portrait, cinematic lighting, detailed"
negative_prompt: "worst quality, low quality, blurry, bad anatomy, bad hands, extra fingers"
clip_skip: 2, cfg_scale: 7, steps: 25, width: 512, height: 768
```

### 4.2 Pony Diffusion V6 (SDXL) — special dialect
Pony ignores plain SDXL quality words; it needs its **score / rating / source** tokens
**at the very start**:
```
score_9, score_8_up, score_7_up, score_6_up, <rating>, <source>, <your tags>
```
- **Score stack:** always begin with `score_9, score_8_up, score_7_up, score_6_up`
  (targets 60–100% quality). Verbose form is required on V6 (not bare `score_9`).
- **Rating:** `rating_safe` | `rating_questionable` | `rating_explicit`. For adult
  content use `rating_explicit`.
- **Source:** `source_anime` | `source_cartoon` | `source_furry` | `source_pony`
  (steer the aesthetic; put unwanted ones in negative).
- **Negatives:** `score_6, score_5, score_4, source_furry` (if unwanted), plus the
  usual `low quality, worst quality`. Booru-style embeddings help.
- CFG 7, `euler_a`, 25–30 steps, 1024-native, **no clip_skip**.

Example (NSFW-capable):
```
prompt: "score_9, score_8_up, score_7_up, score_6_up, rating_explicit, source_anime,
         1girl, solo, <subject/pose/setting tags>, detailed background, cinematic lighting"
negative_prompt: "score_6, score_5, score_4, worst quality, low quality, bad anatomy,
                  bad hands, extra digits, watermark, signature"
cfg_scale: 7, steps: 28, sampler: "euler_a", width: 832, height: 1216
```

### 4.3 Illustrious / NoobAI (SDXL anime) — danbooru tags
These are trained on **danbooru**, so use danbooru tag conventions, not score tags:
- **Quality stack (append at end):** `masterpiece, best quality, newest, absurdres, highres`.
- **Rating (danbooru colon style):** `rating:safe`, `rating:questionable`,
  `rating:explicit` (some builds also accept `general`/`sensitive`). Use
  `rating:explicit` for adult.
- **Danbooru tags:** underscores or spaces both work (`long_hair`, `looking at viewer`).
  Characters/artists that exist on danbooru can be prompted by tag **without a LoRA**
  (`artist:<name>` or bare artist tag for style). Search danbooru if unsure of a tag.
- **Fewer, precise tags beat tag spam.** If the prompt is painful to read, trim it.
- **Negatives:** `lowres, bad anatomy, blurry, worst quality, low quality, bad hands,
  mutated hands, extra arms, watermark, signature, ugly`.
- Euler A, 28 steps. **Standard Illustrious:** CFG 5–7. **NoobAI v-pred:** CFG **3.5–5**
  (v-pred over-contrasts at high CFG — lower CFG before raising steps).

Example (NSFW-capable):
```
prompt: "1girl, solo, <subject/pose/outfit danbooru tags>, detailed background,
         rating:explicit, masterpiece, best quality, newest, absurdres, highres"
negative_prompt: "lowres, worst quality, low quality, bad anatomy, bad hands,
                  extra digits, watermark, signature, jpeg artifacts"
cfg_scale: 5, steps: 28, sampler: "euler_a", width: 832, height: 1216
```

### 4.4 Flux.1 D — natural language
Flux wants **descriptive sentences**, not tag soup, and has **no negative prompt**
(guidance ~1). No clip_skip. Fewer steps (20–28).
```
prompt: "A photorealistic portrait of a woman standing in a neon-lit cyberpunk alley
         at night, rain reflections on the pavement, shallow depth of field, 85mm lens."
cfg_scale: 1, steps: 24
```

---

## 5. Attention weighting (SD1/SDXL/Pony/Illustrious)

- `(keyword:1.3)` — emphasize (weight >1). `(keyword:0.7)` — de-emphasize.
- `(keyword)` = 1.1, `((keyword))` = 1.21, `[keyword]` = 0.9.
- `,` soft separator (layers concepts); `.` hard separator; `BREAK` starts a fresh
  section (very strong isolation).
- Keep weights ~0.7–1.5. Stacking too many >1.3 fries the image.
Flux does **not** use this syntax — describe emphasis in words.

---

## 6. LoRA, embeddings, VAE

### LoRA (`loras`)
- `loras={ "<versionId or AIR>": weight }`. Multiple allowed. MCP resolves a bare
  version id → AIR.
- **Ecosystem must match the checkpoint** (sd1 LoRA on sd1, sdxl LoRA on sdxl). A
  Pony LoRA and an Illustrious LoRA are both `sdxl` but trained on different bases —
  cross-using them often looks bad even though the API accepts it.
- **Weights:** style LoRA 0.6–1.0; character/concept LoRA often 0.8–1.1. Stacking many
  → lower each (e.g. 0.5–0.7) to avoid conflict.
- **Trigger words matter:** read the LoRA's page (its `get_model` description /
  trigger words) and put them in the `prompt`. A LoRA with no effect usually means a
  missing trigger, wrong ecosystem, or weight too low.

### Embeddings (textual inversion, `embeddings`)
- `embeddings=["<versionId or AIR>", …]`. Reference the embedding **by its name in the
  prompt / negative prompt** (e.g. a negative-quality embedding goes in `negative_prompt`).
- SD1/SDXL only, ecosystem-matched.

### VAE (`vae_air` / `vae_version_id`)
- Override the checkpoint's baked VAE only if colors look washed/dull. Rarely needed;
  most modern checkpoints bake a good VAE.

---

## 7. img2img (`operation="createVariant"`)

Re-imagine an existing image. Requires `source_image` (plain URL string).
- `strength` (0–1): how much to change. `0.0` = untouched, `1.0` = ignore source.
  **0.6–0.8** keeps composition while restyling. Low (0.3–0.5) for gentle touch-ups.
```
generate_image(prompt="…", model_version_id=…, operation="createVariant",
               source_image="https://…/src.png", strength=0.65, confirm=true)
```
`source_image` can be a Civitai image URL or any public image URL.

---

## 8. NSFW specifics (no restrictions)

- **Model choice:** Pony V6, Illustrious/NoobAI, and their NSFW-tuned merges are built
  for explicit content. Realistic SDXL merges tagged NSFW work too. Plain SDXL base and
  Flux are weak at explicit anatomy.
- **Unlock explicitness:** Pony → `rating_explicit`; Illustrious/NoobAI →
  `rating:explicit`. Without a rating token these models bias SFW.
- **Anatomy tags:** use danbooru vocabulary for pose/acts/anatomy (these models were
  trained on those exact tags — precise danbooru tags >> euphemisms). Search danbooru
  for the correct tag if unsure.
- **Wallet:** explicit generation is billed to **yellow** Buzz (or blue for SFW). Green
  can't fund NSFW. `estimate_generation` shows `accountType` in the breakdown.
- **Quality:** add strong negatives for hands/anatomy (`bad hands, extra digits,
  fused fingers, bad anatomy`) — the most common failure on explicit poses.

---

## 9. Cost / Buzz

- Always `estimate_generation` (whatif) before a real run — it's free.
- Rough: SD1 512² @20 ≈ 4 Buzz; SDXL 1024² @25 ≈ 8–10 Buzz. Cost ≈
  `base × (W·H / nativePixels) × (steps / refSteps) × quantity`.
- `generate_image` without `confirm=true` returns the price only. Check
  `get_buzz_balance()` if unsure funds exist; `insufficientBuzz` is checked before spend.

---

## 10. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `400 clipSkip not valid` | clip_skip on SDXL/Flux | remove it (MCP auto-omits; heed `warnings`) |
| Duplicated / mirrored subjects | resolution too far from native | use a §2 bucket; upscale later |
| Burnt / oversaturated / HDR halos | CFG too high (esp. v-pred) | lower CFG before raising steps |
| Mushy Turbo/LCM output | CFG/steps tuned for base | CFG 1–2, steps 4–8, `sampler="lcm"` |
| LoRA has no effect | missing trigger / wrong ecosystem / low weight | add trigger word, match ecosystem, raise weight |
| Pony looks generic / low quality | missing score stack | prepend `score_9, score_8_up, score_7_up, score_6_up` |
| Anime model won't go explicit | no rating token | add `rating_explicit` (Pony) / `rating:explicit` (Illustrious) |
| Flux ignores negative prompt | Flux has no negative | put constraints in the positive sentence |
| Bad hands on explicit poses | anatomy failure | strong hand/anatomy negatives; try another seed |
| Step `failed reason=blocked` | prompt hit moderation | change input; don't retry identical |

---

## 11. Copy-paste recipes (against the `civitai` MCP)

**Realistic SDXL portrait**
```
generate_image(prompt="a photorealistic portrait of a woman, natural light, 85mm, detailed skin",
  negative_prompt="worst quality, low quality, blurry, bad anatomy",
  model_version_id=<sdxl_ckpt_ver>, width=832, height=1216, steps=28,
  cfg_scale=6, engine="comfy", sampler="dpmpp_2m", scheduler="karras", confirm=true)
```

**Pony anime, explicit, with a LoRA**
```
generate_image(
  prompt="score_9, score_8_up, score_7_up, score_6_up, rating_explicit, source_anime, 1girl, solo, <tags>",
  negative_prompt="score_6, score_5, score_4, worst quality, bad hands, extra digits, watermark",
  model_version_id=<pony_ver>, loras={"<lora_ver>":0.8},
  width=832, height=1216, steps=28, cfg_scale=7, sampler="euler_a", confirm=true)
```

**NoobAI v-pred anime, explicit**
```
generate_image(
  prompt="1girl, solo, <danbooru tags>, rating:explicit, masterpiece, best quality, newest, absurdres, highres",
  negative_prompt="lowres, worst quality, bad anatomy, bad hands, extra digits, watermark, signature",
  model_version_id=<noobai_ver>, width=832, height=1216, steps=28,
  cfg_scale=4, sampler="euler_a", confirm=true)
```

**img2img restyle**
```
generate_image(prompt="<new style/scene>", model_version_id=<ver>,
  operation="createVariant", source_image="https://…/src.png", strength=0.65, confirm=true)
```

---

## Sources
- Pony score/source/rating syntax: civitai.com/articles/8547, /articles/4248
- Illustrious/NoobAI tips: civitai.com/articles/19107 (login), NoobAI v-pred settings
- Orchestration params & sampler/scheduler enums: developer.civitai.com/orchestration
