# CLAUDE.md — Gruppe 7 Objektrekonstruktion

Course project (FH Technikum Wien, Applied Robotics): **photos of an object + a short text description → 3D model
(STL)**. Method is fixed by the assignment: 3D Gaussian Splatting (nerfstudio `splatfacto`), language-guided
segmentation with Grounding DINO + SAM2, export to STL; optional evaluation against a CAD reference (Chamfer).

## Working with the owner

- Answer in **Russian**, short and direct. Project documents (README, notebooks, paper) are in **German**.
- Everything heavy runs on **Google Colab (T4 GPU)**. You cannot run Colab cells: hand over **one code cell at a
  time**, wait for the pasted output, then continue.
- Don't edit a notebook while it is open in the owner's VS Code/Colab tab (a stale tab overwrites disk edits).
- Never commit input photos (`pictures/`, `figurine_raw/` are git-ignored) or notebook outputs that list the
  owner's Google Drive. The repo is public.

## Layout

| Path | What |
|---|---|
| `app/` | **Web app** (Gradio): upload 20–30 photos + description → 3D preview + STL |
| `app/colab_start.py` | One Colab cell that builds everything and starts the site (public `*.gradio.live` link) |
| `app/app.py` | UI + job flow with sanity checks; `app/style.py` = theme, CSS, header/footer, easter eggs |
| `app/pipeline.py` | Steps as functions (photo prep, COLMAP, masks, training); `app/hull.py` = visual-hull mesh |
| `notebooks/photos_to_stl.ipynb` | Same pipeline as a clean step-by-step notebook |
| `notebooks/web_app.ipynb` | Colab notebook that starts the web app |
| `pipeline_clean.ipynb` | Full experiment log (truck dataset, sparse-view tests, Chamfer, ABO, figurine) |
| `results/` | STL results (truck, figurine, dragon, ABO chair) |
| `eval/chamfer_eval.py` | Align reconstruction to a CAD STL (similarity + symmetric ICP) and report Chamfer / F-score |
| `PROJECT_STATUS.md` | Detailed status, numbers, known problems, open items |
| `paper/` | Scientific paper (IEEE template, 3 pages): full text, figures, CAD evaluation; `main.pdf` compiled |

## Pipeline (what works and why)

1. **Photos**: EXIF-transpose (iPhone), downscale to 1600 px (COLMAP runs on CPU), drop < 1000 px thumbnails.
2. **COLMAP** via `ns-process-data images --matching-method exhaustive --no-gpu` with `QT_QPA_PLATFORM=offscreen`
   (Colab's apt COLMAP has no display/CUDA). Own figurine photos: 43/43 registered.
3. **Masks**: Grounding DINO box from the prompt → SAM2 (`Sam2Model`/`Sam2Processor`, transformers, in the
   kernel). HF `Sam2Processor` has no `reshaped_input_sizes`; `post_process_masks` takes only `original_sizes`.
4. **Training**: `ns-train splatfacto … colmap --colmap-path colmap/sparse/0 --masks-path masks --downscale-factor 1`.
5. **Mesh** (`hull.py`): masks stop background Gaussians from being *trained* but don't delete them (they come
   from the room-wide COLMAP init), so plain DBSCAN + Poisson fails on real photos. Instead, in a 288³ GPU voxel
   grid: silhouettes (inside the mask in ≥ 95 % of views) → cut below the **floor plane** (RANSAC on background
   Gaussians) → free space from **rendered 3DGS depth** (≥ 80 % of trusted views see past the voxel, margin 6
   voxels) → opening, largest component, fill holes → blurred field → marching cubes → watertight STL.
   `--variants` writes a parameter sweep. Run with `TORCHDYNAMO_DISABLE=1` (torch.compile fails on Colab).
   Cameras and Gaussians come from the same `eval_setup` pipeline (same coordinate frame; nerfstudio cameras are
   OpenGL: z backwards, y up). Units are arbitrary (COLMAP has no scale).

### Environment (exact versions matter)

nerfstudio lives in a separate venv `/content/nsenv` (Python 3.10) because the Colab kernel is too new:
`torch==2.1.2+cu118`, `gsplat==1.4.0+pt21cu118` (prebuilt wheel; source build OOMs), `setuptools<70`,
`nerfstudio==1.1.5`, then **`numpy==1.26.4` last**. `/content` is wiped on every runtime reset.

## Web app design

Palette (in `app/style.py`): off-white `#F4F2ED`, white, dark grey `#2D3034`, light blue `#A8CBE6`/`#7FB0D6`,
small accents of burgundy `#7B2232` and emerald `#1E7B5E`. Headings Cormorant Garamond, body Inter.
Gradio 6: `theme`, `css`, `js`, `head` go to `launch()`, not `gr.Blocks()`. Gradio prefixes custom CSS with
`.gradio-container .contain` — never target `body`; toggle classes on `.hero` instead.

**Taurus easter eggs** (the owner is a Taurus ♉ — keep them, keep them subtle, add more in the same spirit):
click ♉ in the header twice → Taurus card (emerald = Taurus birthstone, which is why "done" is emerald);
hover the burgundy star → Aldebaran; Konami code → "Venus mode" with drifting ♉/♀; 20 Apr – 20 May → emerald
"Taurus Season" chip; the running spinner is a turning ♉; prompts like "bull"/"taurus"/"stier"/"телец" get a
special line; greeting in the browser console. Use `♉︎` (U+2649 + U+FE0E) so it renders as text, not emoji.

## Open items

- CAD evaluation is done on the printed dragon page holder (`eval/chamfer_eval.py`, v4: Chamfer 1.01 mm vs the
  whole object); the reference STL is third-party (MakerWorld) and stays out of git. Multi-part objects need a
  multi-phrase prompt ("a gold dragon. a black plate"). White/plain backgrounds make COLMAP fail.
- Scale the figurine STL to millimetres once the real height is known.
- Run the web app end-to-end once on Colab (built and UI-tested locally, not yet with a real job).
- Paper: keep numbers in sync with README when results change (MiKTeX is installed locally for builds).
