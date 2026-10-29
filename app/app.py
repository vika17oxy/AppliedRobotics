"""Web page: upload 30-45 photos + a short description -> 3D preview + STL download.

Start on Colab (GPU) from notebooks/web_app.ipynb; Gradio prints a public *.gradio.live link.
Look and Taurus easter eggs live in style.py.
"""
import os
import shutil
import sys
import time
import traceback

import gradio as gr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import style as S  # noqa: E402

JOBS = "/content/jobs"
DRIVE_RESULTS = "/content/drive/MyDrive/Applied_Robotics/results"  # used only if Drive is mounted
BULL_WORDS = ("bull", "taurus", "stier", "ox", "телец", "бык")


def reconstruct(files, prompt, iters):
    import pipeline as P  # heavy (torch, transformers) - only when a job starts

    log_lines = []

    def log(line):
        log_lines.append(line)

    def state(html, preview=None, model=None, stl=None):
        download = gr.DownloadButton(value=stl, visible=stl is not None)
        return html, "\n".join(log_lines[-40:]), preview, model, download

    if not files or len(files) < 15:
        yield state(S.status_html(1, "Bitte mindestens 15 Fotos hochladen (empfohlen 30–45).", "err"))
        return
    prompt = (prompt or "").strip().lower()
    if not prompt:
        yield state(S.status_html(1, "Bitte beschreiben, was auf den Fotos ist — z. B. „a figurine“, „a mug“.", "err"))
        return
    if not prompt.endswith("."):
        prompt += "."
    bull = any(w in prompt for w in BULL_WORDS)

    job = f"{JOBS}/{time.strftime('%Y%m%d_%H%M%S')}"
    paths = [f if isinstance(f, str) else f.name for f in files]
    t0 = time.time()
    preview = None
    step = 1
    try:
        yield state(S.status_html(1, "Fotos werden vorbereitet …", note="🐂 Ein Stier? Hier bist du richtig." if bull else ""))
        n = P.prepare_photos(paths, f"{job}/raw")
        if n < 15:
            yield state(S.status_html(1, f"Nur {n} brauchbare Fotos (zu klein oder kein Bild).", "err"))
            return

        step = 2
        yield state(S.status_html(2, f"Kamerapositionen werden berechnet — {n} Fotos, ca. 10–15 min …",
                                  note="COLMAP sucht gemeinsame Punkte zwischen allen Fotos."))
        reg = P.colmap(f"{job}/raw", f"{job}/processed", log)
        log(f"COLMAP: {reg}/{n} Fotos registriert")
        if reg < 0.6 * n:
            yield state(S.status_html(2, f"Nur {reg} von {n} Fotos konnten zugeordnet werden.", "err",
                                      note="Mehr Überlappung zwischen den Fotos, Untergrund mit Muster, Objekt nicht drehen."))
            return

        step = 3
        yield state(S.status_html(3, f"„{prompt[:-1]}“ wird auf jedem Foto gesucht …",
                                  note=f"{reg}/{n} Fotos zugeordnet. Grounding DINO findet das Objekt, SAM2 schneidet es aus."))
        no_box, preview = P.make_masks(f"{job}/processed", prompt, log)
        if len(no_box) > 0.3 * reg:
            yield state(S.status_html(3, f"„{prompt[:-1]}“ wurde auf {len(no_box)} von {reg} Fotos nicht gefunden.", "err",
                                      note="Andere, kurze englische Beschreibung probieren: „a toy robot“, „a mug“."), preview)
            return

        step = 4
        yield state(S.status_html(4, f"3D Gaussian Splatting wird trainiert — {iters} Iterationen, ca. 10 min …",
                                  note="Oben links siehst du, was als Objekt erkannt wurde."), preview)
        cfg = P.train(f"{job}/processed", f"{job}/outputs", iters, log)

        step = 5
        yield state(S.status_html(5, "Das Mesh wird aus den Silhouetten gebaut …"), preview)
        stl = P.hull(cfg, f"{job}/object.stl", log)

        mins = (time.time() - t0) / 60
        note = f"{reg}/{n} Fotos verwendet · wasserdicht, bereit für den Slicer."
        if os.path.isdir(DRIVE_RESULTS):  # Colab sessions die; keep every result on Google Drive
            safe = "".join(c if c.isalnum() else "_" for c in prompt[:-1]).strip("_")[:40] or "object"
            kept = f"{DRIVE_RESULTS}/{safe}_{os.path.basename(job)}.stl"
            shutil.copy(stl, kept)
            log(f"saved to Drive: {kept}")
            note += " Kopie auf Google Drive: Applied_Robotics/results/."
        if bull:
            note += " ♉ Ein Stier, rekonstruiert unter dem Zeichen des Stiers."
        yield state(S.status_html(5, f"Fertig in {mins:.0f} min", "ok", note), preview, stl, stl)
    except Exception as e:
        log(traceback.format_exc())
        yield state(S.status_html(step, f"Fehler: {e}", "err", note="Details im Log unten."), preview)


with gr.Blocks(title="Fotos → 3D-Modell ♉") as demo:
    gr.HTML(S.HERO)
    with gr.Row(equal_height=False):
        with gr.Column(scale=5):
            files = gr.File(label="Fotos (30–45)", file_count="multiple",
                            file_types=["image", ".heic", ".HEIC"], elem_classes="card")
            prompt = gr.Textbox(label="Was ist auf den Fotos? (kurz, englisch)", placeholder="a figurine",
                                elem_classes="card")
            with gr.Accordion("Einstellungen", open=False):
                iters = gr.Slider(3000, 15000, value=7000, step=1000, label="Trainingsiterationen",
                                  info="mehr = feiner, aber langsamer")
            go = gr.Button("3D-Modell erstellen", variant="primary", size="lg", elem_id="go-btn")
            gr.HTML(S.TIPS)
        with gr.Column(scale=6):
            status = gr.HTML(S.IDLE)
            model = gr.Model3D(label="Ergebnis", clear_color=[0.957, 0.949, 0.929, 1], height=420,
                               elem_classes="card")
            stl = gr.DownloadButton("STL herunterladen", visible=False, variant="primary", size="lg")
            with gr.Row():
                preview = gr.Image(label="Erkanntes Objekt", type="pil", height=170, elem_classes="card")
            with gr.Accordion("Log", open=False):
                log = gr.Textbox(show_label=False, lines=12, max_lines=12)
    gr.HTML(S.FOOT)
    go.click(reconstruct, [files, prompt, iters], [status, log, preview, model, stl])

if __name__ == "__main__":
    os.makedirs(JOBS, exist_ok=True)
    demo.queue(default_concurrency_limit=1).launch(
        share="--local" not in sys.argv, debug=True,
        theme=S.theme(), css=S.CSS, head=S.HEAD, js=S.JS,
        allowed_paths=[JOBS],  # Gradio 6 only serves files from explicitly allowed folders
    )
