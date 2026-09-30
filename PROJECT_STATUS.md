# Gruppe 7 — Objektrekonstruktion (Neural Rendering, VLM) — Projektstatus

_Übergabedokument für die Weiterarbeit durch den Projektpartner. Stand: 17.09.2026._

---

## 0. Über das Projekt

Kursprojekt an der **FH Technikum Wien**, im Zweierteam bearbeitet. Die ursprüngliche
Frist für den Code war der 15.09.2026 — sie wurde verschoben (genaues neues Datum bitte
im Team klären). Das gesamte Projekt ist auf **3 Wochen** angelegt.

### Ziel (Endergebnis)

Die Nutzerin fotografiert ein beliebiges physisches Objekt aus mehreren Blickwinkeln
(**5–30 Fotos, weniger ist besser für die Bewertung der Methode**) und gibt eine
Textbeschreibung des Objekts an (zum Beispiel "the toy truck"). Das Programm soll:

1. Das Objekt aus diesen Fotos in 3D rekonstruieren.
2. Automatisch genau das beschriebene Objekt segmentieren und den Hintergrund entfernen.
3. Das Ergebnis als **STL** exportieren.
4. (Optional, ca. 10 % der Bewertung, "Evaluierung") Falls ein CAD-Referenzmodell
   existiert: die STL-Geometrie damit vergleichen (Chamfer-Distanz oder ähnlich).

### Geforderte Pipeline-Architektur (Methode ist fest vorgegeben)

```
Fotos (5–30) → Kameraposen schätzen (COLMAP, oder vorgegebene Posen)
   → 3D-Gaussian-Splatting-Szene trainieren (nerfstudio / splatfacto)
   → parallel: sprachgesteuerte Segmentierung — Grounding DINO + SAM2 → 2D-Maske per Text
   → Masken auf die Eingabebilder anwenden, VOR/WÄHREND dem Training (Hintergrund wird
     nie trainiert)
   → Mesh aus den gefilterten Gaussians extrahieren (ns-export gaussian-splat + Poisson)
   → Mesh nachbearbeiten (Open3D/trimesh: Bereinigung, Decimation, Watertight-Fix)
   → als STL exportieren
   → (optional) Chamfer-Distanz vs. CAD-Referenzmodell
```

**Methode: ausschließlich 3D Gaussian Splatting (NICHT NeRF, kein Methodenvergleich).**

---

## 1. Status nach Punkten der Aufgabenstellung

| # | Schritt | Status | Kommentar |
|---|---|---|---|
| 1 | 3DGS-Rekonstruktion (`ns-train splatfacto`) | ✅ fertig | mehrfach getestet, 7000 und 30000 Iterationen |
| 2 | Sprachgesteuerte Segmentierung (Text → Maske) | ✅ fertig (SAM2) | Grounding DINO + **SAM2**, erfolgreich und mehrfach reproduziert: 251/251 Masken |
| 3 | Masken werden vor dem Training angewendet | ✅ fertig | `ns-train ... --masks-path masks` |
| 4 | Mesh-Extraktion + STL-Export | ✅ fertig | `ns-export gaussian-splat` → Poisson → STL |
| 5 | Evaluierung (Chamfer-Distanz) | ⚠️ teilweise | siehe Abschnitt 5 — kein externes CAD-Modell verfügbar, stattdessen Selbstvergleich (Proxy) durchgeführt |

**Gesamtbild:** Die Pipeline funktioniert technisch vollständig und wurde mehrfach
reproduziert — aber bisher nur am bequemen Test-Datensatz **`truck`** (Tanks & Temples,
**251** dichte Aufnahmen), nicht am eigentlichen Zielszenario der Aufgabe (5–30 Fotos
eines beliebigen, selbst fotografierten Objekts). Was das für die Praxis bedeutet, siehe
Abschnitt 4 (Sparse-View-Test) und Abschnitt 8 (offene Punkte).

---

## 2. Infrastruktur und Umgebung

- **Umgebung:** Google Colab, GPU **Tesla T4** (15 GB), über die Colab-Erweiterung mit
  **VS Code** verbunden. Kernel des Notebooks: System-Python 3.13 / torch 2.11 / CUDA 12.8.
- **Kaggle als Alternative:** Das kostenlose GPU-Kontingent von Colab ist begrenzt und war
  mehrfach erschöpft. Kaggle Notebooks bieten ebenfalls kostenlos GPU (T4×2, ca.
  30 Stunden/Woche) — Einstellungen dort: Accelerator = GPU T4×2, Internet = An. Wurde als
  Ausweichoption vorbereitet, aber am Ende meist doch mit Colab weitergearbeitet.
- **Kernproblem:** Nerfstudio 1.1.5 ist nicht direkt mit Python 3.13/torch 2.11
  kompatibel. Lösung: **eigenes venv `/content/nsenv`** mit Python 3.10; alle
  `ns-*`-Befehle laufen als Unterprozess über `/content/nsenv/bin/ns-train`.
- **Die Segmentierung (Grounding DINO + SAM2)** läuft dagegen **im Kernel** (Python 3.13 /
  torch 2.11), nicht im venv — dort wird eine aktuelle torch-Version gebraucht.
- **`/content` ist flüchtig — das ist das größte praktische Problem dieses Projekts.** Bei
  jedem Neustart/Timeout der Laufzeitumgebung geht *alles* verloren: venv, Datensatz,
  Masken, Trainingsergebnisse. Das ist über mehrere Tage hinweg **sehr häufig** passiert
  (nicht nur beim expliziten Neustart, auch scheinbar "von selbst" zwischen zwei
  Nachrichten im Chat, vermutlich durch Colab-Idle-Timeouts). Ein Neuaufbau von venv +
  Datensatz + Masken dauert **ca. 20–25 Minuten**.
- **Wichtigste Lehre daraus (siehe auch Abschnitt 6):** *Nicht* versuchen, das große venv
  (ca. 3,8 GB) als ein einziges Archiv auf Google Drive zu sichern — das ist mehrfach
  fehlgeschlagen, weil Colabs Drive-Mount große Dateien nur lokal gepuffert schreibt und
  bei Sitzungsende der Sync abbricht, ohne Fehlermeldung. Stattdessen: venv **bei jedem
  Neustart neu aufbauen** (automatisiert, ~10 Min., unkritisch), aber **kleine, wertvolle
  Zwischenergebnisse** (Masken, `.ply`-Exporte, fertige `.stl`-Dateien — jeweils nur
  einige MB) **sofort nach dem Entstehen** einzeln auf Drive kopieren und die
  Dateigröße direkt danach verifizieren.
- Es gibt ein **eigenständiges Notebook `C:\Users\HP\Downloads\pipeline_clean.ipynb`** mit
  Schritten, die sich selbst überspringen, wenn sie schon erledigt sind. Die Schritte
  darin sind als **Teil 1 bis Teil 8** durchnummeriert. In der Praxis wurde in den
  letzten Tagen aber oft direkt im Chat mit Claude Schritt für Schritt gearbeitet
  (Ad-hoc-Zellen), nicht immer über dieses Notebook synchronisiert — die Notebook-Datei
  ist also nicht zu 100 % aktuell gegenüber dem, was tatsächlich zuletzt lief. Beim
  Weiterarbeiten am besten: aktuelle Zell-Inhalte aus diesem Dokument (Abschnitt 3–5)
  nehmen, nicht blind dem Notebook-Stand vertrauen.

### Rezept für das venv (Python 3.10 + nerfstudio), in Ausführungsreihenfolge

```bash
pip install -q uv
uv python install 3.10
uv venv --python 3.10 /content/nsenv
uv pip install --python /content/nsenv/bin/python "torch==2.1.2" "torchvision==0.16.2" --index-url https://download.pytorch.org/whl/cu118
uv pip install --python /content/nsenv/bin/python "setuptools<70" wheel
uv pip install --python /content/nsenv/bin/python "gsplat==1.4.0+pt21cu118" --index-url https://docs.gsplat.studio/whl/pt21cu118 --extra-index-url https://pypi.org/simple --index-strategy unsafe-best-match
uv pip install --python /content/nsenv/bin/python "nerfstudio==1.1.5" open3d trimesh
uv pip install --python /content/nsenv/bin/python "numpy==1.26.4"
```

| Paket | Version | Warum genau so |
|---|---|---|
| Python (venv) | 3.10.21 | fertige gsplat-Wheels gibt es nur für cp310 |
| torch/torchvision | 2.1.2+cu118 / 0.16.2+cu118 | Version, für die Nerfstudio 1.1.5 geschrieben ist |
| setuptools | `<70` | torch 2.1.2 importiert in `cpp_extension.py` `pkg_resources.packaging`, das ab setuptools ≥70 fehlt |
| gsplat | `1.4.0+pt21cu118`, vorkompiliertes Wheel | Build aus dem Quellcode führt auf dem Colab-RAM zu OOM-Kill (exit 137); Installation mit `--index-strategy unsafe-best-match`, sonst findet uv das Wheel wegen Dependency-Confusion-Schutz nicht |
| nerfstudio | 1.1.5 | eine andere Auflösung fällt auf die alte Version 0.1.15 zurück, scheitert an `aiohttp==3.8.1` unter Python 3.13 |
| numpy | `1.26.4`, als LETZTES installieren | Abhängigkeiten ziehen sonst numpy 2.x → `_ARRAY_API not found` beim Import von torch |

### Datensatz

`truck` aus Tanks & Temples:
```bash
wget https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/input/tandt_db.zip -O /content/tandt_db.zip
cd /content && unzip -q -o tandt_db.zip
```
→ `/content/tandt/truck/images/` (251 Aufnahmen, 979×546) + `/content/tandt/truck/sparse/0/`
(fertiges COLMAP-Modell).

**Fix für die Kameraparameter (für diesen Datensatz nötig, nur für den vollen
251-Bilder-Datensatz mit mitgeliefertem COLMAP-Modell — nicht nötig, wenn COLMAP-Posen
selbst neu berechnet werden, siehe Abschnitt 4):** `images/` ist auf 979×546 verkleinert,
aber `cameras.bin` (PINHOLE) enthält die Parameter der vollen Größe 1957×1091 →
`AssertionError` in nerfstudio. Behoben durch ein Skript mit
`nerfstudio.data.utils.colmap_parsing_utils` (`read_cameras_binary`/`write_cameras_binary`/
`Camera`): Breite/Höhe und fx, fy, cx, cy werden mit ~0.5 multipliziert, das Original wird
als `cameras.bin.orig` gesichert.

---

## 3. Was technisch umgesetzt wurde

### Teil 1 — Grundlauf ohne Segmentierung (ganze Szene, 7000 Iterationen)

```bash
yes | TORCHDYNAMO_DISABLE=1 /content/nsenv/bin/ns-train splatfacto \
  --data /content/tandt/truck --max-num-iterations 7000 --steps-per-save 2000 \
  --output-dir outputs --pipeline.model.cull-alpha-thresh 0.1 \
  colmap --images-path images --colmap-path sparse/0 --downscale-factor 1
```
Ca. 5 Minuten auf T4. → 466 517 Gaussians → Poisson (`depth=9`) → STL
`truck_scene.stl`: 115 479 Ecken / 228 999 Dreiecke, nicht wasserdicht (offene Kanten der
gesamten Szene, kein Hintergrund entfernt). Frühester Prototyp, nur zur Machbarkeitsprüfung.

### Teil 2 — Objektsegmentierung mit Grounding DINO + SAM2

Läuft **im Kernel** von Colab (transformers, aktuelle torch-Version). Ursprünglich mit
**SAM v1** umgesetzt, dann auf **SAM2** umgestellt, weil die Aufgabenstellung SAM2 explizit
vorgibt — SAM2 ist die aktuell verwendete, produktive Version:

- `IDEA-Research/grounding-dino-base` — Bounding Box per Text (Standard: `"a truck."`,
  Fallback `"object. vehicle. truck."` bei leerem Ergebnis), Schwelle 0.3/0.22, Boxen
  außerhalb von 2–99 % der Bildfläche werden verworfen.
- `facebook/sam2-hiera-large` (`Sam2Model`/`Sam2Processor`, HuggingFace `transformers`) —
  Maske pro Box (mehrere Kandidaten, der mit höchstem IoU wird genommen), Masken mehrerer
  Boxen werden vereinigt.
- **API-Unterschied zu SAM v1** (wichtig, falls der Code nochmal angepasst wird):
  `Sam2Processor` liefert — anders als `SamProcessor` bei SAM v1 — keinen Schlüssel
  `reshaped_input_sizes`. `post_process_masks(masks, original_sizes, ...)` braucht dort
  nur `original_sizes`.
- Ergebnis (mehrfach reproduziert, zuletzt 17.09.2026): **251/251 Masken**, 0 Aufnahmen
  ohne Box, ca. 6 Minuten auf T4.
- **Format-Hinweis:** Der Nerfstudio-1.1.5-COLMAP-Parser baut den Maskenpfad als
  `data/masks_path/<image_name>` **mit erzwungenem `.with_suffix(".png")`** — die Maske
  muss `<stem>.png` heißen, unabhängig von der Dateiendung des Originalbildes.

**Vollständiger Code** (Modelle laden + Masken erzeugen) liegt in `pipeline_clean.ipynb`,
Teil 4, und ist außerdem als Ad-hoc-Zellen mehrfach im Chat-Verlauf dokumentiert.

### Teil 3 — Training mit Masken + Mesh nur vom Objekt

Gleicher Befehl wie Teil 1, zusätzlich `--masks-path masks`:
```bash
yes | TORCHDYNAMO_DISABLE=1 /content/nsenv/bin/ns-train splatfacto \
  --data /content/tandt/truck --max-num-iterations 7000 --steps-per-save 2000 \
  --output-dir outputs_masked --pipeline.model.cull-alpha-thresh 0.1 \
  colmap --images-path images --colmap-path sparse/0 --masks-path masks --downscale-factor 1
```

**Mesh-Skript** (Open3D + trimesh, Vorlage für alle weiteren Läufe in diesem Projekt):
```python
pcd = o3d.io.read_point_cloud(ply)
pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
# größter zusammenhängender Cluster per DBSCAN entfernt Ausreißer/Hintergrundreste:
nn_dists = [...]  # Median-Nachbarabstand über eine Stichprobe der Punkte
labels = np.array(pcd.cluster_dbscan(eps=5*median_nn, min_points=10))
pcd = pcd.select_by_index(np.where(labels == np.bincount(labels[labels>=0]).argmax())[0])
pcd.estimate_normals(...); pcd.orient_normals_consistent_tangent_plane(30)
mesh, dens = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=10)
mesh.remove_vertices_by_mask(dens <= np.quantile(dens, 0.1))
# + remove_degenerate_triangles / remove_duplicated_vertices / remove_duplicated_triangles
# + remove_non_manifold_edges / compute_triangle_normals
# danach in trimesh: fill_holes() + fix_normals() + process(validate=True)
```

**Es gibt mehrere Versionen dieses "vollständigen" Ergebnisses aus verschiedenen
Sitzungen — wichtig für die Übergabe, welche davon noch existiert:**

| Datei | Segmentierung | Iterationen | Ecken / Dreiecke | Bounding Box | Noch vorhanden? |
|---|---|---|---|---|---|
| `truck_object_30k.stl` | SAM v1 | 30 000 | 176 537 / 342 342 | 1.055 × 0.426 × 0.325 | ✅ ja — lokal `C:\Users\HP\Downloads\` und Drive `nerf_day2_30k/` |
| `truck_object.stl` (14.09.) | SAM2 | 7 000 | 216 950 / 421 559 | 1.068 × 0.434 × 0.365 | ❌ nein — nur in Colab existiert, nie gesichert, durch Sitzungsverlust weg. Nur die Zahlen sind noch bekannt. |
| `truck_object_ref.stl` (17.09., aktuell) | SAM2 | 7 000 | 222 017 / 430 796 | *(nicht separat gemessen)* | ✅ ja — Drive `nerf_cache/truck_object_ref.stl` |

Alle drei sind sich in Proportion und Gaußzahl sehr ähnlich (SAM2 reproduziert SAM v1
zuverlässig) — als **aktuelle Referenz für Weiterarbeit `truck_object_ref.stl` verwenden**,
das ist die einzige aktuelle SAM2-Version, die tatsächlich noch als Datei existiert.
`watertight=False` bei allen drei: die Unterseite des Objekts wurde nie fotografiert,
Poisson kann sie nicht ergänzen. Reparatur wäre ein manueller Schritt in
Blender/MeshLab/Meshmixer, nicht in Colab gemacht (siehe Abschnitt 8).

*(Hinweis: Die installierte `trimesh`-Version hat keine Methoden
`remove_duplicate_faces()`/`remove_degenerate_faces()` — stattdessen
`mesh.process(validate=True)` verwenden.)*

### Teil 4 — Interaktiver 3D-Viewer

Ein Artifact wurde veröffentlicht (Three.js, STL als Vertex-Positionen in Base64-JSON, da
`.stl` nicht direkt als Asset hochgeladen werden kann) — zeigt `truck_object_30k.stl`
(die SAM-v1-Version), mit der Maus drehbar, mit Pipeline-Schema und einem ehrlichen
Hinweis, was das Ergebnis ist und was nicht (kein trainiertes Bild→CAD-Modell, sondern
eine Optimierung pro Szene; STL ist kein parametrisches CAD).
URL: `https://claude.ai/code/artifact/c1549258-aad1-45cf-b776-7d2af0b0972c` (nur
innerhalb der Claude-Organisation zugänglich — falls der Kollege dort keinen Zugriff hat,
stattdessen direkt eine `.stl`-Datei schicken, z. B. mit einem lokalen STL-Viewer oder
Blender öffnen).

---

## 4. Sparse-View-Test: funktioniert die Pipeline mit 5–30 Fotos?

Die Aufgabenstellung zielt auf 5–30 Fotos ab; Teil 1–4 oben wurden mit 251 geprüft.

### 4a. Erster Test: vorgegebene Posen, nur subsampled (12/24 Fotos)

Aus den exakten Posen des vollständigen 251-Bilder-COLMAP-Modells wurden **12 und 24
Aufnahmen** ausgewählt (bewusst mit vorhandenen, "geschummelten" Posen — um die Frage
"hält 3DGS+Maske bei wenigen Blickwinkeln?" von der separaten Frage "findet COLMAP selbst
Posen bei wenigen echten Fotos?" zu trennen).

| Aufnahmen | Maske | Gaussians | Nach DBSCAN | Bounding Box (L×B×H) | Bewertung |
|---|---|---|---|---|---|
| 12 | nein | 339 218 | 83 320 | 1.70 × 1.61 × 0.61 | unlesbarer Fleck, mit dem Boden verschmolzen |
| 12 | ja | 84 703 | 42 879 | 1.22 × 0.51 × 0.40 | Proportionen erkennbar, aber grob (~15–20 % Abweichung) |
| 24 | ja | 104 500 | 55 276 | 1.16 × 0.50 × 0.39 | am nächsten am Referenzwert (~9–15 % Abweichung) |
| 251 (Referenz) | ja | 116 351 | 69 980 | 1.06 × 0.43 × 0.33 | sauber |

**Schlussfolgerungen:** Die Maske ist bei wenigen Aufnahmen unverzichtbar (ohne sie
zerfällt die Geometrie komplett). Mit Maske steigt die Genauigkeit mit der Anzahl der
Aufnahmen. 12 Fotos (untere Grenze der Aufgabenstellung) sind bereits problematisch —
Empfehlung für den Bericht: **20–30 Fotos** anstreben, nicht das Minimum von 5.

### 4b. Zweiter Test: echtes COLMAP ohne vorgegebene Posen (20/30 Fotos)

Das oben offen gelassene Risiko wurde geprüft: `ns-process-data images
--matching-method exhaustive --no-gpu` (mit `QT_QPA_PLATFORM=offscreen`, siehe Abschnitt 6
für die Fehlerursachen) auf zufällig ausgewählten Rohbildern aus dem 251-Bilder-Datensatz,
**ganz ohne vorgegebene Posen** — das entspricht am ehesten dem echten Nutzungsszenario.

| Fotos | Versuch | Von COLMAP registriert | Quote |
|---|---|---|---|
| 20 | 1 (15./16.09.) | 15 | 75,0 % |
| 20 | 2 (17.09., andere Sitzung) | 4 | 20,0 % |
| 30 | 1 (15./16.09.) | 22 | 73,3 % |
| 30 | 2 (17.09., andere Sitzung) | 22 | 73,3 % |

**Zwei wichtige Ergebnisse:**
1. Von 20 auf 30 Fotos zu erhöhen hat die Registrierungsquote **nicht** verbessert
   (~75 % bei beiden). Das Problem liegt nicht primär an der Foto-**Anzahl**, sondern
   daran, dass eine **zufällige** Teilmenge aus einer dichten, video-artigen Aufnahme
   große Sprünge im Blickwinkel/Belichtung zwischen benachbarten Bildern hat — genau das
   bringt COLMAP-Matching zum Scheitern. Eine echte Nutzerin, die bewusst gut verteilte
   Fotos rund um ein Objekt aufnimmt, sollte besser abschneiden — **aber das ist nicht
   getestet**, würde eine echte Fotoaufnahme brauchen (siehe Abschnitt 8).
2. **COLMAP ist bei n=20 nicht deterministisch:** derselbe 20-Bilder-Satz ergab beim
   zweiten Versuch nur 4/20 statt 15/20 registrierte Bilder (RANSAC-basierte
   Verfahrensschritte in COLMAP sind nicht exakt reproduzierbar). Bei n=30 war das Ergebnis
   dagegen beide Male identisch (22/30) — stabiler, vermutlich weil mehr Bildpaare mehr
   Redundanz für die Bündelausgleichung liefern. **Für den Bericht wichtig:** ein einzelner
   Testlauf ist nicht unbedingt repräsentativ, v. a. bei kleinerem n.

---

## 5. Evaluierung (Chamfer-Distanz)

Die Aufgabenstellung sieht optional (~10 % der Bewertung) einen Vergleich der
rekonstruierten STL-Geometrie mit einem **externen CAD-Referenzmodell** vor. Das war
nicht direkt umsetzbar:

- Kein physisches Objekt mit auffindbarem CAD/STL-Modell war verfügbar.
- Zufällige Internetfotos verschiedener LKWs wurden bewusst verworfen — Photogrammetrie
  braucht mehrere Fotos **desselben** physischen Objekts, nicht verschiedener Exemplare.
- Der wissenschaftliche Standard-Benchmark für sowas (DTU-MVS-Datensatz) erfordert einen
  manuellen Download von einer älteren Uni-Webseite plus ein eigenes Ausrichtungsprotokoll
  (ObsMask, ICP mit vorgegebener Transformation) — als zu aufwändig/riskant eingeschätzt
  angesichts der ohnehin sehr instabilen Colab-Umgebung (siehe Abschnitt 2).
- Selbst ein CAD-Modell herunterladen und daraus synthetische Fotos rendern (z. B. mit
  Open3D headless) wäre möglich gewesen, birgt aber ein ähnliches Risiko wie die
  OpenGL-Probleme, die schon beim COLMAP-Test aufgetreten sind (Abschnitt 6).

**Gewählter Ersatz-Ansatz (Selbstvergleich/Proxy, mit der Nutzerin abgestimmt):** Die
volle 251-Foto-Rekonstruktion (`truck_object_ref.stl`) dient als Pseudo-Referenz. Die
Sparse-Rekonstruktion aus dem 30-Foto-COLMAP-Test (Abschnitt 4b, 22 registrierte Bilder,
3000 Iterationen, `sparse30_object.stl`, 126 168 Ecken / 245 182 Dreiecke) wird damit
verglichen.

**Methode:**
1. Von beiden Meshes je ~30 000 Oberflächenpunkte sampeln.
2. Zentrieren und auf Einheitsradius normalisieren (nötig, weil jede COLMAP-Rekonstruktion
   einen willkürlichen eigenen Maßstab/Ursprung hat).
3. Gierbrute-Force-Suche über Rotationswinkel um die Hochachse (0–345° in 15°-Schritten),
   jeweils mit ICP verfeinert, bester Winkel nach ICP-RMSE gewählt (hier: 150°).
4. Finaler ICP-Verfeinerungsschritt.
5. Bidirektionale Chamfer-Distanz: mittlerer nächster-Nachbar-Abstand in beide Richtungen.

**Ergebnis:** ICP-Inlier-RMSE nach Ausrichtung = **0,0263** (die gut zueinander passende
Kernform ist auf ~2,6 % der Objektgröße genau ausgerichtet). Bidirektionale
**Chamfer-Distanz = 0,2497** (normalisiert auf die Objektgröße) — deutlich höher als die
RMSE, das heißt: die Grundform stimmt gut überein, aber ein erheblicher Teil der
Sparse-Geometrie (Rauschen, fehlende/zusätzliche Bereiche) weicht stark von der Referenz
ab. Passt zur qualitativen Einschätzung aus Abschnitt 4a ("grob, aber LKW-förmig").

**Wichtig für den Bericht:** Das ist **kein** Vergleich mit einem externen CAD-Modell wie
im Aufgabentext beschrieben, sondern ein Selbstkonsistenz-Test gegen die eigene beste
Rekonstruktion. Im Bericht sollte diese Einschränkung transparent benannt werden (siehe
auch Abschnitt 8, Punkt 2).

---

## 6. Bekannte Probleme (Lösungen geprüft, keine Zeit erneut investieren)

| Symptom | Ursache | Lösung |
|---|---|---|
| `pip install nerfstudio` installiert 0.1.15, scheitert an `aiohttp==3.8.1` | Python 3.13 im Kernel | venv mit Python 3.10 + `nerfstudio==1.1.5` explizit |
| `AssertionError: image (979,546) != camera (1957,1091)` | Bilder verkleinert, cameras.bin nicht | cameras.bin neu schreiben (Abschnitt 2) |
| `torch._dynamo`-Fehler beim Trainingsstart | JIT-Kompilierung des Graphen | `TORCHDYNAMO_DISABLE=1` |
| `ModuleNotFoundError: pkg_resources` / `from pkg_resources import packaging` | setuptools ≥70 | `setuptools<70` im venv |
| `ninja: build stopped` / `FAILED [code=137] Killed` beim Import von gsplat | Build der CUDA-Kernel aus dem Quellcode → OOM | Python 3.10 + vorkompiliertes Wheel `gsplat==1.4.0+pt21cu118` |
| uv: `No solution ... gsplat==1.4.0+pt21cu118` | Schutz von uv vor Dependency-Confusion | `--index-strategy unsafe-best-match` |
| `Failed to initialize NumPy: _ARRAY_API not found` | numpy 2.x, torch für 1.x gebaut | `numpy==1.26.4` als letzten Schritt installieren |
| `AttributeError: 'Trimesh' object has no attribute 'remove_duplicate_faces'` | Methode in dieser trimesh-Version umbenannt/entfernt | `mesh.process(validate=True)` verwenden |
| `KeyError: 'reshaped_input_sizes'` bei SAM2-Maskenerzeugung | `Sam2Processor` liefert diesen Schlüssel nicht (anders als SAM-v1-`SamProcessor`) | `post_process_masks(masks, original_sizes)` ohne den dritten Parameter aufrufen |
| "Server Colab GPU T4 has been removed" / "Failed to start the Kernel" / Buttons reagieren nicht | bekannter Fehler vscode-jupyter #17094, oder die Laufzeitumgebung wurde wirklich beendet | Notebook-Tab schließen und neu öffnen; wenn das nicht hilft — neue Laufzeitumgebung, alles neu aufbauen |
| "Unable to assign server. Insufficient quota" | kostenloses GPU-Kontingent von Colab aufgebraucht | warten (Kontingent erneuert sich meist am nächsten Tag) oder auf Kaggle ausweichen |
| `colmap feature_extractor` stürzt ab: `qt.qpa.plugin: Could not load ... "xcb"` | Qt-Plugin-Konflikt mit den Qt-Dateien von `opencv-python` in der headless-Umgebung (kein X11-Display) | `QT_QPA_PLATFORM=offscreen` vor dem Befehl setzen |
| `colmap feature_extractor` stürzt ab: `Check failed: context_.create()` (OpenGL) | `--SiftExtraction.use_gpu 1` (Standard) versucht einen OpenGL-Kontext zu öffnen; das apt-Paket `colmap` hat kein CUDA, headless gibt es keinen OpenGL-Kontext | zusätzlich `--no-gpu` an `ns-process-data` übergeben |
| `nerfstudio-data`-Parser lehnt `--masks-path` ab | Dieses Argument gibt es nur beim `colmap`-Parser, nicht bei `nerfstudio-data` | bei einem bereits mit `ns-process-data` verarbeiteten Ordner den `colmap`-Parser mit Pfad `colmap/sparse/0` verwenden (nicht `nerfstudio-data`) |
| Großer Cache-Ordner (venv, 3,8 GB) auf Drive "erfolgreich gespeichert", aber beim nächsten Mal weg | Colabs Drive-FUSE-Mount schreibt zunächst nur lokal gepuffert; endete die Laufzeitumgebung vor Sync-Ende, ist die Datei verloren, obwohl `os.path.getsize()` lokal Erfolg zeigte | **nur kleine Dateien (Masken, `.ply`, `.stl`, jeweils einzeln, wenige MB) auf Drive cachen**, das venv selbst jedes Mal neu bauen (siehe Abschnitt 2) |
| STL-Datei nach `o3d.io.write_triangle_mesh` leer/kaputt, `trimesh.load()` gibt eine `Scene` statt `Trimesh` zurück | `mesh.compute_triangle_normals()` vor dem Schreiben vergessen | diesen Aufruf immer vor `write_triangle_mesh` einfügen |

---

## 7. Ordner für die Übergabe an den Kollegen

Das ist die Liste, was übergeben werden sollte und wo es liegt.

### Google Drive

| Ordner | Inhalt | Wichtig? |
|---|---|---|
| `MyDrive/nerf_cache/` | **Aktuellster Stand (17.09.):** `masks_251.tar.gz` (alle 251 SAM2-Masken, 750 KB), `splat_251_ref.ply` (Gaussian-Export der vollen Referenz, 29,8 MB), `truck_object_ref.stl` (aktuelle Referenz-STL, SAM2, 7000 Iter.), `sparse30_processed.tar.gz` (COLMAP-Ergebnis für den 30-Foto-Sparse-Test), `splat_sparse30.ply`, `sparse30_object.stl` (Sparse-Rekonstruktion, 30 Fotos) | ⭐ **Wichtigster Ordner, hier weiterarbeiten** |
| `MyDrive/nerf_day1/` | Erster Prototyp, ganze Szene ohne Maske, 7000 Iter. | niedrig — nur zur Historie |
| `MyDrive/nerf_day2/` | Erster maskierter Lauf (SAM v1), 7000 Iter. | niedrig — überholt |
| `MyDrive/nerf_day2_30k/` | SAM-v1-Referenzlauf, 30000 Iter., `truck_object_30k.stl` | mittel — bisher bester/vollständigster Einzellauf, aber mit altem SAM v1 |

### Lokal (`C:\Users\HP\Downloads\...`)

| Datei/Ordner | Inhalt | Wichtig? |
|---|---|---|
| `pipeline_clean.ipynb` | Aufbau-Notebook (venv, Datensatz, Segmentierung, Training, Export, Mesh), auf Deutsch, Teil 1–8. **Nicht zu 100 % aktuell** gegenüber dem letzten Chat-Stand (siehe Abschnitt 2). **Liegt in `C:\Users\HP\Downloads\`, NICHT in `AppliedRobotics\`** — erscheint deshalb nicht automatisch im VS-Code-Explorer, wenn nur der `AppliedRobotics`-Ordner geöffnet ist; manuell öffnen mit Strg+O und vollem Pfad | hoch — Ausgangspunkt zum Weiterarbeiten |
| `truck_positions.json` (in `AppliedRobotics\`) | Hilfsdaten für den 3D-Viewer (Base64-Vertexpositionen für das Artifact) | **nicht übergeben nötig**, kein Pipeline-Code |
| `truck_object_30k.stl` | Lokal heruntergeladene Kopie des SAM-v1-30k-Ergebnisses | mittel |
| `AppliedRobotics\PROJECT_STATUS.md` | Dieses Dokument | hoch — zuerst lesen |
| `pipeline.ipynb` | Alter, sehr großer Verlaufs-Notebook (SAM v1, unaufgeräumt) | **nicht übergeben**, nicht mehr relevant |

### Wichtig: Drive-Freigabe nicht vergessen

Die obigen Google-Drive-Ordner sind nur für das eigene Google-Konto sichtbar. **Vor der
Übergabe die Ordner `nerf_cache/`, `nerf_day1/`, `nerf_day2/`, `nerf_day2_30k/` (oder
gleich den ganzen `MyDrive`-Bereich, falls einfacher) explizit mit dem Google-Konto des
Kollegen teilen** (Rechtsklick → Freigeben) — sonst kann er trotz korrekter Pfadangaben
nichts davon öffnen.

### Sonstiges

- **3D-Viewer (Artifact):** `https://claude.ai/code/artifact/c1549258-aad1-45cf-b776-7d2af0b0972c`
  — zeigt `truck_object_30k.stl`. Nur innerhalb der Claude-Organisation sichtbar; falls
  der Kollege keinen Zugriff hat, ihm stattdessen eine `.stl`-Datei direkt schicken.
- **Claude-Code-Gedächtnis** (übersteht einen Kontext-Reset des Assistenten, nur auf
  diesem Rechner/Account nutzbar): `nerfstudio-colab-splatfacto-recipe` und
  `project-gruppe7-objektrekonstruktion` unter `~/.claude/projects/.../memory/`.

---

## 8. Was noch zu tun bzw. zu prüfen ist

1. **Echter Fototest mit einem selbst fotografierten Objekt.** Alles bisher lief auf dem
   `truck`-Testdatensatz. Der eigentliche Use Case der Aufgabe — eine Nutzerin fotografiert
   5–30 Mal ein beliebiges reales Objekt — wurde noch nie end-to-end mit echten, frisch
   aufgenommenen Fotos getestet, nur mit Teilmengen aus einem bestehenden dichten
   Datensatz. Erwartung laut Abschnitt 4b: echte, bewusst gut verteilte Fotos sollten
   besser abschneiden als die zufälligen Teilmengen hier — aber unbestätigt.
2. **Echte CAD-Evaluierung nachholen oder die Selbstvergleichs-Lösung im Bericht
   sauber begründen.** Abschnitt 5 beschreibt einen Ersatz-Ansatz (Selbstvergleich statt
   externem CAD). Falls volle Punktzahl bei diesem Aufgabenteil wichtig ist: entweder ein
   reales Objekt mit auffindbarem CAD-Modell besorgen und fotografieren (siehe Punkt 1),
   oder den DTU-MVS-Benchmark sauber aufsetzen (aufwändiger, siehe Abschnitt 5).
3. **20-Foto-Sparse-Test mit Chamfer-Distanz nachholen.** Für n=30 liegt ein Chamfer-Wert
   vor (Abschnitt 5); für n=20 nicht, weil der letzte COLMAP-Lauf nur 4/20 Bilder
   registriert hat (Abschnitt 4b) — zu wenig für eine sinnvolle Rekonstruktion. Lauf
   wiederholen (COLMAP ist bei n=20 nicht-deterministisch, ein neuer Versuch kann
   besser ausfallen) und denselben Chamfer-Vergleich durchführen.
4. **Watertight-Reparatur des Mesh.** Alle bisherigen STL-Exporte sind nicht wasserdicht
   (Unterseite des Objekts nie fotografiert). Für einen 3D-Druck wäre ein manueller
   2-Minuten-Schritt in Blender/MeshLab/Meshmixer nötig — bewusst nicht in Colab gemacht.
5. **Colab-Stabilität im Blick behalten.** Die Laufzeitumgebung setzt sich sehr häufig
   zurück (Abschnitt 2). Vor jeder längeren Arbeitssitzung zuerst prüfen, ob venv/Datensatz/
   Masken noch da sind, und im Zweifel zuerst den Cache aus `nerf_cache/` wiederherstellen
   (Masken/Ergebnisse in Sekunden statt in Minuten), statt sofort alles neu zu bauen.
6. **`pipeline_clean.ipynb` mit dem tatsächlichen Chat-Verlauf synchronisieren.** Die
   Notebook-Datei enthält nicht alle zuletzt verwendeten Codezellen (z. B. den
   Sparse-View-COLMAP-Test aus Abschnitt 4b und das Evaluierungsskript aus Abschnitt 5
   fehlen dort noch) — bei Gelegenheit als eigene Notebook-Abschnitte nachtragen, damit
   das Notebook allein zum Reproduzieren reicht.
7. **Schriftlichen Bericht verfassen.** Dieses Dokument ist ein technisches
   Übergabeprotokoll, kein fertiger Abgabetext — die Ergebnisse (Abschnitte 1–5) müssen
   noch in die vom Kurs geforderte Berichtsform gebracht werden.
