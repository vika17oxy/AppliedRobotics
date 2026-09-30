# Gruppe 7 — Objektrekonstruktion (Neural Rendering, VLM)

Kursprojekt, FH Technikum Wien. Ziel: aus 5–30 Fotos eines Objekts automatisch ein
3D-Modell (STL) rekonstruieren — mit 3D Gaussian Splatting (nerfstudio/splatfacto) und
sprachgesteuerter Segmentierung (Grounding DINO + SAM2), damit nur das benannte Objekt
(z. B. "the toy truck") im Mesh landet, nicht der Hintergrund.

Vollständiger Projektstatus, Architektur, Testergebnisse und offene Punkte:
siehe **[PROJECT_STATUS.md](PROJECT_STATUS.md)**.

## Dateien

| Datei | Inhalt |
|---|---|
| [`pipeline_clean.ipynb`](pipeline_clean.ipynb) | Hauptnotebook der Pipeline (mit Zellausgaben aus dem letzten erfolgreichen Lauf) — Kameraposen (COLMAP) → 3DGS-Training (splatfacto) → SAM2-Maskierung → Mesh-Export (Poisson) → STL |
| [`pipeline.ipynb`](pipeline.ipynb) | Frühere Version des Notebooks |
| [`PROJECT_STATUS.md`](PROJECT_STATUS.md) | Ausführliches Übergabedokument: Status, Umgebung, Ergebnisse, Evaluierung |
| [`truck_object_30k.stl`](truck_object_30k.stl) | Exportiertes Beispiel-Mesh (Testobjekt "truck", 30k Trainingsiterationen) |
| [`truck_positions.json`](truck_positions.json) | Kameraposen-Daten des Testdatensatzes |

## Ausführen

Das Notebook läuft auf **Google Colab** (GPU Tesla T4):

1. `pipeline_clean.ipynb` in Colab öffnen (z. B. über die VS-Code-Colab-Erweiterung oder
   direkt auf [colab.research.google.com](https://colab.research.google.com)).
2. Laufzeit mit GPU (T4) verbinden.
3. Zellen der Reihe nach ausführen — bereits erledigte Schritte überspringen sich selbst.

Details zur Umgebung (venv-Aufbau, Abhängigkeiten, bekannte Stolpersteine) stehen in
[PROJECT_STATUS.md](PROJECT_STATUS.md), Abschnitt 2.
