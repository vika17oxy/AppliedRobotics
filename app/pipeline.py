"""Photos + text prompt -> STL. Same steps as notebooks/photos_to_stl.ipynb, as functions.

Runs in the Colab kernel. nerfstudio (COLMAP wrapper, training, hull) runs in the
separate venv /content/nsenv through subprocesses; Grounding DINO + SAM2 run in the
kernel via transformers.
"""
import glob
import hashlib
import json
import os
import shutil
import subprocess

import numpy as np
import torch
from PIL import Image, ImageOps

NS = "/content/nsenv/bin"
HERE = os.path.dirname(os.path.abspath(__file__))

try:  # iPhone photos often arrive as HEIC
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    pass


def run(cmd, log):
    """Run a shell command, stream its output lines to log(), raise on failure."""
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    tail = []
    for line in p.stdout:
        tail = (tail + [line.rstrip()])[-30:]
        log(line.rstrip())
    if p.wait():
        raise RuntimeError(f"command failed ({p.returncode}):\n" + "\n".join(tail))


def prepare_photos(paths, raw_dir, max_side=1600):
    """EXIF-rotate, downscale (COLMAP runs on CPU), drop broken thumbnails and duplicate files.

    Landscape shots are turned to portrait: COLMAP is run with one shared camera model, which breaks
    when some images are 1600x1200 and others 1200x1600. A 90° turn is just a rolled camera.
    """
    os.makedirs(raw_dir, exist_ok=True)
    seen, n = set(), 0
    for i, p in enumerate(sorted(paths)):
        digest = hashlib.md5(open(p, "rb").read()).hexdigest()
        if digest in seen:
            continue
        seen.add(digest)
        try:
            im = Image.open(p)
        except Exception:
            continue
        if max(im.size) < 1000:
            continue
        im = ImageOps.exif_transpose(im).convert("RGB")
        if im.width > im.height:
            im = im.transpose(Image.Transpose.ROTATE_90)
        im.thumbnail((max_side, max_side), Image.LANCZOS)
        im.save(f"{raw_dir}/img_{i:03d}.jpg", quality=95)
        n += 1
    return n


def colmap(raw_dir, processed_dir, log):
    shutil.rmtree(processed_dir, ignore_errors=True)
    run(f"QT_QPA_PLATFORM=offscreen {NS}/ns-process-data images --data {raw_dir} "
        f"--output-dir {processed_dir} --matching-method exhaustive --no-gpu", log)
    tf = f"{processed_dir}/transforms.json"
    if not os.path.exists(tf):
        raise RuntimeError("COLMAP could not register the photos (no transforms.json)")
    return len(json.load(open(tf))["frames"])


_models = {}


def _load_models():
    if not _models:
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection, Sam2Model, Sam2Processor
        gd, sam = "IDEA-Research/grounding-dino-base", "facebook/sam2-hiera-large"
        _models["gd_proc"] = AutoProcessor.from_pretrained(gd)
        _models["gd"] = AutoModelForZeroShotObjectDetection.from_pretrained(gd).cuda().eval()
        _models["sam_proc"] = Sam2Processor.from_pretrained(sam)
        _models["sam"] = Sam2Model.from_pretrained(sam).cuda().eval()
    return _models


def _boxes(img, text, thr):
    m = _load_models()
    inp = m["gd_proc"](images=img, text=text, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = m["gd"](**inp)
    tgt = [img.size[::-1]]
    try:
        r = m["gd_proc"].post_process_grounded_object_detection(out, inp.input_ids, threshold=thr, text_threshold=0.25, target_sizes=tgt)[0]
    except TypeError:
        r = m["gd_proc"].post_process_grounded_object_detection(out, inp.input_ids, box_threshold=thr, text_threshold=0.25, target_sizes=tgt)[0]
    return r["boxes"].cpu().numpy()


def _sam(img, boxes):
    m = _load_models()
    proc = m["sam_proc"]
    pp = getattr(proc, "post_process_masks", None) or proc.image_processor.post_process_masks
    inp = proc(img, input_boxes=[[list(map(float, b)) for b in boxes]], return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = m["sam"](**inp, multimask_output=True)
    masks = pp(out.pred_masks.cpu(), inp["original_sizes"].cpu())[0].numpy()
    iou = out.iou_scores.cpu().numpy()[0]
    return np.stack([masks[i, iou[i].argmax()] for i in range(masks.shape[0])]).any(0)


def make_masks(processed_dir, prompt, log):
    """Grounding DINO box from the text prompt -> SAM2 mask, one PNG per training image."""
    img_dir, mask_dir = f"{processed_dir}/images", f"{processed_dir}/masks"
    os.makedirs(mask_dir, exist_ok=True)
    names = sorted(os.listdir(img_dir))
    no_box, preview = [], []
    for i, n in enumerate(names):
        img = Image.open(f"{img_dir}/{n}").convert("RGB")
        W, H = img.size
        b = _boxes(img, prompt, 0.3)
        if len(b) == 0:
            b = _boxes(img, prompt + " object.", 0.22)
        keep = [bb for bb in b if 0.02 < ((bb[2] - bb[0]) * (bb[3] - bb[1])) / (W * H) < 0.99]
        if keep:
            mask = _sam(img, keep)
        else:
            no_box.append(n)
            mask = np.ones((H, W), bool)
        Image.fromarray((mask * 255).astype(np.uint8)).save(f"{mask_dir}/{os.path.splitext(n)[0]}.png")
        if i % max(1, len(names) // 8) == 0 and len(preview) < 8:
            ov = np.array(img)
            ov[~mask] = (ov[~mask] * 0.2).astype(np.uint8)
            preview.append(Image.fromarray(ov).resize((180, int(180 * H / W))))
        log(f"mask {i + 1}/{len(names)}")
    sheet = Image.new("RGB", (180 * len(preview), max(p.size[1] for p in preview)))
    for k, p in enumerate(preview):
        sheet.paste(p, (180 * k, 0))
    return no_box, sheet


def train(processed_dir, out_dir, iters, log):
    run(f"cd {os.path.dirname(processed_dir)} && yes | TORCHDYNAMO_DISABLE=1 {NS}/ns-train splatfacto "
        f"--data {processed_dir} --max-num-iterations {iters} --steps-per-save {iters} "
        f"--vis tensorboard --viewer.quit-on-train-completion True "
        f"--output-dir {out_dir} --pipeline.model.cull-alpha-thresh 0.1 "
        f"colmap --images-path images --colmap-path colmap/sparse/0 --masks-path masks --downscale-factor 1", log)
    cfgs = sorted(glob.glob(f"{out_dir}/*/splatfacto/*/config.yml"), key=os.path.getmtime)
    if not cfgs:
        raise RuntimeError("training finished without config.yml")
    return cfgs[-1]


def hull(config, stl_path, log, voxels=256):
    run(f"{NS}/python {HERE}/hull.py {config} {stl_path} {voxels}", log)
    return stl_path
