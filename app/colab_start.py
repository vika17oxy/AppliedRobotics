# ♉ Fotos → 3D-Modell — start the web app on Google Colab.
# Paste this whole file into ONE Colab cell (Runtime → Change runtime type → T4 GPU) and run it.
# First run ~30-40 min (builds the nerfstudio environment), later runs ~2 min.
# At the end Gradio prints "Running on public URL: https://….gradio.live" — that is the website.
import os
import subprocess

REPO = "https://github.com/vika17oxy/AppliedRobotics"
CODE = "/content/AppliedRobotics"
NS = "/content/nsenv"


def sh(cmd, quiet=True):
    """Run a shell command and stream its output into the cell (Colab hides raw subprocess output)."""
    print("▸", cmd if len(cmd) < 110 else cmd[:107] + "…", flush=True)
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    tail = []
    for line in p.stdout:
        tail = (tail + [line])[-25:]
        if not quiet:
            print(line, end="", flush=True)
    if p.wait():
        print("".join(tail))
        raise SystemExit(f"❌ failed: {cmd}")


# 1) GPU present?
if subprocess.run("nvidia-smi -L", shell=True, capture_output=True).returncode:
    raise SystemExit("❌ No GPU. Runtime → Change runtime type → T4 GPU, then run again.")

# 2) nerfstudio in its own Python 3.10 venv (Colab's kernel is too new for nerfstudio 1.1.5).
#    Exact versions matter: gsplat must come as a prebuilt wheel, numpy must be 1.x and installed last.
if not os.path.exists(f"{NS}/bin/ns-train"):
    py = f"{NS}/bin/python"
    sh("pip install -q uv")
    sh("uv python install 3.10")
    sh(f"uv venv --python 3.10 {NS}")
    sh(f'uv pip install --python {py} "torch==2.1.2" "torchvision==0.16.2" --index-url https://download.pytorch.org/whl/cu118')
    sh(f'uv pip install --python {py} "setuptools<70" wheel')
    sh(f'uv pip install --python {py} "gsplat==1.4.0+pt21cu118" --index-url https://docs.gsplat.studio/whl/pt21cu118 '
       f'--extra-index-url https://pypi.org/simple --index-strategy unsafe-best-match')
    sh(f'uv pip install --python {py} "nerfstudio==1.1.5" open3d trimesh scikit-image')
    sh(f'uv pip install --python {py} "numpy==1.26.4"')
sh(f'uv pip install -q --python {NS}/bin/python scikit-image "numpy==1.26.4"')
sh("apt-get -qq install -y colmap > /dev/null")

# 3) app code + kernel packages (Grounding DINO / SAM2 run in the kernel via transformers)
if os.path.isdir(f"{CODE}/.git"):
    sh(f"git -C {CODE} pull -q")
else:
    sh(f"git clone -q --depth 1 {REPO} {CODE}")
sh('pip install -q -U "gradio==6.30.0" transformers pillow-heif')

# 4) start the website (this cell keeps running while the site is online)
print("\n♉ Starting… wait for 'Running on public URL'.\n", flush=True)
sh(f"cd {CODE}/app && python -u app.py", quiet=False)
