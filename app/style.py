"""Look of the web app: theme, CSS, header/footer HTML and the Taurus easter eggs.

Palette: off-white / white / dark grey / light blue, small accents of burgundy and emerald.
Emerald is the Taurus birthstone, so "done" is emerald on purpose.

Easter eggs (♉ Taurus, the author's sign) — keep them subtle, they are meant to be found:
  1. click the ♉ in the header twice (Taurus = 2nd zodiac sign) -> a small Taurus card
  2. hover the reddish star in the header constellation -> "Aldebaran — das Auge des Stiers"
  3. Konami code (↑↑↓↓←→←→ B A) -> "Venus mode": ♉ glyphs drift down, emerald glow
  4. between 20 April and 20 May an emerald "Taurus Season" chip appears in the header
  5. the running-status spinner is a turning ♉
  6. a describe-prompt like "bull" / "taurus" / "stier" / "телец" gets a special line (app.py)
  7. a greeting in the browser dev-tools console
"""
import inspect

import gradio as gr

OFF_WHITE = "#F4F2ED"
WHITE = "#FFFFFF"
DARK = "#2D3034"
GREY = "#6B7076"
LINE = "#E2DED6"
BLUE = "#A8CBE6"
BLUE_DEEP = "#7FB0D6"
BLUE_SOFT = "#E6F0F8"
BURGUNDY = "#7B2232"
EMERALD = "#1E7B5E"


def theme():
    t = gr.themes.Base(
        font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
        font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "monospace"],
        radius_size=gr.themes.sizes.radius_lg,
    )
    values = dict(
        body_background_fill=OFF_WHITE,
        body_text_color=DARK,
        body_text_color_subdued=GREY,
        background_fill_primary=WHITE,
        background_fill_secondary=OFF_WHITE,
        block_background_fill=WHITE,
        block_border_color=LINE,
        block_border_width="1px",
        block_label_background_fill=WHITE,
        block_label_text_color=GREY,
        block_title_text_color=DARK,
        block_shadow="0 1px 2px rgba(45,48,52,.04), 0 4px 16px rgba(45,48,52,.04)",
        border_color_primary=LINE,
        border_color_accent=BLUE_DEEP,
        color_accent=BLUE_DEEP,
        color_accent_soft=BLUE_SOFT,
        input_background_fill=OFF_WHITE,
        input_border_color=LINE,
        input_border_color_focus=BLUE_DEEP,
        button_primary_background_fill=BLUE,
        button_primary_background_fill_hover=BLUE_DEEP,
        button_primary_text_color=DARK,
        button_primary_border_color=BLUE_DEEP,
        button_secondary_background_fill=WHITE,
        button_secondary_text_color=DARK,
        slider_color=BLUE_DEEP,
        link_text_color=BURGUNDY,
        loader_color=BLUE_DEEP,
    )
    # the palette is light by design: dark-mode browsers get the same colours
    allowed = inspect.signature(t.set).parameters
    values.update({k + "_dark": v for k, v in values.items() if k + "_dark" in allowed})
    return t.set(**{k: v for k, v in values.items() if k in allowed})


# Gradio prefixes every rule with ".gradio-container .contain", so never target body/html here:
# toggle classes on .hero instead (see JS).
CSS = f"""
:root {{ --burgundy:{BURGUNDY}; --emerald:{EMERALD}; --blue:{BLUE}; --blue-deep:{BLUE_DEEP};
         --blue-soft:{BLUE_SOFT}; --dark:{DARK}; --grey:{GREY}; --line:{LINE}; --offwhite:{OFF_WHITE}; }}
.gradio-container {{ max-width: 1180px !important; margin: 0 auto !important; }}

/* header */
.hero {{ position: relative; overflow: hidden; background: {WHITE}; border: 1px solid var(--line);
         border-radius: 20px; padding: 34px 36px 28px; margin-bottom: 6px; }}
.hero h1 {{ font-family: 'Cormorant Garamond', Georgia, serif; font-weight: 600; font-size: 46px;
            line-height: 1.05; margin: 0 0 10px; color: var(--dark); letter-spacing: -.5px; }}
.hero h1 .arrow {{ color: var(--blue-deep); font-weight: 400; }}
.hero .sub {{ font-size: 16px; color: var(--grey); max-width: 620px; margin: 0 0 16px; }}
.hero .tech {{ display: flex; flex-wrap: wrap; gap: 6px; }}
.hero .tech span {{ font-size: 12px; padding: 4px 10px; border-radius: 99px; background: var(--blue-soft);
                    color: var(--dark); border: 1px solid #D3E4F2; }}
.hero .rule {{ width: 46px; height: 3px; background: var(--burgundy); border-radius: 2px; margin: 0 0 14px; }}
.constellation {{ position: absolute; right: 28px; top: 18px; width: 230px; opacity: .9; }}
.constellation line {{ stroke: var(--blue); stroke-width: .8; }}
.constellation circle {{ fill: var(--dark); opacity: .55; }}
.constellation .aldebaran {{ fill: var(--burgundy); opacity: 1; cursor: help; transition: r .2s; }}
.constellation .aldebaran:hover {{ r: 4.6; }}
.constellation .pleiades circle {{ fill: var(--blue-deep); opacity: .9; }}
.glyph, #taurus-sign, .foot .bull, .falling-bull, .status.run .msg::before {{ font-family: 'Segoe UI Symbol', 'DejaVu Sans', 'Noto Sans Symbols', serif; font-variant-emoji: text; }}
.glyph {{ color: var(--burgundy); }}
#taurus-sign {{ position: absolute; right: 30px; bottom: 18px; font-size: 26px; color: var(--burgundy);
                cursor: pointer; user-select: none; transition: transform .35s; }}
#taurus-sign:hover {{ transform: rotate(-12deg) scale(1.12); }}
#taurus-card {{ display: none; position: absolute; right: 70px; bottom: 14px; background: var(--offwhite);
                border: 1px solid var(--line); border-left: 3px solid var(--emerald); border-radius: 12px;
                padding: 10px 14px; font-size: 13px; color: var(--dark); max-width: 300px;
                box-shadow: 0 6px 24px rgba(45,48,52,.10); }}
#taurus-card.show {{ display: block; animation: rise .4s ease; }}
#taurus-season {{ display: none; font-family: Inter, 'Segoe UI Symbol', sans-serif; font-variant-emoji: text; background: var(--emerald) !important; color: white !important;
                  border-color: var(--emerald) !important; }}
.hero.taurus-season #taurus-season {{ display: inline-block; }}
@media (max-width: 760px) {{ .constellation {{ width: 140px; opacity: .35; }} .hero h1 {{ font-size: 34px; }} }}

/* cards */
.card {{ border-radius: 18px !important; }}
.tips {{ background: var(--offwhite); border-radius: 14px; padding: 4px 16px; border: 1px dashed var(--line); }}
.tips h3 {{ font-family: 'Cormorant Garamond', Georgia, serif; font-size: 22px; margin: 10px 0 4px; }}
.tips ul {{ margin: 0 0 6px; padding-left: 18px; }}
.tips li {{ margin: 3px 0; font-size: 14px; }}
.tips p {{ font-size: 13px; color: var(--grey); margin: 6px 0 10px; }}
#go-btn {{ font-weight: 600; letter-spacing: .2px; }}

/* status */
.status {{ background: {WHITE}; border: 1px solid var(--line); border-radius: 16px; padding: 14px 16px; }}
.status .pills {{ display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 10px; }}
.status .pill {{ font-size: 12px; padding: 4px 10px; border-radius: 99px; border: 1px solid var(--line);
                 color: var(--grey); background: var(--offwhite); }}
.status .pill.active {{ background: var(--blue-soft); border-color: var(--blue-deep); color: var(--dark); }}
.status .pill.done {{ background: #E5F2ED; border-color: #B9DCCD; color: var(--emerald); }}
.status .pill.error {{ background: #F6E7EA; border-color: #E2B9C1; color: var(--burgundy); }}
.status .msg {{ font-size: 15px; color: var(--dark); }}
.status.run .msg::before {{ content: '\\2649\\FE0E'; display: inline-block; color: var(--burgundy); margin-right: 8px;
                            animation: spin 2.4s linear infinite; }}
.status.ok {{ border-left: 4px solid var(--emerald); }}
.status.ok .msg {{ color: var(--emerald); font-weight: 600; }}
.status.err {{ border-left: 4px solid var(--burgundy); }}
.status.err .msg {{ color: var(--burgundy); }}
.status .note {{ margin-top: 6px; font-size: 13px; color: var(--grey); }}
.status.idle .msg {{ color: var(--grey); }}

.foot {{ text-align: center; font-size: 13px; color: var(--grey); padding: 18px 0 6px; }}
.foot .bull {{ color: var(--burgundy); }}

/* Venus mode (Konami) */
.hero.venus {{ box-shadow: 0 0 0 3px var(--emerald), 0 0 40px rgba(30,123,94,.25); transition: box-shadow .4s; }}
.falling-bull {{ position: fixed; top: -40px; z-index: 9999; pointer-events: none; color: var(--burgundy);
                 animation: fall linear forwards; }}

@keyframes spin {{ to {{ transform: rotate(360deg); }} }}
@keyframes rise {{ from {{ opacity: 0; transform: translateY(6px); }} to {{ opacity: 1; transform: none; }} }}
@keyframes fall {{ to {{ transform: translateY(110vh) rotate(200deg); opacity: .2; }} }}
"""

HEAD = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@500;600&display=swap" rel="stylesheet">
"""

# Taurus: Aldebaran (eye), Hyades "V", horns to Elnath and zeta Tau, Pleiades cluster
HERO = """
<div class="hero">
  <svg class="constellation" viewBox="0 0 230 130" aria-hidden="true">
    <line x1="92" y1="70" x2="108" y2="62"/><line x1="108" y1="62" x2="120" y2="70"/>
    <line x1="120" y1="70" x2="112" y2="80"/><line x1="112" y1="80" x2="92" y2="70"/>
    <line x1="108" y1="62" x2="150" y2="22"/><line x1="92" y1="70" x2="60" y2="30"/>
    <line x1="112" y1="80" x2="100" y2="112"/>
    <circle cx="108" cy="62" r="2"/><circle cx="120" cy="70" r="1.8"/><circle cx="112" cy="80" r="2"/>
    <circle cx="150" cy="22" r="2.6"/><circle cx="60" cy="30" r="2.2"/><circle cx="100" cy="112" r="1.8"/>
    <circle class="aldebaran" cx="92" cy="70" r="3.6"><title>Aldebaran — das Auge des Stiers</title></circle>
    <g class="pleiades"><circle cx="190" cy="92" r="1.6"/><circle cx="197" cy="88" r="1.3"/>
      <circle cx="194" cy="98" r="1.2"/><circle cx="202" cy="95" r="1.4"/><circle cx="186" cy="99" r="1.1"/>
      <circle cx="199" cy="103" r="1.1"/><circle cx="205" cy="90" r="1"/></g>
  </svg>
  <h1>Fotos <span class="arrow">→</span> 3D-Modell</h1>
  <div class="rule"></div>
  <p class="sub">Fotografiere ein Objekt von allen Seiten, sag in ein paar Worten, was es ist —
     daraus entsteht eine druckbare STL-Datei.</p>
  <div class="tech"><span>COLMAP</span><span>Grounding DINO + SAM2</span>
     <span>3D Gaussian Splatting</span><span>Visual Hull</span>
     <span id="taurus-season">Taurus Season ♉︎</span></div>
  <div id="taurus-card"><span class="glyph">♉︎</span> <b>Stier</b> · Erde · Venus · Smaragd<br>
     Smaragd ist der Geburtsstein des Stiers — deshalb leuchtet hier „fertig“ grün.
     Und Geduld, die Stier-Tugend, braucht Gaussian Splatting auch.</div>
  <div id="taurus-sign" title="">♉︎</div>
</div>
"""

TIPS = """
<div class="tips">
  <h3>So fotografieren</h3>
  <ul>
    <li><b>30–45 Fotos</b>, das Objekt bleibt stehen — <b>du gehst herum</b></li>
    <li>2–3 Runden: seitlich, schräg von oben (~45°), fast von oben</li>
    <li>Untergrund mit Muster (Zeitung, Holz, Teppich) — kein weißes Papier</li>
    <li>gleichmäßiges Licht, kein Blitz, kein Zoom, alles scharf</li>
    <li>Originaldateien hochladen (nicht über Messenger komprimiert)</li>
  </ul>
  <p>Glänzendes, Durchsichtiges und sehr Dünnes klappt schlecht. Dauer: <b>ca. 30 min</b>.</p>
</div>
"""

FOOT = """<div class="foot">Gruppe 7 · FH Technikum Wien · Objektrekonstruktion<br>
gebaut mit Geduld — einer Stier-Tugend <span class="bull">♉︎</span></div>"""

JS = """
() => {
  console.log('%c♉  Taurus was here.', 'color:#7B2232;font-size:16px;font-weight:bold');
  console.log('%cBeständig wie ein 3D-Druck. Tipp: ↑ ↑ ↓ ↓ ← → ← → B A', 'color:#1E7B5E');

  const now = new Date(), m = now.getMonth() + 1, d = now.getDate();
  const hero = () => document.querySelector('.hero');
  if ((m === 4 && d >= 20) || (m === 5 && d <= 20)) hero()?.classList.add('taurus-season');

  let clicks = 0;
  document.addEventListener('click', (e) => {
    if (!e.target.closest('#taurus-sign')) return;
    clicks += 1;
    e.target.closest('#taurus-sign').style.transform = `rotate(${clicks * 180}deg)`;
    if (clicks % 2 === 0) document.getElementById('taurus-card')?.classList.toggle('show');
  });

  const code = ['ArrowUp','ArrowUp','ArrowDown','ArrowDown','ArrowLeft','ArrowRight','ArrowLeft','ArrowRight','b','a'];
  let pos = 0;
  document.addEventListener('keydown', (e) => {
    pos = (e.key === code[pos] || e.key.toLowerCase() === code[pos]) ? pos + 1 : 0;
    if (pos < code.length) return;
    pos = 0;
    hero()?.classList.add('venus');
    const box = document.querySelector('.gradio-container .contain') || document.body;
    for (let i = 0; i < 28; i++) {
      const s = document.createElement('div');
      s.className = 'falling-bull';
      s.textContent = i % 5 === 0 ? '\u2640\uFE0E' : '\u2649\uFE0E';
      s.style.left = Math.random() * 100 + 'vw';
      s.style.fontSize = (14 + Math.random() * 22) + 'px';
      s.style.animationDuration = (3 + Math.random() * 3) + 's';
      s.style.animationDelay = (Math.random() * 1.5) + 's';
      box.appendChild(s);
      setTimeout(() => s.remove(), 8000);
    }
    setTimeout(() => hero()?.classList.remove('venus'), 6000);
  });
}
"""

STEPS = ["Fotos", "Kameraposen", "Masken", "Training", "Mesh"]


def status_html(step, msg, kind="run", note=""):
    """step: 1..5 = current step; kind: idle | run | ok | err."""
    pills = []
    for i, name in enumerate(STEPS, 1):
        if kind == "ok" or i < step:
            cls = "done"
        elif i == step:
            cls = {"run": "active", "err": "error"}.get(kind, "")
        else:
            cls = ""
        pills.append(f'<span class="pill {cls}">{i} · {name}</span>')
    note_html = f'<div class="note">{note}</div>' if note else ""
    return f'<div class="status {kind}"><div class="pills">{"".join(pills)}</div><div class="msg">{msg}</div>{note_html}</div>'


IDLE = status_html(0, "Fotos hochladen, Objekt beschreiben, Start drücken.", "idle")
