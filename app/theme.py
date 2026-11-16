"""
Visual theme for the dashboard: an editorial look (warm paper and ink, serif headlines, hairline rules,
one green accent) with light and dark variants and purposeful motion.

Pure CSS and system fonts only: no internet, no external files. Animations switch off automatically
for people who prefer reduced motion.
"""

from urllib.parse import quote

LIGHT = {
    "bg": "#f4f0e8", "panel": "#fbf9f4", "ink": "#1d1c19", "ink2": "#5f5b52", "ink3": "#8b867a",
    "rule": "#dcd6c8", "rule2": "#bdb6a4", "input": "#fffdf8",
    "accent": "#1d6b53", "accent_soft": "rgba(29,107,83,0.10)",
    "amber": "#a35f0b", "amber_soft": "rgba(163,95,11,0.12)",
    "red": "#b3372f", "red_soft": "rgba(179,55,47,0.10)",
    "hatch": "rgba(29,28,25,0.26)", "row_hover": "rgba(29,28,25,0.04)", "grain": 0.05,
}
DARK = {
    "bg": "#12110e", "panel": "#1a1915", "ink": "#f1ede3", "ink2": "#b3ada0", "ink3": "#837d70",
    "rule": "#2e2c26", "rule2": "#4a463d", "input": "#201f1a",
    "accent": "#62c9a3", "accent_soft": "rgba(98,201,163,0.12)",
    "amber": "#e8a64a", "amber_soft": "rgba(232,166,74,0.14)",
    "red": "#f07a70", "red_soft": "rgba(240,122,112,0.14)",
    "hatch": "rgba(241,237,227,0.28)", "row_hover": "rgba(241,237,227,0.05)", "grain": 0.07,
}

BASE_CSS = """
@property --n { syntax: "<integer>"; initial-value: 0; inherits: false; }

.stApp {
  --serif: "Iowan Old Style", "Palatino Linotype", Palatino, "Book Antiqua", Charter, Georgia, serif;
  --sans: -apple-system, BlinkMacSystemFont, "Segoe UI Variable Text", "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
  background-color: var(--bg); background-image: var(--grain); color: var(--ink); font-family: var(--sans);
  transition: background-color .5s ease, color .5s ease;
}
[data-testid="stAppViewContainer"] { position: relative; z-index: 1; }
[data-testid="stHeader"] { background: transparent; }
#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"],
[data-testid="stStatusWidget"] { display: none !important; }
.block-container { max-width: 1120px; padding: 1rem 1.4rem 4.5rem; }

/* ---------- Text defaults for Streamlit's own elements ---------- */
[data-testid="stMarkdownContainer"] p, [data-testid="stWidgetLabel"] p, [data-testid="stWidgetLabel"] label,
[data-testid="stCheckbox"] label span, [data-testid="stCaptionContainer"] { color: var(--ink); }
[data-testid="stCaptionContainer"] { color: var(--ink2); }
[data-testid="stWidgetLabel"] p { font-size: .88rem; font-weight: 600; letter-spacing: .01em; }
[data-testid="stToggle"] label p, [data-testid="stToggle"] label span { color: var(--ink2) !important; font-weight: 600; font-size: .85rem; }
:focus-visible { outline: 2px solid var(--accent); outline-offset: 3px; }

/* ---------- Masthead and hero ---------- */
.mast { display: flex; align-items: baseline; flex-wrap: wrap; gap: .3rem 1.1rem; padding: 1.1rem 0 .85rem; }
.wordmark { font-family: var(--serif); font-size: 1.4rem; font-weight: 700; letter-spacing: -0.01em; color: var(--ink); }
.wordmark i { display: inline-block; width: .58em; height: .58em; background: var(--accent); margin-right: .55rem;
              transform: rotate(45deg); animation: turn .9s cubic-bezier(.2,.8,.2,1) both; }
.mast-sub { font-size: .78rem; letter-spacing: .08em; text-transform: uppercase; color: var(--ink2); font-weight: 600; }
.rule { height: 1px; background: var(--rule2); transform-origin: left; }
.rule.draw { animation: rule 1.2s cubic-bezier(.2,.8,.2,1) both; }

.hero { padding: 2.6rem 0 1.2rem; }
.kicker { font-size: .78rem; letter-spacing: .14em; text-transform: uppercase; color: var(--accent); font-weight: 700; }
.hero h1 { font-family: var(--serif); font-size: clamp(2.7rem, 6.4vw, 4.9rem); line-height: 1.02; letter-spacing: -0.03em;
           font-weight: 600; margin: .9rem 0 1.3rem; color: var(--ink); padding: 0; }
.hero .u { background-image: linear-gradient(var(--accent), var(--accent)); background-repeat: no-repeat;
           background-size: 100% .075em; background-position: 0 94%; animation: underline 1.3s .7s cubic-bezier(.2,.8,.2,1) both; }
.hero p { font-size: clamp(1.02rem, 1.5vw, 1.2rem); line-height: 1.6; color: var(--ink2); max-width: 41rem; margin: 0; }
.r1, .r2, .r3 { animation: rise .95s cubic-bezier(.2,.8,.2,1) both; }
.r1 { animation-delay: .05s; } .r2 { animation-delay: .16s; } .r3 { animation-delay: .32s; }

/* ---------- Notes ---------- */
.readfirst { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 1.6rem; margin: 1.6rem 0 .6rem;
             padding-top: 1.1rem; border-top: 1px solid var(--rule); animation: rise .9s .45s cubic-bezier(.2,.8,.2,1) both; }
.readfirst div { font-size: .9rem; line-height: 1.55; color: var(--ink2); }
.readfirst b { display: block; color: var(--ink); font-size: .74rem; letter-spacing: .12em; text-transform: uppercase; margin-bottom: .3rem; }
.note { padding: .9rem 1.1rem; border-left: 3px solid var(--accent); background: var(--accent_soft); border-radius: 0 10px 10px 0;
        font-size: .93rem; line-height: 1.5; color: var(--ink2); animation: rise .6s cubic-bezier(.2,.8,.2,1) both; }
.note b { color: var(--ink); }
.note.warn { border-color: var(--amber); background: var(--amber_soft); }
.note.error { border-color: var(--red); background: var(--red_soft); }
.tag { display: inline-flex; align-items: center; gap: .5rem; padding: .22rem .6rem; border-radius: 4px; border: 1px solid var(--rule2);
       font-size: .7rem; letter-spacing: .1em; text-transform: uppercase; font-weight: 700; color: var(--ink2); }
.tag i { width: 7px; height: 7px; background: currentColor; display: inline-block; animation: blink 2.4s ease-in-out infinite; }
.tag.good { color: var(--accent); border-color: var(--accent); }
.tag.warn { color: var(--amber); border-color: var(--amber); }

/* ---------- Panels and sections ---------- */
.panel, [class*="st-key-card_"] { background: var(--panel); border: 1px solid var(--rule); border-radius: 14px; padding: 1.7rem 1.8rem;
  transition: border-color .35s ease, transform .35s cubic-bezier(.2,.8,.2,1), background-color .5s ease; }
.panel:hover, [class*="st-key-card_"]:hover { border-color: var(--rule2); transform: translateY(-2px); }
.sec-title { font-family: var(--serif); font-size: 1.55rem; font-weight: 600; letter-spacing: -0.015em; color: var(--ink); line-height: 1.15; }
.sec-sub { font-size: .92rem; color: var(--ink2); margin: .35rem 0 1.1rem; line-height: 1.5; }
.label { font-size: .72rem; letter-spacing: .13em; text-transform: uppercase; color: var(--ink2); font-weight: 700; }
.fine { font-size: .86rem; color: var(--ink2); margin: 1.3rem 0 0; line-height: 1.55; }
.fine b { color: var(--ink); }
.fade { animation: rise .8s cubic-bezier(.2,.8,.2,1) both; animation-delay: calc(var(--d, 0) * 90ms); }

/* ---------- Score numeral, count-up ---------- */
.score-line { display: flex; align-items: baseline; gap: .8rem; margin: .5rem 0 0; }
.numeral { font-size: clamp(4.2rem, 9vw, 6.6rem); font-weight: 300; letter-spacing: -0.055em; line-height: 1; color: var(--ink);
           font-variant-numeric: tabular-nums; }
.numeral .dec { font-size: .38em; letter-spacing: -0.02em; color: var(--ink2); font-weight: 400; }
.outof { font-size: 1rem; color: var(--ink2); }
.count { --n: var(--target); counter-reset: num var(--n); animation: countup .9s cubic-bezier(.2,.8,.2,1); }
.count::after { content: counter(num); }

/* ---------- Ruler ---------- */
.ruler { position: relative; height: 96px; margin: 1.6rem 1.3rem .4rem; }
.ruler-axis { position: absolute; left: 0; right: 0; top: 44px; height: 1px; background: var(--ink2);
              transform-origin: left; animation: rule .8s cubic-bezier(.2,.8,.2,1) both; }
.ruler::before { content: ""; position: absolute; left: 0; right: 0; top: 44px; height: 17px;
  background: repeating-linear-gradient(90deg, var(--ink2) 0 1px, transparent 1px 10%); animation: wipe .9s .1s cubic-bezier(.2,.8,.2,1) both; }
.ruler::after { content: ""; position: absolute; left: 0; right: 0; top: 44px; height: 8px;
  background: repeating-linear-gradient(90deg, var(--ink3) 0 1px, transparent 1px 5%); animation: wipe .9s .2s cubic-bezier(.2,.8,.2,1) both; }
.ruler-cap { position: absolute; right: 0; top: 44px; width: 1px; height: 17px; background: var(--ink2); }
.ruler-hatch { position: absolute; top: 22px; height: 22px; left: var(--a); width: calc(var(--b) - var(--a));
  background: repeating-linear-gradient(135deg, var(--hatch) 0 1px, transparent 1px 6px); border-left: 1px solid var(--ink2);
  border-right: 1px solid var(--ink2); animation: wipe .9s .3s cubic-bezier(.2,.8,.2,1) both; }
.ruler-hatch span { position: absolute; left: 6px; top: 46px; font-size: .68rem; letter-spacing: .1em; text-transform: uppercase;
                    color: var(--ink2); white-space: nowrap; font-weight: 650; }
.ruler-hatch.r span { left: auto; right: 6px; }
.ruler-mark { position: absolute; left: var(--x); top: 14px; height: 54px; border-left: 1px dashed var(--ink); }
.ruler-mark span { position: absolute; top: 58px; left: 0; transform: translateX(-50%); font-size: .74rem; color: var(--ink);
                   white-space: nowrap; font-weight: 650; }
.ruler-mark.amber { border-color: var(--amber); } .ruler-mark.amber span { color: var(--amber); }
.ruler-pin { position: absolute; top: 0; left: var(--x); width: 0; animation: pinmove .75s .3s cubic-bezier(.2,.9,.2,1.12) both; }
.ruler-pin::before { content: ""; position: absolute; left: -1px; top: 22px; width: 2px; height: 23px; background: var(--accent); }
.ruler-pin::after { content: ""; position: absolute; left: -6px; top: 39px; width: 12px; height: 12px; border-radius: 50%;
                    background: var(--accent); border: 3px solid var(--panel); }
.ruler-pin span { position: absolute; left: 0; top: -2px; transform: translateX(-50%); background: var(--accent); color: var(--bg);
                  padding: .18rem .6rem; border-radius: 4px; font-size: .82rem; font-weight: 700; white-space: nowrap;
                  font-variant-numeric: tabular-nums; }
.ruler-ends { display: flex; justify-content: space-between; margin: 0 1.3rem; font-size: .74rem; color: var(--ink2); }

/* ---------- Support card ---------- */
.status { font-family: var(--serif); font-size: clamp(2rem, 4.4vw, 3.3rem); font-weight: 600; letter-spacing: -0.025em; line-height: 1.08;
          margin: 1rem 0 .7rem; color: var(--ink); animation: wipe 1s .1s cubic-bezier(.2,.8,.2,1) both; }
.lead { color: var(--ink2); font-size: 1.04rem; line-height: 1.6; max-width: 40rem; }
.stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(230px, 1fr)); gap: 0; margin-top: 1.8rem; border-top: 1px solid var(--rule); }
.stat { padding: 1.1rem 1.2rem 0 0; }
.stat + .stat { padding-left: 1.2rem; border-left: 1px solid var(--rule); }
.stat b { display: block; font-size: 2.4rem; font-weight: 300; letter-spacing: -0.04em; color: var(--ink); font-variant-numeric: tabular-nums; }
.stat span { font-size: .84rem; color: var(--ink2); line-height: 1.45; display: block; margin-top: .2rem; }

/* ---------- Figures (KPIs) ---------- */
.figures { display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); border-top: 1px solid var(--rule2); }
.figure { padding: 1.2rem 1.2rem .4rem 0; animation: rise .8s cubic-bezier(.2,.8,.2,1) both; animation-delay: calc(var(--d, 0) * 100ms); }
.figure + .figure { padding-left: 1.3rem; border-left: 1px solid var(--rule); }
.fig-val { font-size: 2.7rem; font-weight: 300; letter-spacing: -0.045em; color: var(--ink); margin: .35rem 0 .15rem;
           font-variant-numeric: tabular-nums; line-height: 1.1; }
.fig-val.text { font-family: var(--serif); font-size: 1.85rem; font-weight: 600; letter-spacing: -0.02em; padding: .35rem 0 .2rem; }
.fig-sub { font-size: .84rem; color: var(--ink2); line-height: 1.45; }

/* ---------- Bars ---------- */
.bars .row { display: grid; grid-template-columns: 150px 1fr 54px; align-items: center; gap: 1rem; margin: 1rem 0; font-size: .92rem; }
.bars .name { color: var(--ink); font-weight: 600; }
.bars .val { color: var(--ink2); text-align: right; font-variant-numeric: tabular-nums; }
.bars .track { position: relative; height: 6px; background: var(--rule); }
.bars .fill { position: absolute; left: 0; top: 0; bottom: 0; width: var(--w); background: var(--ink3);
              animation: grow 1s cubic-bezier(.2,.8,.2,1) both; animation-delay: calc(var(--d, 0) * 110ms); }
.bars .fill.sel { background: var(--accent); }
.bars .base { position: absolute; top: -6px; bottom: -6px; width: 0; border-left: 1px dashed var(--amber); }

/* ---------- Tables ---------- */
.tbl-wrap { overflow: auto; max-height: 440px; border: 1px solid var(--rule); border-radius: 12px; background: var(--panel); }
.tbl { width: 100%; border-collapse: separate; border-spacing: 0; font-size: .9rem; }
.tbl th { position: sticky; top: 0; z-index: 1; text-align: left; padding: .8rem 1rem; font-size: .7rem; text-transform: uppercase; letter-spacing: .11em;
          color: var(--ink2); background: var(--panel); border-bottom: 1px solid var(--rule2); white-space: nowrap; font-weight: 700; }
.tbl td { padding: .72rem 1rem; color: var(--ink); border-bottom: 1px solid var(--rule); white-space: nowrap; }
.tbl tr:last-child td { border-bottom: 0; }
.tbl tbody tr { animation: rise .55s cubic-bezier(.2,.8,.2,1) both; animation-delay: calc(var(--i, 0) * 35ms); transition: background-color .25s ease; }
.tbl tbody tr:hover td { background: var(--row_hover); }
.tbl tr.sel td { font-weight: 700; background: var(--accent_soft); }
.tbl tr.sel td:first-child { box-shadow: inset 3px 0 0 var(--accent); }
.tbl td.num { font-variant-numeric: tabular-nums; }

/* ---------- Streamlit widgets ---------- */
button[role="tab"] { background: transparent !important; border-radius: 0 !important; padding: .75rem 1.15rem !important; margin-right: .2rem;
                     color: var(--ink2) !important; transition: color .3s ease; }
button[role="tab"]:hover { color: var(--ink) !important; }
button[role="tab"][aria-selected="true"] { color: var(--ink) !important; }
button[role="tab"] p { color: inherit !important; font-size: 1rem; font-weight: 650; }
[data-baseweb="tab-highlight"] { background-color: var(--accent) !important; height: 3px !important; }
[data-baseweb="tab-border"] { background-color: var(--rule) !important; height: 1px !important; }
[data-baseweb="tab-panel"] { padding-top: 1.8rem; animation: rise .6s cubic-bezier(.2,.8,.2,1) both; }

div[data-baseweb="input"], div[data-baseweb="base-input"], div[data-baseweb="select"] > div {
  background: var(--input) !important; border-color: var(--rule2) !important; border-radius: 10px !important;
  transition: border-color .25s ease; }
div[data-baseweb="input"]:focus-within, div[data-baseweb="select"] > div:focus-within { border-color: var(--accent) !important; }
div[data-baseweb="input"] input, div[data-baseweb="base-input"] input, div[data-baseweb="select"] * {
  color: var(--ink) !important; -webkit-text-fill-color: var(--ink) !important; }
div[data-baseweb="select"] svg, [data-testid="stNumberInput"] button svg { fill: var(--ink2) !important; color: var(--ink2) !important; }
[data-testid="stNumberInput"] button { background: transparent !important; }
[data-baseweb="popover"] > div, [data-baseweb="popover"] ul, [data-baseweb="menu"] { background: var(--panel) !important; color: var(--ink) !important;
  border-radius: 10px !important; }
[data-baseweb="popover"] li, [data-baseweb="popover"] li * { color: var(--ink) !important; }
[data-baseweb="popover"] li:hover, [data-baseweb="popover"] li[aria-selected="true"] { background: var(--accent_soft) !important; }
[data-testid="stSlider"] [data-testid="stTickBarMin"], [data-testid="stSlider"] [data-testid="stTickBarMax"] { color: var(--ink2) !important; }
[data-testid="stSliderThumbValue"] { color: var(--accent) !important; font-weight: 700; }
[data-testid="stSlider"]:has([aria-disabled="true"]) { opacity: .4; }
[data-testid="stCheckbox"] label span { color: var(--ink) !important; }

.stButton > button, .stDownloadButton > button {
  background: var(--ink); border: 1px solid var(--ink); border-radius: 10px; padding: .6rem 1.35rem; font-weight: 650;
  transition: background-color .25s ease, border-color .25s ease, transform .2s ease; }
.stButton > button p, .stDownloadButton > button p { color: var(--bg) !important; font-weight: 650; }
.stButton > button:hover, .stDownloadButton > button:hover { background: var(--accent); border-color: var(--accent); transform: translateY(-1px); }
.stButton > button:active, .stDownloadButton > button:active { transform: translateY(1px); }
.stDownloadButton > button p::after { content: "\\2193"; display: inline-block; margin-left: .55rem; transition: transform .25s ease; }
.stDownloadButton > button:hover p::after { transform: translateY(3px); }
[class*="st-key-preset_"] button { background: transparent !important; border: 1px solid var(--rule2) !important; padding: .4rem .6rem; }
[class*="st-key-preset_"] button p { color: var(--ink) !important; font-size: .86rem; }
[class*="st-key-preset_"] button:hover { border-color: var(--accent) !important; background: var(--accent_soft) !important; transform: none; }
[class*="st-key-preset_"] button:hover p { color: var(--accent) !important; }

[data-testid="stFileUploaderDropzone"] { background: var(--input); border: 1.5px dashed var(--rule2); border-radius: 14px; padding: 2rem;
  transition: border-color .3s ease, background-color .3s ease; }
[data-testid="stFileUploaderDropzone"]:hover { border-color: var(--accent); background: var(--accent_soft); }
[data-testid="stFileUploaderDropzone"] * { color: var(--ink2) !important; }
[data-testid="stFileUploaderDropzone"] button { background: var(--ink) !important; border-radius: 10px; }
[data-testid="stFileUploaderDropzone"] button * { color: var(--bg) !important; }
[data-testid="stFileUploaderFile"] * { color: var(--ink) !important; }
[data-testid="stExpander"] { background: var(--panel); border: 1px solid var(--rule); border-radius: 12px; }
[data-testid="stExpander"] summary p { color: var(--ink); font-weight: 650; }

/* ---------- Motion ---------- */
@keyframes rise { from { opacity: 0; transform: translateY(16px); } to { opacity: 1; transform: none; } }
@keyframes rule { from { transform: scaleX(0); } to { transform: scaleX(1); } }
@keyframes wipe { from { clip-path: inset(0 100% 0 0); } to { clip-path: inset(0 0 0 0); } }
@keyframes grow { from { width: 0; } }
@keyframes underline { from { background-size: 0% .075em; } }
@keyframes countup { from { --n: 0; } }
@keyframes pinmove { from { left: 0%; } }
@keyframes turn { from { transform: rotate(-90deg) scale(.2); opacity: 0; } to { transform: rotate(45deg) scale(1); opacity: 1; } }
@keyframes blink { 0%, 100% { opacity: 1; } 50% { opacity: .3; } }

@media (max-width: 760px) {
  .block-container { padding: .8rem .9rem 3rem; }
  .bars .row { grid-template-columns: 100px 1fr 48px; gap: .6rem; }
  .stat + .stat, .figure + .figure { padding-left: 0; border-left: 0; }
  .ruler { margin-left: 1rem; margin-right: 1rem; }
}
@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after { animation: none !important; transition: none !important; }
}
"""


def _grain(alpha: float) -> str:
    """A tiny inline paper-grain texture (no external file)."""
    svg = ("<svg xmlns='http://www.w3.org/2000/svg' width='180' height='180'><filter id='n'>"
           "<feTurbulence type='fractalNoise' baseFrequency='.85' numOctaves='2' stitchTiles='stitch'/>"
           f"<feColorMatrix values='0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 {alpha} 0'/></filter>"
           "<rect width='100%' height='100%' filter='url(#n)'/></svg>")
    return f'url("data:image/svg+xml,{quote(svg)}")'


def build_css(dark: bool) -> str:
    palette = DARK if dark else LIGHT
    variables = {k: v for k, v in palette.items() if k != "grain"}
    variables["grain"] = _grain(palette["grain"])
    css_vars = ";".join(f"--{k}:{v}" for k, v in variables.items())
    scheme = "dark" if dark else "light"
    return f"<style>:root{{{css_vars};color-scheme:{scheme}}}{BASE_CSS}</style>"
