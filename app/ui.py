"""
HTML building blocks for the dashboard (no Streamlit imports).
Every piece of text that can come from an uploaded file is escaped with esc() before it is placed
into HTML, so a CSV cannot inject markup or scripts.
"""

import html


def esc(value) -> str:
    return html.escape(str(value), quote=True)


def h(markup: str) -> str:
    """Collapse indented HTML to one line, so Markdown never mistakes it for a code block."""
    return " ".join(line.strip() for line in markup.strip().splitlines() if line.strip())


def rule() -> str:
    return '<div class="rule draw"></div>'


def header() -> str:
    return h("""
    <div class="mast">
      <div class="wordmark"><i></i>Early Learning Support</div>
      <div class="mast-sub">Career Solutions &middot; Prototype</div>
    </div>""")


def hero() -> str:
    return h("""
    <div class="hero">
      <div class="kicker r1">Prototype &middot; Synthetic data</div>
      <h1 class="r2">Who might need <span class="u">a hand</span>?</h1>
      <p class="r3">Enter what a learner did in the first two weeks. The model estimates a final score and says whether a
      check-in is worth making. Upload a class file to do the same for everyone at once.</p>
    </div>""")


def banners() -> str:
    return h("""
    <div class="readfirst">
      <div><b>Synthetic data</b>Demonstrates implementation, not real-learner effectiveness. Every model in this
      prototype was trained and tested on generated records.</div>
      <div><b>Use with care</b>Predictions support human review and optional learning assistance only. They must
      not decide admission, certification, grades or access to learning opportunities.</div>
    </div>""")


def note(kind: str, title: str, body: str = "") -> str:
    cls = {"info": "", "warn": " warn", "error": " error", "ok": ""}[kind]
    body_html = f" {esc(body)}" if body else ""
    return h(f'<div class="note{cls}"><b>{esc(title)}</b>{body_html}</div>')


def card_title(title: str, sub: str = "") -> str:
    sub_html = f'<div class="sec-sub">{esc(sub)}</div>' if sub else '<div style="height:.9rem"></div>'
    return h(f'<div class="sec-title">{esc(title)}</div>{sub_html}')


def _ruler(pin: float, pin_label: str, mark: float = None, mark_label: str = "", mark_tone: str = "",
           hatch: tuple = None, hatch_right: bool = False) -> str:
    """A ruled 0-100 scale with a moving pin, an optional dashed marker and an optional hatched zone."""
    parts = ['<div class="ruler"><div class="ruler-axis"></div><div class="ruler-cap"></div>']
    if hatch:
        a, b, label = hatch
        parts.append(f'<div class="ruler-hatch{" r" if hatch_right else ""}" style="--a:{a:.1f}%;--b:{b:.1f}%">'
                     f'<span>{esc(label)}</span></div>')
    if mark is not None:
        parts.append(f'<div class="ruler-mark {mark_tone}" style="--x:{mark:.1f}%"><span>{esc(mark_label)}</span></div>')
    parts.append(f'<div class="ruler-pin" style="--x:{pin:.1f}%"><span>{esc(pin_label)}</span></div></div>')
    parts.append('<div class="ruler-ends"><span>0</span><span>100</span></div>')
    return "".join(parts)


def score_card(score: float, mae: float, benchmark: float) -> str:
    s = max(0.0, min(100.0, round(float(score), 1)))
    whole = int(s)
    frac = f"{s - whole:.1f}"[1:]
    ruler = _ruler(s, f"{s:.1f}", mark=benchmark, mark_label=f"Benchmark {benchmark:g}",
                   hatch=(0, benchmark, "Below benchmark"))
    return h(f"""
    <div class="panel fade" style="--d:1">
      <div class="label">Estimated final score</div>
      <div class="score-line">
        <div class="numeral" role="img" aria-label="Estimated final score {s} out of 100"><span class="count" style="--target:{whole}"></span><span class="dec">{frac}</span></div>
        <div class="outof">out of 100</div>
      </div>
      {ruler}
      <p class="fine">On held-out data, predicted scores were off by <b>{mae:g} score points</b> on average (MAE).
      MAE is an average error, not a confidence interval.</p>
    </div>""")


def support_card(flagged: bool, label: str, clf_name: str, score: float, score_type: str, show_probability: bool,
                 threshold: float, benchmark: float, recall: float, precision: float) -> str:
    tone = "warn" if flagged else "good"
    lead = (f"This learner's first two weeks look similar to those of learners who often finished below the "
            f"benchmark of {benchmark:g}. A coordinator may want to check in and offer help."
            if flagged else
            f"This learner's first two weeks look similar to those of learners who finished at or above the "
            f"benchmark of {benchmark:g}. The model suggests no extra support for now.")
    if show_probability:
        p = max(0.0, min(1.0, float(score))) * 100
        t = max(0.0, min(1.0, float(threshold))) * 100
        middle = (f'<div class="label" style="margin-top:1.9rem">Estimated likelihood of finishing below the benchmark</div>'
                  + _ruler(p, f"{p:.0f}%", mark=t, mark_label=f"Review at {t:.0f}%", mark_tone="amber",
                           hatch=(t, 100, "Flagged for review"), hatch_right=True)
                  + f'<p class="fine">Model estimate: <b>{p:.0f}%</b>. Learners at or above {t:.0f}% are flagged as a priority '
                    f'for human review. This is an estimate from synthetic data, not a certainty.</p>')
    else:
        middle = (f'<div class="stats"><div class="stat"><b>{float(score):+.2f}</b><span>Decision score (a signed margin, '
                  f'not a probability). Learners at or above {float(threshold):+.2f} are flagged.</span></div></div>')
    return h(f"""
    <div class="panel fade" style="--d:1">
      <span class="tag {tone}"><i></i>{esc(clf_name)}</span>
      <div class="status">{esc(label)}</div>
      <div class="lead">{esc(lead)}</div>
      {middle}
      <div class="stats">
        <div class="stat"><b>{recall * 100:.0f}%</b><span>of learners who finished below the benchmark were flagged
        (recall, held-out data)</span></div>
        <div class="stat"><b>{precision * 100:.0f}%</b><span>of flagged learners did finish below the benchmark
        (precision, held-out data)</span></div>
      </div>
      <p class="fine">A flag is a priority for human review, not a judgement of the learner.</p>
    </div>""")


def _is_number(text: str) -> bool:
    try:
        float(str(text).replace("%", "").replace(",", ""))
        return True
    except ValueError:
        return False


def kpis(items) -> str:
    """items: list of (label, value, sub)."""
    cells = "".join(
        f'<div class="figure" style="--d:{i}"><div class="label">{esc(l)}</div>'
        f'<div class="fig-val{"" if _is_number(v) else " text"}">{esc(v)}</div><div class="fig-sub">{esc(s)}</div></div>'
        for i, (l, v, s) in enumerate(items))
    return h(f'<div class="figures">{cells}</div>')


def count_kpis(items) -> str:
    """items: list of (label, integer, sub). Numbers count up on load."""
    cells = "".join(
        f'<div class="figure" style="--d:{i}"><div class="label">{esc(l)}</div>'
        f'<div class="fig-val"><span class="count" style="--target:{int(v)}"></span></div>'
        f'<div class="fig-sub">{esc(s)}</div></div>'
        for i, (l, v, s) in enumerate(items))
    return h(f'<div class="figures">{cells}</div>')


def bars(title: str, items, baseline: float = None, baseline_label: str = "") -> str:
    """items: list of (label, value 0-1, is_selected)."""
    rows = []
    for i, (label, value, sel) in enumerate(items):
        v = max(0.0, min(1.0, float(value))) * 100
        base = f'<div class="base" style="left:{baseline * 100:.1f}%"></div>' if baseline is not None else ""
        rows.append(f'<div class="row"><div class="name">{esc(label)}</div>'
                    f'<div class="track"><div class="fill {"sel" if sel else ""}" style="--w:{v:.1f}%;--d:{i}"></div>{base}</div>'
                    f'<div class="val">{value:.3f}</div></div>')
    foot = (f'<p class="fine">Dashed line: {esc(baseline_label)} ({baseline:.3f}).</p>' if baseline is not None else "")
    return h(f'<div class="panel bars fade"><div class="sec-title" style="font-size:1.25rem;margin-bottom:.6rem">{esc(title)}</div>'
             f'{"".join(rows)}{foot}</div>')


def html_table(rows, columns, selected_col: str = None, selected_value=None, numeric_cols=()) -> str:
    """rows: list of dicts. Every cell is escaped."""
    head = "".join(f"<th>{esc(c)}</th>" for c in columns)
    body = []
    for i, r in enumerate(rows):
        cls = ' class="sel"' if selected_col and r.get(selected_col) == selected_value else ""
        cells = "".join(f'<td class="{"num" if c in numeric_cols else ""}">{esc(r.get(c, ""))}</td>' for c in columns)
        body.append(f'<tr{cls} style="--i:{min(i, 14)}">{cells}</tr>')
    return h(f'<div class="tbl-wrap"><table class="tbl"><thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table></div>')
