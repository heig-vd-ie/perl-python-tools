# =============================================================================
#  pi_tuner_lib.py
#
#  Library of all functions used by PI_Tuner.ipynb.
#  Import with:  from pi_tuner_lib import *
#
#  Dependencies: numpy, control, plotly, ipywidgets
#
#  No global configuration variables live in this file.
#  Every function receives what it needs as explicit arguments.
# =============================================================================

import warnings
import numpy as np
import control as ct
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ipywidgets as widgets
from IPython.display import display

warnings.filterwarnings("ignore")

# Laplace variable — re-exported so the notebook can write transfer functions
# using natural syntax:  G = 1 / (L*s + R)
s = ct.tf('s')

PALETTE = [
    "#2196F3", "#FF5722", "#4CAF50", "#9C27B0", "#FF9800",
    "#00BCD4", "#E91E63", "#8BC34A", "#795548", "#607D8B",
]


# =============================================================================
#  SECTION 1 — PI CONTROLLER BUILDER
# =============================================================================

def build_pi(Kp, Ki):
    """
    Build a PI controller transfer function.

        C(s) = Kp + Ki/s = (Kp·s + Ki) / s

    Parameters
    ----------
    Kp : float — proportional gain
    Ki : float — integral gain

    Returns
    -------
    C : control.TransferFunction

    Notes
    -----
    When Ki ≈ 0 the integrator is omitted and a pure proportional
    controller is returned to avoid a numerical pole at the origin.
    """
    if abs(Ki) < 1e-14:
        return ct.tf([Kp], [1])
    return ct.tf([Kp, Ki], [1, 0])


# =============================================================================
#  SECTION 2 — ANALYSIS COMPUTATIONS
# =============================================================================

def compute_bode(sys_tf, freqs_hz):
    """
    Evaluate the Bode response of a transfer function.

    Parameters
    ----------
    sys_tf   : control.TransferFunction
    freqs_hz : ndarray — frequency vector in Hz

    Returns
    -------
    mag_db    : ndarray — magnitude in dB
    phase_deg : ndarray — unwrapped phase in degrees
    """
    omega = 2.0 * np.pi * np.asarray(freqs_hz)
    mag, phase, _ = ct.bode(sys_tf, omega, plot=False)
    mag_db    = 20.0 * np.log10(np.maximum(np.abs(mag), 1e-15))
    phase_deg = np.degrees(np.unwrap(phase))
    return mag_db, phase_deg


def compute_margins(sys_ol):
    """
    Compute gain margin, phase margin and crossover frequencies.

    Parameters
    ----------
    sys_ol : control.TransferFunction — open-loop TF G_OL(s)

    Returns
    -------
    dict with keys
        gm_db   : float — gain margin [dB]  (∞ if no phase crossover)
        pm_deg  : float — phase margin [°]
        wgc_hz  : float — gain crossover frequency [Hz]  (|G_OL| = 1)
        wpc_hz  : float — phase crossover frequency [Hz] (∠G_OL = −180°)
        stable  : bool  — True when both margins are positive
    """
    try:
        gm, pm, wpc, wgc  = ct.margin(sys_ol)

        gm_db  = float(20.0 * np.log10(gm)) \
                 if (gm is not None and np.isfinite(gm) and gm > 0) else np.inf
        pm_deg = float(pm)  if (pm  is not None and np.isfinite(pm))          else np.nan
        wgc_hz = float(wgc) / (2.0 * np.pi) \
                 if (wgc is not None and np.isfinite(wgc) and wgc > 0) else np.nan
        wpc_hz = float(wpc) / (2.0 * np.pi) \
                 if (wpc is not None and np.isfinite(wpc) and wpc > 0) else np.nan

        stable = bool(
            (not np.isnan(pm_deg)) and (pm_deg > 0) and
            (np.isinf(gm_db) or gm_db > 0)
        )
        return dict(gm_db=gm_db, pm_deg=pm_deg, wgc_hz=wgc_hz, wpc_hz=wpc_hz, stable=stable)

    except Exception as exc:
        return dict(gm_db=np.nan, pm_deg=np.nan, wgc_hz=np.nan, wpc_hz=np.nan,
                    stable=False, error=str(exc))


def _step_metrics(t, y):
    """
    Internal helper — extract rise time, settling time, overshoot and
    steady-state gain from a step response array.
    """
    if len(y) == 0:
        return dict(rise_time=np.nan, settling_time=np.nan,
                    overshoot_pct=np.nan, steady_state=np.nan)

    y_ss  = float(y[-1])
    y_max = float(np.max(y))

    # Overshoot
    if abs(y_ss) > 1e-12:
        overshoot = (y_max - y_ss) / abs(y_ss) * 100.0
    else:
        overshoot = 0.0

    # Rise time 10 % → 90 %
    try:
        idx10 = np.where(y >= 0.1 * y_ss)[0]
        idx90 = np.where(y >= 0.9 * y_ss)[0]
        rise_time = float(t[idx90[0]] - t[idx10[0]]) \
                    if (len(idx10) > 0 and len(idx90) > 0) else np.nan
    except Exception:
        rise_time = np.nan

    # Settling time — last sample outside the 2 % band
    try:
        band    = 0.02 * abs(y_ss)
        outside = np.where(np.abs(y - y_ss) > band)[0]
        settling_time = float(t[outside[-1]]) if len(outside) > 0 else float(t[0])
    except Exception:
        settling_time = np.nan

    return dict(rise_time=rise_time, settling_time=settling_time,
                overshoot_pct=overshoot, steady_state=y_ss)


def compute_step_response(sys_cl, t_end, n_pts):
    """
    Compute the closed-loop unit step response.

    Parameters
    ----------
    sys_cl : control.TransferFunction — closed-loop system
    t_end  : float — simulation end time [s]
    n_pts  : int   — number of time samples

    Returns
    -------
    t       : ndarray — time vector [s]
    y       : ndarray — output vector
    metrics : dict    — rise_time, settling_time, overshoot_pct, steady_state
    """
    t_arr = np.linspace(0.0, t_end, n_pts)
    try:
        t_out, y_out = ct.step_response(sys_cl, T=t_arr)
        t_out = np.asarray(t_out).flatten()
        y_out = np.asarray(y_out).flatten()
    except Exception:
        t_out = t_arr
        y_out = np.zeros_like(t_arr)
    return t_out, y_out, _step_metrics(t_out, y_out)


def get_poles_zeros(sys_tf):
    """
    Return the poles and zeros of a transfer function.

    Returns
    -------
    poles : ndarray (complex)
    zeros : ndarray (complex)
    """
    return np.array(sys_tf.poles()), np.array(sys_tf.zeros())


# =============================================================================
#  SECTION 3 — SHARED PLOTLY STYLE HELPERS
# =============================================================================

_LAYOUT_BASE = dict(
    plot_bgcolor  = 'white',
    paper_bgcolor = 'white',
    font          = dict(family='Arial, monospace', size=11),
    hoverlabel    = dict(bgcolor='white', font_size=11),
)


def _axis(title, log=False, **kw):
    """Return a dict suitable for fig.update_xaxes / update_yaxes."""
    d = dict(
        title_text  = title,
        showgrid    = True,
        gridcolor   = '#ebebeb',
        zeroline    = True,
        zerolinecolor = '#cccccc',
        zerolinewidth = 1.5,
        linecolor   = '#bbbbbb',
        showline    = True,
        mirror      = True,
        **kw,
    )
    if log:
        d['type'] = 'log'
    return d


# =============================================================================
#  SECTION 4 — FIGURE BUILDERS  (return go.Figure, not FigureWidget)
# =============================================================================

def _build_ol_bode(mag_db, phase_deg, freqs_hz, margins, height):
    """
    Build the open-loop Bode diagram go.Figure.

    Trace layout (stable indices for in-place updates):
        data[0] — |G_OL| magnitude
        data[1] — ∠G_OL phase
        data[2] — gain margin vertical line (on magnitude subplot)
        data[3] — phase margin vertical line (on phase subplot)
    """
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.10,
        subplot_titles=['Magnitude', 'Phase'],
    )

    # ── Main traces ──────────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=freqs_hz, y=mag_db, mode='lines',
        name='|G_OL(jω)|',
        line=dict(color='#FF5722', width=2.2),
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=freqs_hz, y=phase_deg, mode='lines',
        name='∠G_OL(jω)',
        line=dict(color='#FF9800', width=2.2),
    ), row=2, col=1)

    # ── Margin vertical lines (always present; hidden when NaN) ──────────────
    gm  = margins.get('gm_db',  np.nan)
    pm  = margins.get('pm_deg', np.nan)
    wpc = margins.get('wpc_hz', np.nan)
    wgc = margins.get('wgc_hz', np.nan)

    wpc_x = [wpc, wpc] if (_valid_freq(wpc)) else [None, None]
    wgc_x = [wgc, wgc] if (_valid_freq(wgc)) else [None, None]

    fig.add_trace(go.Scatter(
        x=wpc_x, y=[-400, 400], mode='lines',
        name=_gm_label(gm),
        line=dict(color='#E91E63', width=2.0, dash='dot'),
        legendgroup='gm',
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=wgc_x, y=[-720, 360], mode='lines',
        name=_pm_label(pm),
        line=dict(color='#00BCD4', width=2.0, dash='dot'),
        legendgroup='pm',
    ), row=2, col=1)

    # ── Reference lines (shapes — unaffected by trace updates) ───────────────
    fig.add_hline(y=0,    line_dash='dash', line_color='#aaaaaa',
                  line_width=1.2, row=1, col=1)
    fig.add_hline(y=-180, line_dash='dash', line_color='#aaaaaa',
                  line_width=1.2, row=2, col=1)

    # ── Axes ─────────────────────────────────────────────────────────────────
    fig.update_xaxes(**_axis('',               log=True), row=1, col=1)
    fig.update_xaxes(**_axis('Frequency (Hz)', log=True), row=2, col=1)
    fig.update_yaxes(**_axis('Magnitude (dB)'),           row=1, col=1)
    fig.update_yaxes(**_axis('Phase (°)'),                row=2, col=1)
    fig.update_yaxes(range=[min(mag_db) - 20, max(mag_db) + 20], row=1, col=1)
    fig.update_yaxes(range=[min(phase_deg) - 10, max(phase_deg) + 10], row=2, col=1)

    fig.update_layout(
        **_LAYOUT_BASE,
        height=height,
        title_text='Open-Loop Bode Diagram',
        title_font=dict(size=13, color='#222'),
        margin=dict(l=65, r=15, t=60, b=15),
        legend=dict(orientation='h', y=-0.08, x=0,
                    bgcolor='rgba(255,255,255,0.85)', bordercolor='#ddd',
                    borderwidth=1),
        showlegend=True,
    )
    return fig


def _build_cl_bode(mag_db, phase_deg, freqs_hz, height):
    """
    Build the closed-loop Bode diagram go.Figure.

    Trace layout:
        data[0] — |G_CL| magnitude
        data[1] — ∠G_CL phase
    """
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.10,
        subplot_titles=['Magnitude', 'Phase'],
    )

    fig.add_trace(go.Scatter(
        x=freqs_hz, y=mag_db, mode='lines',
        name='|G_CL(jω)|',
        line=dict(color='#4CAF50', width=2.2),
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=freqs_hz, y=phase_deg, mode='lines',
        name='∠G_CL(jω)',
        line=dict(color='#8BC34A', width=2.2),
    ), row=2, col=1)

    # −3 dB bandwidth reference
    fig.add_hline(y=-3, line_dash='dash', line_color='#aaaaaa',
                  line_width=1.2, row=1, col=1)
    fig.add_annotation(
        x=0, y=-3, xref='paper', yref='y1',
        text='−3 dB', showarrow=False,
        font=dict(size=9, color='#999999'), xanchor='left',
    )

    fig.update_xaxes(**_axis('',               log=True), row=1, col=1)
    fig.update_xaxes(**_axis('Frequency (Hz)', log=True), row=2, col=1)
    fig.update_yaxes(**_axis('Magnitude (dB)'),           row=1, col=1)
    fig.update_yaxes(**_axis('Phase (°)'),                row=2, col=1)

    fig.update_layout(
        **_LAYOUT_BASE,
        height=height,
        title_text='Closed-Loop Bode Diagram',
        title_font=dict(size=13, color='#222'),
        margin=dict(l=65, r=15, t=60, b=15),
        legend=dict(orientation='h', y=-0.08, x=0,
                    bgcolor='rgba(255,255,255,0.85)', bordercolor='#ddd',
                    borderwidth=1),
        showlegend=True,
    )
    return fig


def _build_step(t, y, metrics, height):
    """
    Build the step response go.Figure.

    Trace layout:
        data[0] — output y(t)
    Annotations are managed separately by _set_step_annotations().
    """
    t_ms = t * 1e3
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=t_ms, y=y, mode='lines',
        name='y(t)',
        line=dict(color='#9C27B0', width=2.2),
    ))

    # Reference and settling band (shapes — stable)
    fig.add_hline(y=1.0, line_dash='dash', line_color='#888888', line_width=1.2)
    fig.add_hrect(y0=0.98, y1=1.02,
                  fillcolor='#4CAF50', opacity=0.07, line_width=0)

    _set_step_annotations(fig, t, y, metrics)

    fig.update_layout(
        **_LAYOUT_BASE,
        height=height,
        title_text='Closed-Loop Step Response',
        title_font=dict(size=13, color='#222'),
        xaxis=_axis('Time (ms)'),
        yaxis=_axis('Amplitude'),
        margin=dict(l=65, r=15, t=55, b=55),
        showlegend=False,
    )
    return fig


def _set_step_annotations(fig, t, y, metrics):
    """
    Replace all annotations on a step figure with current performance labels.
    Called both at creation time and during interactive updates.
    """
    anns = []
    t_ms      = t * 1e3
    overshoot = metrics.get('overshoot_pct',  np.nan)
    settling  = metrics.get('settling_time',  np.nan)
    rise      = metrics.get('rise_time',      np.nan)
    y_ss      = metrics.get('steady_state',   1.0)

    if not np.isnan(overshoot) and overshoot > 0.5:
        idx = int(np.argmax(y))
        anns.append(dict(
            x=float(t_ms[idx]), y=float(y[idx]),
            text=f'<b>OS = {overshoot:.1f} %</b>',
            showarrow=True, arrowhead=2, arrowcolor='#FF5722',
            ax=20, ay=-40,
            font=dict(size=10, color='#FF5722'),
        ))

    if not np.isnan(settling) and settling > 0:
        anns.append(dict(
            x=float(settling * 1e3), y=float(y_ss),
            text=f'<b>Ts = {settling*1e3:.1f} ms</b>',
            showarrow=True, arrowhead=2, arrowcolor='#1565C0',
            ax=20, ay=40,
            font=dict(size=10, color='#1565C0'),
        ))

    if not np.isnan(rise) and rise > 0:
        anns.append(dict(
            x=float(rise * 1e3), y=float(0.5 * y_ss),
            text=f'<b>Tr = {rise*1e3:.1f} ms</b>',
            showarrow=False, xanchor='left',
            font=dict(size=10, color='#2E7D32'),
        ))

    fig.update_layout(annotations=anns)


def _build_pz(G_OL, G_CL, height):
    """
    Build the pole-zero map go.Figure.

    Four traces are always present (even if empty) so that indices
    are stable for in-place updates:
        data[0] — OL zeros   (○ blue)
        data[1] — OL poles   (× orange-red)
        data[2] — CL zeros   (○ green)
        data[3] — CL poles   (× purple)
    """
    poles_ol, zeros_ol = get_poles_zeros(G_OL)
    poles_cl, zeros_cl = get_poles_zeros(G_CL)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=_re(zeros_ol), y=_im(zeros_ol), mode='markers',
        name='OL Zeros',
        marker=dict(symbol='circle-open', size=13,
                    color='#2196F3', line=dict(width=2.5)),
    ))
    fig.add_trace(go.Scatter(
        x=_re(poles_ol), y=_im(poles_ol), mode='markers',
        name='OL Poles',
        marker=dict(symbol='x', size=13,
                    color='#FF5722', line=dict(width=2.5)),
    ))
    fig.add_trace(go.Scatter(
        x=_re(zeros_cl), y=_im(zeros_cl), mode='markers',
        name='CL Zeros',
        marker=dict(symbol='circle-open', size=10,
                    color='#4CAF50', line=dict(width=2.0)),
    ))
    fig.add_trace(go.Scatter(
        x=_re(poles_cl), y=_im(poles_cl), mode='markers',
        name='CL Poles',
        marker=dict(symbol='x', size=10,
                    color='#9C27B0', line=dict(width=2.0)),
    ))

    # Imaginary axis — stability boundary
    fig.add_vline(x=0, line_dash='dash', line_color='#aaaaaa', line_width=1.2)

    fig.update_layout(
        **_LAYOUT_BASE,
        height=height,
        title_text='Pole-Zero Map',
        title_font=dict(size=13, color='#222'),
        xaxis=_axis('Real Part  (σ)'),
        yaxis=dict(**_axis('Imaginary Part  (jω)'),
                   scaleanchor='x', scaleratio=1),
        margin=dict(l=65, r=15, t=55, b=55),
        legend=dict(orientation='h', y=-0.18, x=0,
                    bgcolor='rgba(255,255,255,0.85)', bordercolor='#ddd',
                    borderwidth=1),
        showlegend=True,
    )
    return fig


# =============================================================================
#  SECTION 5 — IN-PLACE FIGURE-WIDGET UPDATERS
# =============================================================================

def _upd_ol_bode(fw, mag_db, phase_deg, margins):
    """Update open-loop Bode FigureWidget without redrawing."""
    gm  = margins.get('gm_db',  np.nan)
    pm  = margins.get('pm_deg', np.nan)
    wpc = margins.get('wpc_hz', np.nan)
    wgc = margins.get('wgc_hz', np.nan)

    wpc_x = [wpc, wpc] if _valid_freq(wpc) else [None, None]
    wgc_x = [wgc, wgc] if _valid_freq(wgc) else [None, None]

    with fw.batch_update():
        fw.data[0].y    = mag_db
        fw.data[1].y    = phase_deg
        fw.data[2].x    = wpc_x
        fw.data[2].name = _gm_label(gm)
        fw.data[3].x    = wgc_x
        fw.data[3].name = _pm_label(pm)
        fw.layout.yaxis.range  = [min(mag_db)   - 20, max(mag_db)   + 20]
        fw.layout.yaxis2.range = [min(phase_deg) - 10, max(phase_deg) + 10]


def _upd_cl_bode(fw, mag_db, phase_deg):
    """Update closed-loop Bode FigureWidget without redrawing."""
    with fw.batch_update():
        fw.data[0].y = mag_db
        fw.data[1].y = phase_deg


def _upd_step(fw, t, y, metrics):
    """Update step response FigureWidget without redrawing."""
    with fw.batch_update():
        fw.data[0].x = (t * 1e3).tolist()
        fw.data[0].y = y.tolist()
    _set_step_annotations(fw, t, y, metrics)


def _upd_pz(fw, G_OL, G_CL):
    """Update pole-zero map FigureWidget without redrawing."""
    poles_ol, zeros_ol = get_poles_zeros(G_OL)
    poles_cl, zeros_cl = get_poles_zeros(G_CL)
    with fw.batch_update():
        fw.data[0].x = _re(zeros_ol);  fw.data[0].y = _im(zeros_ol)
        fw.data[1].x = _re(poles_ol);  fw.data[1].y = _im(poles_ol)
        fw.data[2].x = _re(zeros_cl);  fw.data[2].y = _im(zeros_cl)
        fw.data[3].x = _re(poles_cl);  fw.data[3].y = _im(poles_cl)


# =============================================================================
#  SECTION 6 — INFO PANEL HTML
# =============================================================================

def _info_html(margins, metrics, Kp, Ki):
    """Build the HTML string for the status / metrics info bar."""
    gm  = margins.get('gm_db',  np.nan)
    pm  = margins.get('pm_deg', np.nan)
    wgc = margins.get('wgc_hz', np.nan)
    wpc = margins.get('wpc_hz', np.nan)
    stable = margins.get('stable', False)

    rise      = metrics.get('rise_time',      np.nan)
    settling  = metrics.get('settling_time',  np.nan)
    overshoot = metrics.get('overshoot_pct',  np.nan)
    y_ss      = metrics.get('steady_state',   np.nan)

    def fv(v, unit='', prec=2):
        if v is None or (isinstance(v, float) and (np.isnan(v))): return '—'
        if isinstance(v, float) and np.isinf(v): return '∞'
        return f'{v:.{prec}f} {unit}'.strip()

    def fms(v):
        return fv(v * 1e3 if not np.isnan(v) else v, 'ms')

    st_color = '#1a7f37' if stable else '#cf222e'
    st_text  = '✅ STABLE' if stable else '❌ UNSTABLE'
    gm_str   = '∞' if (isinstance(gm, float) and np.isinf(gm)) else fv(gm, 'dB')

    return f"""
<div style="font-family:monospace; font-size:11.5px; background:#fafafa;
            border:1px solid #ddd; border-radius:6px; padding:8px 18px;
            display:flex; gap:36px; flex-wrap:wrap; line-height:2.1; margin-bottom:4px">
  <div>
    <span style="color:{st_color}; font-weight:bold; font-size:13px">{st_text}</span><br>
    C(s) = {Kp:.4g} + {Ki:.4g}/s
  </div>
  <div>
    <b style="color:#444">Open-Loop Margins</b><br>
    Phase Margin &nbsp;: <b style="color:#00838F">{fv(pm, '°')}</b><br>
    Gain Margin &nbsp;&nbsp;: <b style="color:#AD1457">{gm_str}</b><br>
    ω<sub>gc</sub> = {fv(wgc, 'Hz')} &nbsp;|&nbsp; ω<sub>pc</sub> = {fv(wpc, 'Hz')}
  </div>
  <div>
    <b style="color:#444">Closed-Loop Step Response</b><br>
    Rise time (10–90 %) &nbsp;: <b>{fms(rise)}</b><br>
    Settling time (2 %) &nbsp;: <b>{fms(settling)}</b><br>
    Overshoot &nbsp;: <b>{fv(overshoot, '%')}</b> &nbsp;|&nbsp; SS = {fv(y_ss, '', 4)}
  </div>
</div>"""


# =============================================================================
#  SECTION 7 — MASTER TUNER FUNCTION
# =============================================================================

def create_pi_tuner(
    plant_tf,
    feedback_tf     = None,
    init_Kp         = 1.0,
    init_Ki         = 1.0,
    f_min           = 0.1,
    f_max           = 10_000.0,
    n_freq          = 2000,
    t_end           = 1.0,
    n_time          = 4000,
    Kp_range        = (1e-4, 1e4),
    Ki_range        = (1e-4, 1e5),
    fig_height      = 440,
):
    """
    Launch the interactive PI tuner widget.

    Parameters
    ----------
    plant_tf    : control.TransferFunction  — plant G(s)
    feedback_tf : control.TransferFunction | None
                  Feedback/sensor path H(s).  None → unity feedback (H = 1).
    init_Kp     : float  — initial proportional gain
    init_Ki     : float  — initial integral gain
    f_min       : float  — Bode plot lower frequency bound [Hz]
    f_max       : float  — Bode plot upper frequency bound [Hz]
    n_freq      : int    — number of Bode frequency points
    t_end       : float  — step response simulation end time [s]
    n_time      : int    — number of step response time samples
    Kp_range    : (min, max) — Kp slider bounds
    Ki_range    : (min, max) — Ki slider bounds
    fig_height  : int    — height [px] of each Bode plot panel

    The four interactive plots displayed are:
        • Open-loop Bode   — magnitude + phase, with GM/PM markers
        • Closed-loop Bode — magnitude + phase
        • Step response    — with rise time, settling time, overshoot labels
        • Pole-zero map    — open-loop and closed-loop poles/zeros
    """
    if feedback_tf is None:
        feedback_tf = ct.tf([1], [1])

    freqs_hz = np.geomspace(f_min, f_max, n_freq)

    # ── Initial computation ──────────────────────────────────────────────────
    C    = build_pi(init_Kp, init_Ki)
    G_OL = C * plant_tf
    G_CL = ct.feedback(G_OL, feedback_tf)

    ol_m, ol_p = compute_bode(G_OL, freqs_hz)
    cl_m, cl_p = compute_bode(G_CL, freqs_hz)
    margins    = compute_margins(G_OL)
    t, y, metr = compute_step_response(G_CL, t_end, n_time)

    # ── Figure widgets ───────────────────────────────────────────────────────
    fw_ol   = go.FigureWidget(_build_ol_bode(ol_m, ol_p, freqs_hz, margins, fig_height))
    fw_cl   = go.FigureWidget(_build_cl_bode(cl_m, cl_p, freqs_hz, fig_height))
    fw_step = go.FigureWidget(_build_step(t, y, metr, fig_height - 60))
    fw_pz   = go.FigureWidget(_build_pz(G_OL, G_CL, fig_height - 60))

    # ── Info panel ───────────────────────────────────────────────────────────
    w_info = widgets.HTML(value=_info_html(margins, metr, init_Kp, init_Ki))

    # ── Sliders (log scale — wide gain range) ────────────────────────────────
    _sty = {'description_width': '145px'}
    _lay = widgets.Layout(width='450px')

    w_Kp = widgets.FloatLogSlider(
        value=init_Kp, base=10,
        min=np.log10(max(Kp_range[0], 1e-12)),
        max=np.log10(Kp_range[1]),
        step=0.02,
        description='Kp — prop. gain',
        style=_sty, layout=_lay,
        continuous_update=True,
        readout_format='.4g',
    )
    w_Ki = widgets.FloatLogSlider(
        value=init_Ki, base=10,
        min=np.log10(max(Ki_range[0], 1e-12)),
        max=np.log10(Ki_range[1]),
        step=0.02,
        description='Ki — integ. gain',
        style=_sty, layout=_lay,
        continuous_update=True,
        readout_format='.4g',
    )

    # ── Callback (triggered on every slider change) ──────────────────────────
    def _on_change(_change):
        Kp_, Ki_ = w_Kp.value, w_Ki.value

        C_    = build_pi(Kp_, Ki_)
        G_OL_ = C_ * plant_tf
        G_CL_ = ct.feedback(G_OL_, feedback_tf)

        ol_m_, ol_p_ = compute_bode(G_OL_, freqs_hz)
        cl_m_, cl_p_ = compute_bode(G_CL_, freqs_hz)
        marg_        = compute_margins(G_OL_)
        t_, y_, met_ = compute_step_response(G_CL_, t_end, n_time)

        _upd_ol_bode(fw_ol,   ol_m_, ol_p_, marg_)
        _upd_cl_bode(fw_cl,   cl_m_, cl_p_)
        _upd_step   (fw_step, t_, y_, met_)
        _upd_pz     (fw_pz,   G_OL_, G_CL_)
        w_info.value = _info_html(marg_, met_, Kp_, Ki_)

    w_Kp.observe(_on_change, names='value')
    w_Ki.observe(_on_change, names='value')

    # ── Widget layout ────────────────────────────────────────────────────────
    title = widgets.HTML(
        '<h3 style="font-family:monospace; color:#111; margin:2px 0 6px 0;">'
        '⚙ PI Controller Tuner</h3>'
    )
    ctrl_box = widgets.VBox([
        widgets.HTML(
            '<b style="font-family:monospace; color:#2196F3; font-size:12px">'
            '── Controller Parameters ──</b>'
        ),
        w_Kp,
        w_Ki,
    ])
    row_top = widgets.HBox([fw_ol, fw_cl],   layout=widgets.Layout(gap='10px'))
    row_bot = widgets.HBox([fw_step, fw_pz], layout=widgets.Layout(gap='10px'))

    display(widgets.VBox(
        [title, ctrl_box, w_info, row_top, row_bot],
        layout=widgets.Layout(gap='4px'),
    ))


# =============================================================================
#  SECTION 8 — STATIC ANALYSIS REPORT
# =============================================================================

def print_analysis(plant_tf, feedback_tf, Kp, Ki, t_end=1.0, n_time=4000):
    """
    Print a detailed stability and performance analysis to stdout.

    Useful to get precise numbers after tuning interactively.

    Parameters
    ----------
    plant_tf    : control.TransferFunction
    feedback_tf : control.TransferFunction | None  (None → unity)
    Kp, Ki      : float — gains to evaluate
    t_end       : float — step simulation duration [s]
    n_time      : int   — number of step time samples
    """
    if feedback_tf is None:
        feedback_tf = ct.tf([1], [1])

    C    = build_pi(Kp, Ki)
    G_OL = C * plant_tf
    G_CL = ct.feedback(G_OL, feedback_tf)

    margins          = compute_margins(G_OL)
    _, _, metrics    = compute_step_response(G_CL, t_end, n_time)
    poles_cl, _      = get_poles_zeros(G_CL)

    gm  = margins.get('gm_db',  np.nan)
    pm  = margins.get('pm_deg', np.nan)
    wgc = margins.get('wgc_hz', np.nan)
    wpc = margins.get('wpc_hz', np.nan)

    def fv(v, unit='', prec=2):
        if v is None or np.isnan(v): return '—'
        if np.isinf(v): return '∞'
        return f'{v:.{prec}f} {unit}'.strip()

    sep = '─' * 54
    print(sep)
    print('  PI CONTROLLER — ANALYSIS REPORT')
    print(sep)
    print(f'  Kp = {Kp:.6g}   |   Ki = {Ki:.6g}')
    print(f'  C(s) = {Kp:.4g} + {Ki:.4g} / s')
    print()
    print('  ── Open-Loop Stability ─────────────────────────────')
    print(f'  {"✓ STABLE" if margins.get("stable") else "✗ UNSTABLE"}')
    print(f'  Phase margin        : {fv(pm, "°")}')
    gm_str = '∞' if (isinstance(gm, float) and np.isinf(gm)) else fv(gm, 'dB')
    print(f'  Gain margin         : {gm_str}')
    print(f'  Gain crossover freq : {fv(wgc, "Hz")}   (|G_OL| = 1)')
    print(f'  Phase crossover freq: {fv(wpc, "Hz")}   (∠G_OL = −180°)')
    print()

    rise      = metrics.get('rise_time',      np.nan)
    settling  = metrics.get('settling_time',  np.nan)
    overshoot = metrics.get('overshoot_pct',  np.nan)
    y_ss      = metrics.get('steady_state',   np.nan)

    print('  ── Closed-Loop Step Response ───────────────────────')
    print(f'  Rise time  (10–90 %) : {fv(None if np.isnan(rise)     else rise     * 1e3, "ms")}')
    print(f'  Settling time  (2 %) : {fv(None if np.isnan(settling) else settling * 1e3, "ms")}')
    print(f'  Overshoot            : {fv(overshoot, "%")}')
    print(f'  Steady-state gain    : {fv(y_ss, "", 4)}')
    print()
    print('  ── Closed-Loop Poles ───────────────────────────────')
    for p in sorted(poles_cl, key=lambda x: -x.real):
        wn   = abs(p)
        zeta = (-p.real / wn) if wn > 1e-12 else 1.0
        print(f'  {p.real:+12.4f} {p.imag:+12.4f}j'
              f'   ζ = {zeta:.3f}   ωn = {wn:.4g} rad/s')
    print(sep)


# =============================================================================
#  PRIVATE HELPERS
# =============================================================================

def _valid_freq(f):
    """Return True when f is a usable positive finite frequency."""
    return (f is not None) and np.isfinite(f) and (f > 0)


def _gm_label(gm):
    return f'GM = {gm:.1f} dB' if np.isfinite(gm) else 'GM = ∞'


def _pm_label(pm):
    return f'PM = {pm:.1f}°' if not np.isnan(pm) else 'PM = ?'


def _re(arr):
    return arr.real.tolist() if len(arr) > 0 else []


def _im(arr):
    return arr.imag.tolist() if len(arr) > 0 else []
