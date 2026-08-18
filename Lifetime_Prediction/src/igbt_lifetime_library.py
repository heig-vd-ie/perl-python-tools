"""
igbt_lifetime_library.py
────────────────────────
Helper functions for IGBT power module lifetime prediction from junction
temperature waveforms obtained via PLECS simulation.

Pipeline overview
─────────────────
  PLECS Tj waveform
      │
      ▼
  Rainflow cycle counting          → (ΔT, T_mean, count) DataFrame
      │
      ▼
  3-D cycle histogram              → plot: N (count) vs ΔT vs T_mean
      │
      ▼
  Lifetime model  (Nf per cycle)   → Miner's rule → total damage → lifetime
      │
      ▼
  Monte Carlo on model parameters  → lifetime distribution histogram

Lifetime models implemented
───────────────────────────
  1. Coffin-Manson
  2. Modified Coffin-Manson
  3. Norris-Landzberg
  4. Bayerer (2008)
  5. Semikron (2013)

Reference: Table 6.1 – IGBT Lifetime Models (see notebook documentation).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 DAMAGE SUMMATION — Miner's linear damage rule
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  D = Σ  n_i / N_f,i

  where n_i   is the cycle count in bin i  (from rainflow counting)
        N_f,i is the number of cycles to failure at (ΔT_i, T_m,i) from model

  Time-to-failure = T_sim / D   where T_sim is the mission-profile duration.
"""

import copy
import random
import warnings

import numpy as np
import pandas as pd
import matplotlib.cm as cm          # used only for colour interpolation
import matplotlib.colors as mcolors # used only for colour interpolation
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import norm as sp_norm


# ─── Physical constants (SI) ──────────────────────────────────────────────────
K_B = 1.380_649e-23   # Boltzmann constant  [J / K]  (CODATA 2018 exact value)
                      # NOTE: activation energies Ea must therefore be in Joules.
                      # Convert from eV: Ea_J = Ea_eV × 1.602_176_634e-19


# ===============================================================================
#  MODEL REGISTRY
# ===============================================================================

LIFETIME_MODELS = {
    # ── 1. Coffin-Manson ──────────────────────────────────────────────────────
    'coffin_manson': {
        'name': 'Coffin-Manson',
        'year': '(classic)',
        'description': (
            "The original Coffin-Manson model relates the number of cycles to "
            "failure solely to the junction temperature swing ΔTj. It is the "
            "simplest model and serves as a baseline. It does NOT account for "
            "mean temperature effects, making it suitable only when temperature "
            "swings are the dominant ageing driver and operating conditions are "
            "roughly constant."
        ),
        'formula': 'Nf = A · ΔTj^(-n)',
        'required_fields': ['delta_T'],
        'parameters': {
            'A': {
                'default': 3.44e14,
                'description': 'Pre-exponential constant  (fit to device data)',
                'unit': '–',
            },
            'n': {
                'default': 5.0,
                'description': 'Coffin-Manson exponent  (typically 4–7 for IGBTs)',
                'unit': '–',
            },
        },
    },

    # ── 2. Modified Coffin-Manson ─────────────────────────────────────────────
    'modified_coffin_manson': {
        'name': 'Modified Coffin-Manson',
        'year': '(classic+Ea)',
        'description': (
            "Extends the Coffin-Manson model by adding an Arrhenius activation-"
            "energy term that captures the temperature-dependent acceleration of "
            "failure mechanisms (e.g. bond-wire fatigue, solder delamination). "
            "It requires the mean junction temperature T_jm [°C] in addition to "
            "ΔTj. A good compromise between simplicity and accuracy for most "
            "applications."
        ),
        'formula': 'Nf = A · ΔTj^(-n) · exp(Ea / (kB · T_jm))',
        'required_fields': ['delta_T', 'T_mean'],
        'parameters': {
            'A': {
                'default': 3.44e14,
                'description': 'Pre-exponential constant',
                'unit': '–',
            },
            'n': {
                'default': 5.0,
                'description': 'Coffin-Manson exponent',
                'unit': '–',
            },
            'Ea': {
                'default': 1.009e-19,   # 0.63 eV × 1.602176634e-19 J/eV
                'description': 'Activation energy  (0.4–0.9 eV → 6.4e-20–1.44e-19 J)',
                'unit': 'J',
            },
        },
    },

    # ── 3. Norris-Landzberg ───────────────────────────────────────────────────
    'norris_landzberg': {
        'name': 'Norris-Landzberg',
        'year': '(1969)',
        'description': (
            "Adds a cycling frequency f [Hz] dependence to the Modified Coffin-"
            "Manson model. Slower cycles (lower f) typically cause more damage "
            "per cycle because the material has more time to creep and diffuse. "
            "Best used when the thermal cycling frequency is known and spans a "
            "wide range (e.g. mission profiles mixing fast load transients with "
            "slow day-night cycles)."
        ),
        'formula': 'Nf = A · ΔTj^(-n) · exp(Ea / (kB · T_jm)) · f^(-m)',
        'required_fields': ['delta_T', 'T_mean', 'f'],
        'parameters': {
            'A': {
                'default': 3.44e14,
                'description': 'Pre-exponential constant',
                'unit': '–',
            },
            'n': {
                'default': 5.0,
                'description': 'Coffin-Manson exponent',
                'unit': '–',
            },
            'Ea': {
                'default': 1.009e-19,   # 0.63 eV × 1.602176634e-19 J/eV
                'description': 'Activation energy',
                'unit': 'J',
            },
            'm': {
                'default': 0.5,
                'description': 'Frequency exponent  (typically 0.3–0.7)',
                'unit': '–',
            },
            'f': {
                'default': 1.0,
                'description': 'Cycling frequency  (derived from rainflow or set manually)',
                'unit': 'Hz',
            },
        },
    },

    # ── 4. Bayerer (2008) ─────────────────────────────────────────────────────
    'bayerer_2008': {
        'name': 'Bayerer',
        'year': '(2008)',
        'description': (
            "A physics-of-failure model developed by Infineon / Siemens based on "
            "power-cycling tests of IGBT modules. It accounts for six independent "
            "variables: ΔTj, minimum junction temperature T_jmin, heating time "
            "ton, current per bond-wire I, blocking voltage V, and bond-wire "
            "diameter D. Recommended when detailed device and operating-condition "
            "data are available. Parameters β1–β6 are device-specific and are "
            "obtained from curve-fitting to manufacturer test data."
        ),
        'formula': 'Nf = A · ΔTj^β1 · exp(β2 / T_jmin) · ton^β3 · I^β4 · V^β5 · D^β6',
        'required_fields': ['delta_T', 'T_min'],
        'parameters': {
            'A': {
                'default': 9.34e14,
                'description': 'Pre-exponential constant',
                'unit': '–',
            },
            'beta1': {
                'default': -4.416,
                'description': 'ΔTj exponent',
                'unit': '–',
            },
            'beta2': {
                'default': 1285.0,
                'description': 'Activation temperature  [K]  (= Ea/kB)',
                'unit': 'K',
            },
            'beta3': {
                'default': -0.463,
                'description': 'Heating-time exponent',
                'unit': '–',
            },
            'beta4': {
                'default': -0.716,
                'description': 'Current exponent',
                'unit': '–',
            },
            'beta5': {
                'default': -0.761,
                'description': 'Voltage exponent',
                'unit': '–',
            },
            'beta6': {
                'default': -0.5,
                'description': 'Bond-wire diameter exponent',
                'unit': '–',
            },
            'ton': {
                'default': 1.0,
                'description': 'Heating time per cycle  (set globally or from rainflow)',
                'unit': 's',
            },
            'I': {
                'default': 100.0,
                'description': 'Current per bond-wire',
                'unit': 'A',
            },
            'V': {
                'default': 600.0,
                'description': 'Blocking voltage of the chip',
                'unit': 'V',
            },
            'D': {
                'default': 0.375e-3,
                'description': 'Bond-wire diameter  (typically 300–500 µm)',
                'unit': 'm',
            },
        },
    },

    # ── 5. Semikron (2013) ────────────────────────────────────────────────────
    'semikron_2013': {
        'name': 'Semikron (ηρ-model)',
        'year': '(2013)',
        'description': (
            "Developed by Scheuermann & Schmidt (Semikron, CIPS 2014). Extends "
            "the Modified Coffin-Manson model with two additional physical "
            "drivers: the bond-wire aspect ratio ar (geometry factor capturing "
            "different loop heights) and the heating time ton via a saturation "
            "function (C + ton^γ)/C that correctly reproduces the non-linear "
            "ton dependence observed experimentally. fdiode accounts for mixed "
            "IGBT/diode failure contributions. The most comprehensive and "
            "accurate model for modern discrete IGBT modules when full "
            "device characterisation data are available."
        ),
        'formula': 'Nf = A · ΔTj^β0 · (ar)^β1 · [(C + ton^γ)/C] · exp(Ea/(kB·T_jm)) · fdiode',
        'required_fields': ['delta_T', 'T_mean'],
        'parameters': {
            'A': {
                'default': 3.4368e14,
                'description': 'Pre-exponential constant',
                'unit': '–',
            },
            'beta0': {
                'default': -4.923,
                'description': 'ΔTj exponent',
                'unit': '–',
            },
            'beta1': {
                'default': -9.012,
                'description': 'Bond-wire aspect-ratio exponent',
                'unit': '–',
            },
            'C': {
                'default': 1.434,
                'description': 'Saturation constant for ton influence',
                'unit': '–',
            },
            'gamma': {
                'default': 0.5,
                'description': 'ton exponent in saturation function',
                'unit': '–',
            },
            'Ea': {
                'default': 9.613e-21,   # 0.06 eV × 1.602176634e-19 J/eV
                'description': 'Activation energy  (bond-wire mechanism, ~0.06 eV)',
                'unit': 'J',
            },
            'ar': {
                'default': 3.1e-3,
                'description': 'Bond-wire aspect ratio  (height / horizontal span)',
                'unit': '–',
            },
            'ton': {
                'default': 1.0,
                'description': 'Heating time per cycle',
                'unit': 's',
            },
            'fdiode': {
                'default': 1.0,
                'description': 'Diode failure-fraction factor  (1.0 = IGBT only)',
                'unit': '–',
            },
        },
    },
}


# ===============================================================================
#  RAINFLOW COUNTING  (ASTM E1049-85)
# ===============================================================================

def _turning_points(signal: np.ndarray) -> list:
    """
    Extract turning points (local maxima and minima) from a 1-D signal.

    Endpoints are always included. Consecutive equal values are collapsed.

    Parameters
    ----------
    signal : np.ndarray  shape (n,)

    Returns
    -------
    pts : list of float
    """
    arr = np.asarray(signal, dtype=float)
    if arr.size < 2:
        return list(arr)

    # Remove flat segments (keep first occurrence of consecutive equal values)
    diff  = np.diff(arr)
    nonzero_mask = diff != 0.0
    # Indices where value changes; include the last index
    change_idx = np.concatenate(([0], np.where(nonzero_mask)[0] + 1))
    arr = arr[change_idx]

    if arr.size < 2:
        return list(arr)
    if arr.size == 2:
        return [arr[0], arr[-1]]

    # Keep peaks and valleys (sign changes in 1st derivative)
    d    = np.diff(arr)
    keep = np.concatenate(([True], d[:-1] * d[1:] < 0, [True]))
    return list(arr[keep])


def rainflow_counting(signal) -> pd.DataFrame:
    """
    Rainflow cycle counting using the ASTM E1049-85 three-point algorithm.

    Each thermal cycle is characterised by:
      range (ΔT)  – temperature swing  [°C or K]
      mean  (T_m) – mean temperature   [°C or K]
      count       – 0.5 (half-cycle) or 1.0 (full cycle)

    Parameters
    ----------
    signal : array-like
        1-D junction temperature waveform.

    Returns
    -------
    cycles : pd.DataFrame  with columns ['range', 'mean', 'count']
        'range'  – ΔT of the cycle  [same unit as input signal]
        'mean'   – mean temperature (T_min + T_max) / 2
        'count'  – 0.5 or 1.0
    """
    pts = _turning_points(np.asarray(signal, dtype=float))

    if len(pts) < 2:
        return pd.DataFrame(columns=['range', 'mean', 'count'])

    cycles = []
    stack  = []

    for p in pts:
        stack.append(p)

        while len(stack) >= 3:
            X = abs(stack[-1] - stack[-2])   # current range
            Y = abs(stack[-2] - stack[-3])   # range to test

            if X >= Y:
                rng  = Y
                mean = (stack[-3] + stack[-2]) / 2.0

                if len(stack) == 3:
                    # Half-cycle at the sequence start – advance past it
                    cycles.append({'range': rng, 'mean': mean, 'count': 0.5})
                    stack.pop(0)
                else:
                    # Full cycle – extract the two inner points
                    cycles.append({'range': rng, 'mean': mean, 'count': 1.0})
                    del stack[-3:-1]   # removes elements at positions -3 and -2
            else:
                break

    # Remaining turning points form residual half-cycles
    for i in range(len(stack) - 1):
        rng  = abs(stack[i + 1] - stack[i])
        mean = (stack[i] + stack[i + 1]) / 2.0
        cycles.append({'range': rng, 'mean': mean, 'count': 0.5})

    if not cycles:
        return pd.DataFrame(columns=['range', 'mean', 'count'])

    df = pd.DataFrame(cycles)
    # Drop trivially small cycles (numerical noise)
    df = df[df['range'] > 1e-9].reset_index(drop=True)
    return df


def bin_cycles(
    cycles: pd.DataFrame,
    n_delta_T_bins: int = 10,
    n_T_mean_bins: int  = 10,
    delta_T_range: tuple = None,
    T_mean_range:  tuple = None,
) -> tuple:
    """
    Aggregate rainflow cycles into a 2-D histogram grid.

    Parameters
    ----------
    cycles        : pd.DataFrame  output of rainflow_counting()
    n_delta_T_bins: int  number of bins along ΔT axis
    n_T_mean_bins : int  number of bins along T_mean axis
    delta_T_range : (min, max) or None  (auto if None)
    T_mean_range  : (min, max) or None  (auto if None)

    Returns
    -------
    counts      : np.ndarray  shape (n_T_mean_bins, n_delta_T_bins)
                  sum of cycle counts in each 2-D bin
    delta_T_edges : np.ndarray  shape (n_delta_T_bins + 1,)
    T_mean_edges  : np.ndarray  shape (n_T_mean_bins  + 1,)
    """
    if cycles.empty:
        raise ValueError("cycles DataFrame is empty — no cycles to bin.")

    dt_min, dt_max = delta_T_range or (cycles['range'].min(), cycles['range'].max())
    tm_min, tm_max = T_mean_range  or (cycles['mean'].min(),  cycles['mean'].max())

    # Avoid zero-width bins
    if dt_min == dt_max:
        dt_min, dt_max = dt_min * 0.9, dt_max * 1.1 + 1.0
    if tm_min == tm_max:
        tm_min, tm_max = tm_min * 0.9, tm_max * 1.1 + 1.0

    delta_T_edges = np.linspace(dt_min, dt_max, n_delta_T_bins + 1)
    T_mean_edges  = np.linspace(tm_min, tm_max, n_T_mean_bins  + 1)

    counts, _, _ = np.histogram2d(
        cycles['mean'].values,
        cycles['range'].values,
        bins    = [T_mean_edges, delta_T_edges],
        weights = cycles['count'].values,
    )
    return counts, delta_T_edges, T_mean_edges


# ===============================================================================
#  3-D CYCLE HISTOGRAM  (Plotly — fully interactive)
# ===============================================================================

def _hex_color_from_value(value: float, colormap: str = 'viridis') -> str:
    """
    Map a normalised scalar [0, 1] to a CSS hex colour using a matplotlib
    colormap.  matplotlib is used only for colour look-up, not for plotting.
    """
    try:
        cmap = cm.get_cmap(colormap)
        rgba = cmap(float(np.clip(value, 0.0, 1.0)))
        return mcolors.to_hex(rgba)
    except Exception:
        # Simple purple→yellow fallback (viridis-like)
        v = float(np.clip(value, 0.0, 1.0))
        r = int(68  + v * 185)
        g = int(1   + v * 230)
        b = int(84  - v * 47)
        return f'#{r:02x}{g:02x}{b:02x}'


def _bar_mesh3d(
    x_c: float, y_c: float,
    z_top: float,
    dx: float, dy: float,
    color_hex: str,
    label: str,
    opacity: float,
) -> go.Mesh3d:
    """
    Build one rectangular-prism Mesh3d trace representing a single 3-D bar.

    Vertex layout (8 corners)
    ─────────────────────────
      Bottom face (z = 0) :  0(x0,y0) 1(x1,y0) 2(x1,y1) 3(x0,y1)
      Top face (z = z_top):  4(x0,y0) 5(x1,y0) 6(x1,y1) 7(x0,y1)

    12 triangular faces (2 triangles per rectangular face, 6 faces total):
      bottom: 0-1-2, 0-2-3   top:   4-5-6, 4-6-7
      front:  0-1-5, 0-5-4   back:  2-3-7, 2-7-6
      left:   0-3-7, 0-7-4   right: 1-2-6, 1-6-5
    """
    x0, x1 = x_c - dx / 2.0, x_c + dx / 2.0
    y0, y1 = y_c - dy / 2.0, y_c + dy / 2.0

    vx = [x0, x1, x1, x0,   x0, x1, x1, x0]
    vy = [y0, y0, y1, y1,   y0, y0, y1, y1]
    vz = [0,  0,  0,  0,    z_top, z_top, z_top, z_top]

    fi = [0, 0,   4, 4,   0, 0,   2, 2,   0, 0,   1, 1]
    fj = [1, 2,   5, 6,   1, 5,   3, 7,   3, 7,   2, 6]
    fk = [2, 3,   6, 7,   5, 4,   7, 6,   7, 4,   6, 5]

    return go.Mesh3d(
        x=vx, y=vy, z=vz,
        i=fi, j=fj, k=fk,
        color=color_hex,
        opacity=opacity,
        showscale=False,
        showlegend=False,
        name=label,
        hovertemplate=label + '<extra></extra>',
        flatshading=True,
        lighting=dict(ambient=0.85, diffuse=0.5, specular=0.15, roughness=0.5),
        lightposition=dict(x=1, y=1, z=2),
    )


def plot_3d_cycle_histogram(
    cycles: pd.DataFrame,
    n_delta_T_bins: int   = 8,
    n_T_mean_bins:  int   = 8,
    delta_T_range:  tuple = None,
    T_mean_range:   tuple = None,
    title:          str   = 'Rainflow Cycle Count Histogram',
    colormap:       str   = 'viridis',
    opacity:        float = 0.88,
    bar_gap:        float = 0.12,
    label:          str   = '',
) -> go.Figure:
    """
    Interactive 3-D bar chart of rainflow cycle counts (Plotly Mesh3d).

    Each non-empty bin is rendered as a solid rectangular prism, coloured by
    cycle count.  The chart is fully interactive: rotate, zoom, hover to read
    exact values.

    Parameters
    ----------
    cycles        : pd.DataFrame  — output of rainflow_counting()
    n_delta_T_bins: int           — number of bins along ΔT axis (Y)
    n_T_mean_bins : int           — number of bins along T_mean axis (X)
    delta_T_range : (min, max) or None  — ΔT axis range, auto if None
    T_mean_range  : (min, max) or None  — T_mean axis range, auto if None
    title         : str
    colormap      : str  — any matplotlib colormap name
                    ('viridis', 'plasma', 'Blues', 'YlOrRd', 'coolwarm', …)
    opacity       : float  [0, 1]
    bar_gap       : float  — fractional spacing between bars (0 = flush)
    label         : str   — appended to the title (used when called in a loop)

    Returns
    -------
    fig : plotly.graph_objects.Figure
    """
    counts, delta_T_edges, T_mean_edges = bin_cycles(
        cycles, n_delta_T_bins, n_T_mean_bins, delta_T_range, T_mean_range
    )

    dT_centers = 0.5 * (delta_T_edges[:-1] + delta_T_edges[1:])
    Tm_centers = 0.5 * (T_mean_edges[:-1]  + T_mean_edges[1:])
    dT_width   = (delta_T_edges[1] - delta_T_edges[0]) * (1.0 - bar_gap)
    Tm_width   = (T_mean_edges[1]  - T_mean_edges[0])  * (1.0 - bar_gap)

    z_max = float(counts.max())
    if z_max <= 0:
        print("[3D Histogram] All bins are empty — nothing to plot.")
        return go.Figure()

    # ── One Mesh3d per non-empty bin ─────────────────────────────────────────
    traces = []
    for i_Tm, Tm_c in enumerate(Tm_centers):
        for i_dT, dT_c in enumerate(dT_centers):
            z = float(counts[i_Tm, i_dT])   # counts shape: (n_Tm, n_dT)
            if z <= 0.0:
                continue

            color_hex = _hex_color_from_value(z / z_max, colormap)
            hover_lbl = (f"T_m = {Tm_c:.1f} °C<br>"
                         f"ΔT  = {dT_c:.1f} °C<br>"
                         f"N   = {z:.1f} cycles")

            traces.append(_bar_mesh3d(
                x_c=Tm_c, y_c=dT_c, z_top=z,
                dx=Tm_width, dy=dT_width,
                color_hex=color_hex,
                label=hover_lbl,
                opacity=opacity,
            ))

    # ── Invisible scatter trace — only purpose is to show the colorbar ────────
    traces.append(go.Scatter3d(
        x=[None], y=[None], z=[None],
        mode='markers',
        marker=dict(
            size=0,
            color=[0.0, z_max],
            colorscale=colormap,
            showscale=True,
            colorbar=dict(title='Cycle count', thickness=16, len=0.7, x=1.01),
        ),
        showlegend=False,
        hoverinfo='skip',
    ))

    full_title = f"{title}  {label}".strip()
    fig = go.Figure(data=traces)
    fig.update_layout(
        title=dict(text=full_title, x=0.5, xanchor='center'),
        scene=dict(
            xaxis=dict(title='T_m (°C)',
                       backgroundcolor='rgb(240,240,248)', showbackground=True,
                       gridcolor='white'),
            yaxis=dict(title='ΔT (°C)',
                       backgroundcolor='rgb(230,240,252)', showbackground=True,
                       gridcolor='white'),
            zaxis=dict(title='N (Cycle Count)',
                       backgroundcolor='rgb(235,235,248)', showbackground=True,
                       gridcolor='white'),
            camera=dict(eye=dict(x=-1.55, y=-1.8, z=1.2)),
            aspectmode='cube',
        ),
        template='plotly_white',
        margin=dict(l=0, r=60, t=60, b=0),
        height=600,
    )

    n_bars = len(traces) - 1
    total  = float(cycles['count'].sum())
    print(f"[3D Histogram] '{full_title}'")
    print(f"  Total cycles  : {total:.1f}  |  "
          f"Non-empty bins : {n_bars} / {n_delta_T_bins * n_T_mean_bins}")
    print(f"  ΔT range      : [{delta_T_edges[0]:.2f}, {delta_T_edges[-1]:.2f}] °C")
    print(f"  T_m range     : [{T_mean_edges[0]:.2f},  {T_mean_edges[-1]:.2f}] °C")

    fig.show()
    return fig


# ===============================================================================
#  LIFETIME MODEL FUNCTIONS
# ===============================================================================

def _to_kelvin(T_celsius: float) -> float:
    """Convert Celsius to Kelvin."""
    return T_celsius + 273.15


def coffin_manson(delta_T, *, A, n, **_kwargs) -> np.ndarray:
    """
    Coffin-Manson:  Nf = A · ΔTj^(-n)

    Parameters
    ----------
    delta_T : float or np.ndarray  [°C or K]
    A, n    : model parameters

    Returns
    -------
    Nf : np.ndarray  number of cycles to failure
    """
    dT = np.asarray(delta_T, dtype=float)
    if np.any(dT <= 0):
        warnings.warn("delta_T contains non-positive values; clipping to 1e-9.")
        dT = np.clip(dT, 1e-9, None)
    return A * dT ** (-n)


def modified_coffin_manson(delta_T, T_mean, *, A, n, Ea, **_kwargs) -> np.ndarray:
    """
    Modified Coffin-Manson:  Nf = A · ΔTj^(-n) · exp(Ea / (kB · T_jm))

    Parameters
    ----------
    delta_T : float or array  [°C]
    T_mean  : float or array  [°C]  — mean junction temperature
    A, n    : Coffin-Manson parameters
    Ea      : activation energy  [eV]

    Returns
    -------
    Nf : np.ndarray
    """
    dT  = np.asarray(delta_T, dtype=float)
    Tm  = np.asarray(T_mean,  dtype=float)
    dT  = np.clip(dT, 1e-9, None)
    TmK = _to_kelvin(Tm)
    return A * dT ** (-n) * np.exp(Ea / (K_B * TmK))


def norris_landzberg(delta_T, T_mean, f_cycle, *, A, n, Ea, m, **_kwargs) -> np.ndarray:
    """
    Norris-Landzberg:  Nf = A · ΔTj^(-n) · exp(Ea/(kB·T_jm)) · f^(-m)

    Parameters
    ----------
    delta_T : float or array  [°C]
    T_mean  : float or array  [°C]
    f_cycle : float or array  [Hz]  — cycling frequency
    A, n, Ea, m : model parameters

    Returns
    -------
    Nf : np.ndarray
    """
    dT  = np.asarray(delta_T,  dtype=float)
    Tm  = np.asarray(T_mean,   dtype=float)
    fc  = np.asarray(f_cycle,  dtype=float)
    dT  = np.clip(dT, 1e-9, None)
    fc  = np.clip(fc, 1e-9, None)
    TmK = _to_kelvin(Tm)
    return A * dT ** (-n) * np.exp(Ea / (K_B * TmK)) * fc ** (-m)


def bayerer_2008(
    delta_T, T_min, *, A, beta1, beta2, beta3, beta4, beta5, beta6,
    ton, I, V, D, **_kwargs
) -> np.ndarray:
    """
    Bayerer (2008):  Nf = A · ΔTj^β1 · exp(β2/T_jmin) · ton^β3 · I^β4 · V^β5 · D^β6

    T_jmin is the minimum junction temperature per cycle [°C] → converted to K.

    Parameters
    ----------
    delta_T          : float or array  [°C]
    T_min            : float or array  [°C]  — minimum junction temperature per cycle
    A, beta1 … beta6 : Bayerer model parameters
    ton              : float  [s]   heating time
    I                : float  [A]   current per bond-wire
    V                : float  [V]   blocking voltage
    D                : float  [m]   bond-wire diameter

    Returns
    -------
    Nf : np.ndarray
    """
    dT    = np.asarray(delta_T, dtype=float)
    Tmin  = np.asarray(T_min,   dtype=float)
    dT    = np.clip(dT, 1e-9, None)
    TminK = _to_kelvin(Tmin)
    return (
        A
        * dT    ** beta1
        * np.exp(beta2 / TminK)
        * ton   ** beta3
        * I     ** beta4
        * V     ** beta5
        * D     ** beta6
    )


def semikron_2013(
    delta_T, T_mean, *, A, beta0, beta1, C, gamma, Ea, ar, ton, fdiode, **_kwargs
) -> np.ndarray:
    """
    Semikron ηρ-model (Scheuermann & Schmidt, 2013/2014):
      Nf = A · ΔTj^β0 · (ar)^β1 · [(C + ton^γ) / C] · exp(Ea/(kB·T_jm)) · fdiode

    Parameters
    ----------
    delta_T       : float or array  [°C]
    T_mean        : float or array  [°C]  — mean junction temperature per cycle
    A, beta0, beta1, C, gamma, Ea : model parameters
    ar            : float  bond-wire aspect ratio (height / distance)
    ton           : float  [s]  heating time
    fdiode        : float  diode failure-fraction factor

    Returns
    -------
    Nf : np.ndarray
    """
    dT  = np.asarray(delta_T, dtype=float)
    Tm  = np.asarray(T_mean,  dtype=float)
    dT  = np.clip(dT, 1e-9, None)
    TmK = _to_kelvin(Tm)

    ton_factor = (C + ton ** gamma) / C
    return (
        A
        * dT     ** beta0
        * ar     ** beta1
        * ton_factor
        * np.exp(Ea / (K_B * TmK))
        * fdiode
    )


# ── Model dispatch map ────────────────────────────────────────────────────────
_MODEL_FUNCTIONS = {
    'coffin_manson':         coffin_manson,
    'modified_coffin_manson': modified_coffin_manson,
    'norris_landzberg':      norris_landzberg,
    'bayerer_2008':          bayerer_2008,
    'semikron_2013':         semikron_2013,
}


# ===============================================================================
#  COMPUTE Nf PER CYCLE  (model dispatch)
# ===============================================================================

def compute_nf_per_cycle(
    cycles: pd.DataFrame,
    model_key: str,
    params: dict,
) -> np.ndarray:
    """
    Compute the cycles-to-failure Nf for every row in the rainflow DataFrame.

    Parameters
    ----------
    cycles    : pd.DataFrame  output of rainflow_counting()
                Must have columns: 'range', 'mean', 'count'
    model_key : str  one of the keys of LIFETIME_MODELS
    params    : dict  parameter values for the selected model

    Returns
    -------
    Nf : np.ndarray  shape (len(cycles),)  — cycles to failure per row
    """
    if model_key not in _MODEL_FUNCTIONS:
        raise ValueError(
            f"Unknown model '{model_key}'. "
            f"Choose from: {list(_MODEL_FUNCTIONS.keys())}"
        )

    dT   = cycles['range'].values
    Tm   = cycles['mean'].values
    Tmin = Tm - dT / 2.0     # minimum junction temperature per cycle

    model_fn = _MODEL_FUNCTIONS[model_key]

    if model_key == 'coffin_manson':
        Nf = model_fn(dT, **params)

    elif model_key == 'modified_coffin_manson':
        Nf = model_fn(dT, Tm, **params)

    elif model_key == 'norris_landzberg':
        # Derive frequency from count / (simulation duration) if not per-cycle
        f_val = params.get('f', 1.0)
        Nf    = model_fn(dT, Tm, f_val, **params)

    elif model_key == 'bayerer_2008':
        Nf = model_fn(dT, Tmin, **params)

    elif model_key == 'semikron_2013':
        Nf = model_fn(dT, Tm, **params)

    else:
        raise ValueError(f"No dispatch rule for model '{model_key}'.")

    return np.asarray(Nf, dtype=float)


# ===============================================================================
#  MINER'S RULE  →  Damage  →  Lifetime
# ===============================================================================

def miner_damage(cycles: pd.DataFrame, Nf: np.ndarray) -> float:
    """
    Apply Miner's linear damage rule.

    D = Σ  n_i / N_f,i

    Parameters
    ----------
    cycles : pd.DataFrame  (must have 'count' column)
    Nf     : np.ndarray    cycles to failure per cycle  (from compute_nf_per_cycle)

    Returns
    -------
    D : float  total accumulated damage  (failure when D ≥ 1)
    """
    n_i   = cycles['count'].values
    Nf_i  = np.asarray(Nf, dtype=float)
    Nf_i  = np.clip(Nf_i, 1.0, None)   # guard against Nf < 1
    return float(np.sum(n_i / Nf_i))


def lifetime_from_damage(
    damage: float,
    T_sim_hours: float,
    operating_hours_per_year: float = 8760.0,
) -> dict:
    """
    Convert accumulated damage per mission profile into lifetime estimates.

    Parameters
    ----------
    damage                  : float  — Miner's damage per simulation run
    T_sim_hours             : float  — duration of one simulation run  [hours]
    operating_hours_per_year: float  — annual operating hours  (default: 8760 = 24/7)

    Returns
    -------
    result : dict with keys
        'damage'      – damage per simulation run
        'hours'       – estimated time to failure  [hours]
        'days'        – estimated time to failure  [days]
        'years'       – estimated time to failure  [years]
        'T_sim_hours' – mission profile duration used
    """
    if damage <= 0:
        return {
            'damage': 0.0,
            'hours': np.inf,
            'days':  np.inf,
            'years': np.inf,
            'T_sim_hours': T_sim_hours,
        }

    hours_to_failure = T_sim_hours / damage
    days_to_failure  = hours_to_failure / 24.0
    years_to_failure = hours_to_failure / operating_hours_per_year

    return {
        'damage':      damage,
        'hours':       hours_to_failure,
        'days':        days_to_failure,
        'years':       years_to_failure,
        'T_sim_hours': T_sim_hours,
    }


def print_lifetime_result(result: dict, label: str = '') -> None:
    """Pretty-print a lifetime result dict."""
    tag = f"  [{label}]" if label else ''
    print(f"\n{'─'*55}")
    print(f"  Lifetime Estimate{tag}")
    print(f"{'─'*55}")
    print(f"  Mission-profile duration : {result['T_sim_hours']:.4g} h")
    print(f"  Damage per run           : {result['damage']:.4e}")
    print(f"  ─────────────────────────────────────────────────")
    if np.isinf(result['hours']):
        print("  Time to failure          : ∞  (damage = 0)")
    else:
        print(f"  Time to failure          : {result['hours']:.4e} h")
        print(f"                             {result['days']:.4e} days")
        print(f"                             {result['years']:.4e} years")
    print(f"{'─'*55}\n")


# ===============================================================================
#  MONTE CARLO  —  Lifetime parameter uncertainty
# ===============================================================================

def _sample_params(nominal_params: dict, mc_config: dict, rng: np.random.Generator) -> dict:
    """
    Draw one realisation of model parameters from their distributions.

    Parameters
    ----------
    nominal_params : dict  { param_name : nominal_value }
    mc_config      : dict  { param_name : (relative_tolerance, distribution) }
        distribution : 'uniform' or 'normal'
            'uniform' → flat in [nominal*(1-tol), nominal*(1+tol)]
            'normal'  → Gaussian with sigma = |nominal| * tol

    Returns
    -------
    sampled : dict  — same keys as nominal_params with perturbed values
    """
    sampled = copy.deepcopy(nominal_params)
    for name, (tol, dist) in mc_config.items():
        if name not in nominal_params:
            warnings.warn(f"MC config key '{name}' not found in nominal_params — skipped.")
            continue
        nominal = nominal_params[name]
        if dist == 'uniform':
            sampled[name] = nominal * (1.0 + rng.uniform(-tol, tol))
        elif dist == 'normal':
            sampled[name] = rng.normal(nominal, abs(nominal) * tol)
        else:
            raise ValueError(
                f"Unknown distribution '{dist}' for parameter '{name}'. "
                "Use 'uniform' or 'normal'."
            )
    return sampled


def run_montecarlo_lifetime(
    cycles: pd.DataFrame,
    model_key: str,
    nominal_params: dict,
    mc_config: dict,
    T_sim_hours: float,
    n_runs: int = 200,
    seed: int = None,
    operating_hours_per_year: float = 8760.0,
) -> pd.DataFrame:
    """
    Monte Carlo uncertainty analysis on lifetime model parameters.

    For each run, a perturbed set of parameters is drawn, the model is
    evaluated, Miner's damage is computed, and the lifetime is derived.

    Parameters
    ----------
    cycles        : pd.DataFrame  — rainflow output (range, mean, count)
    model_key     : str           — one of the keys in LIFETIME_MODELS
    nominal_params: dict          — nominal parameter values
    mc_config     : dict          — { param_name : (rel_tolerance, distribution) }
        Example:
          {'A' : (0.10, 'uniform'),   # A varies ±10 %
           'n' : (0.05, 'normal')}    # n varies with σ = 5 % of nominal
    T_sim_hours   : float         — simulation duration  [hours]
    n_runs        : int           — number of Monte Carlo draws
    seed          : int or None   — random seed for reproducibility
    operating_hours_per_year : float

    Returns
    -------
    mc_results : pd.DataFrame  with columns
        'run'           – run index
        'damage'        – Miner's damage per simulation run
        'lifetime_h'    – lifetime in hours
        'lifetime_days' – lifetime in days
        'lifetime_years'– lifetime in years
        + one column per MC parameter with its sampled value
    """
    rng = np.random.default_rng(seed)

    print(f"[MC Lifetime] model='{model_key}'  n_runs={n_runs}"
          + (f"  seed={seed}" if seed is not None else ""))
    print(f"  Perturbed parameters: {list(mc_config.keys())}")
    print(f"  Mission profile: {T_sim_hours:.4g} h")

    records = []
    for i in range(n_runs):
        sampled_p = _sample_params(nominal_params, mc_config, rng)

        try:
            Nf     = compute_nf_per_cycle(cycles, model_key, sampled_p)
            damage = miner_damage(cycles, Nf)
            lt     = lifetime_from_damage(damage, T_sim_hours,
                                          operating_hours_per_year)
        except Exception as exc:
            warnings.warn(f"Run {i} failed: {exc}")
            continue

        row = {
            'run':            i,
            'damage':         lt['damage'],
            'lifetime_h':     lt['hours'],
            'lifetime_days':  lt['days'],
            'lifetime_years': lt['years'],
        }
        for k in mc_config:
            row[f'p_{k}'] = sampled_p.get(k, np.nan)

        records.append(row)

        if (i + 1) % max(1, n_runs // 10) == 0:
            print(f"  [{i+1:>4}/{n_runs}]  lifetime = {lt['days']:.2f} days")

    mc_df = pd.DataFrame(records)
    print(f"[MC Lifetime] Done. {len(mc_df)} successful runs.")
    return mc_df


# ===============================================================================
#  LIFETIME HISTOGRAM  (Monte Carlo result)
# ===============================================================================

def plot_lifetime_histogram(
    mc_results: pd.DataFrame,
    column:     str   = 'lifetime_days',
    n_bins:     int   = 25,
    fit_normal: bool  = True,
    title:      str   = 'Monte Carlo Lifetime Distribution',
    xlabel:     str   = 'Lifetime (days)',
    percentiles: list = [5, 50, 95],
    color:      str   = 'steelblue',
) -> go.Figure:
    """
    Interactive Plotly histogram of Monte Carlo lifetime results with an
    optional fitted normal distribution overlay.

    Reproduces the style of the reference figure (Image 3 in the notebook).

    Parameters
    ----------
    mc_results  : pd.DataFrame  — output of run_montecarlo_lifetime()
    column      : str           — which lifetime column to plot
                  ('lifetime_days', 'lifetime_years', 'lifetime_h')
    n_bins      : int
    fit_normal  : bool          — overlay a fitted normal PDF curve
    title       : str
    xlabel      : str
    percentiles : list[int]     — percentile lines to draw  (e.g. [5, 50, 95])
    color       : str           — bar fill colour

    Returns
    -------
    fig : plotly.graph_objects.Figure
    """
    data = mc_results[column].replace([np.inf, -np.inf], np.nan).dropna().values

    if len(data) == 0:
        print("[Histogram] No finite lifetime values to plot.")
        return go.Figure()

    mu, sigma = float(np.mean(data)), float(np.std(data))
    median    = float(np.median(data))
    n         = len(data)

    # Build histogram trace
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x         = data,
        nbinsx    = n_bins,
        name      = 'Lifetime',
        marker    = dict(color=color, line=dict(color='white', width=0.6)),
        opacity   = 0.80,
        histnorm  = 'probability density',
        showlegend= True,
    ))

    # Fitted normal PDF overlay
    if fit_normal and sigma > 0:
        x_pdf = np.linspace(data.min(), data.max(), 400)
        y_pdf = sp_norm.pdf(x_pdf, loc=mu, scale=sigma)
        fig.add_trace(go.Scatter(
            x          = x_pdf,
            y          = y_pdf,
            mode       = 'lines',
            name       = f'Normal fit  µ={mu:.4g}, σ={sigma:.4g}',
            line       = dict(color='tomato', width=2.5),
            showlegend = True,
        ))

    # Percentile vertical lines
    pct_colors = {5: 'orange', 50: 'red', 95: 'orange'}
    for p in percentiles:
        pct_val = float(np.percentile(data, p))
        fig.add_vline(
            x                 = pct_val,
            line_dash         = 'dash' if p != 50 else 'dot',
            line_color        = pct_colors.get(p, 'grey'),
            line_width        = 1.5,
            annotation_text   = f'P{p}={pct_val:.4g}',
            annotation_position = 'top right' if p >= 50 else 'top left',
        )

    fig.update_layout(
        title       = title,
        xaxis_title = xlabel,
        yaxis_title = 'Probability density',
        template    = 'plotly_white',
        bargap      = 0.04,
        legend      = dict(x=0.01, y=0.99, xanchor='left', yanchor='top'),
        annotations = [dict(
            text       = (f'n={n}   µ={mu:.4g}   σ={sigma:.4g}<br>'
                          f'Median={median:.4g}   '
                          f'[P5={np.percentile(data,5):.4g}, '
                          f'P95={np.percentile(data,95):.4g}]'),
            xref='paper', yref='paper',
            x=0.99, y=0.97,
            xanchor='right', yanchor='top',
            showarrow=False,
            bordercolor='lightgrey',
            borderwidth=1,
            bgcolor='white',
            font=dict(size=11),
        )],
    )

    # Print summary
    print(f"\n[MC Lifetime Histogram]  column='{column}'  n={n}")
    print(f"  Mean   : {mu:.4g}")
    print(f"  Std    : {sigma:.4g}  ({100*sigma/mu:.1f}% of mean)")
    print(f"  Median : {median:.4g}")
    for p in percentiles:
        print(f"  P{p:<3}   : {np.percentile(data,p):.4g}")

    fig.show()
    return fig


# ===============================================================================
#  MODEL TABLE  (display)
# ===============================================================================

def print_model_table() -> None:
    """
    Print a formatted summary table of all available lifetime models.

    Call this before choosing a model to get a concise overview.
    """
    divider = '═' * 72
    print(f"\n{divider}")
    print("  IGBT LIFETIME MODELS  —  Overview")
    print(divider)
    for key, info in LIFETIME_MODELS.items():
        print(f"\n  [{key}]")
        print(f"  Name    : {info['name']}  {info['year']}")
        print(f"  Formula : {info['formula']}")
        print(f"  Inputs  : {', '.join(info['required_fields'])}")
        # Word-wrap description at 68 chars
        desc = info['description']
        words, line_buf = desc.split(), ''
        lines = []
        for w in words:
            if len(line_buf) + len(w) + 1 > 68:
                lines.append(line_buf)
                line_buf = w
            else:
                line_buf = (line_buf + ' ' + w).strip()
        if line_buf:
            lines.append(line_buf)
        for l in lines:
            print(f"  {'':>10}{l}")
        print(f"\n  Parameters:")
        for pname, pinfo in info['parameters'].items():
            print(f"    {pname:<10} default={pinfo['default']:<12}  "
                  f"{pinfo['description']}  [{pinfo['unit']}]")
    print(f"\n{divider}\n")


def print_model_parameters(model_key: str) -> dict:
    """
    Print the parameters for a specific model and return a dict of defaults.

    Use the returned dict as a starting point to fill in your values.

    Parameters
    ----------
    model_key : str

    Returns
    -------
    params : dict  { param_name : default_value }
    """
    if model_key not in LIFETIME_MODELS:
        raise ValueError(
            f"Unknown model key '{model_key}'. "
            f"Available: {list(LIFETIME_MODELS.keys())}"
        )
    info   = LIFETIME_MODELS[model_key]
    params = {}

    print(f"\n{'─'*60}")
    print(f"  Model : {info['name']}  {info['year']}")
    print(f"  {info['formula']}")
    print(f"{'─'*60}")
    print(f"  {'Parameter':<12}  {'Default':>14}  {'Unit':<6}  Description")
    print(f"  {'─'*9:<12}  {'─'*10:>14}  {'─'*4:<6}  {'─'*28}")

    for pname, pinfo in info['parameters'].items():
        default = pinfo['default']
        params[pname] = default
        print(f"  {pname:<12}  {default:>14.4g}  "
              f"{pinfo['unit']:<6}  {pinfo['description']}")

    print(f"{'─'*60}\n")
    return params
