# =============================================================================
#  signal_analysis_lib.py
#
#  Library of all functions used by signal_analysis.ipynb.
#  Import with:  from signal_analysis_lib import *
#
#  No global configuration variables are used inside this file.
#  Every function receives what it needs as explicit arguments.
# =============================================================================

import os
import io
import re
import warnings
import time as _time

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from IPython.display import display

warnings.filterwarnings("ignore", category=RuntimeWarning)

# Default colour palette (overridable at call sites)
PALETTE = [
    "#2196F3", "#FF5722", "#4CAF50", "#9C27B0", "#FF9800",
    "#00BCD4", "#E91E63", "#8BC34A", "#795548", "#607D8B",
]


# =============================================================================
#  SECTION 1 — CSV LOADERS
# =============================================================================

def _read_head(path, n=12):
    with open(path, "r", errors="replace") as fh:
        return [fh.readline() for _ in range(n)]


def _coerce(df):
    return df.apply(pd.to_numeric, errors="coerce").dropna(how="all")


def _load_lecroy(path):
    df = pd.read_csv(path, skiprows=5, header=0)
    df.columns = [c.strip() for c in df.columns]
    cols = list(df.columns)
    cols[0] = "Time"
    if len(cols) > 1 and cols[1].lower() in ("ampl", ""):
        cols[1] = "Ampl"
    df.columns = cols
    return _coerce(df)


def _load_tektronix(path):
    lines = open(path, "r", errors="replace").readlines()
    hr = next(
        (i for i, l in enumerate(lines) if l.strip().upper().startswith("TIME")),
        None,
    )
    if hr is None:
        raise ValueError("TIME header not found")
    df = pd.read_csv(path, skiprows=hr, header=0)
    df.columns = [c.strip() for c in df.columns]
    df = df[[c for c in df.columns if "peak detect" not in c.lower()]]
    df = _coerce(df)
    df.rename(columns={df.columns[0]: "Time"}, inplace=True)
    return df


def _load_siglent(path):
    head = _read_head(path, 2)
    src = [c.strip() for c in head[0].split(",")]
    units = [c.strip() for c in head[1].split(",")]
    skip = 2 if any(u.lower() in ("volt", "second", "a", "amp") for u in units) else 1
    df = pd.read_csv(path, skiprows=skip, header=None, names=src)
    df = _coerce(df)
    df.rename(columns={df.columns[0]: "Time"}, inplace=True)
    return df


def _load_plecs(path):
    df = pd.read_csv(path, header=0)
    df.columns = [c.strip().strip('"') for c in df.columns]
    df = _coerce(df)
    df.rename(columns={df.columns[0]: "Time"}, inplace=True)
    return df


def _row_numeric_ratio(line, sep):
    parts = line.strip().split(sep)
    if len(parts) < 2:
        return 0.0
    ok = sum(1 for p in parts if _is_float(p.strip()))
    return ok / len(parts)


def _is_float(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def _find_data_start(lines, sep):
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if _row_numeric_ratio(stripped, sep) >= 0.6:
            hr = None
            for j in range(i - 1, max(i - 5, -1), -1):
                prev = lines[j].strip()
                if prev and not prev.startswith("#"):
                    if _row_numeric_ratio(prev, sep) < 0.5:
                        hr = j
                    break
            return hr, i
    return None, None


def _find_time_column(df):
    time_kw = {"time", "t", "time(s)", "time/s", "time_s", "s", "second", "seconds"}
    for c in df.columns:
        if str(c).lower().strip() in time_kw:
            return c
        if "time" in str(c).lower():
            return c
    for c in df.columns:
        col = pd.to_numeric(df[c], errors="coerce").dropna()
        if len(col) > 2 and (col.diff().dropna() > 0).mean() > 0.88:
            return c
    return df.columns[0]


def _load_generic(path):
    raw = open(path, "r", errors="replace").readlines()
    best = None
    for sep in (",", ";", "\t", " "):
        hr, dr = _find_data_start(raw, sep)
        if dr is None:
            continue
        names = None
        if hr is not None:
            names = [
                c.strip().strip('"').strip("'") or f"Col{i}"
                for i, c in enumerate(raw[hr].strip().split(sep))
            ]
        try:
            df = pd.read_csv(
                io.StringIO("".join(raw[dr:])),
                sep=sep,
                header=None,
                names=names,
                engine="python",
                skipinitialspace=True,
                on_bad_lines="skip",
            )
            df = df.apply(pd.to_numeric, errors="coerce")
            df = df.dropna(axis=1, how="all").dropna(how="all")
            df = df.dropna(subset=["Time"])  # removes the 110 rows with date strings in Time
            if df.shape[1] >= 2 and len(df) > 2:
                if best is None or df.shape[1] > best.shape[1]:
                    best = df
        except Exception:
            continue
    if best is None:
        raise ValueError(f"Cannot parse '{path}' as a numeric CSV.")
    if list(best.columns) == list(range(best.shape[1])):
        best.columns = [f"Col{i}" for i in range(best.shape[1])]
    tc = _find_time_column(best)
    best.rename(columns={tc: "Time"}, inplace=True)
    rn = {
        c: f"CH{i + 1}"
        for i, c in enumerate([x for x in best.columns if x != "Time"])
        if re.match(r"^Col\d+$", str(c))
    }
    best.rename(columns=rn, inplace=True)
    return best


def _sniff(path):
    head = _read_head(path)
    j = " ".join(head).lower()
    if "lecroy" in j:
        return "lecroy"
    if "time,ampl" in j.replace(" ", "") and "#1," in j:
        return "lecroy"
    if "dpo" in j or ("sample interval" in j and "record length" in j):
        return "tektronix"
    if "second,volt" in j.replace(" ", "") or (
        "source," in j and "second" in head[1].lower()
    ):
        return "siglent"
    if "time / s" in j or "time/s" in j.replace(" ", ""):
        return "plecs"
    return "generic"


_LOADERS = dict(
    lecroy=_load_lecroy,
    tektronix=_load_tektronix,
    siglent=_load_siglent,
    plecs=_load_plecs,
    generic=_load_generic,
)


def load_csv(path):
    """Auto-detect format and load CSV into a DataFrame with a 'Time' column."""
    fmt = _sniff(path)
    print(f"  Format detected : {fmt.upper()}")
    try:
        df = _LOADERS[fmt](path)
    except Exception as e:
        print(f"  Warning: {fmt} loader failed ({e}) — trying generic loader")
        df = _load_generic(path)
    return df.sort_values("Time").reset_index(drop=True)


# =============================================================================
#  SECTION 2 — TIME-DOMAIN PLOT
# =============================================================================

def _leg(ch, file_label, n_files, legend_names=None):
    if legend_names and ch in legend_names:
        return legend_names[ch]
    return ch if n_files == 1 else f"{file_label} – {ch}"


def _ylab(ch, yaxis_labels=None):
    if yaxis_labels:
        return yaxis_labels.get(ch, ch)
    return ch


def _build_groups(sig_cols, pg):
    if pg is None:
        return [[c] for c in sig_cols]
    if pg == "all":
        return [list(sig_cols)]
    assigned = {c for grp in pg for c in grp}
    extra = [[c] for c in sig_cols if c not in assigned]
    return list(pg) + extra


def plot_time_domain(
    all_data,
    plot_groups=None,
    title_override="",
    height_per_ch=220,
    fig_width=None,
    legend_names=None,
    yaxis_labels=None,
    palette=None,
):
    """
    Plot time-domain signals from all loaded files.

    Parameters
    ----------
    all_data        : dict {file_label: DataFrame}
    plot_groups     : None | "all" | [[ch, ...], ...]
    title_override  : str  — figure title; "" = auto
    height_per_ch   : int  — px per subplot row
    fig_width       : int | None
    legend_names    : dict {ch: display_name}
    yaxis_labels    : dict {ch: y-axis label}
    palette         : list of hex colour strings
    """
    if palette is None:
        palette = PALETTE
    n_files = len(all_data)

    for file_label, df in all_data.items():
        sig = [c for c in df.columns if c != "Time"]
        grps = _build_groups(sig, plot_groups)
        n = len(grps)
        title = title_override or (
            f"Time-domain — {file_label}" if n_files > 1 else "Time-domain signals"
        )

        fig = make_subplots(
            rows=n, cols=1, shared_xaxes=True, vertical_spacing=0.04
        )
        ci = 0
        for ri, grp in enumerate(grps):
            for ch in grp:
                if ch not in df.columns:
                    continue
                fig.add_trace(
                    go.Scattergl(
                        x=df["Time"],
                        y=df[ch],
                        mode="lines",
                        name=_leg(ch, file_label, n_files, legend_names),
                        line=dict(width=1.4, color=palette[ci % len(palette)]),
                    ),
                    row=ri + 1,
                    col=1,
                )
                ci += 1
            ylab = " / ".join(
                _ylab(c, yaxis_labels) for c in grp if c in df.columns
            )
            fig.update_yaxes(
                title_text=ylab,
                row=ri + 1,
                col=1,
                showgrid=True,
                gridcolor="#e0e0e0",
            )
        fig.update_xaxes(
            title_text="Time (s)", row=n, col=1, showgrid=True, gridcolor="#e0e0e0"
        )
        layout_kw = dict(
            height=height_per_ch * n + 80,
            title_text=title,
            hovermode="x unified",
            plot_bgcolor="white",
            paper_bgcolor="white",
            legend=dict(
                orientation="h", y=-0.06, bgcolor="rgba(255,255,255,0.85)"
            ),
            margin=dict(l=70, r=20, t=60, b=70),
        )
        if fig_width is not None:
            layout_kw["width"] = fig_width
        fig.update_layout(**layout_kw)
        fig.show(config={"toImageButtonOptions": {"format": "svg"}})


# =============================================================================
#  SECTION 3 — STATISTICAL ANALYSIS
# =============================================================================

def _fmt(v):
    if pd.isna(v):
        return "—"
    av = abs(v)
    if av == 0:
        return "0"
    return f"{v:.5g}" if av >= 0.001 else f"{v:.4e}"


def compute_statistics(all_data, dc_channels=None):
    """
    Compute RMS, mean, AC-RMS, max, min, peak-peak, std for every channel.
    If a channel is listed in dc_channels, DC ripple (pp and %) are added.

    Displays a formatted table and returns the raw DataFrame.
    """
    if dc_channels is None:
        dc_channels = []
    rows = []
    for fl, df in all_data.items():
        for ch in [c for c in df.columns if c != "Time"]:
            y = df[ch].dropna().values
            mean = np.mean(y)
            rms = np.sqrt(np.mean(y ** 2))
            pp = float(np.max(y) - np.min(y))
            row = {
                "File": fl,
                "Channel": ch,
                "RMS": rms,
                "Mean (DC)": mean,
                "AC RMS": np.sqrt(np.mean((y - mean) ** 2)),
                "Max": float(np.max(y)),
                "Min": float(np.min(y)),
                "Peak-Peak": pp,
                "Std dev": float(np.std(y)),
            }
            if ch in dc_channels:
                row["DC Ripple (pp)"] = pp
                row["DC Ripple (%)"] = pp / abs(mean) * 100 if mean else np.nan
            rows.append(row)

    stat_df = pd.DataFrame(rows).set_index(["File", "Channel"])
    print("=== Statistical Analysis ===")
    display(stat_df.map(_fmt))
    return stat_df


# =============================================================================
#  SECTION 4 — POWER ANALYSIS
# =============================================================================

def compute_power(all_data, power_pairs):
    """
    Compute active P, apparent S, reactive Q and power factor for each
    (voltage, current) pair listed in power_pairs.

    power_pairs : list of (voltage_col, current_col) tuples
    """
    if not power_pairs:
        print("Power analysis skipped — set POWER_PAIRS in cell 1.")
        return

    print("=== Power Analysis ===")
    rows = []
    for fl, df in all_data.items():
        for vc, ic in power_pairs:
            if vc not in df.columns or ic not in df.columns:
                missing = [c for c in (vc, ic) if c not in df.columns]
                print(f"  Skip {fl}: {missing} not found")
                continue
            V, I = df[vc].values, df[ic].values
            P = np.mean(V * I)
            Vr = np.sqrt(np.mean(V ** 2))
            Ir = np.sqrt(np.mean(I ** 2))
            S = Vr * Ir
            Q = np.sqrt(max(S ** 2 - P ** 2, 0))
            rows.append(
                {
                    "File": fl,
                    "V ch": vc,
                    "I ch": ic,
                    "V_rms (V)": Vr,
                    "I_rms (A)": Ir,
                    "P active (W)": P,
                    "S apparent (VA)": S,
                    "Q reactive (VAR)": Q,
                    "Power factor": P / S if S else np.nan,
                }
            )
    if rows:
        display(
            pd.DataFrame(rows)
            .set_index(["File", "V ch", "I ch"])
            .map(lambda v: f"{v:.5g}" if not pd.isna(v) else "—")
        )


# =============================================================================
#  SECTION 5 — FFT: RESAMPLING + COMPUTATION
# =============================================================================

def robust_resample(t, y, max_points=500_000):
    """
    Prepare a non-uniform time series for FFT:
      1. Remove duplicate / non-monotone timestamps.
      2. Estimate fs from the 10th-percentile of dt (robust against variable-
         step simulator splinter steps near switching events).
      3. Cap at max_points samples for speed.
      4. Interpolate onto a uniform grid.

    Returns
    -------
    y_uniform   : ndarray
    fs          : float  — uniform sample rate (Hz)
    t_span      : float  — original signal duration (s)
    """
    _, idx = np.unique(t, return_index=True)
    t, y = t[idx], y[idx]
    t_span = t[-1] - t[0]

    dt_pos = np.diff(t)
    dt_pos = dt_pos[dt_pos > 0]
    if not len(dt_pos):
        raise ValueError("All timestamps are identical.")

    dt_est = float(f"{np.percentile(dt_pos, 10):.6g}")
    fs_natural = 1.0 / dt_est
    fs_budget = max_points / t_span
    fs = float(f"{min(fs_natural, fs_budget):.6g}")

    N = int(round(t_span * fs))
    t_u = np.linspace(t[0], t[-1], N, endpoint=False)
    y_u = interp1d(
        t, y, kind="linear", bounds_error=False, fill_value=(y[0], y[-1])
    )(t_u)
    return y_u, fs, t_span


def compute_fft(t, y, f0=None, max_harm=20, window="rectangular", max_points=500_000):
    """
    Compute the single-sided harmonic spectrum of signal y(t).

    Parameters
    ----------
    t           : ndarray  — time axis (s), may be non-uniform
    y           : ndarray  — signal values
    f0          : float | None  — fundamental frequency (Hz); None = auto-detect
    max_harm    : int   — number of harmonics to extract (H1 … Hmax)
    window      : str   — "rectangular" (default, PLECS-compatible, zero leakage
                          on exact integer periods) | "hann" (better sidelobe
                          rejection but adds ±0.25 copies at adjacent bins)
    max_points  : int   — max resampled points (speed vs. resolution trade-off)

    Returns
    -------
    freqs  : ndarray  — frequency axis of the full one-sided spectrum (Hz)
    amps   : ndarray  — one-sided amplitudes corresponding to freqs
    f0     : float    — fundamental frequency used
    thd    : float    — THD in % computed from harmonics H2 … H(max_harm+1)

    Raises
    ------
    ValueError  if the signal window is too short to contain even 1 period of f0
    """
    t0_wall = _time.perf_counter()

    y_u, fs, t_span = robust_resample(t, y, max_points=max_points)
    N = len(y_u)

    # Auto-detect f0
    if f0 is None:
        Ya = np.abs(np.fft.rfft(y_u - y_u.mean()))
        fa = np.fft.rfftfreq(N, d=1.0 / fs)
        f0 = float(fa[np.argmax(Ya[1:]) + 1])
        print(f"  Auto f0   : {f0:.4g} Hz")

    # Validate window length
    periods = t_span * f0
    if periods >= 1.0 - 1e-12:
        n_periods = 1
    else:
        n_periods = int(periods)
    if n_periods < 1:
        T_needed = 1.0 / f0
        raise ValueError(
            f"\n  ⚠  Window too short for f0 = {f0:.4g} Hz\n"
            f"     Signal duration : {t_span:.4g} s\n"
            f"     One period of f0: {T_needed:.4g} s  ({T_needed * 1000:.3g} ms)\n"
            f"     → Either set FUNDAMENTAL_FREQ to the correct fundamental\n"
            f"       (e.g. 50 Hz if the signal is 50 Hz, not {f0:.4g} Hz),\n"
            f"       or provide a longer signal containing ≥ 1 full period."
        )

    # Exact bin alignment: N_trim = n_periods × N_per ensures k·f0 lands on bin k
    N_per = max(1, int(round(fs / f0)))
    N_trim = min(n_periods * N_per, N)

    # Apply window
    seg = y_u[:N_trim]
    win_name = (window or "rectangular").lower()
    if win_name == "hann":
        _win = np.hanning(N_trim)
        seg_w = seg * _win
        scale = _win.mean()
    else:
        win_name = "rectangular"
        seg_w = seg
        scale = 1.0

    Y = np.fft.rfft(seg_w)
    freqs = np.fft.rfftfreq(N_trim, d=1.0 / fs)
    amps = 2.0 * np.abs(Y) / (N_trim * scale)
    amps[0] /= 2.0  # DC bin is not doubled

    df_actual = freqs[1] if len(freqs) > 1 else 0.0
    print(f"  Resampled : {N:,} pts   fs = {fs:.4g} Hz")
    print(f"  n_periods : {n_periods}   N_per = {N_per}   N_trim = {N_trim:,}")
    print(f"  Window    : {win_name}")
    print(f"  df        : {df_actual:.6f} Hz  (target {f0 / n_periods:.6f} Hz)")

    # THD
    ha = []
    for k in range(1, max_harm + 2):
        fh = k * f0
        if fh > freqs[-1]:
            break
        ha.append(amps[np.argmin(np.abs(freqs - fh))])
    V1 = ha[0] if ha else 0.0
    thd = np.sqrt(np.sum(np.array(ha[1:]) ** 2)) / V1 * 100 if V1 > 0 else np.nan

    elapsed = _time.perf_counter() - t0_wall
    thd_str = f"{thd:.3g} %" if not np.isnan(thd) else "N/A"
    print(f"  FFT done  : [{elapsed:.2f} s]   THD = {thd_str}")
    return freqs, amps, f0, thd


def run_fft_all(all_data, fundamental_freq, max_harm, window, max_points):
    """
    Run compute_fft on every channel of every loaded file.

    Returns
    -------
    fft_results : dict  {(file_label, channel): (freqs, amps, f0, thd)}
    """
    print("=== FFT Analysis ===")
    fft_results = {}
    for fl, df in all_data.items():
        t = df["Time"].values
        for ch in [c for c in df.columns if c != "Time"]:
            print(f"\n  {fl} / {ch}")
            try:
                fr, am, f0_out, thd = compute_fft(
                    t,
                    df[ch].values,
                    f0=fundamental_freq,
                    max_harm=max_harm,
                    window=window,
                    max_points=max_points,
                )
                fft_results[(fl, ch)] = (fr, am, f0_out, thd)
            except ValueError as e:
                print(e)
            except Exception as e:
                print(f"  ERROR: {e}")
    print("\nFFT done.")
    return fft_results


# =============================================================================
#  SECTION 6 — FFT HARMONIC BAR CHART
# =============================================================================

def _hex_to_rgba(hex_col, alpha=1.0):
    r, g, b = int(hex_col[1:3], 16), int(hex_col[3:5], 16), int(hex_col[5:7], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _tick_stride(n_harm):
    """Return label stride so at most ~8 harmonic labels are shown at once."""
    if n_harm <= 8:
        return 1
    if n_harm <= 16:
        return 2
    if n_harm <= 40:
        return 5
    return 10


def _plot_harmonics_subplot(
    fig, ri, ch, freqs, amps, f0, thd, max_harm, show_dc, show_grid, file_label,
    legend_names=None, palette=None,
):
    """Add one harmonic bar chart to a subplot row."""
    if palette is None:
        palette = PALETTE
    base_col = palette[ri % len(palette)]
    r0, g0, b0 = int(base_col[1:3], 16), int(base_col[3:5], 16), int(base_col[5:7], 16)
    dc_col = f"rgba({r0},{g0},{b0},0.55)"

    bar_x, bar_y, bar_w, bar_c, bar_hover = [], [], [], [], []

    if show_dc:
        dc_amp = float(amps[0])
        dc_x = -0.5 * f0
        bar_x.append(dc_x)
        bar_y.append(dc_amp)
        bar_w.append(0.35 * f0)
        bar_c.append(dc_col)
        bar_hover.append(f"<b>DC</b>  —  0 Hz<br>Amplitude: {dc_amp:.4g}")

    h_k_list, h_f_list = [], []
    for k in range(1, max_harm + 1):
        fh = k * f0
        if fh > freqs[-1]:
            break
        idx = np.argmin(np.abs(freqs - fh))
        # Only plot if the closest bin is actually close enough
        if np.abs(freqs[idx] - fh) > 0.5:  # same tolerance as your mask
            continue
        a = float(amps[idx])
        b = float(freqs[idx])
        bar_x.append(b)#(fh)
        bar_y.append(a)
        bar_w.append(0.40 * f0)
        bar_c.append(
            f"rgba({r0},{g0},{b0},0.88)"
            if k == 1
            else f"rgba(229,57,53,{max(0.30, 0.88 - (k - 2) * 0.035):.2f})"
        )
        bar_hover.append(f"<b>H{k}</b>  —  {fh:.4g} Hz<br>Amplitude: {a:.4g}")
        h_k_list.append(k)
        h_f_list.append(fh)

    if not h_k_list:
        return

    n_harm = len(h_k_list)
    stride = _tick_stride(n_harm)
    tick_vals, tick_text = [], []

    if show_dc:
        tick_vals.append(-0.5 * f0)
        tick_text.append('DC<br><span style="font-size:9px">0 Hz</span>')

    for k, fh in zip(h_k_list, h_f_list):
        if k == 1 or k % stride == 0:
            tick_vals.append(float(fh))
            tick_text.append(
                f'H{k}<br><span style="font-size:9px">{fh:.4g} Hz</span>'
            )

    thd_s = f"{thd:.2f}%" if not np.isnan(thd) else "N/A"
    gc = "#e0e0e0" if show_grid else "rgba(0,0,0,0)"

    leg_name = _leg(ch, file_label, 1, legend_names)
    fig.add_trace(
        go.Bar(
            x=bar_x,
            y=bar_y,
            width=bar_w,
            marker=dict(color=bar_c, line=dict(width=0.5, color="rgba(0,0,0,0.12)")),
            name=leg_name,
            hovertext=bar_hover,
            hoverinfo="text",
        ),
        row=ri + 1,
        col=1,
    )

    x_min = (-0.9 * f0) if show_dc else (-0.3 * f0)
    x_max = h_f_list[-1] + f0 if h_f_list else f0 * 2

    fig.update_yaxes(
        title_text=f"Amplitude  [THD={thd_s}]",
        row=ri + 1,
        col=1,
        showgrid=show_grid,
        gridcolor=gc,
    )
    fig.update_xaxes(
        title_text="Frequency (Hz)",
        tickmode="array",
        tickvals=tick_vals,
        ticktext=tick_text,
        tickangle=0,
        row=ri + 1,
        col=1,
        showgrid=show_grid,
        gridcolor=gc,
        range=[x_min, x_max],
    )


def plot_fft(
    fft_results,
    max_harm=20,
    show_dc=True,
    show_grid=True,
    height_per_ch=280,
    fig_width=None,
    title_override="",
    legend_names=None,
    palette=None,
):
    """
    Render one interactive harmonic bar chart figure per file.

    Parameters
    ----------
    fft_results   : dict  {(file_label, ch): (freqs, amps, f0, thd)}
    max_harm      : int   — harmonics shown (H1 … Hmax)
    show_dc       : bool  — include the DC bar
    show_grid     : bool  — show axis grid lines
    height_per_ch : int   — px per subplot
    fig_width     : int | None
    title_override: str   — "" = auto title
    legend_names  : dict  — {ch: display_name}
    palette       : list  — hex colour list
    """
    if palette is None:
        palette = PALETTE

    by_file = {}
    for (fl, ch), val in fft_results.items():
        by_file.setdefault(fl, {})[ch] = val

    for fl, ch_dict in by_file.items():
        n = len(ch_dict)
        title = title_override or f"FFT Harmonic Spectrum — {fl}"
        v_spc = max(0.02, min(0.10, 0.60 / max(n - 1, 1)))

        fig = make_subplots(
            rows=n,
            cols=1,
            shared_xaxes=False,
            vertical_spacing=v_spc,
            subplot_titles=list(ch_dict.keys()),
        )

        for ri, (ch, (freqs, amps, f0, thd)) in enumerate(ch_dict.items()):
            _plot_harmonics_subplot(
                fig, ri, ch, freqs, amps, f0, thd,
                max_harm, show_dc, show_grid, fl,
                legend_names=legend_names, palette=palette,
            )

        kw = dict(
            height=height_per_ch * n + 120,
            title_text=title,
            bargap=0.15,
            plot_bgcolor="white",
            paper_bgcolor="white",
            hovermode="closest",
            legend=dict(
                orientation="h", y=-0.03, bgcolor="rgba(255,255,255,0.85)"
            ),
            margin=dict(l=80, r=30, t=70, b=90),
        )
        if fig_width is not None:
            kw["width"] = fig_width
        fig.update_layout(**kw)
        fig.show()


# =============================================================================
#  SECTION 7 — FFT HARMONIC SUMMARY TABLE
# =============================================================================

def table_harmonics(fft_results, top_n=5, max_harm=None):
    """
    Display a table of the top_n strongest harmonics per channel,
    ranked by amplitude, with partial THD from those harmonics.
    Only harmonics up to max_harm are included.

    Parameters
    ----------
    fft_results : dict  {(file_label, ch): (freqs, amps, f0, thd)}
    top_n       : int   — number of strongest harmonics to display
    max_harm    : int | None  — max harmonic number (e.g., 1500). If None, uses full FFT range.
    """
    rows = []
    for (fl, ch), (freqs, amps, f0, _) in fft_results.items():
        # Determine max harmonic number; if max_harm is specified, use it; else use full range
        if max_harm is not None:
            max_k = max_harm
        else:
            max_k = int(freqs[-1] / f0) + 1
        
        hlist = []
        for k in range(1, max_k + 1):
            fh = k * f0
            if fh > freqs[-1]:
                break
            # Use searchsorted for O(log n) lookup instead of O(n) argmin
            idx = np.searchsorted(freqs, fh)
            if idx >= len(freqs):
                idx = len(freqs) - 1
            elif idx > 0 and np.abs(freqs[idx - 1] - fh) < np.abs(freqs[idx] - fh):
                idx -= 1
            hlist.append((k, fh, float(amps[idx])))

        hlist.sort(key=lambda x: x[2], reverse=True)
        top = hlist[:top_n]

        row = {"File": fl, "Channel": ch, "f0 (Hz)": f0}
        for r, (k, fh, a) in enumerate(top, 1):
            row[f"#{r} H{k}  ({fh:.4g} Hz)"] = a
        rows.append(row)

    if not rows:
        print("  No FFT results available.")
        return

    df_out = pd.DataFrame(rows).set_index(["File", "Channel"])
    display(
        df_out.map(
            lambda v: f"{v:.4g}"
            if isinstance(v, float) and not pd.isna(v)
            else ("—" if pd.isna(v) else v)
        )
    )


# =============================================================================
#  SECTION 8 — CUSTOM TIME-DOMAIN OVERLAY
# =============================================================================

def resolve_channels(overlay, all_data):
    """
    Resolve an overlay list to (file_label, channel, DataFrame) tuples.

    Overlay entries:
      "ChannelName"             → searched in all loaded files
      ("file_label", "Channel") → specific file
    """
    out = []
    for item in overlay:
        if isinstance(item, tuple):
            fl, ch = item
            if fl in all_data and ch in all_data[fl].columns:
                out.append((fl, ch, all_data[fl]))
            else:
                print(f"  Not found: ({fl}, {ch})")
        else:
            ch = item
            for fl, df in all_data.items():
                if ch in df.columns:
                    out.append((fl, ch, df))
    return out


def plot_custom_overlay(
    overlay,
    all_data,
    title="Custom overlay",
    ylabel="Amplitude",
    xlabel="Time (s)",
    bg_color="white",
    curve_colors=None,
    line_width=1.6,
    line_dash="solid",
    custom_legend=None,
    height=500,
    width=None,
    palette=None,
):
    """
    Plot any selection of channels on a single interactive figure.

    Parameters
    ----------
    overlay      : list — channel entries (str or (file, ch) tuples)
    all_data     : dict {file_label: DataFrame}
    title        : str
    ylabel       : str
    xlabel       : str
    bg_color     : str  — CSS colour; dark backgrounds auto-invert text/grid
    curve_colors : list — one hex colour per trace; [] = palette
    line_width   : float
    line_dash    : str | list — "solid" | "dot" | "dash" | "longdash" | "dashdot"
    custom_legend: dict — {ch: display_name}
    height       : int  — figure height (px)
    width        : int | None
    palette      : list — default colour palette
    """
    if palette is None:
        palette = PALETTE
    if curve_colors is None:
        curve_colors = []
    if custom_legend is None:
        custom_legend = {}

    if not overlay:
        print("overlay is empty.  Available channels:")
        for fl, df in all_data.items():
            chs = [c for c in df.columns if c != "Time"]
            print(f"  '{fl}' : {chs}")
        return

    traces = resolve_channels(overlay, all_data)
    if not traces:
        print("No matching channels found.")
        return

    dark = bg_color.lower() not in ("white", "#ffffff", "#f5f5f5", "#fafafa", "#fffffe")
    txtcol = "#f0f0f0" if dark else "#1a1a1a"
    gridcl = "#3a3a3a" if dark else "#e0e0e0"
    linecl = "#555555" if dark else "#cccccc"

    fig = go.Figure()
    for i, (fl, ch, df) in enumerate(traces):
        color = curve_colors[i] if i < len(curve_colors) else palette[i % len(palette)]
        dash = (
            line_dash[i]
            if isinstance(line_dash, list) and i < len(line_dash)
            else (line_dash if isinstance(line_dash, str) else "solid")
        )
        leg = custom_legend.get(
            ch, ch if len(all_data) == 1 else f"{fl} – {ch}"
        )
        fig.add_trace(
            go.Scattergl(
                x=df["Time"],
                y=df[ch],
                mode="lines",
                name=leg,
                line=dict(width=line_width, color=color, dash=dash),
            )
        )

    layout_kw = dict(
        title_text=title,
        title_font=dict(size=16, color=txtcol),
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        height=height,
        hovermode="x unified",
        plot_bgcolor=bg_color,
        paper_bgcolor=bg_color,
        font=dict(color=txtcol),
        xaxis=dict(
            showgrid=True,
            gridcolor=gridcl,
            linecolor=linecl,
            tickfont=dict(color=txtcol),
            title_font=dict(color=txtcol),
            zeroline=False,
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor=gridcl,
            linecolor=linecl,
            tickfont=dict(color=txtcol),
            title_font=dict(color=txtcol),
            zeroline=True,
            zerolinecolor=linecl,
            zerolinewidth=1,
        ),
        legend=dict(
            orientation="h",
            y=-0.18,
            bgcolor="rgba(0,0,0,0)",
            font=dict(color=txtcol),
            bordercolor=gridcl,
            borderwidth=1,
        ),
        margin=dict(l=70, r=20, t=65, b=100),
    )
    if width is not None:
        layout_kw["width"] = width
    fig.update_layout(**layout_kw)
    fig.show()


# =============================================================================
#  SECTION 9 — FFT OVERLAY PLOT
# =============================================================================

def resolve_fft_channels(overlay_list, fft_results):
    """
    Resolve an FFT overlay list to (file_label, ch, freqs, amps, f0, thd) tuples.

    Entries:
      "ChannelName"             → all files that contain this channel
      ("file_label", "Channel") → specific file
    """
    out = []
    for item in overlay_list:
        if isinstance(item, tuple):
            fl, ch = item
            key = (fl, ch)
            if key in fft_results:
                out.append((fl, ch, *fft_results[key]))
            else:
                print(f"  Not found: {key}")
        else:
            ch = item
            matched = [
                (fl, ch, *v)
                for (fl, c), v in fft_results.items()
                if c == ch
            ]
            if matched:
                out.extend(matched)
            else:
                print(f"  Channel not found: {ch!r}")
    return out


def plot_fft_overlay(
    fft_overlay,
    fft_results,
    max_harm=20,
    show_dc=True,
    show_grid=True,
    fmin=None,
    fmax=None,
    title_override="",
    height=None,
    width=None,
    height_per_ch_fallback=280,
    palette=None,
):
    """
    Plot harmonics of multiple channels side-by-side on one figure.

    Parameters
    ----------
    fft_overlay   : list — channel entries (str or (file, ch) tuples)
    fft_results   : dict {(file_label, ch): (freqs, amps, f0, thd)}
    max_harm      : int
    show_dc       : bool
    show_grid     : bool
    fmin / fmax   : float | None — frequency zoom range (Hz)
    title_override: str
    height / width: int | None — figure dimensions (px)
    palette       : list
    """
    if palette is None:
        palette = PALETTE

    if not fft_overlay:
        print("FFT_OVERLAY is empty. Available channels:")
        for fl, ch in fft_results:
            print(f"  {fl!r}  →  {ch!r}")
        return

    traces = resolve_fft_channels(fft_overlay, fft_results)
    if not traces:
        print("No matching channels found.")
        return

    title = title_override or "FFT Overlay — " + ", ".join(
        ch for _, ch, *_ in traces
    )

    fig = go.Figure()
    gc = "#e0e0e0" if show_grid else "rgba(0,0,0,0)"

    for i, (fl, ch, freqs, amps, f0, thd) in enumerate(traces):
        col = palette[i % len(palette)]
        label = ch if len({t[0] for t in traces}) == 1 else f"{fl} – {ch}"

        if show_dc:
            dc_amp = float(amps[0]) / 2.0
            fig.add_trace(
                go.Bar(
                    x=[-0.5 * f0],
                    y=[dc_amp],
                    width=[0.30 * f0],
                    marker_color=col,
                    opacity=0.40,
                    name=f"{label} DC",
                    legendgroup=label,
                    hovertext=[f"<b>{label} — DC</b><br>0 Hz<br>A={dc_amp:.4g}"],
                    hoverinfo="text",
                    showlegend=True,
                )
            )

        hx, hy, hh = [], [], []
        for k in range(1, max_harm + 1):
            fh = k * f0
            if fh > freqs[-1]:
                break
            idx = np.argmin(np.abs(freqs - fh))
            a = float(amps[idx])
            hx.append(fh)
            hy.append(a)
            hh.append(f"<b>{label} H{k}</b>  {fh:.4g} Hz<br>A={a:.4g}")

        fig.add_trace(
            go.Bar(
                x=hx,
                y=hy,
                width=0.40 * f0,
                marker_color=col,
                opacity=0.85,
                name=label,
                legendgroup=label,
                hovertext=hh,
                hoverinfo="text",
                showlegend=True,
            )
        )

    all_f0 = [t[4] for t in traces]
    max_f0 = max(all_f0)
    x_min = fmin if fmin is not None else -0.5 * max_f0
    x_max = fmax if fmax is not None else max_harm * max_f0 + max_f0

    fig_height = height if height is not None else max(height_per_ch_fallback, 420)

    layout_kw = dict(
        title_text=title,
        barmode="group",
        bargap=0.10,
        bargroupgap=0.05,
        xaxis=dict(
            title="Frequency (Hz)",
            range=[x_min, x_max],
            showgrid=show_grid,
            gridcolor=gc,
        ),
        yaxis=dict(title="Amplitude", showgrid=show_grid, gridcolor=gc),
        plot_bgcolor="white",
        paper_bgcolor="white",
        hovermode="closest",
        legend=dict(
            orientation="h", y=-0.12, bgcolor="rgba(255,255,255,0.85)"
        ),
        height=fig_height,
        margin=dict(l=80, r=30, t=70, b=120),
    )
    if width is not None:
        layout_kw["width"] = width
    fig.update_layout(**layout_kw)
    fig.show()
