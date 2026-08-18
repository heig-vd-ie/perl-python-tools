"""
plecs_sim_library.py
────────────────────
Helper functions for running PLECS simulations from Python via XML-RPC.
Supports parameter sweeps and Monte Carlo runs.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 HOW PLECS SIGNALS REACH PYTHON  (read this once)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 In your PLECS model, add an "Outport" block on the TOP-LEVEL schematic
 (Component > Signal Sources > Outport, or search "Out").
 Wire every signal you want to export into that single Outport.
 The Outport block accepts a vector, so you can bundle many signals together
 using a Mux block.

 The signals come back to Python as rows of a 2-D numpy array called
 `values` (shape = n_signals x n_samples):

   values[0]  ->  first signal wired into the Outport   (signal_index = 0)
   values[1]  ->  second signal                          (signal_index = 1)
   values[2]  ->  third signal                           (signal_index = 2)
   ...

 The order is the order signals enter the Outport (top to bottom if you use
 a Mux block, or the port number if you wire signals directly).

 To discover which index corresponds to which signal, call:
     inspect_signals(results)
 after your first simulation run.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 BASE_VARS MODES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 PLECS only overrides the variables you explicitly send in ModelVars.
 Every variable you do NOT include keeps the value defined inside the
 PLECS model (initialization commands / component parameters).

  Mode A — "Python controls everything"
      BASE_VARS = { 'R': 0.5, 'L': 3e-3, 'fc': 20e3, ... }
      All variables are defined here; PLECS values are ignored for those keys.

  Mode B — "PLECS controls everything, Python changes only swept params"
      BASE_VARS = {}
      Only the swept / MC parameters are sent; everything else uses the
      values already defined inside the .plecs file.

Inspired by the Plexim tutorial "PLECS RPC Interface and Controller Design in Python".
"""

import xmlrpc.client
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import scipy.io as sio
import io
import os
import copy
import random
import csv
from datetime import datetime


# ===============================================================================
#  CONNECTION
# ===============================================================================

def connect_plecs(host="http://localhost:1080/RPC2"):
    """
    Connect to the PLECS XML-RPC server.

    Parameters
    ----------
    host : str
        URL of the PLECS XML-RPC server (default port 1080).

    Returns
    -------
    server : xmlrpc.client.Server
    """
    server = xmlrpc.client.Server(host)
    print(f"[PLECS] Connected to {host}")
    return server


# ===============================================================================
#  MODEL MANAGEMENT
# ===============================================================================

def load_model(server, model_folder, model_name):
    """
    Load a PLECS model from disk.

    Parameters
    ----------
    server       : xmlrpc.client.Server
    model_folder : str  -- absolute path of the folder containing the .plecs file.
    model_name   : str  -- file name WITHOUT the .plecs extension.
    """
    full_path = os.path.join(model_folder, model_name)
    server.plecs.load(full_path)
    print(f"[PLECS] Model '{model_name}' loaded from '{model_folder}'")


def close_model(server, model_name):
    """
    Close a loaded PLECS model.

    Parameters
    ----------
    server     : xmlrpc.client.Server
    model_name : str  -- model name WITHOUT the .plecs extension.
    """
    server.plecs.close(model_name)
    print(f"[PLECS] Model '{model_name}' closed.")


# ===============================================================================
#  CORE SIMULATION RUNNER  (internal)
# ===============================================================================

def _run_one_simulation(server, model_name, model_vars, mode='simulate', analysis_name=None):
    """
    Run a single PLECS simulation or analysis and return time + signal arrays.

    Parameters
    ----------
    server       : xmlrpc.client.Server
    model_name   : str
    model_vars   : dict or None
    mode         : str  -- 'simulate' or 'analyze'
    analysis_name: str or None  -- required if mode='analyze'

    Returns
    -------
    time   : np.ndarray  shape (n_samples,)
    values : np.ndarray  shape (n_signals, n_samples)
        Each row is one signal wired into the top-level Outport block,
        in the order the signals were connected (0-based row index).
    """
    opts = {
        "ModelVars"    : model_vars if model_vars else {},
        "OutputFormat" : "MatFile",
    }
    if mode == 'simulate':
        data_raw = server.plecs.simulate(model_name, opts)
    elif mode == 'analyze':
        if analysis_name is None:
            raise ValueError("analysis_name must be provided when mode='analyze'")
        data_raw = server.plecs.analyze(model_name, analysis_name, opts)
    else:
        raise ValueError("mode must be 'simulate' or 'analyze'")
    data     = sio.loadmat(io.BytesIO(data_raw.data))
    time     = data["Time"][0]   # shape (n_samples,)
    values   = data["Values"]    # shape (n_signals, n_samples)
    return time, values


# ===============================================================================
#  SIGNAL INSPECTOR
# ===============================================================================

def inspect_signals(results):
    """
    Print a summary of all signals available in the results.

    Call this after your first simulation to find out which signal_index
    corresponds to which signal in your PLECS model.

    Parameters
    ----------
    results : list of dict  -- output of run_sweep() or run_montecarlo()
    """
    res       = results[0]
    n_signals = res['values'].shape[0]
    n_samples = res['values'].shape[1]
    t_start   = res['time'][0]
    t_end     = res['time'][-1]

    print("Signals available in results (first run):")
    print(f"  Total signals  : {n_signals}  "
          f"-> use signal_index = 0 to {n_signals - 1}")
    print(f"  Total samples  : {n_samples}")
    print(f"  Time range     : {t_start:.6g} s  to  {t_end:.6g} s")
    print()
    print(f"  {'index':<8}  {'min':>14}  {'max':>14}  {'mean':>14}")
    print(f"  {'-------':<8}  {'-------------':>14}  {'-------------':>14}  {'-------------':>14}")
    for i in range(n_signals):
        row = res['values'][i]
        print(f"  {i:<8}  {row.min():>14.5g}  {row.max():>14.5g}  {row.mean():>14.5g}")


# ===============================================================================
#  PARAMETER SWEEP
# ===============================================================================

def run_sweep(server, model_name, base_vars, sweep_params, mode='simulate', analysis_name=None):
    """
    Sweep one or several PLECS model parameters over a list of values.

    Parameters
    ----------
    server        : xmlrpc.client.Server
    model_name    : str
    base_vars     : dict or None
        Variables sent to PLECS on every run.
        Populated dict -> Mode A (Python overrides those variables every run).
        {} or None     -> Mode B (only swept params are sent; everything else
                          comes from the PLECS model initialization).
    sweep_params  : dict  { param_name : [val_0, val_1, ...] }
        Each key is a PLECS variable name; the value is a list of values to try.
        When multiple keys are given they are stepped together (zip-style),
        so all lists must have the same length.
    mode          : str  -- 'simulate' or 'analyze'
    analysis_name : str or None  -- required if mode='analyze'

    Returns
    -------
    results : list of dict, one per step:
        'params'  -- dict of the swept parameter values used in this step
        'time'    -- np.ndarray (n_samples,)
        'values'  -- np.ndarray (n_signals, n_samples)
    """
    base_vars   = base_vars or {}
    param_names = list(sweep_params.keys())
    n_steps     = len(sweep_params[param_names[0]])

    for name in param_names:
        if len(sweep_params[name]) != n_steps:
            raise ValueError(
                f"Sweep list for '{name}' has length {len(sweep_params[name])}, "
                f"expected {n_steps}."
            )

    mode_label = ("Python + PLECS vars" if base_vars
                  else "PLECS vars only (base_vars empty)")
    print(f"[Sweep] {n_steps} step(s) | swept: {param_names} | mode: {mode_label}")
    if base_vars:
        print(f"  base_vars sent every run: {list(base_vars.keys())}")

    results = []
    for i in range(n_steps):
        vars_i   = copy.deepcopy(base_vars)
        params_i = {}
        for name in param_names:
            vars_i[name]   = sweep_params[name][i]
            params_i[name] = sweep_params[name][i]

        time, values = _run_one_simulation(server, model_name, vars_i, mode, analysis_name)
        results.append({"params": params_i, "time": time, "values": values})
        print(f"  step {i+1:>3}/{n_steps}  ->  {params_i}")

    print("[Sweep] Done.")
    return results


# ===============================================================================
#  MONTE CARLO
# ===============================================================================

def run_montecarlo(server, model_name, base_vars, mc_params, n_runs, seed=None, mode='simulate', analysis_name=None):
    """
    Monte Carlo simulation: randomly vary one or several PLECS parameters.

    Parameters
    ----------
    server     : xmlrpc.client.Server
    model_name : str
    base_vars  : dict or None
        Same semantics as run_sweep(). {} / None = Mode B.
    mc_params  : dict  { param_name : (nominal, tolerance, distribution) }
        nominal      : float  -- nominal (center) value
        tolerance    : float  -- relative tolerance (e.g. 0.10 = +/-10 %)
        distribution : str    -- 'uniform' or 'normal'
            'uniform' -> flat in [nominal*(1-tol), nominal*(1+tol)]
            'normal'  -> Gaussian with sigma = nominal * tol
    n_runs     : int
    seed       : int or None  -- fixed seed for reproducibility
    mode       : str  -- 'simulate' or 'analyze'
    analysis_name : str or None  -- required if mode='analyze'

    Returns
    -------
    results : list of dict  (same structure as run_sweep)
    """
    base_vars = base_vars or {}

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    mode_label = ("Python + PLECS vars" if base_vars
                  else "PLECS vars only (base_vars empty)")
    print(f"[MC] {n_runs} run(s) | varied: {list(mc_params.keys())} "
          f"| mode: {mode_label}"
          + (f" | seed={seed}" if seed is not None else ""))
    if base_vars:
        print(f"  base_vars sent every run: {list(base_vars.keys())}")

    results = []
    for i in range(n_runs):
        vars_i   = copy.deepcopy(base_vars)
        params_i = {}
        for name, (nominal, tol, dist) in mc_params.items():
            if dist == 'uniform':
                val = nominal * (1.0 + random.uniform(-tol, tol))
            elif dist == 'normal':
                val = random.gauss(nominal, abs(nominal) * tol)
            else:
                raise ValueError(
                    f"Unknown distribution '{dist}'. Choose 'uniform' or 'normal'."
                )
            vars_i[name]   = val
            params_i[name] = val

        time, values = _run_one_simulation(server, model_name, vars_i, mode, analysis_name)
        results.append({"params": params_i, "time": time, "values": values})
        print(f"  run  {i+1:>3}/{n_runs}  ->  "
              + "  ".join(f"{k}={v:.4g}" for k, v in params_i.items()))

    print("[MC] Done.")
    return results


# ===============================================================================
#  CSV EXPORT
# ===============================================================================

def save_results_csv(results, signal_names, save_folder, filename=None):
    """
    Save all simulation time-series results to a single CSV file.

    One row per time sample per run. A 'run_index' column and one column per
    swept parameter identify each run. Columns then follow for time and every
    signal.

    Parameters
    ----------
    results      : list of dict  -- output of run_sweep() or run_montecarlo()
    signal_names : list of str
        Human-readable names for every signal, in the same order as the
        rows of `values`  (signal_names[i] labels signal_index i).
        Must match the number of signals in the data.
    save_folder  : str   -- folder where the CSV is written (created if needed)
    filename     : str or None
        CSV filename (without path). If None, a timestamped name is used:
        sim_results_YYYYMMDD_HHMMSS.csv

    Returns
    -------
    filepath : str

    Example
    -------
    >>> save_results_csv(
    ...     results,
    ...     signal_names = ['Id_ref_A', 'Id_meas_A'],
    ...     save_folder  = r'C:\\Users\\you\\sim_data',
    ... )
    """
    os.makedirs(save_folder, exist_ok=True)

    if filename is None:
        ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"sim_results_{ts}.csv"

    filepath   = os.path.join(save_folder, filename)
    n_signals  = results[0]['values'].shape[0]

    if len(signal_names) != n_signals:
        raise ValueError(
            f"signal_names has {len(signal_names)} entries but the data has "
            f"{n_signals} signals. They must match."
        )

    param_keys = list(results[0]['params'].keys())

    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        # Header
        writer.writerow(
            ['run_index'] +
            [f'param_{k}' for k in param_keys] +
            ['time_s'] +
            signal_names
        )
        # Rows
        for run_idx, res in enumerate(results):
            time      = res['time']
            values    = res['values']
            param_row = [res['params'].get(k, '') for k in param_keys]
            for s in range(len(time)):
                writer.writerow(
                    [run_idx] +
                    param_row +
                    [time[s]] +
                    [values[sig_i][s] for sig_i in range(n_signals)]
                )

    print(f"[CSV] Saved {len(results)} run(s), {results[0]['values'].shape[1]} "
          f"samples/run  ->  {filepath}")
    return filepath


def save_montecarlo_stats_csv(results, signal_names, t_eval_list,
                               save_folder, filename=None):
    """
    Save a compact Monte Carlo statistics table to CSV.

    One row per run. For each (signal, time-instant) pair the sampled value
    is stored, plus the parameter values for that run. Useful for
    post-processing distributions without loading the full time-series.

    Parameters
    ----------
    results      : list of dict
    signal_names : list of str   -- names matching signal row indices
    t_eval_list  : list of float -- time instants (seconds) at which to sample
    save_folder  : str
    filename     : str or None

    Returns
    -------
    filepath : str

    Example
    -------
    >>> save_montecarlo_stats_csv(
    ...     results,
    ...     signal_names = ['Id_ref_A', 'Id_meas_A'],
    ...     t_eval_list  = [0.230, 0.232, 0.234],
    ...     save_folder  = r'C:\\Users\\you\\sim_data',
    ... )
    """
    os.makedirs(save_folder, exist_ok=True)

    if filename is None:
        ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"mc_stats_{ts}.csv"

    filepath   = os.path.join(save_folder, filename)
    param_keys = list(results[0]['params'].keys())

    value_cols = [
        f"{sname}@t={t:.6g}s"
        for t in t_eval_list
        for sname in signal_names
    ]

    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(
            ['run_index'] +
            [f'param_{k}' for k in param_keys] +
            value_cols
        )
        for run_idx, res in enumerate(results):
            time       = res['time']
            values     = res['values']
            param_row  = [res['params'].get(k, '') for k in param_keys]
            sample_row = []
            for t in t_eval_list:
                idx = int(np.argmin(np.abs(time - t)))
                for sig_i in range(values.shape[0]):
                    sample_row.append(values[sig_i][idx])
            writer.writerow([run_idx] + param_row + sample_row)

    print(f"[CSV] MC stats saved ({len(results)} runs, "
          f"{len(t_eval_list)} time points)  ->  {filepath}")
    return filepath


# ===============================================================================
#  DOWNSAMPLING  (min-max bucketing — vectorized, no Python loop)
# ===============================================================================
#
#  Why min-max instead of LTTB?
#  ─────────────────────────────
#  LTTB is accurate but sequential: it has a Python for-loop that runs
#  n_out times, which becomes the bottleneck for large datasets or many
#  MC runs (e.g. 500 runs × 2 signals × 10k loop iterations = 10M Python
#  iterations → minutes).
#
#  Min-max bucketing:
#  • Splits the signal into n_out/2 equal buckets.
#  • Keeps the min AND max sample from each bucket (2 points/bucket).
#  • Fully vectorised with numpy — no Python loop at all.
#  • For power electronics this is actually BETTER than LTTB because it
#    always preserves voltage/current spikes, never smoothing them away.
#
#  Typical speedup vs LTTB: 50–200×.

def _minmax_downsample(t, y, n_out):
    """
    Downsample a time-series to ≤ n_out points using min-max bucketing.

    Fully vectorised (numpy only, no Python for-loop).
    Always preserves peaks and troughs — ideal for power-electronics waveforms.

    Parameters
    ----------
    t, y   : np.ndarray  shape (n,)
    n_out  : int  -- target number of output points (actual output <= n_out)

    Returns
    -------
    t_ds, y_ds : np.ndarray
    """
    n = len(t)
    if n_out >= n:
        return t, y

    n_out    = max(n_out, 4)
    n_buckets = n_out // 2          # each bucket contributes min + max = 2 pts

    # Trim to a length exactly divisible by n_buckets
    trim     = (n // n_buckets) * n_buckets
    bsize    = trim // n_buckets    # samples per bucket

    # Reshape into (n_buckets, bsize)
    y_mat    = y[:trim].reshape(n_buckets, bsize)

    # Global index of min and max within each bucket
    offsets  = np.arange(n_buckets) * bsize
    idx_min  = offsets + np.argmin(y_mat, axis=1)
    idx_max  = offsets + np.argmax(y_mat, axis=1)

    # Interleave in chronological order within each bucket
    pairs    = np.stack([idx_min, idx_max], axis=1)   # (n_buckets, 2)
    pairs.sort(axis=1)
    all_idx  = pairs.ravel()

    # Remove duplicates that arise when min == max (flat region)
    all_idx  = all_idx[np.concatenate(
        [[True], all_idx[1:] != all_idx[:-1]]
    )]

    # Include all remaining samples from trim to the end (prevents gaps at signal end)
    if trim < n:
        tail_idx = np.arange(trim, n)
        all_idx  = np.concatenate([all_idx, tail_idx])

    return t[all_idx], y[all_idx]


def _downsample(t, y, max_points):
    """
    Apply min-max downsampling if the series exceeds max_points.
    Returns (t_out, y_out, was_downsampled).
    """
    if max_points is None or len(t) <= max_points:
        return t, y, False
    t_ds, y_ds = _minmax_downsample(t, y, max_points)
    return t_ds, y_ds, True


# ===============================================================================
#  PLOTLY PLOTTING
# ===============================================================================
#
#  Performance design
#  ──────────────────
#  Three layers work together:
#
#  1. Time-window masking first
#     Applied before downsampling so only the visible portion of the data
#     is processed. A narrow time_window on a long simulation dramatically
#     reduces the work done by the downsampler.
#
#  2. Min-max downsampling (vectorised numpy, no Python loop)
#     Reduces each trace to max_points_per_figure / n_traces points.
#     The budget is shared across traces so the total data sent to the
#     browser stays bounded regardless of how many runs there are.
#     CSV export always uses the full raw data — display only.
#
#  3. go.Scattergl (WebGL GPU rendering)
#     10–20× faster browser rendering than SVG for large point counts.
#
#  Choosing max_points (total budget for the whole figure)
#  ───────────────────────────────────────────────────────
#  50 000   fast default — good for sweep (few traces, high detail)
#  20 000   recommended for 20–100 MC runs
#  10 000   recommended for 100–500 MC runs
#  None     no downsampling (raw data) — only for very small datasets

def plot_signals(results, plot_config, sim_mode="sweep", label_param=None,
                 max_points=50_000):
    """
    Plot simulation results using Plotly (interactive: zoom, pan, hover).

    Fast for large datasets: uses WebGL rendering (Scattergl) and vectorised
    min-max downsampling. The total point budget (max_points) is shared across
    all traces in the figure so browser load stays constant regardless of how
    many runs or signals are plotted. CSV export is never affected — raw data.

    Creates one figure per entry in plot_config.

    Parameters
    ----------
    results     : list of dict  -- output of run_sweep() or run_montecarlo()

    plot_config : list of dict, one dict per figure:

        signal_index : int or list[int]
            Which signal(s) to plot (0-based row index of the PLECS Outport).
            Use inspect_signals(results) to find the available indices.

        signal_name : str or list[str]
            Display name(s). Must match length of signal_index.

        time_window : (t_start, t_end) or None
            Time range in seconds.  None -> full simulation time.

        title       : str            (default: 'Simulation Results')
        xlabel      : str            (default: 'Time (s)')
        ylabel      : str            (default: 'Value')

        show_legend : bool           (default: True)
            Set to False when many runs make the legend unreadable.

    sim_mode    : 'sweep' or 'montecarlo'

    label_param : str or None
        Which swept parameter to show in the legend (sweep mode).

    max_points  : int or None
        Total point budget for the whole figure (shared across all traces).
        Each trace gets  max_points // n_traces  points.
        Recommended values:
          50 000  default — good balance for sweep with few traces
          20 000  recommended for 20–100 MC runs
          10 000  recommended for 100–500 MC runs
          None    no downsampling — only for already-small datasets
    """
    import time as _time

    n_runs  = len(results)
    palette = px.colors.qualitative.Plotly      # 10 distinct colours
    if n_runs > len(palette):
        palette = px.colors.qualitative.Alphabet    # 26 colours
    if n_runs > len(palette):
        palette = [palette[i % len(palette)] for i in range(n_runs)]

    # Resolve label_param once
    lp = label_param
    if lp is None and results and results[0]['params']:
        lp = list(results[0]['params'].keys())[0]

    for cfg in plot_config:

        # Normalise signal_index / signal_name to lists
        sig_indices = cfg['signal_index']
        sig_names   = cfg.get('signal_name', None)
        if isinstance(sig_indices, int):
            sig_indices = [sig_indices]
        if sig_names is None:
            sig_names = [f"Signal {i}" for i in sig_indices]
        if isinstance(sig_names, str):
            sig_names = [sig_names]

        time_window = cfg.get('time_window', None)
        title       = cfg.get('title',       'Simulation Results')
        xlabel      = cfg.get('xlabel',      'Time (s)')
        ylabel      = cfg.get('ylabel',      'Value')
        show_legend = cfg.get('show_legend', True)

        # Total traces in this figure
        n_traces = n_runs * len(sig_indices)

        # Per-trace point budget — divide total budget among all traces
        pts_per_trace = (
            max(200, max_points // n_traces) if max_points is not None else None
        )

        # Hover mode: 'x unified' lags with many traces
        hovermode = 'x unified' if n_traces <= 20 else 'x'

        t0              = _time.perf_counter()
        traces          = []            # build list first, then create Figure
        any_downsampled = False
        raw_total       = 0
        ds_total        = 0

        for run_idx, res in enumerate(results):
            time   = res['time']
            values = res['values']
            params = res['params']

            # 1. Apply time window FIRST (reduces data before downsampling)
            if time_window is not None:
                mask  = (time >= time_window[0]) & (time <= time_window[1])
                t_win = time[mask]
            else:
                mask  = slice(None)     # no copy — just a view
                t_win = time

            # Build legend label
            if sim_mode == 'sweep' and lp and lp in params:
                lp_val = params[lp]
                if isinstance(lp_val, (int, float, np.floating, np.integer)):
                    run_label = f"{lp}={lp_val:.4g}"
                else:
                    run_label = f"{lp}={lp_val}"
            else:
                run_label = f"Run {run_idx + 1}"

            color = palette[run_idx % len(palette)]

            for sig_idx, sig_name in zip(sig_indices, sig_names):
                y_win = values[sig_idx][mask]

                # 2. Vectorised min-max downsampling (display only)
                t_plot, y_plot, was_ds = _downsample(t_win, y_win, pts_per_trace)

                raw_total += len(t_win)
                ds_total  += len(t_plot)
                if was_ds:
                    any_downsampled = True

                trace_name = (
                    f"{sig_name} [{run_label}]" if n_runs > 1 else sig_name
                )

                # 3. Scattergl = WebGL GPU rendering
                traces.append(go.Scattergl(
                    x           = t_plot,
                    y           = y_plot,
                    mode        = 'lines',
                    name        = trace_name,
                    line        = dict(color=color),
                    legendgroup = run_label,
                    showlegend  = show_legend,
                ))

        t_build = _time.perf_counter() - t0

        # Create figure with all traces at once (faster than add_trace in loop)
        fig = go.Figure(data=traces)
        fig.update_layout(
            title       = title,
            xaxis_title = xlabel,
            yaxis_title = ylabel,
            hovermode   = hovermode,
            template    = 'plotly_white',
            legend      = dict(
                orientation = "v",
                x=1.01, xanchor="left",
                y=1,    yanchor="top",
            ),
        )
        if time_window is not None:
            fig.update_xaxes(range=list(time_window))

        t_render = _time.perf_counter() - t0

        # Summary line
        if any_downsampled:
            ratio = raw_total / ds_total if ds_total else 1
            print(f"[Plot] '{title}'\n"
                  f"       {n_traces} trace(s) | "
                  f"{raw_total:,} raw pts -> {ds_total:,} rendered "
                  f"({ratio:.0f}x reduction) | "
                  f"build {t_build:.2f}s | render {t_render:.2f}s")
        else:
            print(f"[Plot] '{title}'\n"
                  f"       {n_traces} trace(s) | "
                  f"{raw_total:,} pts (no downsampling) | "
                  f"build {t_build:.2f}s | render {t_render:.2f}s")

        fig.show()


def plot_montecarlo_histogram(results, signal_index, t_eval,
                               signal_name="Signal",
                               title="Monte Carlo Histogram",
                               xlabel="Value",
                               n_bins=20,
                               show_legend=True):
    """
    Interactive Plotly histogram of signal values sampled at a specific time.

    Parameters
    ----------
    results      : list of dict
    signal_index : int    -- OutPort row index (0-based)
    t_eval       : float  -- time instant (seconds) at which to sample each run
    signal_name  : str
    title        : str
    xlabel       : str
    n_bins       : int
    show_legend  : bool
    """
    sampled = []
    for res in results:
        idx = int(np.argmin(np.abs(res['time'] - t_eval)))
        sampled.append(float(res['values'][signal_index][idx]))

    arr        = np.array(sampled)
    mean_val   = float(np.mean(arr))
    median_val = float(np.median(arr))

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x          = sampled,
        nbinsx     = n_bins,
        name       = signal_name,
        marker     = dict(color='steelblue',
                          line=dict(color='white', width=0.5)),
        opacity    = 0.85,
        showlegend = show_legend,
    ))
    fig.add_vline(x=mean_val,   line_dash='dash', line_color='red',
                  annotation_text=f"Mean={mean_val:.4g}",
                  annotation_position="top right")
    fig.add_vline(x=median_val, line_dash='dot',  line_color='orange',
                  annotation_text=f"Median={median_val:.4g}",
                  annotation_position="top left")
    fig.update_layout(
        title       = title,
        xaxis_title = xlabel,
        yaxis_title = "Count",
        template    = "plotly_white",
        bargap      = 0.05,
    )
    fig.show()

    print(f"[Histogram] {signal_name}  at  t = {t_eval:.4g} s")
    print(f"  n      = {len(sampled)}")
    print(f"  Mean   = {mean_val:.4g}")
    print(f"  Std    = {np.std(arr):.4g}")
    print(f"  Min    = {arr.min():.4g}")
    print(f"  Max    = {arr.max():.4g}")
