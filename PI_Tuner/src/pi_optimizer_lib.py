"""
pi_optimizer_lib.py
───────────────────
Library for automated PI controller optimisation using PLECS simulations.

Workflow
────────
1. PLECS runs a closed-loop step-response simulation for given Kp / Ki values.
2. Python extracts the measured output and computes step-response metrics
   (rise time, overshoot, settling time, steady-state error, ISE, ITAE).
3. A scalar cost function turns those metrics into one number to minimise.
4. An optimiser (grid search → differential evolution → Nelder-Mead) searches
   the (Kp, Ki) space and drives the cost to zero.

Reuses
──────
- plecs_sim_library.py  - _run_one_simulation()
- signal_analysis_lib.py - no direct dependency; metrics are computed here

Usage
─────
    from pi_optimizer_lib import (
        compute_step_response_metrics,
        build_cost_function,
        grid_search_pi,
        optimize_pi_differential_evolution,
        optimize_pi_nelder_mead,
        plot_cost_landscape,
        plot_convergence,
        plot_best_response,
        print_optimization_summary,
    )
"""

import copy
import time as _time
import warnings
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy.optimize import differential_evolution, minimize

# Import the PLECS runner (must be in the same folder)
from src.plecs_sim_library import _run_one_simulation

warnings.filterwarnings("ignore", category=RuntimeWarning)


# ══════════════════════════════════════════════════════════════════════════════
#  DATA STRUCTURES
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class StepMetrics:
    """
    Scalar metrics extracted from a single closed-loop step response.

    All times are in seconds; amplitudes in the same unit as the signal.
    """
    rise_time      : float   # time from 10% to 90% of the step amplitude
    overshoot_pct  : float   # peak overshoot as % of step amplitude (0 if none)
    settling_time  : float   # time to enter and stay within ±settling_band of setpoint
    sse            : float   # steady-state error (absolute, mean of last window)
    ise            : float   # integral of squared error  ∫ e² dt
    itae           : float   # integral of time × |error|  ∫ t·|e| dt

    def as_dict(self) -> Dict[str, float]:
        return {
            "rise_time"     : self.rise_time,
            "overshoot_pct" : self.overshoot_pct,
            "settling_time" : self.settling_time,
            "sse"           : self.sse,
            "ise"           : self.ise,
            "itae"          : self.itae,
        }


@dataclass
class OptimHistory:
    """
    Running log of every (Kp, Ki, cost) evaluation during optimisation.
    """
    kp_list   : List[float] = field(default_factory=list)
    ki_list   : List[float] = field(default_factory=list)
    cost_list : List[float] = field(default_factory=list)
    tag_list  : List[str]   = field(default_factory=list)   # 'grid' | 'de' | 'nm'

    def append(self, kp: float, ki: float, cost: float, tag: str = ""):
        self.kp_list.append(float(kp))
        self.ki_list.append(float(ki))
        self.cost_list.append(float(cost))
        self.tag_list.append(tag)

    def best(self) -> Tuple[float, float, float]:
        """Return (kp, ki, cost) of the lowest-cost evaluation so far."""
        idx = int(np.argmin(self.cost_list))
        return self.kp_list[idx], self.ki_list[idx], self.cost_list[idx]

    def __len__(self):
        return len(self.cost_list)


# ══════════════════════════════════════════════════════════════════════════════
#  STEP-RESPONSE METRICS
# ══════════════════════════════════════════════════════════════════════════════

def compute_step_response_metrics(
    time         : np.ndarray,
    measured     : np.ndarray,
    setpoint     : float,
    initial_value: float,
    t_step       : float,
    settling_band: float = 0.02,
    ss_window    : float = 0.05,
) -> StepMetrics:
    """
    Compute closed-loop step-response metrics from simulation data.

    Parameters
    ----------
    time          : 1-D array - simulation time vector (seconds)
    measured      : 1-D array - output signal to evaluate (same length as time)
    setpoint      : float - target (final) value after the step
    initial_value : float - value before the step (used to compute amplitude)
    t_step        : float - time at which the setpoint step occurs (seconds)
    settling_band : float - fraction of step amplitude defining the settling
                    criterion band (default 0.02 = ±2 %)
    ss_window     : float - fraction of the post-step window used to compute
                    steady-state error (default 0.05 = last 5 %)

    Returns
    -------
    StepMetrics dataclass
    """
    # ── Isolate the post-step portion ──────────────────────────────────────
    mask     = time >= t_step
    t_post   = time[mask]
    y_post   = measured[mask]

    if len(t_post) < 10:
        # Not enough data after the step — return worst-case metrics
        return StepMetrics(
            rise_time=np.inf, overshoot_pct=np.inf,
            settling_time=np.inf, sse=np.inf,
            ise=np.inf, itae=np.inf,
        )

    amplitude = setpoint - initial_value
    if abs(amplitude) < 1e-12:
        raise ValueError("Step amplitude is zero: setpoint == initial_value.")

    # ── Rise time (10 % → 90 % of step amplitude) ──────────────────────────
    lo = initial_value + 0.10 * amplitude
    hi = initial_value + 0.90 * amplitude

    # Handle both rising and falling steps
    if amplitude > 0:
        idx_10 = np.argmax(y_post >= lo)
        idx_90 = np.argmax(y_post >= hi)
    else:
        idx_10 = np.argmax(y_post <= lo)
        idx_90 = np.argmax(y_post <= hi)

    rise_time = (
        float(t_post[idx_90] - t_post[idx_10])
        if (idx_90 > 0 and idx_10 >= 0)
        else float(t_post[-1] - t_post[0])   # fallback: full window
    )

    # ── Overshoot ───────────────────────────────────────────────────────────
    if amplitude > 0:
        peak_val = float(y_post.max())
    else:
        peak_val = float(y_post.min())

    overshoot_abs = peak_val - setpoint
    if amplitude > 0:
        overshoot_pct = max(0.0, overshoot_abs / abs(amplitude) * 100.0)
    else:
        overshoot_pct = max(0.0, -overshoot_abs / abs(amplitude) * 100.0)

    # ── Settling time (±settling_band of amplitude around setpoint) ─────────
    band          = abs(settling_band * amplitude)
    outside       = np.abs(y_post - setpoint) > band
    # Find the last index where the signal is still outside the band
    outside_idx   = np.where(outside)[0]

    if len(outside_idx) == 0:
        settling_time = 0.0   # already settled from the start
    else:
        last_out      = outside_idx[-1]
        settling_time = float(t_post[last_out] - t_post[0])

    # ── Steady-state error ──────────────────────────────────────────────────
    n_ss    = max(1, int(ss_window * len(t_post)))
    ss_mean = float(np.mean(y_post[-n_ss:]))
    sse     = abs(ss_mean - setpoint)

    # ── Integral criteria ───────────────────────────────────────────────────
    error = y_post - setpoint
    dt    = np.diff(t_post, prepend=t_post[0])

    ise  = float(np.sum(error**2 * dt))
    itae = float(np.sum(t_post * np.abs(error) * dt))

    return StepMetrics(
        rise_time     = rise_time,
        overshoot_pct = overshoot_pct,
        settling_time = settling_time,
        sse           = sse,
        ise           = ise,
        itae          = itae,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  COST FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def build_cost_function(
    weights           : Dict[str, float],
    normalisation     : Optional[Dict[str, float]] = None,
) -> Callable[[StepMetrics], float]:
    """
    Return a scalar cost function  f(metrics) → float.

    The cost is a weighted sum of normalised metrics:

        cost = Σ  weights[k] * (metrics[k] / normalisation[k])

    Parameters
    ----------
    weights : dict
        Keys must be a subset of:
            'rise_time', 'overshoot_pct', 'settling_time', 'sse', 'ise', 'itae'
        Values are non-negative floats; they are automatically normalised so
        they sum to 1 (relative importance only matters).

    normalisation : dict or None
        Scale factors for each metric (same keys as weights).
        Divide each metric by its scale factor before weighting so that all
        terms contribute similarly.
        If None, safe defaults are used (tuned for typical power-electronics
        current / voltage control loops).

    Returns
    -------
    cost_fn : callable  -  cost_fn(metrics: StepMetrics) → float
    """
    # Default normalisation constants (adjust for your application)
    _defaults = {
        "rise_time"     : 5e-3,    # 5 ms
        "overshoot_pct" : 10.0,    # 10 %
        "settling_time" : 20e-3,   # 20 ms
        "sse"           : 0.1,     # 0.1 (same unit as setpoint)
        "ise"           : 1e-3,    # 1e-3 (unit² · s)
        "itae"          : 1e-4,    # 1e-4 (unit · s²)
    }

    norm = {**_defaults, **(normalisation or {})}

    # Normalise weights so they sum to 1
    total = sum(weights.values())
    if total <= 0:
        raise ValueError("At least one weight must be positive.")
    w = {k: v / total for k, v in weights.items()}

    def cost_fn(m: StepMetrics) -> float:
        d = m.as_dict()
        cost = 0.0
        for key, weight in w.items():
            val = d.get(key, 0.0)
            # Cap inf / nan so the optimiser can still make progress
            if not np.isfinite(val):
                val = 1e6
            cost += weight * (val / norm[key])
        return float(cost)

    return cost_fn


# ══════════════════════════════════════════════════════════════════════════════
#  SINGLE EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_pi(
    server          ,
    model_name      : str,
    base_vars       : dict,
    kp              : float,
    ki              : float,
    kp_var_name     : str,
    ki_var_name     : str,
    setpoint        : float,
    initial_value   : float,
    t_step          : float,
    signal_idx_meas : int,
    cost_fn         : Callable,
    settling_band   : float = 0.02,
    ss_window       : float = 0.05,
    verbose         : bool  = False,
) -> Tuple[StepMetrics, float]:
    """
    Run one PLECS simulation with (Kp, Ki), compute metrics and cost.

    Parameters
    ----------
    server           : xmlrpc.client.Server
    model_name       : str
    base_vars        : dict - variables always sent to PLECS (Mode A or B)
    kp / ki          : float - controller gains to test
    kp_var_name      : str - PLECS variable name for proportional gain
    ki_var_name      : str - PLECS variable name for integral gain
    setpoint         : float - target value after the step
    initial_value    : float - value before the step
    t_step           : float - time at which step occurs (seconds)
    signal_idx_meas  : int - Outport row index of the measured output signal
    cost_fn          : callable built by build_cost_function()
    settling_band    : float - see compute_step_response_metrics()
    ss_window        : float - see compute_step_response_metrics()
    verbose          : bool - print per-evaluation details

    Returns
    -------
    (metrics, cost) : (StepMetrics, float)
    """
    vars_i = copy.deepcopy(base_vars)
    vars_i[kp_var_name] = float(kp)
    vars_i[ki_var_name] = float(ki)

    t0 = _time.perf_counter()
    try:
        time, values = _run_one_simulation(server, model_name, vars_i)
    except Exception as exc:
        if verbose:
            print(f"    [PLECS ERROR] Kp={kp:.4g}, Ki={ki:.4g}: {exc}")
        # Return worst-case metrics on simulation failure
        bad = StepMetrics(np.inf, np.inf, np.inf, np.inf, np.inf, np.inf)
        return bad, 1e6

    measured = values[signal_idx_meas]

    metrics = compute_step_response_metrics(
        time          = time,
        measured      = measured,
        setpoint      = setpoint,
        initial_value = initial_value,
        t_step        = t_step,
        settling_band = settling_band,
        ss_window     = ss_window,
    )
    cost = cost_fn(metrics)
    elapsed = _time.perf_counter() - t0

    if verbose:
        print(
            f"    Kp={kp:8.4g}  Ki={ki:8.4g}  |  "
            f"rise={metrics.rise_time*1e3:.1f}ms  "
            f"OS={metrics.overshoot_pct:.1f}%  "
            f"settle={metrics.settling_time*1e3:.1f}ms  "
            f"SSE={metrics.sse:.4g}  "
            f"cost={cost:.4f}  ({elapsed:.2f}s)"
        )

    return metrics, cost


# ══════════════════════════════════════════════════════════════════════════════
#  GRID SEARCH
# ══════════════════════════════════════════════════════════════════════════════

def grid_search_pi(
    server          ,
    model_name      : str,
    base_vars       : dict,
    kp_range        : Tuple[float, float],
    ki_range        : Tuple[float, float],
    n_kp            : int,
    n_ki            : int,
    kp_var_name     : str,
    ki_var_name     : str,
    setpoint        : float,
    initial_value   : float,
    t_step          : float,
    signal_idx_meas : int,
    cost_fn         : Callable,
    log_scale       : bool  = True,
    settling_band   : float = 0.02,
    ss_window       : float = 0.05,
    history         : Optional[OptimHistory] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, OptimHistory]:
    """
    Exhaustive 2-D grid search over Kp x Ki.

    Useful for visualising the cost landscape and choosing good starting
    bounds for the global optimiser.

    Parameters
    ----------
    kp_range : (kp_min, kp_max)
    ki_range : (ki_min, ki_max)
    n_kp     : number of Kp grid points
    n_ki     : number of Ki grid points
    log_scale: use logarithmically spaced grid (recommended for gains)

    Returns
    -------
    KP       : 2-D array (n_ki, n_kp)  - Kp values at each grid point
    KI       : 2-D array (n_ki, n_kp)  - Ki values at each grid point
    COST     : 2-D array (n_ki, n_kp)  - cost at each grid point
    history  : OptimHistory
    """
    if history is None:
        history = OptimHistory()

    _fn = np.geomspace if log_scale else np.linspace
    kp_vals = _fn(kp_range[0], kp_range[1], n_kp)
    ki_vals = _fn(ki_range[0], ki_range[1], n_ki)
    KP, KI  = np.meshgrid(kp_vals, ki_vals)
    COST    = np.full_like(KP, np.nan)

    total   = n_kp * n_ki
    print(f"[Grid] {total} evaluations  "
          f"(Kp: {n_kp} pts  Ki: {n_ki} pts  "
          f"{'log' if log_scale else 'linear'} scale)")

    for i in range(n_ki):
        for j in range(n_kp):
            kp = KP[i, j]
            ki = KI[i, j]
            idx = i * n_kp + j + 1
            print(f"  {idx:>4}/{total}  Kp={kp:.4g}  Ki={ki:.4g}", end="  ")

            _, cost = evaluate_pi(
                server, model_name, base_vars,
                kp, ki, kp_var_name, ki_var_name,
                setpoint, initial_value, t_step,
                signal_idx_meas, cost_fn,
                settling_band, ss_window, verbose=False,
            )
            COST[i, j] = cost
            history.append(kp, ki, cost, tag="grid")
            print(f"cost={cost:.4f}")

    best_kp, best_ki, best_cost = history.best()
    print(f"\n[Grid] Best  ->  Kp={best_kp:.4g}  Ki={best_ki:.4g}  cost={best_cost:.4f}")
    return KP, KI, COST, history


# ══════════════════════════════════════════════════════════════════════════════
#  GLOBAL OPTIMISATION — DIFFERENTIAL EVOLUTION
# ══════════════════════════════════════════════════════════════════════════════

def optimize_pi_differential_evolution(
    server          ,
    model_name      : str,
    base_vars       : dict,
    kp_bounds       : Tuple[float, float],
    ki_bounds       : Tuple[float, float],
    kp_var_name     : str,
    ki_var_name     : str,
    setpoint        : float,
    initial_value   : float,
    t_step          : float,
    signal_idx_meas : int,
    cost_fn         : Callable,
    max_iter        : int   = 30,
    popsize         : int   = 8,
    tol             : float = 1e-3,
    seed            : int   = 42,
    log_scale       : bool  = True,
    settling_band   : float = 0.02,
    ss_window       : float = 0.05,
    history         : Optional[OptimHistory] = None,
) -> Tuple[float, float, OptimHistory]:
    """
    Global optimisation with Differential Evolution (scipy).

    DE is a gradient-free global search — it does not need a good starting
    point and handles multimodal cost landscapes well.

    Parameters
    ----------
    kp_bounds : (kp_min, kp_max)  -  search bounds for Kp
    ki_bounds : (ki_min, ki_max)  -  search bounds for Ki
    max_iter  : maximum number of DE generations
    popsize   : population size multiplier (total = popsize x 2)
    tol       : convergence tolerance on the cost function
    seed      : random seed for reproducibility
    log_scale : optimise in log-space (recommended for gains spanning decades)

    Returns
    -------
    best_kp, best_ki : float - optimised gains
    history          : OptimHistory
    """
    if history is None:
        history = OptimHistory()

    # Work in log space so the search is scale-invariant
    if log_scale:
        bounds_opt = [
            (np.log10(kp_bounds[0]), np.log10(kp_bounds[1])),
            (np.log10(ki_bounds[0]), np.log10(ki_bounds[1])),
        ]
    else:
        bounds_opt = [kp_bounds, ki_bounds]

    eval_count = [0]

    def _objective(x):
        kp = 10**x[0] if log_scale else x[0]
        ki = 10**x[1] if log_scale else x[1]
        eval_count[0] += 1
        print(f"  [DE #{eval_count[0]:>3}]  Kp={kp:.4g}  Ki={ki:.4g}", end="  ")
        _, cost = evaluate_pi(
            server, model_name, base_vars,
            kp, ki, kp_var_name, ki_var_name,
            setpoint, initial_value, t_step,
            signal_idx_meas, cost_fn,
            settling_band, ss_window, verbose=False,
        )
        history.append(kp, ki, cost, tag="de")
        print(f"cost={cost:.4f}")
        return cost

    print(f"[DE] Starting  bounds Kp={kp_bounds}  Ki={ki_bounds}  "
          f"max_iter={max_iter}  popsize={popsize}  seed={seed}")

    result = differential_evolution(
        _objective,
        bounds   = bounds_opt,
        maxiter  = max_iter,
        popsize  = popsize,
        tol      = tol,
        seed     = seed,
        polish   = False,      # local polish done separately with Nelder-Mead
        disp     = False,
    )

    best_kp_log, best_ki_log = result.x
    best_kp = 10**best_kp_log if log_scale else best_kp_log
    best_ki = 10**best_ki_log if log_scale else best_ki_log

    print(f"\n[DE] Done  ->  Kp={best_kp:.4g}  Ki={best_ki:.4g}  "
          f"cost={result.fun:.4f}  "
          f"({eval_count[0]} evaluations)")
    return best_kp, best_ki, history


# ══════════════════════════════════════════════════════════════════════════════
#  LOCAL REFINEMENT — NELDER-MEAD
# ══════════════════════════════════════════════════════════════════════════════

def optimize_pi_nelder_mead(
    server          ,
    model_name      : str,
    base_vars       : dict,
    kp_start        : float,
    ki_start        : float,
    kp_var_name     : str,
    ki_var_name     : str,
    setpoint        : float,
    initial_value   : float,
    t_step          : float,
    signal_idx_meas : int,
    cost_fn         : Callable,
    max_iter        : int   = 100,
    xatol           : float = 1e-4,
    fatol           : float = 1e-4,
    initial_simplex_scale: float = 0.15,
    log_scale       : bool  = True,
    settling_band   : float = 0.02,
    ss_window       : float = 0.05,
    history         : Optional[OptimHistory] = None,
) -> Tuple[float, float, OptimHistory]:
    """
    Local refinement with Nelder-Mead simplex (gradient-free).

    Run this after DE to converge precisely onto the minimum.

    Parameters
    ----------
    kp_start / ki_start          : starting point (e.g. best result from DE)
    initial_simplex_scale        : fraction of starting values used to build
                                   the initial simplex (default 15 %)
    xatol / fatol                : stopping tolerances on gain change and cost

    Returns
    -------
    best_kp, best_ki : float
    history          : OptimHistory
    """
    if history is None:
        history = OptimHistory()

    eval_count = [0]

    def _objective(x):
        kp = 10**x[0] if log_scale else x[0]
        ki = 10**x[1] if log_scale else x[1]
        eval_count[0] += 1
        print(f"  [NM #{eval_count[0]:>3}]  Kp={kp:.4g}  Ki={ki:.4g}", end="  ")
        _, cost = evaluate_pi(
            server, model_name, base_vars,
            kp, ki, kp_var_name, ki_var_name,
            setpoint, initial_value, t_step,
            signal_idx_meas, cost_fn,
            settling_band, ss_window, verbose=False,
        )
        history.append(kp, ki, cost, tag="nm")
        print(f"cost={cost:.4f}")
        return cost

    if log_scale:
        x0  = np.array([np.log10(kp_start), np.log10(ki_start)])
        s   = initial_simplex_scale
        # Build a 3-vertex simplex around x0
        simplex = np.array([
            x0,
            x0 + np.array([s, 0.0]),
            x0 + np.array([0.0, s]),
        ])
    else:
        x0      = np.array([kp_start, ki_start])
        s       = initial_simplex_scale
        simplex = np.array([
            x0,
            x0 * (1 + s),
            x0 * np.array([1 + s, 1 - s]),
        ])

    print(f"[NM] Starting from Kp={kp_start:.4g}  Ki={ki_start:.4g}  "
          f"max_iter={max_iter}")

    result = minimize(
        _objective,
        x0      = x0,
        method  = "Nelder-Mead",
        options = {
            "maxiter"        : max_iter,
            "xatol"          : xatol,
            "fatol"          : fatol,
            "initial_simplex": simplex,
            "disp"           : False,
        },
    )

    best_kp = 10**result.x[0] if log_scale else result.x[0]
    best_ki = 10**result.x[1] if log_scale else result.x[1]

    print(f"\n[NM] Done  ->  Kp={best_kp:.6g}  Ki={best_ki:.6g}  "
          f"cost={result.fun:.4f}  "
          f"({'converged' if result.success else 'max iter reached'}, "
          f"{eval_count[0]} evals)")
    return best_kp, best_ki, history


# ══════════════════════════════════════════════════════════════════════════════
#  PLOTS — COST LANDSCAPE
# ══════════════════════════════════════════════════════════════════════════════

def plot_cost_landscape(
    KP           : np.ndarray,
    KI           : np.ndarray,
    COST         : np.ndarray,
    history      : Optional[OptimHistory] = None,
    title        : str = "PI Cost Landscape",
    log_scale    : bool = True,
    colorscale   : str = "Viridis",
):
    """
    Interactive heatmap of the cost landscape from a grid search.

    Optionally overlays the evaluation trajectory from the history log.

    Parameters
    ----------
    KP / KI / COST : outputs of grid_search_pi()
    history        : OptimHistory - if provided, scatter points are overlaid
    log_scale      : use log10 axes (matches the grid search scale)
    colorscale     : Plotly colorscale name
    """
    kp_axis = KP[0, :]   # 1-D Kp values
    ki_axis = KI[:, 0]   # 1-D Ki values

    # Cap cost for better colour contrast (outliers skew the scale)
    cost_plot = np.clip(COST, 0, np.nanpercentile(COST[np.isfinite(COST)], 95))

    fig = go.Figure()

    fig.add_trace(go.Heatmap(
        x          = np.log10(kp_axis) if log_scale else kp_axis,
        y          = np.log10(ki_axis) if log_scale else ki_axis,
        z          = cost_plot,
        colorscale = colorscale,
        colorbar   = dict(title="Cost"),
        hovertemplate = (
            "Kp=10^%{x:.2f}=%{customdata[0]:.3g}<br>"
            "Ki=10^%{y:.2f}=%{customdata[1]:.3g}<br>"
            "Cost=%{z:.4f}<extra></extra>"
            if log_scale else
            "Kp=%{x:.4g}<br>Ki=%{y:.4g}<br>Cost=%{z:.4f}<extra></extra>"
        ),
        customdata = np.stack([KP, KI], axis=-1) if log_scale else None,
    ))

    # Overlay optimisation history
    if history and len(history) > 0:
        tag_palette = {"grid": "gray", "de": "orange", "nm": "red"}
        for tag, color in tag_palette.items():
            idx = [i for i, t in enumerate(history.tag_list) if t == tag]
            if not idx:
                continue
            kp_pts = [history.kp_list[i] for i in idx]
            ki_pts = [history.ki_list[i] for i in idx]
            c_pts  = [history.cost_list[i] for i in idx]
            fig.add_trace(go.Scatter(
                x    = np.log10(kp_pts) if log_scale else kp_pts,
                y    = np.log10(ki_pts) if log_scale else ki_pts,
                mode = "markers",
                name = tag.upper(),
                marker = dict(
                    color  = c_pts,
                    colorscale = "Reds_r",
                    size   = 8 if tag == "grid" else 12,
                    symbol = "circle" if tag == "grid" else "star",
                    line   = dict(color=color, width=1),
                ),
                text = [f"cost={c:.4f}" for c in c_pts],
                hovertemplate = (
                    f"{tag.upper()}<br>Kp=%{{customdata[0]:.4g}}<br>"
                    "Ki=%{customdata[1]:.4g}<br>Cost=%{text}<extra></extra>"
                ),
                customdata = list(zip(kp_pts, ki_pts)),
            ))

        # Mark the global best
        bkp, bki, bcost = history.best()
        fig.add_trace(go.Scatter(
            x    = [np.log10(bkp) if log_scale else bkp],
            y    = [np.log10(bki) if log_scale else bki],
            mode = "markers",
            name = "Best",
            marker = dict(
                symbol = "star",
                size   = 20,
                color  = "yellow",
                line   = dict(color="black", width=2),
            ),
            text           = [f"Kp={bkp:.4g}, Ki={bki:.4g}, cost={bcost:.4f}"],
            hovertemplate  = "%{text}<extra></extra>",
        ))

    axis_label = lambda base: f"log₁₀({base})" if log_scale else base
    fig.update_layout(
        title       = title,
        xaxis_title = axis_label("Kp"),
        yaxis_title = axis_label("Ki"),
        template    = "plotly_white",
        height      = 550,
    )
    fig.show()


# ══════════════════════════════════════════════════════════════════════════════
#  PLOTS — CONVERGENCE HISTORY
# ══════════════════════════════════════════════════════════════════════════════

def plot_convergence(
    history    : OptimHistory,
    title      : str = "Optimisation Convergence",
    show_best  : bool = True,
):
    """
    Interactive line + scatter plot of cost vs. evaluation number.

    Shows all evaluations colour-coded by optimiser stage (grid / DE / NM)
    and a running-minimum envelope.
    """
    palette = {"grid": "#aaaaaa", "de": "#FF9800", "nm": "#E91E63"}

    fig = go.Figure()

    # Running minimum envelope
    if show_best:
        running_min = np.minimum.accumulate(history.cost_list)
        fig.add_trace(go.Scatter(
            x    = list(range(1, len(running_min) + 1)),
            y    = running_min,
            mode = "lines",
            name = "Best so far",
            line = dict(color="steelblue", width=2, dash="dash"),
        ))

    # Per-tag scatter
    for tag, color in palette.items():
        idx = [i for i, t in enumerate(history.tag_list) if t == tag]
        if not idx:
            continue
        fig.add_trace(go.Scatter(
            x    = [i + 1 for i in idx],
            y    = [history.cost_list[i] for i in idx],
            mode = "markers",
            name = tag.upper(),
            marker = dict(color=color, size=8),
            text = [
                f"Kp={history.kp_list[i]:.4g}  Ki={history.ki_list[i]:.4g}  "
                f"cost={history.cost_list[i]:.4f}"
                for i in idx
            ],
            hovertemplate = "%{text}<extra></extra>",
        ))

    fig.update_layout(
        title       = title,
        xaxis_title = "Evaluation #",
        yaxis_title = "Cost",
        template    = "plotly_white",
        height      = 400,
    )
    fig.show()


# ══════════════════════════════════════════════════════════════════════════════
#  PLOTS — BEST STEP RESPONSE
# ══════════════════════════════════════════════════════════════════════════════

def plot_best_response(
    server           ,
    model_name       : str,
    base_vars        : dict,
    best_kp          : float,
    best_ki          : float,
    kp_var_name      : str,
    ki_var_name      : str,
    setpoint         : float,
    initial_value    : float,
    t_step           : float,
    signal_idx_meas  : int,
    cost_fn          : Callable,
    signal_idx_ref   : Optional[int] = None,
    signal_name_meas : str   = "Measured output",
    signal_name_ref  : str   = "Setpoint",
    time_window      : Optional[Tuple[float, float]] = None,
    settling_band    : float = 0.02,
    ss_window        : float = 0.05,
    title            : str   = "Best PI Response",
) -> StepMetrics:
    """
    Re-run the simulation with the best gains and plot the step response.

    Annotates rise time, settling time, and overshoot on the figure.

    Parameters
    ----------
    signal_idx_ref : int or None
        Outport row index of the reference (setpoint) signal.
        If None, a dashed line at `setpoint` is drawn instead.
    time_window    : (t_start, t_end) or None  -  zoom in seconds

    Returns
    -------
    metrics : StepMetrics  -  metrics of the best response
    """
    vars_i = copy.deepcopy(base_vars)
    vars_i[kp_var_name] = float(best_kp)
    vars_i[ki_var_name] = float(best_ki)

    print(f"[Plot] Running best simulation  Kp={best_kp:.4g}  Ki={best_ki:.4g} ...")
    time, values = _run_one_simulation(server, model_name, vars_i)
    measured     = values[signal_idx_meas]

    metrics = compute_step_response_metrics(
        time, measured, setpoint, initial_value, t_step,
        settling_band=settling_band, ss_window=ss_window,
    )
    cost = cost_fn(metrics)

    # Apply time window
    if time_window is not None:
        mask = (time >= time_window[0]) & (time <= time_window[1])
        t_plt = time[mask]
        y_plt = measured[mask]
    else:
        t_plt = time
        y_plt = measured

    fig = go.Figure()

    # Measured output
    fig.add_trace(go.Scatter(
        x    = t_plt,
        y    = y_plt,
        mode = "lines",
        name = signal_name_meas,
        line = dict(color="#2196F3", width=2),
    ))

    # Reference
    if signal_idx_ref is not None:
        ref_sig = values[signal_idx_ref]
        ref_plt = ref_sig[mask] if time_window else ref_sig
        fig.add_trace(go.Scatter(
            x    = t_plt,
            y    = ref_plt,
            mode = "lines",
            name = signal_name_ref,
            line = dict(color="#FF5722", width=2, dash="dash"),
        ))
    else:
        fig.add_hline(
            y               = setpoint,
            line_dash       = "dash",
            line_color      = "#FF5722",
            annotation_text = signal_name_ref,
        )

    # Settling band shading
    band = abs(settling_band * (setpoint - initial_value))
    fig.add_hrect(
        y0          = setpoint - band,
        y1          = setpoint + band,
        fillcolor   = "green",
        opacity     = 0.08,
        line_width  = 0,
        annotation_text = f"±{settling_band*100:.0f}% band",
        annotation_position = "top right",
    )

    # Annotations
    t_axis_start = t_step
    if np.isfinite(metrics.rise_time):
        fig.add_annotation(
            x    = t_step + metrics.rise_time,
            y    = setpoint * 0.90,
            text = f"Rise: {metrics.rise_time*1e3:.1f} ms",
            showarrow = False,
            bgcolor   = "white",
            bordercolor = "#2196F3",
        )
    if np.isfinite(metrics.settling_time) and metrics.settling_time > 0:
        fig.add_vline(
            x               = t_step + metrics.settling_time,
            line_dash       = "dot",
            line_color      = "green",
            annotation_text = f"Settle: {metrics.settling_time*1e3:.1f} ms",
            annotation_position = "top left",
        )
    if metrics.overshoot_pct > 0.1:
        mask_post = time >= t_step
        if (setpoint - initial_value) > 0:
            pk_idx = int(np.argmax(measured[mask_post]))
        else:
            pk_idx = int(np.argmin(measured[mask_post]))
        pk_time = time[mask_post][pk_idx]
        pk_val  = measured[mask_post][pk_idx]
        fig.add_annotation(
            x    = pk_time,
            y    = pk_val,
            text = f"OS: {metrics.overshoot_pct:.1f}%",
            showarrow = True,
            arrowhead = 2,
            bgcolor   = "lightyellow",
        )

    fig.update_layout(
        title       = (f"{title}  —  Kp={best_kp:.4g}  Ki={best_ki:.4g}  "
                       f"cost={cost:.4f}"),
        xaxis_title = "Time (s)",
        yaxis_title = "Amplitude",
        template    = "plotly_white",
        height      = 450,
    )
    if time_window:
        fig.update_xaxes(range=list(time_window))
    fig.show()

    return metrics


# ══════════════════════════════════════════════════════════════════════════════
#  SUMMARY PRINTER
# ══════════════════════════════════════════════════════════════════════════════

def print_optimization_summary(
    best_kp  : float,
    best_ki  : float,
    metrics  : StepMetrics,
    cost     : Optional[float] = None,
):
    """
    Print a formatted table of the optimisation result.

    Parameters
    ----------
    best_kp / best_ki : float - optimised gains
    metrics           : StepMetrics - from compute_step_response_metrics()
    cost              : float or None - scalar cost (optional)
    """
    sep = "─" * 50
    print(sep)
    print("  PI OPTIMISATION RESULT")
    print(sep)
    print(f"  Kp              = {best_kp:.6g}")
    print(f"  Ki              = {best_ki:.6g}")
    print(sep)
    print(f"  Rise time       = {metrics.rise_time * 1e3:.2f} ms")
    print(f"  Overshoot       = {metrics.overshoot_pct:.2f} %")
    print(f"  Settling time   = {metrics.settling_time * 1e3:.2f} ms")
    print(f"  Steady-state Δ  = {metrics.sse:.6g}  (abs)")
    print(f"  ISE             = {metrics.ise:.4e}")
    print(f"  ITAE            = {metrics.itae:.4e}")
    if cost is not None:
        print(sep)
        print(f"  Cost (weighted) = {cost:.6f}")
    print(sep)
