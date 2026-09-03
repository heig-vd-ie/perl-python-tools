# PI Controller Tuner

An interactive PI tuning tool for Jupyter.
Enter your plant transfer function, move the Kp and Ki sliders, and watch four live Plotly charts update in real time — open-loop Bode with stability margins, closed-loop Bode, step response, and pole-zero map. Optionally add a sensor transfer function to simulate a realistic feedback path.

---

## Table of Contents

1. [Requirements](#requirements)
2. [Quick Start](#quick-start)
3. [Plant Transfer Function Syntax](#plant-transfer-function-syntax)
4. [Configuration Reference](#configuration-reference)
5. [Notebook Sections](#notebook-sections)
6. [Interactive Charts](#interactive-charts)
7. [Stability Margins & Info Panel](#stability-margins--info-panel)
8. [Feedback Transfer Function](#feedback-transfer-function)
9. [Analysis Report](#analysis-report)
10. [Custom Bode Overlay](#custom-bode-overlay)
11. [Troubleshooting](#troubleshooting)

---

## Requirements

```
numpy >= 1.24
control >= 0.9
plotly >= 5.14
ipywidgets >= 8.0
ipython >= 8.0
jupyterlab >= 4.0
```

Install with:

```bash
uv pip install requirements.txt
```

> **JupyterLab is required** for `ipywidgets` sliders and `plotly` `FigureWidget` to render correctly.
> Classic Jupyter Notebook also works if the `widgetsnbextension` extension is enabled.

---

## Quick Start

1. Open `PI_Tuner.ipynb` in JupyterLab.
2. Edit **Cell 1 (Configuration)** — set the initial gains and frequency range for your system.
3. Edit **Cell 3 (Plant & Feedback TF)** — define your plant using the `s` variable.
4. Run all cells (`Kernel → Restart & Run All`).
5. Use the **Kp** and **Ki** sliders in Cell 4 to tune interactively.

> **Only Cell 1 and Cell 3 ever need to be edited** for a normal tuning session.
> Cells 5 and 6 have their own self-contained configuration at the top.

---

## Plant Transfer Function Syntax

The library exports the Laplace variable `s = ct.tf('s')`. Use it to write transfer functions in natural algebraic notation — no need to manually specify numerator and denominator arrays.

### Examples

```python
# RL filter — G(s) = 1 / (L·s + R)
L = 1e-3
R = 0.5
plant_tf = 1 / (L*s + R)

# Second-order system
wn   = 2 * np.pi * 50     # natural frequency [rad/s]
zeta = 0.3                 # damping ratio
plant_tf = wn**2 / (s**2 + 2*zeta*wn*s + wn**2)

# Integrating plant  G(s) = K/s
plant_tf = 5.0 / s

# DC motor (back-EMF model)  G(s) = K / (τ·s + 1)
K   = 0.8
tau = 50e-3
plant_tf = K / (tau*s + 1)

# From numerator / denominator coefficient lists
plant_tf = ct.tf([1], [1e-3, 0.01, 0])   # 1 / (0.001s² + 0.01s)

# With a first-order Padé dead-time approximation  e^(-Td·s)
T_d   = 500e-6
delay = ct.tf([-T_d/2, 1], [T_d/2, 1])
plant_tf = delay / (L*s + R)
```

The PI controller is built internally as:

$$C(s) = K_p + \frac{K_i}{s} = \frac{K_p s + K_i}{s}$$

and combined with the plant to form:

$$G_{OL}(s) = C(s) \cdot G(s) \qquad G_{CL}(s) = \frac{G_{OL}(s)}{1 + H(s)\,G_{OL}(s)}$$

---

## Configuration Reference

All settings are in **Cell 1**. Only the gains and frequency bounds typically need adjusting for each new system.

### Initial Gains

| Parameter | Default | Description |
|-----------|---------|-------------|
| `Kp_INIT` | `1.0` | Initial proportional gain displayed when the tuner opens |
| `Ki_INIT` | `10.0` | Initial integral gain displayed when the tuner opens |

### Slider Ranges

| Parameter | Default | Description |
|-----------|---------|-------------|
| `Kp_RANGE` | `(1e-4, 1e4)` | `(min, max)` bounds of the Kp slider. The slider is logarithmically scaled to cover wide gain ranges |
| `Ki_RANGE` | `(1e-4, 1e5)` | `(min, max)` bounds of the Ki slider |

> Set the range to span at least two decades above and below your expected operating gains. The slider readout always shows the true value, not the log exponent.

### Bode Plot Settings

| Parameter | Default | Description |
|-----------|---------|-------------|
| `F_MIN` | `0.1` | Lower frequency bound for Bode plots [Hz] |
| `F_MAX` | `10_000.0` | Upper frequency bound for Bode plots [Hz]. Should comfortably exceed the gain crossover frequency |
| `N_FREQ` | `2000` | Number of frequency points. More points → smoother curves, slightly slower updates |

### Step Response Settings

| Parameter | Default | Description |
|-----------|---------|-------------|
| `T_END` | `1.0` | Simulation end time [s]. Set to at least 5 × the expected settling time |
| `N_TIME` | `4000` | Number of time samples. Reduce to `2000` for faster slider response on slow machines |

### Figure Size

| Parameter | Default | Description |
|-----------|---------|-------------|
| `FIG_HEIGHT` | `440` | Height [px] of each Bode panel (open-loop and closed-loop). The step response and pole-zero panels are 60 px shorter |

---

## Notebook Sections

| Section | Cell | Description |
|---------|------|-------------|
| **1 — Configuration** | 1 | The only cell that needs editing for a normal session |
| **2 — Imports** | 2 | Library imports — run once |
| **3 — Plant & Feedback TF** | 3 | Define `plant_tf` and `feedback_tf` using the `s` variable |
| **4 — Interactive PI Tuner** | 4 | Launches the live tuner with all four charts and the two sliders |
| **5 — Analysis Report** | 5 | Prints a precise numerical report at any specified gain pair |
| **6 — Custom Open-Loop Bode** | 6 | Overlays multiple gain combinations on one Bode figure for comparison |

---

## Interactive Charts

Running Cell 4 displays four live `plotly` `FigureWidget` panels arranged in a 2 × 2 grid. Every slider movement instantly recomputes all four charts without re-running any cell.

### Open-Loop Bode Diagram

Magnitude (dB) and phase (°) of $G_{OL}(s) = C(s) \cdot G(s)$ plotted on a logarithmic frequency axis.

Two marker lines update with the sliders:

| Marker | Colour | Meaning |
|--------|--------|---------|
| **GM line** | Pink / dashed | Vertical line at the phase crossover frequency $\omega_{pc}$ (where $\angle G_{OL} = -180°$). Label shows the gain margin in dB |
| **PM line** | Cyan / dashed | Vertical line at the gain crossover frequency $\omega_{gc}$ (where $|G_{OL}| = 0\,\text{dB}$). Label shows the phase margin in degrees |

Reference lines at 0 dB and −180° are always visible.

### Closed-Loop Bode Diagram

Magnitude and phase of $G_{CL}(s)$ with a dashed −3 dB reference line marking the bandwidth. If a `feedback_tf` is set, it is included in the loop: $G_{CL} = G_{OL} / (1 + H \cdot G_{OL})$.

### Step Response

Unit step response of the closed-loop system. Three performance annotations update automatically:

| Annotation | Colour | Definition |
|-----------|--------|------------|
| **Tr** | Green | Rise time — time for the output to travel from 10 % to 90 % of the steady-state value |
| **Ts** | Blue | Settling time — last instant the output leaves the ±2 % band around the steady-state value |
| **OS** | Red | Overshoot — peak exceedance above the steady-state value, in percent |

A shaded green band marks the ±2 % settling region.

### Pole-Zero Map

Real/imaginary plane showing four groups of markers:

| Marker | Symbol | Colour | Meaning |
|--------|--------|--------|---------|
| OL Zeros | ○ (open circle) | Blue | Open-loop transfer function zeros |
| OL Poles | × | Orange-red | Open-loop transfer function poles |
| CL Zeros | ○ (open circle) | Green | Closed-loop zeros |
| CL Poles | × | Purple | Closed-loop poles (determine stability) |

The imaginary axis (stability boundary) is shown as a dashed vertical line. The axes are locked to equal scaling so the pole-zero geometry is not distorted.

> All four panels are fully interactive: scroll to zoom, drag to pan, hover for exact values, and click legend entries to show or hide individual traces.

---

## Stability Margins & Info Panel

A status bar between the sliders and the charts shows live metrics for the current gain values:

```
✅ STABLE   C(s) = 5.623 + 47.86/s

Open-Loop Margins             Closed-Loop Step Response
Phase Margin : 52.3°          Rise time  : 14.2 ms
Gain Margin  : 18.7 dB        Settling   : 38.1 ms  (2%)
ωgc = 87.4 Hz | ωpc = 341 Hz  Overshoot  : 8.4%  |  SS = 1.0000
```

The status badge changes colour:

| Badge | Condition |
|-------|-----------|
| **✅ STABLE** | Phase margin > 0° and gain margin > 0 dB |
| **❌ UNSTABLE** | Either margin is negative or could not be computed |

> The stability check applies to minimum-phase systems. For non-minimum-phase plants (right-half-plane zeros or open-loop unstable poles), verify stability from the closed-loop pole positions in the pole-zero map.

---

## Feedback Transfer Function

By default the tuner uses **unity feedback** ($H(s) = 1$). Set `feedback_tf` in Cell 3 to simulate a sensor or measurement filter in the feedback path.

```python
# Low-pass sensor filter  H(s) = 1 / (τ·s + 1)
tau_sensor  = 100e-6          # 100 µs time constant
feedback_tf = 1 / (tau_sensor*s + 1)

# Band-limited current sensor
feedback_tf = ct.tf([1], [1e-5, 1])

# Unity feedback (default)
feedback_tf = None
```

The feedback TF appears in the denominator of the closed-loop expression and shifts both the closed-loop Bode and the step response. Open-loop margins are computed from $G_{OL} = C \cdot G$ independently of $H$, consistent with classical loop-gain analysis.

---

## Analysis Report

After tuning interactively, run Cell 5 to get a complete numerical report at any gain pair. The report is printed directly in the notebook output.

```python
Kp_REPORT = 5.623    # paste your tuned values here
Ki_REPORT = 47.86
```

Example output:

```
──────────────────────────────────────────────────────
  PI CONTROLLER — ANALYSIS REPORT
──────────────────────────────────────────────────────
  Kp = 5.623   |   Ki = 47.86
  C(s) = 5.623 + 47.86 / s

  ── Open-Loop Stability ─────────────────────────────
  ✓ STABLE
  Phase margin        : 52.31°
  Gain margin         : 18.74 dB
  Gain crossover freq : 87.4 Hz   (|G_OL| = 1)
  Phase crossover freq: 341.2 Hz  (∠G_OL = −180°)

  ── Closed-Loop Step Response ───────────────────────
  Rise time  (10–90 %) : 14.20 ms
  Settling time  (2 %) : 38.06 ms
  Overshoot            : 8.41 %
  Steady-state gain    : 1.0000

  ── Closed-Loop Poles ───────────────────────────────
    -218.7422   +491.3318j   ζ = 0.407   ωn = 537.4 rad/s
    -218.7422   -491.3318j   ζ = 0.407   ωn = 537.4 rad/s
──────────────────────────────────────────────────────
```

---

## Custom Bode Overlay

Cell 6 plots the open-loop Bode diagram for multiple gain combinations on the same figure — useful for visually comparing the effect of gain changes on loop shape and margins.

Edit the `COMPARE_GAINS` list and re-run that cell independently:

```python
COMPARE_GAINS = [
    (Kp_INIT, Ki_INIT, '#2196F3', 'Initial'),
    (5.0,     50.0,    '#FF5722', 'Tuned'),
    (10.0,    100.0,   '#4CAF50', 'Aggressive'),
]
```

Each row is `(Kp, Ki, hex_color, legend_label)`. The figure is a static Plotly chart (not a widget) — run the cell again after any edit to refresh it.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| Sliders appear but plots do not update | `ipywidgets` extension not enabled | Run `jupyter labextension install @jupyter-widgets/jupyterlab-manager` or use JupyterLab ≥ 3 which includes it |
| `ModuleNotFoundError: control` | `python-control` not installed | `pip install control` |
| `ModuleNotFoundError: pi_tuner_lib` | Library not in the same folder as the notebook | Place `pi_tuner_lib.py` in the same directory as `PI_Tuner.ipynb` |
| Phase margin shows `—` | No gain crossover found in the plotted frequency range | Increase `F_MAX` until the open-loop magnitude crosses 0 dB |
| Gain margin shows `∞` | Phase never reaches −180° (common for first-order plants with PI) | This is correct — the system has infinite gain margin |
| Step response does not settle within the plot | `T_END` too short | Increase `T_END` in Cell 1 to cover more than 5 settling times |
| Sliders are very slow to respond | `N_TIME` or `N_FREQ` too high | Reduce to `N_TIME = 2000` and `N_FREQ = 1000` for faster updates |
| Pole-zero map axes are very large | Unstable poles far from the origin | Check stability — very large poles indicate the closed-loop is near instability |
| `❌ UNSTABLE` badge for a visually-settling step response | Non-minimum-phase plant | Margins are not reliable for NMP systems; judge stability from the CL pole locations instead |
