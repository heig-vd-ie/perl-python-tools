#  Signal Analysis Tool — Oscilloscope & PLECS

A Jupyter notebook for interactive analysis of electrical signals exported from oscilloscopes (LeCroy, Tektronix, Siglent/Owon) and PLECS simulators. Drop in your CSV files, set a handful of parameters in one configuration cell, and run all cells to get time-domain plots, statistical tables, power analysis, and a harmonic FFT spectrum.

---

## Table of Contents

1. [Requirements](#requirements)
2. [Quick Start](#quick-start)
3. [Supported CSV Formats](#supported-csv-formats)
4. [Configuration Reference](#configuration-reference)
5. [Notebook Sections](#notebook-sections)
6. [FFT Analysis — Window Functions](#fft-analysis--window-functions)
7. [Custom & FFT Overlay Plots](#custom--fft-overlay-plots)
8. [Plot Size & Export](#plot-size--export)
9. [Troubleshooting](#troubleshooting)

---

## Requirements

```
numpy >= 1.24
pandas >= 1.5
scipy >= 1.10
plotly >= 5.14
ipython >= 8.0
jupyterlab >= 4.0
```

Install with:

```bash
uv pip install requirements.txt
```

---

## Quick Start

1. Open `signal_analysis.ipynb` in JupyterLab or Jupyter Notebook.
2. Edit **only Cell 1 (Configuration)**:
   ```python
   CSV_FILES = [
       r"/path/to/your/export.csv",
   ]
   ```
3. Run all cells (`Kernel → Restart & Run All`).
4. All plots, tables, and FFT results appear inline and are fully interactive.

> **Only Cell 1 ever needs to be edited** for a normal analysis run.  
> Sections 10 and 11 (overlay plots) have their own self-contained configuration at the top of their respective cells.

---

## Supported CSV Formats

The loader auto-detects the format from the file header. No manual format selection is needed.

| Format | Typical filename | Signature |
|--------|-----------------|-----------|
| **LeCroy HDO / WS** | `C1test00000.csv` | 5-line header, `Time,Ampl` columns |
| **Tektronix DPO** | `T0000ALL.CSV` | Multi-line metadata, `TIME,CH1,…` header |
| **Siglent / Owon** | `WA000001.CSV` | `Source,CH1,CH2` / `Second,Volt,Volt` header |
| **PLECS variable-step** | `data.csv` | Quoted headers like `"Time / s"`, near-duplicate timestamps |
| **Generic numeric CSV** | anything | Any separator (`,` `;` `TAB` `space`), auto-detected |

The generic fallback handles files with any number of header rows, mixed separators, and anonymous column names (renamed to `CH1`, `CH2`, …).

---

## Configuration Reference

All settings are in **Cell 1**. Every parameter has a default that works out-of-the-box; only `CSV_FILES` and usually `FUNDAMENTAL_FREQ` need to be changed.

### Files

| Parameter | Default | Description |
|-----------|---------|-------------|
| `CSV_FILES` | `["/path/to/file.csv"]` | List of CSV file paths to analyse |

### Time Window

| Parameter | Default | Description |
|-----------|---------|-------------|
| `TIME_START` | `None` | Start of analysis window in seconds (`None` = beginning of file) |
| `TIME_END` | `None` | End of analysis window in seconds (`None` = end of file) |

### Channel Selection

| Parameter | Default | Description |
|-----------|---------|-------------|
| `CHANNELS` | `None` | List of channel names to keep, e.g. `["CH1", "CH2"]`. `None` = all channels |

### Time-Domain Plot Layout

| Parameter | Default | Description |
|-----------|---------|-------------|
| `PLOT_GROUPS` | `None` | `None` = one subplot per channel · `"all"` = all channels overlaid · `[["CH1","CH2"],["CH3"]]` = custom groups |
| `PLOT_HEIGHT_PER_CH` | `220` | Pixel height of each subplot in the time-domain figure |
| `PLOT_WIDTH` | `None` | Total figure width in pixels (`None` = fills the notebook cell) |

### FFT Settings

| Parameter | Default | Description |
|-----------|---------|-------------|
| `FUNDAMENTAL_FREQ` | `None` | Fundamental frequency in Hz. `None` = auto-detected from the dominant spectral peak. **Must be set explicitly** if the signal window is shorter than one full period |
| `FFT_MAX_HARMONIC` | `20` | Number of harmonics to compute and display (H1 … HN) |
| `FFT_SHOW_DC` | `True` | Display the DC component (H0) as the leftmost bar |
| `FFT_TOP_N` | `5` | Number of harmonics listed in the summary table |
| `FFT_WINDOW` | `"rectangular"` | Window function — see [FFT Window Functions](#fft-analysis--window-functions) |
| `FFT_MAX_POINTS` | `500_000` | Maximum samples fed to the FFT after resampling. Reduce for speed, increase for very high frequency resolution |

### FFT Plot Size

| Parameter | Default | Description |
|-----------|---------|-------------|
| `FFT_HEIGHT_PER_CH` | `280` | Pixel height of each channel subplot in the FFT figure |
| `FFT_WIDTH` | `None` | Total FFT figure width in pixels |
| `FFT_SHOW_GRID` | `True` | Show grid lines on FFT figures |

### Power Analysis

| Parameter | Default | Description |
|-----------|---------|-------------|
| `POWER_PAIRS` | `[]` | List of `(voltage_col, current_col)` pairs for active/reactive/apparent power and power factor, e.g. `[("CH1", "CH2")]` |

### DC Ripple

| Parameter | Default | Description |
|-----------|---------|-------------|
| `DC_CHANNELS` | `[]` | Channel names to include in DC ripple analysis (peak-to-peak ripple and ripple %) |

### Label Customisation

| Parameter | Default | Description |
|-----------|---------|-------------|
| `PLOT_TITLE` | `""` | Override the time-domain figure title (`""` = auto) |
| `FFT_TITLE` | `""` | Override the FFT figure title (`""` = auto) |
| `YAXIS_LABELS` | `{}` | Per-channel y-axis label overrides, e.g. `{"CH1": "Voltage (V)"}` |
| `LEGEND_NAMES` | `{}` | Per-channel legend name overrides, e.g. `{"Ampl": "Output voltage"}` |

### FFT Overlay (Section 11)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `FFT_OVERLAY` | `[]` | Channels to compare — see [FFT Overlay](#custom--fft-overlay-plots). Leave `[]` to print available names |
| `FFT_OVERLAY_FMIN` | `None` | Lower frequency bound for zoom (Hz) |
| `FFT_OVERLAY_FMAX` | `None` | Upper frequency bound for zoom (Hz) |
| `FFT_OVERLAY_TITLE` | `""` | Figure title (`""` = auto) |
| `FFT_OVERLAY_HEIGHT` | `None` | Figure height in pixels |
| `FFT_OVERLAY_WIDTH` | `None` | Figure width in pixels |

---

## Notebook Sections

| Section | Cell | Description |
|---------|------|-------------|
| **1 — Configuration** | 2 | The only cell that needs editing |
| **2 — Imports** | 4 | Library imports (run once) |
| **3 — CSV Loader** | 6 | Universal auto-detecting loader |
| **4 — Load Data** | 8 | Reads all `CSV_FILES`, applies time window and channel filter |
| **5 — Time-Domain Plot** | 10 | One interactive Plotly figure per file with shared x-axis zoom |
| **6 — Statistical Analysis** | 12 | RMS, mean (DC), AC RMS, min/max, peak-to-peak, std dev, DC ripple |
| **7 — Power Analysis** | 14 | Active P, apparent S, reactive Q, power factor (requires `POWER_PAIRS`) |
| **8 — FFT Analysis** | 16–17 | Harmonic spectrum computation and bar chart |
| **9 — FFT Summary Table** | 19 | Top-N harmonics ranked by amplitude with partial THD |
| **10 — Custom Overlay** | 21 | Free-form time-domain overlay with full style control |
| **11 — FFT Overlay** | 23 | Side-by-side harmonic comparison across channels or files |

---

## FFT Analysis — Window Functions

### Why it matters

The FFT decomposes a signal into frequency bins of width `df = f0 / n_periods`, where `n_periods` is the number of complete fundamental periods in the signal window. A tone at frequency `f` only lands **exactly on a bin** if `f` is a multiple of `df`. Any energy between bins leaks into neighbouring bins — how much depends on the window function.

### `"rectangular"` (default — recommended)

No weighting is applied. When the signal window contains an **exact integer number of fundamental periods**, every harmonic `k·f0` lands precisely on bin `k`, and neighbouring bins receive **zero energy** by the orthogonality of the DFT.

```python
FFT_WINDOW = "rectangular"
```

Use this for all periodic power-electronics signals where the window is aligned to the fundamental.

### `"hann"`

The Hann window is defined as `w[n] = 0.5 − 0.5·cos(2πn/N)`. Its DFT is equivalent to three shifted rectangular spectra at weights `[0.5, −0.25, −0.25]`, which means every tone always creates ±0.25 copies in the adjacent bins `(k±1)·df`. For a 692 V fundamental these copies appear at 346 V — they look like real components but carry no physical information.

```python
FFT_WINDOW = "hann"
```

Use this only for burst signals, transients, or signals that do not fill the window with complete periods and where sidelobe suppression at far frequencies is important.

### Window-too-short guard

If `FUNDAMENTAL_FREQ` is set to a frequency whose period is longer than the signal window, the code raises a clear error instead of computing silently wrong results:

```
⚠  Window too short for f0 = 5 Hz
   Signal duration : 0.1 s
   One period of f0: 0.2 s
   → Set FUNDAMENTAL_FREQ to the actual fundamental (e.g. 50 Hz),
     or provide a longer signal containing ≥ 1 full period.
```

---

## Custom & FFT Overlay Plots

### Section 10 — Custom Time-Domain Overlay

Edit the variables at the top of cell 21 and run that cell independently (no need to re-run the whole notebook):

```python
OVERLAY = ["CH1", "CH2"]                    # channel names (all files searched)
# or
OVERLAY = [("data5", "PCC Vab"), ("data7", "PCC Vab")]  # file-specific

CUSTOM_TITLE  = "PCC voltage comparison"
CUSTOM_YLABEL = "Voltage (V)"
CUSTOM_XLABEL = "Time (s)"
BG_COLOR      = "white"       # or dark: "#1e1e1e", "#0d1117", …
CURVE_COLORS  = []            # [] = palette; or ["#E91E63", "#00BCD4"]
LINE_WIDTH    = 1.6
LINE_DASH     = "solid"       # "solid" | "dot" | "dash" | "longdash" | "dashdot"
CUSTOM_LEGEND = {}            # {"CH1": "Grid voltage"}
CUSTOM_HEIGHT = 500           # px
CUSTOM_WIDTH  = None          # px (None = fill cell)
```

Leave `OVERLAY = []` and run the cell to print all available channel names.

### Section 11 — FFT Harmonic Overlay

Set in Cell 1, then re-run cell 23:

```python
FFT_OVERLAY       = ["PCC Vab", "PCC Vbc", "PCC Vca"]
FFT_OVERLAY_FMIN  = 0      # Hz — zoom from here …
FFT_OVERLAY_FMAX  = 600    # Hz — … to here
FFT_OVERLAY_TITLE = "Phase voltages — harmonic comparison"
FFT_OVERLAY_HEIGHT = 500
FFT_OVERLAY_WIDTH  = None
```

Channels are drawn as **grouped bars** at their true harmonic frequencies so bars for the same harmonic of different channels sit side by side. The frequency zoom is applied directly to the x-axis — no recomputation required.

---

## Plot Size & Export

Every figure is fully interactive (pan, zoom, hover, select). To export a static image for a report:

1. Set the desired dimensions in Cell 1:
   ```python
   FFT_WIDTH         = 950   # px
   FFT_HEIGHT_PER_CH = 300   # px per channel
   ```
2. Run the relevant cells.
3. Click the **camera icon** (📷) in the top-right corner of any Plotly figure to download a PNG at exactly those pixel dimensions.

Size parameters summary:

| Plot | Height | Width |
|------|--------|-------|
| Time-domain | `PLOT_HEIGHT_PER_CH × n_channels` | `PLOT_WIDTH` |
| FFT harmonic | `FFT_HEIGHT_PER_CH × n_channels` | `FFT_WIDTH` |
| Custom overlay (§10) | `CUSTOM_HEIGHT` | `CUSTOM_WIDTH` |
| FFT overlay (§11) | `FFT_OVERLAY_HEIGHT` | `FFT_OVERLAY_WIDTH` |

---

