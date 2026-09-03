# Project Template

Welcome to the project template! This repository follows a structured, modular template designed to keep code clean, reproducible, and easy to collaborate on.

---

## Project Directory Structure

```text
Project_template/
│
├── .github/                   # Project guidelines & best practices
├── .venv/                     # Isolated Python virtual environment (ignored by Git)
├── .vscode/                   # Editor-specific settings, launch configs, and extensions
├── data/                      # Local datasets and data storage
├── docs/                      # Documentation files
├── experiments/               # Main execution scripts & Jupyter Notebooks
├── src/                       # Reusable core Python source code/modules
├── .envrc                     # Environment/shell auto-loader configuration
├── .gitignore                 # Git rules for files to exclude from version control
├── README.md                  # Project overview and usage guidelines
└── requirements.txt           # List of Python dependencies and package versions

```

---

## Detailed Folder & File Breakdown

### **1. Core Directories**

* **`.github/`**
* **Purpose:** Contains repository governance guidelines and engineering best practice documents for contributors.


* **`.venv/`**
* **Purpose:** The local Python virtual environment containing all isolated dependencies and libraries needed for this project.
* *Note:* Automatically created locally; never committed to GitHub.




* **`.vscode/`**
* **Purpose:** Pre-configured visual editor support for VS Code users.


    *  **`extensions.txt`:** Recommends essential extensions (Python, Black Formatter, Direnv, Debugger).


    * **`launch.json`:** Pre-configured test-debugging setups (`pytest`) with custom environment variables.


    * **`settings.json`:** Standardizes auto-formatting on save (`Black`), linting rules, and sets `PYTHONPATH` automatically.




* **`data/`**
* **Purpose:** Dedicated storage for all input raw files, processed data, and model artifacts.


* **`docs/`**
* **Purpose:** Holds supplemental documentation, architectural specs, or generated documentation guides.


* **`experiments/`**
* **Purpose:** The primary space for interactive exploration, prototype scripts, and Jupyter Notebooks (`.ipynb`). Use this space to run tests, visualize results, and execute main workflows.


* **`src/`**
* **Purpose:** The core Python package containing reusable source code (helper functions, custom modules, data parsers, model architectures).


* *Usage:* Import functions from `src` directly into your scripts or notebooks in `experiments/`.





---

### **2. Essential Root Files**

* **`.envrc`**
* **Purpose:** Environment configuration managed via `direnv`. It automatically exports variables into your shell session upon entering the project directory (e.g., configuring `PYTHONPATH=$(pwd)/src:$(pwd):$PYTHONPATH`).


* **How to use:** Ensures all python modules inside `src/` can be smoothly imported across notebooks and scripts without path errors.




* **`.gitignore`**
* **Purpose:** Directs Git on which untracked files or directories to omit from commits.


* **What it excludes:** Python cache (`__pycache__`), virtual environment (`.venv`), temporary logs, environment configurations, and build artifacts to prevent repo bloat and leak of credentials.




* **`requirements.txt`**
* **Purpose:** Lists all external Python packages and specific versions required to execute the code.
* **How to use:** Used to recreate the exact Python environment across different machines.



---

## Getting Started & How to Use

### Prerequisites & Environment Setup


 **Set up Virtual Environment & Install Dependencies**:
Create your local `.venv`, activate it, and install all required packages:


```bash
uv venv .venv
source .venv/Scripts/activate    
uv pip install -r requirements.txt

```



---

## 🛠️ Typical Development Workflow

1. **Develop Reusable Logic (`src/`)**
Write modular, clean Python functions and classes in `src/`.


2. **Run Experiments (`experiments/`)**
Import modules from `src/` into Jupyter Notebooks or Python scripts located in `experiments/` to analyze data, run models, or plot results.


```python
# Example inside an experiment notebook/script
from src.data_loader import load_data

```


3. **Manage Dependencies (`requirements.txt`)**
If you install new packages while developing, update the file so others can run your code:
```bash
uv pip freeze > requirements.txt

```