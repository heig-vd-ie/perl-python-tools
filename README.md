# perl-python_tools

This repository contains several tools related to **Python** analysis and power electronics. There are tools for lifetime prediction, control tuning, and signal processing.

> **Important:** This repository contains **multiple independent projects in the same GitHub repository**.\
> When working with one of the projects in VS Code, you should open **only that project folder** as your VS Code workspace.

> **Environment:** to use the following projects, you can set up either a Linux (WSL) or Windows environment. Instructions for setting up WSL environment are available under [Environment installation](https://github.com/heig-vd-ie/perl-python_tools/blob/main/Environment_Installation.md), and instructions for setting up the Windows environment are available in the [perl-plecs_tools repository](https://github.com/heig-vd-ie/perl-plecs_tools/blob/main/Environment_Installation.md).

## Repository Structure

The repository is organized into several projects:

``` text
perl-python_tools/
├── Images/
├── Lifetime_Prediction/
├── PI_Tuner/
├── Signal_Analysis/
├── Environment_Installation.md
└── README.md

```

* ### Lifetime Prediction



The **Lifetime Prediction** project provides a workflow to predict the lifetime of a transistor based on its junction temperature profile.

* ### PI Tuner



The **PI Tuner** allows you to enter the transfer function of a system, modify the PI controller parameters, and view the system's Bode plot and step response dynamically on a panel.

* ### Signal Analysis



The **Signal Analysis** tool allows you to import a CSV signal file to plot the data and perform analysis including FFT, power calculations, and DC ripple measurement.

## Working with the Projects in VS Code

Because the GitHub repository contains **multiple projects**, it is important to understand how to open them in VS Code.

### Recommended: Open One Project at a Time

If you want to work on the `Lifetime_Prediction` project, open the `Lifetime_Prediction` folder directly in VS Code:

In VS Code, use:

**File → Open Folder... → `Lifetime_Prediction**`

The same principle applies to the other projects:

```text
File → Open Folder... → PI_Tuner
```
...

### Do Not Open the Entire Repository for a Single Project

Avoid opening the root of the GitHub repository when you are working on only one project:

```text
perl-python_tools/
├── Lifetime_Prediction/
├── PI_Tuner/
├── Signal_Analysis/
└── ...

```

Opening the repository root makes VS Code treat all the projects as part of the same workspace. This can cause confusion with:

* Python virtual environments
* Python interpreters
* VS Code settings
* Extensions
* Project-specific paths
* `requirements.txt`
* Source code navigation
* Environment variables
* Debugging and launch configurations

Each project should instead be treated as its **own VS Code project**, even though all projects are stored in the same GitHub repository.

## Why Are Multiple Projects in One Repository?

The projects are related and share a common purpose and development environment, so they are kept together in one repository.

Think of the repository as a container for several projects rather than as one single VS Code project.

## Installation

Each project may have its own dependencies and setup instructions.

For environment setup, see:

`Environment_Installation.md`

For a specific project, check its own documentation and `requirements.txt` file.

## Getting Started

1. Clone the repository.
2. Decide which project you want to work on.
3. Open **only that project's folder** in VS Code.
4. Create or activate the project's Python environment if required.
5. Install the project's dependencies.
6. Follow the project-specific documentation.
7. Run the tools as required.

For example, to work on the Lifetime Prediction tool:

```text
Clone repository
      ↓
Open perl-python_tools/Lifetime_Prediction
      ↓
Select the project's Python environment
      ↓
Install requirements.txt
      ↓
Run the Lifetime Prediction tool

```

## Notes

* Make sure the correct Python environment is selected in VS Code for the project you are working on.
* Do not assume that the Python environment or VS Code configuration of one project applies to another project.
* Keep project-specific dependencies inside the corresponding project.
