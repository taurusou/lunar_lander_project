# Local setup

These instructions use PowerShell on Windows. Run every command from the root
of the `lunar-lander-rl-project` repository.

## 1. Create the virtual environment

```powershell
python -m venv .venv
```

A virtual environment keeps this project's packages separate from packages
used by other Python projects. The `.venv` folder is ignored by Git.

## 2. Activate the virtual environment

```powershell
.\.venv\Scripts\Activate.ps1
```

After activation, PowerShell normally displays `(.venv)` before the prompt.

If PowerShell blocks the activation script, the project can still be used by
calling its Python executable directly, as shown below.

## 3. Install the dependencies

UTF-8 mode prevents a Windows console error caused by the non-English
characters in the full course-folder path.

```powershell
.\.venv\Scripts\python.exe -X utf8 -m pip install --upgrade pip
.\.venv\Scripts\python.exe -X utf8 -m pip install -r requirements.txt
```

## 4. Check the environment

```powershell
.\.venv\Scripts\python.exe -X utf8 src\lunar_lander_rl\environment_check.py
```

A successful run prints the package versions, LunarLander's spaces, one random
episode's reward, and `Environment check passed.`

## JupyterLab note

JupyterLab is not required for the initial project. In this workspace, its
installation exceeds the default Windows path-length limit because the project
is inside a deeply nested course folder. The installed `ipykernel` package is
enough to select `.venv` as a notebook kernel in an editor such as VS Code.

The agent and experiments can also be completed with ordinary Python scripts,
which avoids this path problem entirely.

