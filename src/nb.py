"""Build a .ipynb from a 'percent' script (cells separated by '# %%') and run it.

  python -m src.nb notebooks/L1_exploration.py          # build + execute
Markdown cells start with '# %% [markdown]' and every line is prefixed by '# '.
"""
import sys
from pathlib import Path

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor


def script_to_notebook(src: Path) -> nbformat.NotebookNode:
    cells, kind, buf = [], None, []

    def flush():
        if kind is None:
            return
        text = "\n".join(buf).strip("\n")
        if kind == "md":
            text = "\n".join(l[2:] if l.startswith("# ") else l.lstrip("#") for l in text.splitlines())
            cells.append(nbformat.v4.new_markdown_cell(text))
        elif text:
            cells.append(nbformat.v4.new_code_cell(text))

    for line in src.read_text(encoding="utf-8").splitlines():
        if line.startswith("# %%"):
            flush()
            kind, buf = ("md" if "[markdown]" in line else "code"), []
        else:
            buf.append(line)
    flush()
    nb = nbformat.v4.new_notebook(cells=cells)
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    return nb


def build(src: Path, execute: bool = True, timeout: int = 1800) -> Path:
    nb = script_to_notebook(src)
    if execute:
        ExecutePreprocessor(timeout=timeout, kernel_name="python3").preprocess(
            nb, {"metadata": {"path": str(src.parent.parent)}})   # run from project root
    out = src.with_suffix(".ipynb")
    nbformat.write(nb, out)
    return out


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        print("built", build(Path(arg)))
