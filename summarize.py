
import glob, os, re, pandas as pd, pathlib
from tabulate import tabulate

reports = pathlib.Path("reports")
reports.mkdir(exist_ok=True)

def latest(pattern):
    files = sorted(glob.glob(pattern))
    return files[-1] if files else None

def load_summary(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def main():
    # Try to pair latest sdk and proxy summaries per scenario
    rows = []
    for scenario in ["chat","stream","tools"]:
        sdk = latest(str(reports / f"summary_sdk_{scenario}_*.md"))
        proxy = latest(str(reports / f"summary_proxy_{scenario}_*.md"))
        rows.append([scenario, sdk or "-", proxy or "-"])
    print(tabulate(rows, headers=["Scenario","SDK summary","Proxy summary"]))
    print("\nTip: Open both markdown files to compare numbers side-by-side.")

if __name__ == "__main__":
    main()
