#!/usr/bin/env python3
"""Summarize LoopUS eval JSONs into a benchmark table."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

PPL_COLS = [
    ("Wiki", ["wikitext"], ["word_perplexity", "perplexity"]),
    ("LAMBADA", ["lambada_openai", "lambada", "lambada_standard"], ["perplexity"]),
]

ACC_COLS = [
    ("MMLU", ["mmlu"]),
    ("HS", ["hellaswag"]),
    ("ARC-E", ["arc_easy"]),
    ("ARC-C", ["arc_challenge"]),
    ("PIQA", ["piqa"]),
    ("WG", ["winogrande"]),
    ("OBQA", ["openbookqa"]),
]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("results", nargs="*", default=None,
                   help="Eval JSON files or directories (default: this script's directory)")
    p.add_argument("--metric", default="acc", choices=["acc", "acc_norm"],
                   help="Accuracy field to report; falls back to acc when acc_norm is absent")
    p.add_argument("--baseline", default=None,
                   help="Substring matching the row to compute the delta against (default: first row)")
    p.add_argument("--latex", action="store_true", help="Emit a booktabs LaTeX table")
    p.add_argument("--csv", default=None, help="Also write the table to this CSV path")
    p.add_argument(
        "--custom",
        action="store_true",
        help="Use the hardcoded order/labels in collect_files_custom() instead of scanning a directory",
    )
    return p.parse_args()

def collect_files_custom() -> tuple[list[Path], dict[Path, str]]:
    """Hardcoded (file, setting) order — edit this dict to add/reorder rows."""
    files2setting = {
        "results/LoopUS.json": "LoopUS",
        "results/LoopUS_NI0.0_RI0.0_44000.json": "NI0.0_RI0.0 (44k steps)",
        "results/LoopUS_NI0.0_RI0.0_final.json": "NI0.0_RI0.0 (400M)",
        "results/LoopUS_NI0.0_RI0.0_62000.json": "NI0.0_RI0.0 (62k steps)",

        "results/LoopUS_NI0.001_RI0.0.json": "NI0.001_RI0.0 (100M)",
        "results/LoopUS_NI0.001_RI0.0_30000.json": "NI0.001_RI0.0 (30k steps)",
        "results/LoopUS_NI0.001_RI0.0_cont_final.json": "NI0.001_RI0.0 (400M)",
        "results/LoopUS_NI0.001_RI0.0_87000.json": "NI0.001_RI0.0 (87k steps)",
        "results/LoopUS_NI0.001_RI0.0_1B.json": "NI0.001_RI0.0 (1B)",

        "results/LoopUS_NI0.0_RI0.05.json": "NI0.0_RI0.05 (100M)",
        "results/LoopUS_NI0.0_RI0.05_20000.json": "NI0.0_RI0.05 (20k steps)",
        "results/LoopUS_NI0.0_RI0.05_cont_final.json": "NI0.0_RI0.05 (400M)",
        "results/LoopUS_NI0.0_RI0.05_90000.json": "NI0.0_RI0.05 (90k steps)",

        "results/LoopUS_NI0.001_RI0.05_20000.json": "NI0.001_RI0.05 (20k steps)",
        "results/LoopUS_NI0.001_RI0.05_46000.json": "NI0.001_RI0.05 (46k steps)",
        "results/LoopUS_NI0.001_RI0.05_65000.json": "NI0.001_RI0.05 (65k steps)",
    }

    repo_root = Path(__file__).resolve().parent.parent
    files: list[Path] = []
    overrides: dict[Path, str] = {}
    for raw, setting in files2setting.items():
        path = (repo_root / raw).resolve()
        if not path.is_file():
            raise SystemExit(f"[--custom] file not found: {path}")
        files.append(path)
        overrides[path] = setting
    return files, overrides

def collect_files(paths: list[str] | None) -> list[Path]:
    paths = paths or [str(Path(__file__).resolve().parent)]
    files: list[Path] = []
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            files.extend(sorted(path.glob("*.json")))
        elif path.is_file():
            files.append(path)
    return files


def label_for(data: dict, path: Path) -> tuple[str, str]:
    """Derive (model, setting) labels from eval metadata."""
    model = data.get("model_name") or "?"
    model = model.split("/")[-1].replace("Qwen3-", "Qwen ")

    source = data.get("decomposed_model") or data.get("checkpoint_dir")
    if not source:
        return model, "w/o LoopUS"

    name = os.path.basename(str(source).rstrip("/"))
    if name in {"final", "latest"} or name.startswith("step_"):
        name = os.path.basename(os.path.dirname(str(source).rstrip("/"))) or name
    return model, f"{name} (N={data.get('n_recursion', '?')})"


def lookup(data: dict, names: list[str], fields: list[str]) -> float | None:
    for name in names:
        entry = data.get(name)
        if isinstance(entry, dict):
            for field in fields:
                value = entry.get(field)
                if isinstance(value, (int, float)):
                    return float(value)
    return None


def build_row(path: Path, metric: str, setting_override: str | None = None) -> dict:
    data = json.load(open(path))
    model, setting = label_for(data, path)
    if setting_override:
        setting = setting_override
    row: dict = {"model": model, "setting": setting, "file": path.name}

    for col, names, fields in PPL_COLS:
        row[col] = lookup(data, names, fields)

    # acc_norm is missing for some tasks (e.g. MMLU, WinoGrande) — fall back to acc.
    fields = [metric, "acc"] if metric != "acc" else ["acc"]
    for col, names in ACC_COLS:
        value = lookup(data, names, fields)
        row[col] = None if value is None else value * 100
    return row


def add_averages(rows: list[dict]) -> list[str]:
    """Average over tasks every row has, so AVG and Δ stay comparable."""
    common = [c for c, _ in ACC_COLS if all(r.get(c) is not None for r in rows)]
    for row in rows:
        row["AVG"] = sum(row[c] for c in common) / len(common) if common else None
    return common


def fmt(value: float | None, digits: int = 1) -> str:
    return "--" if value is None else f"{value:.{digits}f}"


def add_deltas(rows: list[dict], baseline: str | None) -> None:
    base_row = rows[0]
    if baseline:
        matches = [r for r in rows if baseline in r["setting"] or baseline in r["file"]]
        if matches:
            base_row = matches[0]
    base = base_row.get("AVG")
    for row in rows:
        if row is base_row or base is None or row["AVG"] is None:
            row["delta"] = None
        else:
            row["delta"] = row["AVG"] - base


def print_table(rows: list[dict], metric: str) -> None:
    ppl = [c[0] for c in PPL_COLS]
    acc = [c[0] for c in ACC_COLS]
    widths = {
        "model": max(5, *(len(r["model"]) for r in rows)),
        "setting": max(7, *(len(r["setting"]) for r in rows)),
    }
    for col in ppl + acc:
        widths[col] = max(len(col), 5)

    head1 = f"{'Model':<{widths['model']}}  {'Setting':<{widths['setting']}}  "
    head1 += " ".join(f"{c:>{widths[c]}}" for c in ppl)
    head1 += " | " + " ".join(f"{c:>{widths[c]}}" for c in acc)
    head1 += f" | {'AVG':>6} {'Δ':>6}"

    span_ppl = sum(widths[c] for c in ppl) + len(ppl) - 1
    span_acc = sum(widths[c] for c in acc) + len(acc) - 1
    head0 = " " * (widths["model"] + widths["setting"] + 4)
    head0 += f"{'ppl ↓':^{span_ppl}} | {'acc ↑ (' + metric + ')':^{span_acc}} |"

    print(head0)
    print(head1)
    print("-" * len(head1))
    for row in rows:
        line = f"{row['model']:<{widths['model']}}  {row['setting']:<{widths['setting']}}  "
        line += " ".join(f"{fmt(row[c], 2):>{widths[c]}}" for c in ppl)
        line += " | " + " ".join(f"{fmt(row[c]):>{widths[c]}}" for c in acc)
        delta = "--" if row["delta"] is None else f"{row['delta']:+.1f}"
        line += f" | {fmt(row['AVG']):>6} {delta:>6}"
        print(line)


def tex(text: str) -> str:
    return text.replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")


def print_latex(rows: list[dict], metric: str) -> None:
    ppl = [c[0] for c in PPL_COLS]
    acc = [c[0] for c in ACC_COLS]
    first, last = 3, 2 + len(ppl) + len(acc)
    print(r"\begin{tabular}{ll" + "c" * (len(ppl) + len(acc) + 2) + "}")
    print(r"\toprule")
    print(f"\\multirow{{2}}{{*}}{{Model}} & \\multirow{{2}}{{*}}{{Setting}} & "
          + " & ".join(ppl + acc)
          + f" & \\multirow{{2}}{{*}}{{AVG}} & \\multirow{{2}}{{*}}{{$\\Delta$}} \\\\")
    print(f"\\cmidrule(lr){{{first}-{first+len(ppl)-1}}} "
          f"\\cmidrule(lr){{{first+len(ppl)}-{last}}}")
    print(f" & & \\multicolumn{{{len(ppl)}}}{{c}}{{\\texttt{{ppl}} $\\downarrow$}} & "
          f"\\multicolumn{{{len(acc)}}}{{c}}{{\\texttt{{{metric}}} $\\uparrow$}} & & \\\\")
    print(r"\midrule")
    for row in rows:
        cells = [tex(row["model"]), tex(row["setting"])]
        cells += [fmt(row[c], 2) for c in ppl]
        cells += [fmt(row[c]) for c in acc]
        cells.append(fmt(row["AVG"]))
        cells.append("--" if row["delta"] is None else f"{row['delta']:+.1f}")
        print(" & ".join(cells) + r" \\")
    print(r"\bottomrule")
    print(r"\end{tabular}")


def write_csv(rows: list[dict], path: str) -> None:
    import csv

    cols = ["model", "setting"] + [c[0] for c in PPL_COLS] + [c[0] for c in ACC_COLS] + ["AVG", "delta"]
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=cols, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({c: row.get(c) for c in cols})
    print(f"\n[csv] wrote {path}")


def main() -> None:
    args = parse_args()
    overrides: dict[Path, str] = {}
    if args.custom:
        files, overrides = collect_files_custom()
    else:
        files = collect_files(args.results)
    if not files:
        raise SystemExit("No eval JSON files found.")

    rows = [build_row(path, args.metric, overrides.get(path.resolve())) for path in files]
    common = add_averages(rows)
    add_deltas(rows, args.baseline)

    if args.latex:
        print_latex(rows, args.metric)
    else:
        print_table(rows, args.metric)

    missing = [c for c, _ in ACC_COLS if c not in common]
    if missing:
        print(f"\n[note] AVG/Δ over {len(common)} shared tasks ({', '.join(common)}); "
              f"excluded (not in every row): {', '.join(missing)}")
    if args.csv:
        write_csv(rows, args.csv)


if __name__ == "__main__":
    main()
