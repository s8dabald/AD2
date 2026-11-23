"""Assemble existing 20x25 curves into one four-seed/five-method report."""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "legacy" / "experiments" / "results"
SEEDS = (42, 7, 123, 2024)
SCHEDULE = "20x25"
REVIEWS = tuple(range(25, 501, 25))
METHODS = (
    "random_query",
    "uncertainty_query",
    "shap_box_tags_no_split",
    "split_direct_only",
    "split_plus_shap_box_tags",
)
METHOD_DISPLAY = {
    "random_query": "1. Random query, full-pool direct-only",
    "uncertainty_query": "2. Uncertainty query, full-pool direct-only",
    "shap_box_tags_no_split": "3. Per-feature SHAP-box tags, no split",
    "split_direct_only": "4. Bottom-50% split, direct-only",
    "split_plus_shap_box_tags": "5. Bottom-50% split + per-feature SHAP-box tags",
}
STEM = RESULTS / "final_active_learning_matrix_20x25_matched4"


def read_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def checkpoint_key(row: dict) -> tuple[int, int, str]:
    fit_pass = row.get("fit_pass") or row.get("checkpoint_type") or "retrain"
    if fit_pass == "":
        fit_pass = "retrain"
    if fit_pass == "final_flush":
        return int(row["seed"]), int(row["review_budget"]), "final_flush"
    return int(row["seed"]), int(row["review_budget"]), "retrain"


def p_and_r(row: dict) -> tuple[float, float]:
    p = row.get("model_metrics_model_full_precision")
    r = row.get("model_metrics_model_full_recall")
    if p in (None, "") or r in (None, ""):
        raise ValueError(f"Missing full-data P/R in checkpoint: {row}")
    return float(p), float(r)


def load_method_rows() -> tuple[dict, dict]:
    rows_by_method: dict[str, dict[tuple[int, int, str], dict]] = {
        method: {} for method in METHODS
    }
    source: dict[str, dict[tuple[int, int, str], dict]] = {
        method: {} for method in METHODS
    }

    fullpool_seed42 = RESULTS / "split_training_full_pool_matrix_matched4_20x25_checkpoints.csv"
    fullpool_random_other = RESULTS / "split_training_full_pool_matrix_random_seeds7_123_2024_checkpoints.csv"
    fullpool_uncertainty_other = RESULTS / "split_training_full_pool_matrix_uncertainty_seeds7_123_2024_checkpoints.csv"
    fullpool_tags = RESULTS / "split_training_full_pool_matrix_feature_tags_matched4_20x25_checkpoints.csv"
    split_arms = RESULTS / "split_training_shap_box_oracle_features_paired_matched4_fit_agreement_checkpoints.csv"

    sources = [
        (fullpool_seed42, {"random_direct_only": "random_query",
                           "uncertainty_direct_only": "uncertainty_query"}, {42}),
        (fullpool_random_other, {"random_direct_only": "random_query"}, {7, 123, 2024}),
        (fullpool_uncertainty_other, {"uncertainty_direct_only": "uncertainty_query"}, {7, 123, 2024}),
        (fullpool_tags, {"feature_tagged_shap_box": "shap_box_tags_no_split"}, set(SEEDS)),
        (split_arms, {"direct_only": "split_direct_only",
                      "feature_tagged_shap_box": "split_plus_shap_box_tags"}, set(SEEDS)),
    ]
    for path, arm_map, wanted_seeds in sources:
        if not path.exists():
            raise FileNotFoundError(f"Expected completed experiment artifact missing: {path}")
        for row in read_rows(path):
            if row.get("schedule") != SCHEDULE:
                continue
            arm = row.get("arm")
            method = arm_map.get(arm)
            if method is None or int(row["seed"]) not in wanted_seeds:
                continue
            key = checkpoint_key(row)
            previous = rows_by_method[method].get(key)
            if previous is not None:
                raise ValueError(f"Duplicate matrix row for {method} {key}: {path}")
            rows_by_method[method][key] = row
            source[method][key] = row

    expected = {
        (seed, review, phase)
        for seed in SEEDS
        for review in REVIEWS
        for phase in ("retrain", "final_flush")
    }
    #final flush applies only at 500 reviews
    expected = {(seed, review, "retrain") for seed in SEEDS for review in REVIEWS}
    expected.update((seed, 500, "final_flush") for seed in SEEDS)
    for method, method_rows in rows_by_method.items():
        missing = expected - set(method_rows)
        extra = set(method_rows) - expected
        if missing or extra:
            raise ValueError(
                f"Incomplete matrix arm {method}: missing={sorted(missing)}, extra={sorted(extra)}"
            )
    return rows_by_method, source


def make_matrix_rows(rows_by_method):
    """One row per method/seed; columns carry P and R for each retrain budget."""
    matrix = []
    for method in METHODS:
        for seed in SEEDS:
            row = {"method": method, "seed": seed, "schedule": SCHEDULE}
            for review in REVIEWS:
                p, r = p_and_r(rows_by_method[method][(seed, review, "retrain")])
                row[f"{review}_P"] = p
                row[f"{review}_R"] = r
            p, r = p_and_r(rows_by_method[method][(seed, 500, "final_flush")])
            row["final_flush_P"] = p
            row["final_flush_R"] = r
            matrix.append(row)
    return matrix


def fit_summary(rows_by_method):
    summary = []
    for method, rows in rows_by_method.items():
        for source_name in ("pseudo", "direct", "propagated"):
            agreement_col = f"fit_target_metrics_{source_name}_agreement"
            loss_col = f"fit_target_metrics_{source_name}_log_loss"
            vals = [row for row in rows.values() if row.get(agreement_col) not in (None, "")]
            if not vals:
                summary.append({"method": method, "source": source_name,
                                "n_checkpoints": 0, "min_agreement": None,
                                "max_agreement": None, "min_log_loss": None,
                                "max_log_loss": None})
                continue
            agreements = [float(row[agreement_col]) for row in vals]
            losses = [float(row[loss_col]) for row in vals if row.get(loss_col) not in (None, "")]
            summary.append({
                "method": method, "source": source_name,
                "n_checkpoints": len(vals),
                "min_agreement": min(agreements), "max_agreement": max(agreements),
                "min_log_loss": min(losses) if losses else None,
                "max_log_loss": max(losses) if losses else None,
            })
    return summary


def fmt(value):
    return "—" if value is None else f"{value:.5f}"


def write_report(matrix, rows_by_method, fit_stats):
    lines = [
        "# Active-learning comparison matrix — 20×25, four run seeds", "",
        "Five methods, 500 direct reviews, per-retrain full-data P/R. Seeds 42, 7, 123, 2024.",
        "Each method block has one row per seed; each retrain budget cell shows P/R.",
        "`Final flush` is final delayed fit after last-block tags.",
        "Method arms: random full-pool direct-only; uncertainty full-pool direct-only; "
        "uncapped per-feature tagged SHAP-box without split; split direct-only; split plus "
        "same tagged SHAP-box policy.", "",
    ]
    for method in METHODS:
        display = METHOD_DISPLAY[method]
        lines.extend([
            f"## {display}", "",
            "| Seed | " + " | ".join(map(str, REVIEWS)) + " | Final flush |",
            "|---:|" + "---:|" * (len(REVIEWS) + 1),
        ])
        for row in (item for item in matrix if item["method"] == method):
            cells = [f"{row[f'{review}_P']:.5f}/{row[f'{review}_R']:.5f}"
                     for review in REVIEWS]
            cells.append(f"{row['final_flush_P']:.5f}/{row['final_flush_R']:.5f}")
            lines.append(f"| {row['seed']} | " + " | ".join(cells) + " |")
        lines.append("")

    lines.extend([
        "", "## Fit-target agreement and log loss", "",
        "Ranges across retrains, by current fit-target source. This is a diagnostic "
        "of model agreement with assigned targets; full per-retrain values are in the "
        "checkpoint CSV.", "",
        "| Method | Source | Checkpoints | Agreement min–max | Log-loss min–max |",
        "|---|---|---:|---:|---:|",
    ])
    for stat in fit_stats:
        agreement = ("—" if stat["n_checkpoints"] == 0 else
                     f"{stat['min_agreement']:.5f}–{stat['max_agreement']:.5f}")
        losses = ("—" if stat["n_checkpoints"] == 0 else
                  f"{stat['min_log_loss']:.5f}–{stat['max_log_loss']:.5f}")
        lines.append(
            f"| {stat['method']} | {stat['source']} | {stat['n_checkpoints']} | "
            f"{agreement} | {losses} |"
        )

    lines.extend([
        "", "## Uncertainty-arm source", "",
        "Matrix uses the saved full-pool uncertainty/direct-only checkpoint curves; "
        "matrix assembly itself runs no trajectories.", "",
        "Source artifacts: random/uncertainty full-pool arms and feature-tagged no-split "
        "arms under `split_training_full_pool_matrix_*`; split arms under "
        "`split_training_shap_box_oracle_features_paired_matched4_fit_agreement_*`.",
        "Companion matrix: `final_active_learning_matrix_20x25_matched4.csv`; fit summary: "
        "`final_active_learning_matrix_20x25_matched4_fit_agreement.csv`.",
    ])
    Path(f"{STEM}_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_excel(matrix, rows_by_method):
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter

    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "P-R Matrix"
    last_col = 2 + len(REVIEWS) + 1
    sheet.merge_cells(start_row=1, start_column=1, end_row=1, end_column=last_col)
    title = sheet.cell(1, 1, "Active-learning matrix — 20×25, four seeds")
    title.font = Font(bold=True, size=14, color="FFFFFF")
    title.fill = PatternFill("solid", fgColor="1F4E78")
    title.alignment = Alignment(horizontal="left")

    sheet.merge_cells(start_row=2, start_column=1, end_row=3, end_column=1)
    sheet.cell(2, 1, "Method")
    sheet.merge_cells(start_row=2, start_column=2, end_row=3, end_column=2)
    sheet.cell(2, 2, "Seed")
    sheet.merge_cells(start_row=2, start_column=3, end_row=2,
                      end_column=2 + len(REVIEWS))
    sheet.cell(2, 3, "Direct reviews at retrain")
    for column, review in enumerate(REVIEWS, start=3):
        sheet.cell(3, column, review)
    flush_col = last_col
    sheet.merge_cells(start_row=2, start_column=flush_col, end_row=3, end_column=flush_col)
    sheet.cell(2, flush_col, "Final flush\n500 reviews")

    row_cursor = 4
    for method in METHODS:
        method_rows = [row for row in matrix if row["method"] == method]
        first = row_cursor
        for row in method_rows:
            sheet.cell(row_cursor, 2, row["seed"])
            for index, review in enumerate(REVIEWS, start=3):
                sheet.cell(row_cursor, index,
                           f"{row[f'{review}_P']:.5f}/{row[f'{review}_R']:.5f}")
            sheet.cell(row_cursor, flush_col,
                       f"{row['final_flush_P']:.5f}/{row['final_flush_R']:.5f}")
            row_cursor += 1
        last = row_cursor - 1
        sheet.merge_cells(start_row=first, start_column=1, end_row=last, end_column=1)
        method_cell = sheet.cell(first, 1, METHOD_DISPLAY[method])
        method_cell.alignment = Alignment(vertical="center", wrap_text=True)
        method_cell.font = Font(bold=True)
        for r in range(first, last + 1):
            if (first - 4) // len(SEEDS) % 2 == 0:
                for c in range(1, last_col + 1):
                    sheet.cell(r, c).fill = PatternFill("solid", fgColor="EAF2F8")

    for row in sheet.iter_rows(min_row=2, max_row=row_cursor - 1,
                               min_col=1, max_col=last_col):
        for cell in row:
            cell.alignment = Alignment(vertical="center", horizontal="center", wrap_text=True)
            if cell.row in (2, 3):
                cell.font = Font(bold=True, color="FFFFFF")
                cell.fill = PatternFill("solid", fgColor="4472C4")
    sheet.column_dimensions["A"].width = 26
    sheet.column_dimensions["B"].width = 10
    for c in range(3, last_col + 1):
        sheet.column_dimensions[get_column_letter(c)].width = 16
    sheet.freeze_panes = "C4"

    fit_sheet = workbook.create_sheet("Fit Agreement")
    headers = ["Method", "Seed", "Reviews", "Fit pass", "Source", "N",
               "Agreement", "Log loss", "Objective-weighted agreement",
               "Objective-weighted log loss"]
    fit_sheet.append(headers)
    for method, rows in rows_by_method.items():
        for key, data in rows.items():
            seed, review, fit_pass = key
            for source in ("pseudo", "direct", "propagated"):
                prefix = f"fit_target_metrics_{source}_"
                n = data.get(prefix + "n")
                if n in (None, "", 0, "0"):
                    continue
                fit_sheet.append([
                    METHOD_DISPLAY[method], seed, review, fit_pass, source,
                    int(float(n)),
                    data.get(prefix + "agreement"),
                    data.get(prefix + "log_loss"),
                    data.get(prefix + "objective_weighted_agreement"),
                    data.get(prefix + "objective_weighted_log_loss"),
                ])
    fit_sheet.freeze_panes = "A2"
    fit_sheet.auto_filter.ref = fit_sheet.dimensions
    for cell in fit_sheet[1]:
        cell.font = Font(bold=True, color="FFFFFF")
        cell.fill = PatternFill("solid", fgColor="4472C4")
    for column, width in enumerate((28, 10, 12, 16, 16, 10, 16, 16, 30, 26), start=1):
        fit_sheet.column_dimensions[get_column_letter(column)].width = width
    workbook.save(f"{STEM}.xlsx")


def main():
    rows_by_method, _ = load_method_rows()
    matrix = make_matrix_rows(rows_by_method)
    fit_stats = fit_summary(rows_by_method)
    matrix_path = Path(f"{STEM}.csv")
    with matrix_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = ["method", "seed", "schedule"]
        for review in REVIEWS:
            fieldnames.extend((f"{review}_P", f"{review}_R"))
        fieldnames.extend(("final_flush_P", "final_flush_R"))
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(matrix)
    fit_path = Path(f"{STEM}_fit_agreement.csv")
    with fit_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fit_stats[0]))
        writer.writeheader()
        writer.writerows(fit_stats)
    write_report(matrix, rows_by_method, fit_stats)
    write_excel(matrix, rows_by_method)
    print(f"Wrote {len(matrix)} method/seed rows: {STEM}_report.md, .csv, and .xlsx")


if __name__ == "__main__":
    main()
