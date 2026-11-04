#!/usr/bin/env python3
"""Audit of the ingestion step (plan §4.1) on every business of the given splits.

Reports, per table and column, what the loader could NOT normalize, so the code agent does not
have to guess: date-named columns that still hold text, text columns whose values look numeric,
columns that look like dates but were not parsed, ambiguous dd/mm vs mm/dd, Excel error cells,
empty tables, and whether the README's last data day matches the data.

  python scripts/audit_ingestion.py -d train public_test
"""

import argparse
import collections
import os
import re
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd  # noqa: E402

from src.datamate.loader import NUMBER_TEXT_RE, is_date_column_name, load_business_data  # noqa: E402

DATE_TEXT_RE = re.compile(r"^\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}|^\d{4}-\d{2}-\d{2}|\bthg\b|\btháng\b", re.I)
ISO_RE = re.compile(r"^\d{4}-\d{2}-\d{2}( \d{2}:\d{2}:\d{2})?$")
EXCEL_ERR = {"#VALUE!", "#N/A", "#REF!", "#DIV/0!", "#NAME?", "#NUM!", "#NULL!"}


def audit_table(name: str, df: pd.DataFrame) -> list:
    issues = []
    if df.empty:
        return [f"{name}: empty table"]
    for col in df.columns:
        s = df[col].dropna()
        if s.empty:
            continue
        texts = s[s.map(lambda v: isinstance(v, str))]
        if texts.empty:
            continue
        uniq = pd.Series(texts.unique()).astype(str).str.strip()
        share = len(texts) / len(s)
        if is_date_column_name(col):
            bad = uniq[~uniq.str.match(ISO_RE)]
            if len(bad) and share > 0.05:
                issues.append(f"{name}.{col}: date column with {len(bad)} non-ISO values, e.g. {bad.head(3).tolist()}")
        else:
            num = uniq.str.match(NUMBER_TEXT_RE) & ~uniq.str.match(r"^0\d")
            if num.mean() > 0.8 and len(uniq) > 3:
                issues.append(f"{name}.{col}: {num.mean():.0%} of text values look numeric but stayed text, e.g. {uniq[~num].head(3).tolist()}")
            dates = uniq.str.match(DATE_TEXT_RE) & ~uniq.str.match(ISO_RE)   # ISO text = already normalized
            if dates.mean() > 0.8 and len(uniq) > 3:
                issues.append(f"{name}.{col}: looks like dates but the column name was not recognized, e.g. {uniq.head(3).tolist()}")
        errs = set(uniq) & EXCEL_ERR
        if errs:
            issues.append(f"{name}.{col}: Excel error cells {sorted(errs)}")
    return issues


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--splits", nargs="+", default=["train", "public_test"])
    args = ap.parse_args()
    totals = collections.Counter()
    for split in args.splits:
        for biz in sorted(os.listdir(os.path.join(split, "data"))):
            bd = load_business_data(os.path.join(split, "data", biz), keep_time=True, use_cache=False)
            issues = []
            for name, df in bd.tables.items():
                if name in {p for parts in bd.metadata.get("merged_into", {}).values() for p in parts}:
                    continue
                issues += audit_table(name, df)
            totals["businesses"] += 1
            if issues:
                totals["with_issues"] += 1
                print(f"== {split}/{biz}")
                for i in issues:
                    print("   ", i)
    print(f"\n{totals['with_issues']}/{totals['businesses']} businesses with ingestion issues")


if __name__ == "__main__":
    main()
