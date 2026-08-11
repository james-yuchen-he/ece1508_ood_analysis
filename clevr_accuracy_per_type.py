"""Compute soft-match accuracy per question type from a CLEVR BLIP-2 results CSV.

Same soft-match rules as clevr_accuracy.py (see clevr_common.py).

Usage:
    python clevr_accuracy_per_type.py [results_val_with_question_type.csv]
"""

import argparse
import csv

from clevr_common import QUESTION_TYPES, accuracy_report, extract_answer, normalize


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", nargs="?", default="results_val_with_question_type_10_tokens.csv")
    args = parser.parse_args()

    totals = [0] * len(QUESTION_TYPES)
    corrects = [0] * len(QUESTION_TYPES)
    with open(args.csv_path, newline="") as f:
        for row in csv.DictReader(f):
            type_id = int(row["question_type_id"])
            pred = extract_answer(row["raw_prediction"], QUESTION_TYPES[type_id])
            totals[type_id] += 1
            corrects[type_id] += pred == normalize(row["ground_truth"])

    print(accuracy_report(corrects, totals))


if __name__ == "__main__":
    main()
