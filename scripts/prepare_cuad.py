"""Download CUAD (510 contracts, 41 clause types, expert-marked spans) and stage it.

  python scripts/prepare_cuad.py                # clones the Atticus repo, writes data/cuad/*

Writes
  data/cuad/contracts.jsonl   {doc_name, text}
  data/cuad/labels.jsonl      {doc_name, category, spans: [{start, end, text}]}   (only categories with >=1 span)
  data/cuad/categories.json   {category: {question, answer_format}}

CUAD is CC BY 4.0 (Hendrycks et al., NeurIPS 2021). Character offsets are exact: text[start:end] == span text.
"""

import argparse
import csv
import json
import subprocess
import zipfile
from pathlib import Path

REPO_URL = "https://github.com/TheAtticusProject/cuad.git"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-dir", default="data/cuad_repo")
    parser.add_argument("--out-dir", default="data/cuad")
    args = parser.parse_args()
    repo, out = Path(args.repo_dir), Path(args.out_dir)

    if not (repo / "data.zip").exists():
        repo.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--depth", "1", REPO_URL, str(repo)], check=True)
    out.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(repo / "data.zip") as z:
        data = json.loads(z.read("CUADv1.json"))["data"]

    contracts, labels, n_spans = [], [], 0
    for item in data:
        para = item["paragraphs"][0]
        text, name = para["context"], item["title"]
        contracts.append({"doc_name": name, "text": text})
        for qa in para["qas"]:
            category = qa["id"].split("__")[-1]
            spans = []
            for a in qa["answers"]:
                start = a["answer_start"]
                assert text[start : start + len(a["text"])] == a["text"], (name, category)
                spans.append({"start": start, "end": start + len(a["text"]), "text": a["text"]})
            if spans:
                labels.append({"doc_name": name, "category": category, "spans": spans})
                n_spans += len(spans)

    categories = {}
    with open(repo / "category_descriptions.csv", encoding="utf-8-sig", newline="") as f:
        for row in csv.reader(f):
            if len(row) < 3 or not row[0].startswith("Category:"):
                continue
            name = row[0].removeprefix("Category:").strip()
            question = row[1].removeprefix("Description:").strip()
            fmt = row[2].removeprefix("Answer Format:").strip()
            categories[name] = {"question": question, "answer_format": fmt}

    # The CSV spells some names differently from the label ids ("Cap on Liability" vs "Cap On Liability").
    spelling = {l["category"].lower(): l["category"] for l in labels}
    categories = {spelling.get(k.lower(), k): v for k, v in categories.items()}

    (out / "contracts.jsonl").write_text(
        "\n".join(json.dumps(c) for c in contracts) + "\n", encoding="utf-8"
    )
    (out / "labels.jsonl").write_text(
        "\n".join(json.dumps(l) for l in labels) + "\n", encoding="utf-8"
    )
    (out / "categories.json").write_text(json.dumps(categories, indent=2), encoding="utf-8")
    print(
        f"{len(contracts)} contracts, {len(labels)} (contract, clause) labels, {n_spans} spans, "
        f"{len(categories)} categories -> {out}"
    )


if __name__ == "__main__":
    main()
