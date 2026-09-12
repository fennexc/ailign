#!/usr/bin/env python3
"""Evaluate predicted TEI lexical alignments against reference TEI links."""

import argparse
import csv
from pathlib import Path
import re
import xml.etree.ElementTree as ET


XML_ID = "{http://www.w3.org/XML/1998/namespace}id"
CONTENT_POS = {"ADJ", "ADV", "NOUN", "NUM", "PROPN", "VERB"}
TSV_COLUMNS = ["scope", "unit", "ref", "predicted", "correct", "precision", "recall", "f1"]


def xml_id(element):
    return element.get(XML_ID) or element.get("xml:id") or element.get("id")


def trailing_number(identifier):
    match = re.search(r"^(.*?)(\d+)$", identifier)
    if match:
        return match.group(1), int(match.group(2))
    return identifier, -1


def read_tei(path):
    root = ET.parse(path).getroot()
    tokens = []
    corresp_ids = []
    links = set()

    for word in root.findall(".//{*}w"):
        tokens.append({
            "id": xml_id(word),
            "text": "".join(word.itertext()).strip(),
            "pos": word.get("pos", "X"),
        })

    for segment in root.findall(".//{*}seg"):
        segment_corresp_ids = [
            value.lstrip("#")
            for value in segment.get("corresp", "").split()
        ]
        corresp_ids.extend(segment_corresp_ids)
        for word in segment.findall(".//{*}w"):
            target_id = xml_id(word)
            for pivot_id in segment_corresp_ids:
                links.add((target_id, pivot_id))

    return {"tokens": tokens, "links": links, "corresp_ids": corresp_ids}


def normalize_pivot_tokens(pivot_doc):
    """Replace copied token ids in pivot files using only pivot TEI corresp ids."""
    current_ids = [token["id"] for token in pivot_doc["tokens"]]
    referenced_ids = sorted(set(pivot_doc["corresp_ids"]), key=trailing_number)
    if not referenced_ids:
        return pivot_doc["tokens"]
    if current_ids == referenced_ids:
        return pivot_doc["tokens"]
    if len(referenced_ids) != len(current_ids):
        raise ValueError(
            f"Cannot normalize pivot IDs: {len(current_ids)} pivot tokens but "
            f"{len(referenced_ids)} referenced IDs."
        )

    return [
        {**token, "id": referenced_id}
        for token, referenced_id in zip(pivot_doc["tokens"], referenced_ids)
    ]


def group_pivots_by_target(links):
    grouped = {}
    for target_id, pivot_id in links:
        grouped.setdefault(target_id, set()).add(pivot_id)
    return grouped


def score_target_tokens(tokens, reference_links, predicted_links):
    token_ids = {token["id"] for token in tokens}
    reference_links = {
        link for link in reference_links
        if link[0] in token_ids
    }
    predicted_links = {
        link for link in predicted_links
        if link[0] in token_ids
    }
    reference_by_target = group_pivots_by_target(reference_links)
    predicted_by_target = group_pivots_by_target(predicted_links)
    correct = sum(
        1
        for target_id in reference_by_target.keys() & predicted_by_target.keys()
        if reference_by_target[target_id] & predicted_by_target[target_id]
    )
    return make_score(
        ref=len(reference_by_target),
        predicted=len(predicted_by_target),
        correct=correct,
    )


def make_score(ref, predicted, correct):
    precision = correct / predicted if predicted else 0
    recall = correct / ref if ref else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
    return {
        "unit": "target_token",
        "ref": ref,
        "predicted": predicted,
        "correct": correct,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def add_scores(scores):
    return make_score(
        ref=sum(score["ref"] for score in scores),
        predicted=sum(score["predicted"] for score in scores),
        correct=sum(score["correct"] for score in scores),
    )


def filter_links(reference_links, predicted_links, target_ids, pivot_ids=None):
    filtered_reference = {
        (target_id, pivot_id)
        for target_id, pivot_id in reference_links
        if target_id in target_ids and (pivot_ids is None or pivot_id in pivot_ids)
    }
    filtered_predicted = {
        (target_id, pivot_id)
        for target_id, pivot_id in predicted_links
        if target_id in target_ids and (pivot_ids is None or pivot_id in pivot_ids)
    }
    return filtered_reference, filtered_predicted


def evaluate_file(ref_target_path, sys_target_path, pivot_tokens):
    ref_target = read_tei(ref_target_path)
    sys_target = read_tei(sys_target_path)

    all_tokens = ref_target["tokens"]
    all_score = score_target_tokens(all_tokens, ref_target["links"], sys_target["links"])

    content_target_ids = {
        token["id"] for token in ref_target["tokens"]
        if token["pos"] in CONTENT_POS
    }
    content_pivot_ids = {
        token["id"] for token in pivot_tokens
        if token["pos"] in CONTENT_POS
    }
    content_reference, content_predicted = filter_links(
        ref_target["links"],
        sys_target["links"],
        content_target_ids,
        content_pivot_ids,
    )
    content_tokens = [
        token for token in ref_target["tokens"]
        if token["id"] in content_target_ids
    ]
    content_score = score_target_tokens(
        content_tokens,
        content_reference,
        content_predicted,
    )

    by_pos = {}
    for pos in sorted({token["pos"] for token in ref_target["tokens"]}):
        pos_target_ids = {
            token["id"] for token in ref_target["tokens"]
            if token["pos"] == pos
        }
        pos_reference, pos_predicted = filter_links(
            ref_target["links"],
            sys_target["links"],
            pos_target_ids,
            content_pivot_ids if pos in CONTENT_POS else None,
        )
        pos_tokens = [
            token for token in ref_target["tokens"]
            if token["id"] in pos_target_ids
        ]
        by_pos[pos] = score_target_tokens(pos_tokens, pos_reference, pos_predicted)

    return {
        "file": ref_target_path.name,
        "all": all_score,
        "content": content_score,
        "by_pos": by_pos,
    }


def find_ref_targets(ref_dir, pivot_file, target):
    for path in sorted(ref_dir.glob("*.xml")):
        if path.resolve() == pivot_file.resolve():
            continue
        if target and target not in {path.name, path.stem}:
            continue
        yield path


def sys_path_for(ref_target_path, sys_dir):
    return sys_dir / f"{ref_target_path.stem}_word_ai.xml"


def format_score(label, score):
    return (
        f"{label:<7} ref={score['ref']} predicted={score['predicted']} "
        f"correct={score['correct']} P={score['precision']:.4f} "
        f"R={score['recall']:.4f} F1={score['f1']:.4f}"
    )


def score_row(scope, score):
    return {
        "scope": scope,
        "unit": score["unit"],
        "ref": score["ref"],
        "predicted": score["predicted"],
        "correct": score["correct"],
        "precision": f"{score['precision']:.4f}",
        "recall": f"{score['recall']:.4f}",
        "f1": f"{score['f1']:.4f}",
    }


def write_summary_tsv(path, all_score, content_score):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=TSV_COLUMNS, delimiter="\t")
        writer.writeheader()
        writer.writerow(score_row("all", all_score))
        writer.writerow(score_row("content", content_score))


def write_details(prefix, file_results, pos_totals):
    by_file_path = prefix.with_name(prefix.name + "_by_file.tsv")
    by_pos_path = prefix.with_name(prefix.name + "_by_pos.tsv")

    with by_file_path.open("w", encoding="utf-8", newline="") as file:
        columns = ["file", *TSV_COLUMNS]
        writer = csv.DictWriter(file, fieldnames=columns, delimiter="\t")
        writer.writeheader()
        for result in file_results:
            writer.writerow({"file": result["file"], **score_row("all", result["all"])})
            writer.writerow({"file": result["file"], **score_row("content", result["content"])})

    with by_pos_path.open("w", encoding="utf-8", newline="") as file:
        columns = ["pos", *TSV_COLUMNS]
        writer = csv.DictWriter(file, fieldnames=columns, delimiter="\t")
        writer.writeheader()
        for pos, score in sorted(pos_totals.items()):
            writer.writerow({"pos": pos, **score_row(pos.lower(), score)})

    return by_file_path, by_pos_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--ref-dir", type=Path)
    input_group.add_argument("--ref-file", type=Path)
    parser.add_argument("--sys-dir", type=Path)
    parser.add_argument("--sys-file", type=Path)
    parser.add_argument("--pivot-file", required=True, type=Path)
    parser.add_argument("--output-prefix", required=True, type=Path)
    parser.add_argument("--target", help="Evaluate one REF target file name or stem")
    parser.add_argument("--details", action="store_true", help="Write by-file and by-POS TSV files")
    parser.add_argument("--verbose", action="store_true", help="Print by-file and by-POS results")
    args = parser.parse_args()

    pivot_doc = read_tei(args.pivot_file)
    pivot_tokens = normalize_pivot_tokens(pivot_doc)

    if args.ref_file:
        if not args.sys_file:
            parser.error("--sys-file is required with --ref-file")
        input_files = [(args.ref_file, args.sys_file)]
    else:
        if not args.sys_dir:
            parser.error("--sys-dir is required with --ref-dir")
        if args.sys_file:
            parser.error("--sys-file can only be used with --ref-file")
        input_files = [
            (ref_target_path, sys_path_for(ref_target_path, args.sys_dir))
            for ref_target_path in find_ref_targets(
                args.ref_dir, args.pivot_file, args.target
            )
        ]

    file_results = []
    for ref_target_path, sys_target_path in input_files:
        if not sys_target_path.is_file():
            continue
        result = evaluate_file(ref_target_path, sys_target_path, pivot_tokens)
        file_results.append(result)
        if args.verbose:
            print(format_score(result["file"], result["all"]))
            print(format_score("CONTENT", result["content"]))

    if not file_results:
        parser.error("No matching REF/SYS target files found.")

    all_score = add_scores([result["all"] for result in file_results])
    content_score = add_scores([result["content"] for result in file_results])
    pos_totals = {
        pos: add_scores([
            result["by_pos"][pos]
            for result in file_results
            if pos in result["by_pos"]
        ])
        for pos in sorted({
            pos for result in file_results for pos in result["by_pos"]
        })
    }

    print(format_score("ALL", all_score))
    print(format_score("CONTENT", content_score))

    summary_path = args.output_prefix.with_name(args.output_prefix.name + "_evaluation.tsv")
    write_summary_tsv(summary_path, all_score, content_score)
    print(f"Wrote {summary_path}")

    if args.details:
        by_file_path, by_pos_path = write_details(args.output_prefix, file_results, pos_totals)
        print(f"Wrote {by_file_path}")
        print(f"Wrote {by_pos_path}")

    if args.verbose:
        for pos, score in sorted(pos_totals.items()):
            print(format_score(pos, score))


if __name__ == "__main__":
    main()
