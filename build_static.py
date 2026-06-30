#!/usr/bin/env python3
"""Build the static leaderboard dataset (``docs/data.json``).

Scans every result JSON under ``results/``, normalizes per-seed records into
a compact schema, and emits a single ``data.json`` consumed by ``index.html``.
Re-run after every ``scripts/sync_results.sh``.
"""
from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
LEADERBOARD_ROOT = Path(__file__).resolve().parent
DEFAULT_RESULTS = REPO_ROOT / "results"
DEFAULT_OUT = LEADERBOARD_ROOT / "docs" / "data.json"

# Keep this in sync with leaderboard/index.html — sources of truth for catalog data.
METHODS_CATALOG: dict[str, dict[str, str]] = {
    "source":      dict(full="Source",      venue="—",         year="—",    family="Baseline",          summary="No adaptation; runs the pretrained model directly."),
    "tent":        dict(full="TENT",        venue="ICLR",      year="2021", family="Entropy",           summary="Entropy minimization on test stream by updating BN affine."),
    "adabn":       dict(full="AdaBN",       venue="PR",        year="2018", family="BN statistics",     summary="Replace source BN statistics with test-batch statistics."),
    "eata":        dict(full="EATA",        venue="ICML",      year="2022", family="Entropy + filter",  summary="Sample-efficient entropy min with reliability+diversity filters and Fisher-anchor regularization."),
    "sar":         dict(full="SAR",         venue="ICLR",      year="2023", family="Entropy + SAM",     summary="Sharpness-aware entropy minimization; robust under non-i.i.d. test streams."),
    "lame":        dict(full="LAME",        venue="CVPR",      year="2022", family="Output",            summary="Laplacian-regularized output correction; does not update the backbone."),
    "t3a":         dict(full="T3A",         venue="NeurIPS",   year="2021", family="Classifier",        summary="Compute test-time class prototypes from feature support sets; no gradient updates."),
    "deyo":        dict(full="DeYO",        venue="ICLR",      year="2024", family="Entropy + filter",  summary="Disentangle reliable from spurious samples and reweight entropy minimization."),
    "memo":        dict(full="MEMO",        venue="NeurIPS",   year="2022", family="Augmentation",      summary="Per-sample episodic adaptation via marginal-entropy minimization over augmentations."),
    "tea":         dict(full="TEA",         venue="CVPR",      year="2024", family="Energy",            summary="Test-time energy adaptation aligning sample energies with the source distribution."),
    "cotta":       dict(full="CoTTA",       venue="CVPR",      year="2022", family="Teacher-student",   summary="Continual TTA: weight-averaged teacher + stochastic restore against error accumulation."),
    "adacontrast": dict(full="AdaContrast", venue="CVPR",      year="2022", family="Self-supervised",   summary="Self-supervised pseudo-label refinement with contrastive consistency."),
    "rotta":       dict(full="RoTTA",       venue="CVPR",      year="2023", family="Memory bank",       summary="Robust TTA with category-balanced sampling memory and timely update for non-i.i.d. streams."),
    "rmt":         dict(full="RMT",         venue="CVPR",      year="2023", family="Robust mean-teacher",summary="Symmetric cross-entropy + contrastive mean-teacher for continual TTA."),
    "becotta":     dict(full="BeCoTTA",     venue="ICML",      year="2024", family="MoE adapters",      summary="Mixture-of-experts low-rank adapters for continual/compositional TTA."),
    "dss":         dict(full="DSS",         venue="WACV",      year="2024", family="Filter",            summary="Decoupled sample selection: reliable + uncertain sample re-weighting."),
    "gold":        dict(full="GOLD",        venue="CVPR",      year="2026", family="Low-rank subspace",  summary="The 'golden subspace': low-rank directional (AGOP) adaptation for efficient continual TTA."),
    "santa":       dict(full="SANTA",       venue="TMLR",      year="2023", family="Source-aligned",    summary="Source-anchored test-time adaptation; aligns features to source-class statistics."),
    "ods":         dict(full="ODS",         venue="ICML",      year="2023", family="Open-world label-shift", summary="Open-world TTA under joint covariate + label-distribution shift: LAME posterior refinement with a class-balanced reservoir memory and learnt BN statistics."),
    "obao":        dict(full="OBAO",        venue="ECCV",      year="2024", family="BN+Output",         summary="Online batch-aware optimization combining BN refresh + output regularization."),
    "note":        dict(full="NOTE",        venue="NeurIPS",   year="2022", family="Memory bank",       summary="Non-i.i.d. test-time adaptation with prediction-balanced reservoir sampling."),
    # Task-native methods for the structured-prediction boards.
    "diga":        dict(full="DIGA",        venue="CVPR",      year="2023", family="Prototype (seg)",    summary="Backward-free segmentation TTA: instance-guided BN-stat mixing (DAM) + EMA class-prototype fusion (SAM)."),
    "meanteacher": dict(full="Mean-Teacher",venue="—",         year="—",    family="Teacher-student (det)", summary="Detection self-training: EMA teacher emits confident pseudo-detections, student distills them query-aligned (class CE + box L1)."),
    "actmad":      dict(full="ActMAD",      venue="CVPR",      year="2023", family="Activation matching", summary="Aligns per-channel activation mean/var to the model's stored source BN statistics; backprop into the backbone feature extractor."),
    "svdp":        dict(full="SVDP",        venue="AAAI",      year="2024", family="Visual prompt (seg)", summary="Sparse visual domain prompts (~0.1% params) on the input image for dense-prediction TTA, with adaptive placement and per-sample updating."),
    "vptta":       dict(full="VPTTA",       venue="CVPR",      year="2024", family="Visual prompt (seg)", summary="Per-image low-frequency prompt trained to align BN statistics; EMA prompt memory bank + source/target stat warm-up."),
    "stfar":       dict(full="STFAR",       venue="arXiv",     year="2023", family="Self-training (det)", summary="Test-time detection via EMA self-training with output-level feature-alignment regularization against noisy pseudo-labels."),
    "memclr":      dict(full="MemCLR",      venue="WACV",      year="2023", family="Memory + contrastive (det)", summary="MemXformer cross-attention memory recording target prototypes; yields pos/neg pairs for a contrastive loss in online detection adaptation."),
    "ioufilter":   dict(full="FullyTTA",    venue="CVPRW",     year="2024", family="Filter (det)",        summary="Single-image fully TTA for detection; two IoU-based indicators complement confidence to filter unreliable pseudo-labels."),
    "sgp":         dict(full="SGP",         venue="CVPR",      year="2025", family="Pruning (det)",       summary="Sensitivity-guided pruning selects where to adapt for efficient continual test-time detection."),
}

# Paper bibliography (title / authors / venue / url / pdf / code / bibtex) lives in
# papers.json next to this script — keeps multi-line BibTeX out of Python source.
# Merged into each method's catalog entry at build time so index.html's Details tab
# can emit \cite{} keys and BibTeX for the LaTeX/CSV exporters.
# `gold` (arXiv 2603.21928, CVPR 2026) verified 2026-06-03 via the arXiv abstract page.
def _load_json(name: str):
    p = LEADERBOARD_ROOT / name
    return json.loads(p.read_text()) if p.exists() else None

PAPER_META: dict[str, dict[str, Any]] = _load_json("papers.json") or {}

# Topic category for the 19 benchmarked methods. Kept on each paper entry for
# potential downstream use (currently informational).
BENCH_CATEGORY: dict[str, str] = {
    "tent": "Entropy / Pseudo-label", "eata": "Entropy / Pseudo-label",
    "sar": "Entropy / Pseudo-label", "deyo": "Entropy / Pseudo-label",
    "adabn": "Normalization",
    "lame": "Output / Prototype", "t3a": "Output / Prototype",
    "memo": "Augmentation", "tea": "Energy", "adacontrast": "Self-supervised",
    "cotta": "Continual / Online TTA", "rotta": "Continual / Online TTA",
    "rmt": "Continual / Online TTA", "becotta": "Continual / Online TTA",
    "dss": "Continual / Online TTA", "gold": "Continual / Online TTA",
    "santa": "Continual / Online TTA", "obao": "Continual / Online TTA",
    "note": "Continual / Online TTA",
}


_JOURNAL_TOKENS = ("tmlr", "ijcv", "jmlr", "tpami", "pami", "pattern recognition",
                   "journal of", "transactions", "tmi", "media", "tip", "tnnls", "imwut")
_CONF_TOKENS = ("iclr", "neurips", "icml", "cvpr", "iccv", "eccv", "aaai", "wacv",
                "aistats", "bmvc", "ijcai", "acl", "emnlp", "naacl", "kdd", "3dv",
                "icra", "miccai", "iros", "interspeech", "asru", "wsdm")


# Verified presentation status (Oral / Spotlight) per slug. Web-verified against
# venue accepted-paper lists / OpenReview decisions; everything else is a poster.
PRESENTATION: dict[str, str] = {
    "ods": "Oral", "owttt": "Oral", "sar": "Oral", "lame": "Oral", "sgp": "Oral",
    "tent": "Spotlight", "eata": "Spotlight", "t3a": "Spotlight", "deyo": "Spotlight",
}


def classify_venue(venue: str) -> tuple[str, str, str]:
    """Return (display_label, venue_name, venue_type). type ∈ conference|journal|other."""
    v = venue or ""
    low = v.lower()
    label = re.sub(r"\s*\((oral|spotlight|notable[^)]*)\)", "", v, flags=re.I).strip()
    # venue_name = venue without year / issue / Workshop tag — used as a search key.
    name = re.sub(r"\b\d{4}\b.*$", "", label).replace("Workshop", "").strip(" ,")
    if "workshop" in low or "arxiv" in low or "preprint" in low:
        vtype = "other"
    elif any(t in low for t in _JOURNAL_TOKENS):
        vtype = "journal"
    elif any(re.search(r"\b" + t + r"\b", low) for t in _CONF_TOKENS):
        vtype = "conference"
    else:
        vtype = "other"
    return label, name, vtype


def build_paper_list() -> list[dict[str, Any]]:
    """Bibliography entries for the 19 benchmarked methods (for Details-tab export)."""
    papers: list[dict[str, Any]] = []

    def entry(slug, full, meta, category, benchmarked):
        label, vname, vtype = classify_venue(meta.get("venue") or "")
        pres = PRESENTATION.get(slug) or meta.get("presentation")
        return {
            "slug": slug, "full": full,
            "title": meta.get("title"), "authors": meta.get("authors"),
            "venue": meta.get("venue"), "venue_label": label, "venue_name": vname,
            "venue_type": vtype, "presentation": pres,
            "category": category,
            "task": meta.get("task") or "Image Classification",
            "url": meta.get("url"), "pdf": meta.get("pdf"),
            "code": meta.get("code"), "bibtex": meta.get("bibtex"),
            "benchmarked": benchmarked,
        }

    for slug, meta in PAPER_META.items():
        cat = METHODS_CATALOG.get(slug, {})
        papers.append(entry(slug, cat.get("full") or slug.upper(), meta,
                            BENCH_CATEGORY.get(slug, "Other"), True))

    papers.sort(key=lambda x: (x["full"] or "").lower())
    return papers

DESIGNED_SETTINGS: dict[str, list[str]] = {
    "source":      ["—"],
    "tent":        ["Online / Uniform"],
    "adabn":       ["Online / Uniform"],
    "eata":        ["Online / Uniform"],
    "sar":         ["Online / Uniform", "Online / Imbalanced"],
    "lame":        ["Online / Uniform", "Online / Imbalanced"],
    "t3a":         ["Online / Uniform"],
    "deyo":        ["Online / Uniform", "Continual"],
    "memo":        ["Episodic"],
    "tea":         ["Online / Uniform", "Continual"],
    "cotta":       ["Continual"],
    "adacontrast": ["Online / Uniform"],
    "rotta":       ["Continual / Non-i.i.d."],
    "rmt":         ["Continual"],
    "becotta":     ["Continual / Compositional"],
    "dss":         ["Continual / Non-i.i.d."],
    "gold":        ["Online / Uniform", "Continual"],
    "santa":       ["Online / Uniform", "Continual"],
    "obao":        ["Online / Uniform", "Continual"],
    "note":        ["Continual / Non-i.i.d."],
}

# Datasets we evaluate against, per task, with the leaderboard's target run
# count and the number of (method × shift) cells expected when complete. The
# frontend coverage panel shows the entry list for the currently-selected task.
TARGET_DATASETS_BY_TASK = {
    "classification": [
        {"hf": "TTA-CIFAR-10-C",  "label": "CIFAR-10-C",  "target": 20 * 6 * 5, "cells": 120},
        {"hf": "TTA-CIFAR-100-C", "label": "CIFAR-100-C", "target": 20 * 6 * 5, "cells": 120},
        {"hf": "TTA-ImageNet-C",  "label": "ImageNet-C",  "target": 20 * 6 * 5, "cells": 120},
    ],
    # Segmentation: core-5 + DIGA, continual protocol, 5 seeds.
    "segmentation": [
        {"hf": "TTA-Cityscapes-C", "label": "Cityscapes-C", "target": 6 * 1 * 5, "cells": 6},
        {"hf": "TTA-ADE20K-C",     "label": "ADE20K-C",     "target": 6 * 1 * 5, "cells": 6},
    ],
    # Detection: source/tent/cotta/eata/sar/meanteacher/actmad, online, 5 seeds.
    "detection": [
        {"hf": "TTA-COCO-C", "label": "COCO-C", "target": 7 * 1 * 5, "cells": 7},
    ],
}

# Protocols (settings) we evaluate every method against.
PROTOCOLS_GLOSSARY = [
    {
        "key": "online_uniform",
        "shift": "Online / Uniform",
        "label": "Online · Uniform",
        "mode": "online",
        "reset_policy": "per_stream",
        "sampling": "uniform i.i.d.",
        "definition": "One corruption stream at a time. Within each stream, batches are drawn i.i.d. The adapter state is reset between corruptions.",
        "stream_order": "15 corruptions are processed independently; order across corruptions does not matter.",
        "batch_distribution": "Each batch reflects the natural class proportions of the dataset.",
        "reset": "Adapter state resets between corruptions (`reset_policy=per_stream`).",
        "originating": "Wang et al., TENT (ICLR 2021), Table 3 setting — the canonical online TTA benchmark.",
        "use_when": "Default evaluation for any TTA method; tests pure adaptation without temporal drift.",
    },
    {
        "key": "continual_uniform",
        "shift": "Continual / Uniform",
        "label": "Continual · Uniform",
        "mode": "continual",
        "reset_policy": "never",
        "sampling": "uniform i.i.d.",
        "definition": "Streams flow back-to-back without resetting adapter state. The model must accumulate adaptation across all 15 corruptions.",
        "stream_order": "Streams concatenated in a fixed order (alphabetical by corruption name).",
        "batch_distribution": "Each batch reflects the natural class proportions of the dataset.",
        "reset": "Adapter state persists across streams (`reset_policy=never`).",
        "originating": "Wang et al., CoTTA (CVPR 2022) — measures resistance to error accumulation under long sequences.",
        "use_when": "Tests whether a method can adapt continuously without catastrophic forgetting or drift.",
    },
    {
        "key": "continual_dir01",
        "shift": "Continual / Dirichlet (α=0.1)",
        "label": "Continual · Dirichlet α=0.1 (strong class correlation)",
        "mode": "continual",
        "reset_policy": "never",
        "sampling": "Dirichlet (α=0.1)",
        "definition": "Continual stream where each batch's class proportions are drawn from a Dirichlet(α) distribution with small α — batches cluster heavily on a few classes.",
        "stream_order": "Same as Continual; corruption-level order fixed, but per-batch class shifts dramatically.",
        "batch_distribution": "Highly skewed (low α) — most batches dominated by 1–2 classes, producing severe label shift over time.",
        "reset": "Adapter state persists.",
        "originating": "Yuan et al., NOTE (NeurIPS 2022) — non-i.i.d. evaluation; small α stresses methods that assume balanced batches.",
        "use_when": "Stress-tests robustness to non-i.i.d. test streams; breaks entropy-min methods relying on diverse predictions.",
    },
    {
        "key": "continual_dir10",
        "shift": "Continual / Dirichlet (α=1.0)",
        "label": "Continual · Dirichlet α=1.0 (mild class correlation)",
        "mode": "continual",
        "reset_policy": "never",
        "sampling": "Dirichlet (α=1.0)",
        "definition": "Same as α=0.1 but with α=1.0: batches still non-uniform, but the class distribution is closer to uniform.",
        "stream_order": "Same as Continual.",
        "batch_distribution": "Moderately skewed — distinguishes methods that overfit to extreme correlations from those that genuinely tolerate non-i.i.d. streams.",
        "reset": "Adapter state persists.",
        "originating": "Companion setting to α=0.1, common in non-i.i.d. TTA papers (RoTTA, NOTE, DSS).",
        "use_when": "Tests sensitivity to milder correlation; complements α=0.1 to characterize the full robustness curve.",
    },
    {
        "key": "continual_twk5",
        "shift": "Continual / Tweak One (γ=5)",
        "label": "Continual · Tweak-One γ=5 (moderate label shift)",
        "mode": "continual",
        "reset_policy": "never",
        "sampling": "Tweak-One (γ=5)",
        "definition": "One 'hot' class per corruption appears γ× more often than other classes — a single dominant class per phase, rotated across corruptions.",
        "stream_order": "Hot class rotates per corruption (per the YAML's `tweak_one_hot_classes` mapping).",
        "batch_distribution": "One class over-represented by 5×; other classes uniform.",
        "reset": "Adapter state persists.",
        "originating": "Niu et al., SAR (ICLR 2023) — 'tweak one' label-shift evaluation isolating single-class oversampling.",
        "use_when": "Tests sensitivity to label shift without the full chaos of Dirichlet sampling.",
    },
    {
        "key": "continual_twk10",
        "shift": "Continual / Tweak One (γ=10)",
        "label": "Continual · Tweak-One γ=10 (strong label shift)",
        "mode": "continual",
        "reset_policy": "never",
        "sampling": "Tweak-One (γ=10)",
        "definition": "Stronger variant: hot class appears 10× more often.",
        "stream_order": "Same as γ=5.",
        "batch_distribution": "Severe class imbalance (90%+ batches from one class).",
        "reset": "Adapter state persists.",
        "originating": "Niu et al., SAR (ICLR 2023) — the most aggressive tweak-one variant.",
        "use_when": "Pushes label-shift further; many entropy-min baselines collapse here.",
    },
]


def dataset_label(hf_repo: str) -> str:
    if "TTA-" in hf_repo:
        return hf_repo.split("TTA-", 1)[1]
    return hf_repo.rsplit("/", 1)[-1] if "/" in hf_repo else hf_repo


# Primary leaderboard metric per task. Returns (metric_key, display_label,
# lower_wins). Classification ranks by error (lower wins); the structured tasks
# rank by mIoU / mAP (higher wins). The frontend reads the per-record `score`
# (the value of metric_key) plus `lower_wins` so one ranking path serves all
# three boards.
TASK_METRIC = {
    "classification": ("error", "Error (%)", True),
    "segmentation":   ("miou",  "mIoU (%)", False),
    "detection":      ("map",   "mAP",      False),
}


def shift_label(mode: str, sampling: dict[str, Any]) -> str:
    stype = (sampling or {}).get("type", "uniform")
    mode_l = (mode or "online").replace("_", " ").title()
    if stype == "dirichlet":
        alpha = sampling.get("dirichlet_alpha")
        return f"{mode_l} / Dirichlet (α={alpha})" if alpha is not None else f"{mode_l} / Dirichlet"
    if stype == "tweak_one":
        gamma = sampling.get("tweak_one_gamma")
        return f"{mode_l} / Tweak One (γ={int(gamma)})" if gamma is not None else f"{mode_l} / Tweak One"
    return f"{mode_l} / Uniform"


def normalize(payload: dict[str, Any], path: Path) -> dict[str, Any] | None:
    benchmark = payload.get("benchmark") or {}
    protocol = payload.get("protocol") or {}
    source_model = payload.get("source_model") or {}
    metrics = payload.get("metrics_aggregate") or {}
    compute = payload.get("compute") or {}
    parameters = payload.get("parameters") or {}
    sampling = benchmark.get("sampling") or {}
    per_stream = payload.get("metrics_per_stream") or {}

    method = payload.get("adapter_name")
    if not method:
        return None
    # tweak_one (γ=5/10) is a deprecated experimental setting that was removed
    # from the benchmark. Stray result files for it keep resurfacing via remote
    # rsync; drop the records outright so the board can never show it again.
    if (sampling.get("type") or "").lower() == "tweak_one":
        return None
    # dss was removed from the leaderboard (impractical: ~10x slower than the
    # next slowest method on ImageNet). Old result files keep resurfacing via
    # rsync from results_bl/; drop them here permanently.
    if method == "dss":
        return None
    dataset = dataset_label(benchmark.get("hf_repo", ""))
    mode = protocol.get("mode") or "online"
    shift = shift_label(mode, sampling)
    seed = payload.get("seed")

    # Timing/memory are only trustworthy from a dedicated profile run (one clean
    # pass on a fixed GPU, result_kind="efficiency"). Performance seeds and the
    # old pre-split layout recorded per-seed timing on mixed machines, which the
    # board would otherwise aggregate into a spurious mean±std. So only let an
    # efficiency record contribute timing; everything else reports null (→ the
    # frontend filters it out, leaving the single clean profile number or blank).
    is_efficiency = payload.get("result_kind") == "efficiency"
    samples_per_s = compute.get("samples_per_sec") if is_efficiency else None
    adapter_s = compute.get("adapter_time_s") if is_efficiency else None
    peak_gpu_mb = compute.get("peak_gpu_mem_mb") if is_efficiency else None

    task = (payload.get("adapter_task_type") or benchmark.get("task_type")
            or "classification")
    metric_key, metric_label, lower_wins = TASK_METRIC.get(
        task, TASK_METRIC["classification"])

    return {
        "method": method,
        "task": task,
        # Unified ranking metric: `score` is the value of the task's primary
        # metric (error / mIoU / mAP); `lower_wins` tells the board which way to
        # rank. `metric` is the column label. Per-task secondary metrics are
        # kept too so tooltips/exports can show them.
        "score": metrics.get(metric_key),
        "metric": metric_label,
        "lower_wins": lower_wins,
        "dataset": dataset,
        "shift": shift,
        "mode": mode,
        "sampling": sampling.get("type") or "uniform",
        "reset": protocol.get("reset_policy") or "—",
        "batch_size": protocol.get("batch_size"),
        "source": source_model.get("name") or "—",
        "seed": seed,
        "error": metrics.get("error"),
        "accuracy": metrics.get("accuracy"),
        "miou": metrics.get("miou"),
        "map": metrics.get("map"),
        "map_50": metrics.get("map_50"),
        "samples_per_s": samples_per_s,
        "adapter_s": adapter_s,
        "peak_gpu_mb": peak_gpu_mb,
        "optimizer_params": parameters.get("optimizer"),
        "total_params": parameters.get("total"),
        "per_stream": {
            stream: {"score": (s or {}).get(metric_key),
                     "samples": (s or {}).get("samples")}
            for stream, s in per_stream.items()
        },
    }


def main():
    results_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_RESULTS
    out_path = Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_OUT
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Collect efficiency timing first (profile.json files). These must win the
    # dedup over same-seed performance records — but old-format perf files under
    # results/cifar10c/METHOD/ sort alphabetically before results/METHOD/profile.json
    # for any method whose name starts after 'c', causing the efficiency record to
    # be silently dropped. Fix: process profile.json files in a first pass so they
    # always register in `seen` before any performance file with the same key.
    records: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str, int]] = set()
    all_files = sorted(results_dir.rglob("*.json"))
    # profile.json first, everything else second (stable within each group)
    all_files.sort(key=lambda f: (0 if f.name == "profile.json" else 1, str(f)))
    for f in all_files:
        if f.name == "merged.json":
            continue
        # Permanently ignore legacy flat-format result files (basename like
        # `<method>__<model>__...__seed<N>.json`, found under results/<dataset>c/<method>/
        # or at the results/ root). They date from an early "seed not applied" bug era —
        # every seed carries identical metrics, faking STD=0 — and keep returning via
        # rsync from .53. They sort before the correct new-layout files
        # (results/<method>/<model>/<bench>/<proto>/seed<N>.json) and win the dedup,
        # silently shadowing real per-seed data. The new layout never puts `__` in a
        # filename, so this is a safe, durable guard.
        if "__" in f.name:
            continue
        try:
            payload = json.loads(f.read_text())
        except Exception:
            continue
        if not isinstance(payload, dict) or "metrics_aggregate" not in payload:
            continue
        rec = normalize(payload, f)
        if rec is None:
            continue
        # Include source model in the dedup key — the model-axis stores multiple
        # source models per (method, dataset, shift, seed); without it they
        # collide and only one model survives (the multi-model view vanishes).
        key = (rec["method"], rec["source"], rec["dataset"], rec["shift"], rec["seed"])
        if key in seen:
            continue
        seen.add(key)
        records.append(rec)

    methods_catalog = {
        slug: {**info, **PAPER_META.get(slug, {})}
        for slug, info in METHODS_CATALOG.items()
    }

    payload = {
        "meta": {
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "n_records": len(records),
            "target_datasets_by_task": TARGET_DATASETS_BY_TASK,
        },
        "methods": methods_catalog,
        "designed_settings": DESIGNED_SETTINGS,
        "protocols": PROTOCOLS_GLOSSARY,
        "papers": build_paper_list(),
        "records": records,
    }
    out_path.write_text(json.dumps(payload, separators=(",", ":")))
    size_kb = out_path.stat().st_size / 1024.0
    n_papers = len(payload["papers"])
    print(f"Wrote {len(records)} records, {n_papers} papers → {out_path} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
