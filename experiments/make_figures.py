"""Figures and summary tables for the Phase 1 report.

    python experiments/make_figures.py results/phase1_haiku
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

# Validated reference categorical palette, slots 1-3 in fixed order.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
STAGE_ORDER = ["research", "plan", "exec"]
FTYPE_ORDER = ["wrong_value", "stale_value", "fabricated", "injected_instruction", "malicious_redirect"]
SHORT = {"wrong_value": "wrong", "stale_value": "stale", "fabricated": "fabricated",
         "injected_instruction": "injected instr.", "malicious_redirect": "redirect"}
GRAN = ["stage", "message", "fact"]

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
    "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "legend.frameon": False, "savefig.dpi": 300, "savefig.bbox": "tight",
})


def _bar(ax, x, h, color, width, label=None):
    return ax.bar(x, h, width=width, color=color, label=label, edgecolor="white", linewidth=1)


def derive(d: pd.DataFrame) -> pd.DataFrame:
    """Outcome columns shared by all summaries; older result files lack the newer checker fields."""
    d = d.copy()
    if "faulty_emails_incorrect" in d:
        d["bad_emails"] = d.faulty_emails_unexpected + d.faulty_emails_incorrect
    else:  # older results: key values differ from the clean run
        d["bad_emails"] = d["emails_wrong"] if "emails_wrong" in d else d["emails_changed"]
    d["unauthorized"] = d.get("faulty_unauthorized_action", d.faulty_safety_violation).astype(bool)
    # With a response cache, a second placement that injects the same content as an earlier one
    # replays the same run: every recorded outcome matches. It is a duplicate, not a new sample.
    same = [c for c in d.columns if c != "fault_seed" and not c.endswith(("_elapsed_s", "_cache_hits"))]
    first = d.sort_values("fault_seed").astype(str).drop_duplicates(same)
    d["duplicate"] = ~d.index.isin(first.index)
    return d


def fig_propagation(d: pd.DataFrame, out: Path) -> None:
    combos = [(s, f) for s in STAGE_ORDER for f in FTYPE_ORDER if ((d.stage == s) & (d.ftype == f)).any()]
    g = d.groupby(["stage", "ftype"])
    labels = [f"{s}: {SHORT[f]}" for s, f in combos]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.0, 1.65), gridspec_kw={"width_ratios": [1, 1.25]})
    y = range(len(combos))[::-1]
    a1.barh(list(y), [g.get_group(c)["downstream_affected"].mean() for c in combos], height=0.7,
            color=[SERIES[STAGE_ORDER.index(s)] for s, _ in combos], edgecolor="white")
    a1.set_yticks(list(y), labels)
    a1.grid(axis="y", visible=False)
    a1.set_title("(a) Downstream units changed (mean)", loc="left", color=INK, fontsize=8.5, pad=3)

    w = 0.27
    metrics = [("faulty_success", "task failed", lambda s: 1 - s.mean()),
               ("unauthorized", "unauthorized action", lambda s: s.mean()),
               ("bad_emails", "wrong email (irreversible)", lambda s: (s > 0).mean())]
    for i, (col, name, fn) in enumerate(metrics):
        vals = [100 * fn(g.get_group(c)[col].astype(float)) for c in combos]
        a2.barh([v + (1 - i) * w for v in y], vals, height=w, color=SERIES[i], label=name,
                edgecolor="white", linewidth=0.5)
    a2.set_yticks(list(y), [""] * len(combos))
    a2.tick_params(axis="y", length=0)
    a2.set_xlim(0, 100)
    a2.grid(axis="y", visible=False)
    a2.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8, handlelength=1.2, borderaxespad=0)
    a2.set_title("(b) Fault runs without recovery (%)", loc="left", color=INK, fontsize=8.5, pad=3)
    fig.tight_layout()
    fig.savefig(out / "fig_propagation.png")
    plt.close(fig)


def fig_granularity(d: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(3.4, 2.2))
    w = 0.26
    metrics = [("precision", "precision"), ("recall", "recall"), ("recompute_frac", "recompute cost / full")]
    for i, (m, name) in enumerate(metrics):
        vals = [d[f"{g}_{m}"].mean() for g in GRAN]
        bars = _bar(ax, [x + (i - 1) * w for x in range(3)], vals, SERIES[i], w, name)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.2f}", ha="center", va="bottom",
                    fontsize=6, color=INK2)
    ax.set_xticks(range(3), [f"{g}\n({d[f'{g}_book_items'].mean():.0f} dep. items)" for g in GRAN])
    ax.set_ylim(0, 1.12)
    ax.grid(axis="x", visible=False)
    ax.legend(ncol=3, fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, 1.2))
    fig.savefig(out / "fig_granularity.png")
    plt.close(fig)


def tables(d: pd.DataFrame, meta: dict, out: Path) -> str:
    if "unauthorized" not in d:
        d = derive(d)
    lines = []
    tasks = meta["tasks"]
    clean_tokens = {t["task_id"]: t["clean_tokens"] for t in tasks}
    denominator = d["clean_tokens"] if "clean_tokens" in d else d.task_id.map(clean_tokens)
    lines.append("Recompute fractions are token estimates, not measured selective recovery.")
    lines.append("Measurement mode: " + meta.get("measurement_mode", "legacy; latency unavailable"))
    if meta.get("verification_repair"):
        lines.append("Always-verify repair: " + meta["verification_repair"])
    legacy = meta.get("legacy_from")
    if legacy and legacy.get("fields"):
        lines.append(f"Carried over from {legacy['dir']} without re-scoring ({legacy['note']}): "
                     + ", ".join(legacy["fields"]))
    if meta.get("failures"):
        lines.append("INCOMPLETE EXPERIMENT: failures are recorded in tasks.json; do not treat as a full matrix.")
    lines.append(f"clean runs: {sum(t['clean_success'] for t in tasks)}/{len(tasks)} pass the checker")
    noise = [n for t in tasks for n in t.get("noise", [])]
    if noise:
        lines.append(f"uncached clean reruns: {len(noise)}, mean units differing from cached clean run "
                     f"{sum(n['units_differ'] for n in noise) / len(noise):.2f}, success "
                     f"{sum(n['success'] for n in noise)}/{len(noise)}")
    cv = [t["clean_verify"] for t in tasks if "clean_verify" in t]
    if cv:
        fp = sum(c["verify_false_pos"] for c in cv) / max(1, sum(c["verify_checked"] for c in cv))
        lines.append(f"always-verify on clean runs: flag rate per non-injected output {fp:.3f} "
                     "(not an adjudicated false-positive rate)")
    lines.append(f"fault runs injected: {len(d)}; benign (no downstream change): {d.fault_benign.mean():.2%}")

    by_stage = d.groupby("stage").agg(
        runs=("task_id", "size"), downstream=("downstream_affected", "mean"),
        fail=("faulty_success", lambda s: 1 - s.mean()), unauthorized=("unauthorized", "mean"),
        attacker=("faulty_safety_violation", "mean"),
        wrong_email=("bad_emails", lambda s: (s > 0).mean())).reindex(STAGE_ORDER)
    lines.append("\nPer injection stage:\n" + by_stage.round(3).to_string())
    if "fault_class" in d:
        classes = d.groupby("fault_class").agg(runs=("task_id", "size"),
            success=("faulty_success", "mean"), unauthorized=("unauthorized", "mean"),
            attacker=("faulty_safety_violation", "mean"),
            propagated=("downstream_affected", lambda s: (s > 0).mean()),
            downstream=("downstream_affected", "mean"))
        lines.append("\nAccidental and adversarial cases separately:\n" + classes.round(3).to_string())
    if "faulty_unauthorized_payments" in d:
        unauth = d.groupby(["stage", "ftype"]).agg(
            attacker=("faulty_safety_violation", "mean"),
            unknown_account=("faulty_unauthorized_payments", lambda s: (s > 0).mean()),
            unexpected_recipient=("faulty_emails_unexpected", lambda s: (s > 0).mean()),
            overpaid=("faulty_overpaid", lambda s: (s > 0).mean()))
        lines.append("\nUnauthorized actions without recovery (unknown account or unexpected recipient;"
                     " attacker is a subset; overpayment to a legitimate payee reported separately):\n"
                     + unauth.round(2).to_string())
    dups = int(d.duplicate.sum())
    lines.append(f"\nDuplicate placements (same injected content as seed 0, replayed from cache): {dups}; "
                 f"independent fault runs: {len(d) - dups}")
    if "clean_success" in d:
        capable = d[d.clean_success.astype(bool)]
        cols = ["faulty_success", "faulty_safety_violation"]
        cols += [c for c in ("verify_success", "verify_safety_violation", "restart_state_ok") if c in capable]
        lines.append(f"\nClean-capable subset: {len(capable)}/{len(d)} fault cases. "
                     "Other cases remain in the full results; inability is not attack resistance.")
        if len(capable):
            lines.append(capable.groupby("fault_class")[cols].mean().round(3).to_string())
    for col in ("faulty_elapsed_s", "restart_elapsed_s", "verify_elapsed_s"):
        if col in d:
            lines.append(f"{col}: median={d[col].median():.4f}, p95={d[col].quantile(.95):.4f}; "
                         "cache hits are recorded per run; cached timings are replay timings")

    gran = pd.DataFrame({g: {
        "precision": d[f"{g}_precision"].mean(), "recall": d[f"{g}_recall"].mean(),
        "under_runs": (d[f"{g}_under"] > 0).mean(), "over_units": d[f"{g}_over"].mean(),
        "recompute_frac": d[f"{g}_recompute_frac"].mean(), "dep_items": d[f"{g}_book_items"].mean(),
        "dep_bytes": d[f"{g}_book_bytes"].mean()} for g in GRAN}).T
    lines.append("\nGranularity (all runs; recall counts the faulty unit, which every policy predicts):\n"
                 + gran.round(3).to_string())
    if "fact_ds_recall" in d:
        for name, sub in (("propagated runs", d[d.downstream_affected > 0]),
                          ("propagated, duplicates removed", d[(d.downstream_affected > 0) & ~d.duplicate])):
            ds = pd.DataFrame({g: {
                "precision": sub[f"{g}_ds_precision"].mean(), "recall": sub[f"{g}_ds_recall"].mean(),
                "under_runs": (sub[f"{g}_ds_under"] > 0).mean(), "over_units": sub[f"{g}_ds_over"].mean(),
                "recompute_frac": sub[f"{g}_recompute_frac"].mean()} for g in GRAN}).T
            lines.append(f"\nGranularity, downstream units only, {name} (n={len(sub)}):\n" + ds.round(3).to_string())
    gs = d.groupby("stage")[[f"{g}_recall" for g in GRAN] + [f"{g}_precision" for g in GRAN]].mean()
    lines.append("\nGranularity by stage:\n" + gs.reindex(STAGE_ORDER).round(3).to_string())

    base = {"no recovery": {"state_ok": 1 - ((d.faulty_missing_payments + d.faulty_extra_payments
                                              + d.faulty_kv_wrong + d.faulty_files_wrong) > 0).mean(),
                            "unauthorized": d.unauthorized.mean(), "attacker": d.faulty_safety_violation.mean(),
                            "wrong_email_runs": (d.bad_emails > 0).mean(), "extra_tokens": 0.0},
            "full restart": {"state_ok": d.restart_state_ok.mean(),
                             "unauthorized": d.get("restart_unauthorized", d.restart_safety_violation).mean(),
                             "attacker": d.restart_safety_violation.mean(),
                             "extra_tokens": (d.restart_tokens / denominator).mean(),
                             "unrecovered_email_runs": (d.restart_unrecovered_emails > 0).mean(),
                             "duplicate_emails": d.restart_duplicate_emails.mean()}}
    if "verify_detected" in d:
        base["always verify"] = {
            "state_ok": d.verify_state_ok.mean(),
            "unauthorized": d.get("verify_unauthorized", d.verify_safety_violation).mean(),
            "attacker": d.verify_safety_violation.mean(),
            "extra_tokens": (d.verify_overhead_tokens / denominator).mean(),
            "detected": d.verify_detected.mean()}
        if "verify_bad_emails" in d:
            base["always verify"]["wrong_email_runs"] = (d.verify_bad_emails > 0).mean()
    lines.append("\nBaselines (extra_tokens relative to one clean run):\n"
                 + pd.DataFrame(base).T.round(3).to_string())
    if "verify_detected" in d:
        lines.append("\nAlways-verify detection by stage x type:\n"
                     + d.groupby(["stage", "ftype"]).verify_detected.mean().round(2).to_string())
    text = "\n".join(line.rstrip() for line in "\n".join(lines).splitlines())
    (out / "summary.txt").write_text(text, encoding="utf-8")
    return text


def main() -> None:
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "results/phase1_haiku")
    df = pd.read_csv(out / "runs.csv")
    d = derive(df[df.injected])
    meta = json.loads((out / "tasks.json").read_text(encoding="utf-8"))
    fig_propagation(d, out)
    fig_granularity(d, out)
    print(tables(d, meta, out))


if __name__ == "__main__":
    main()
