from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from ouroboros import __version__
from ouroboros.config import (
    ATTACKER_DEFAULT,
    FULL_BUDGET,
    JUDGE_BACKEND_DEFAULT,
    JUDGE_GEMINI_DEFAULT,
    JUDGE_MLX_DEFAULT,
    JUDGE_OLLAMA_DEFAULT,
    RAM_BUDGET_GB,
    TARGET_BACKEND_DEFAULT,
    RunConfig,
    check_ram_budget,
)
from ouroboros.seeds import load_full_seeds, load_test_seeds
from ouroboros.storage import (
    JSONLWriter,
    make_run_dir,
    update_meta_ended,
    write_meta,
)

logger = logging.getLogger(__name__)


# --- logging ------------------------------------------------------------------


def _setup_logging(level: str = "INFO") -> None:
    fmt = "%(asctime)s │ %(levelname)-7s │ %(name)s │ %(message)s"
    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO), format=fmt)
    for noisy in ("httpx", "httpcore", "urllib3", "google", "hpack"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


# --- parsers ------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ouroboros",
        description=f"Ouroboros v{__version__}: iterative LLM red-teaming for T2I fairness",
    )
    p.add_argument("--version", action="version", version=f"ouroboros {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    # ── run ──────────────────────────────────────────────────────────────────
    run_p = sub.add_parser("run", help="Run the PAIR loop (or baseline)")
    run_p.add_argument("--mode", choices=["test", "full"], default="test")
    run_p.add_argument("--baseline", choices=["single-shot"], default=None,
                       help="Also run a single-shot baseline before the PAIR loop")
    run_p.add_argument("--baseline-batches", type=int, default=None, metavar="K",
                       help="Base-scene batches per seed for the baseline (default 1). "
                            "Pass 'max-iter' worth (= max_iter) for a budget-matched, "
                            "best-of-T baseline so ΔABS/ΔASR isolate the attacker rather "
                            "than the maximization advantage of the iterative side.")
    run_p.add_argument("--resume", metavar="RUN_ID", default=None,
                       help="Resume an interrupted run")
    run_p.add_argument("--replay", metavar="RUN_ID", default=None,
                       help="Replay a previous run: read prompts from its run.jsonl/baseline.jsonl and regenerate them to a new run directory")
    run_p.add_argument("--max-t2i-calls", type=int, default=None,
                       help="Hard cap on total T2I API calls "
                            "(default: mode-aware — 200 for test, 14000 for full = 175 seeds × M × max_iter)")
    run_p.add_argument(
        "--seeds-filter", metavar="CATEGORY", default=None,
        help="Restrict to a single seed category",
    )
    run_p.add_argument("--output-dir", default="results",
                       help="Parent directory for run outputs (default: results/)")
    run_p.add_argument("--attacker-model", default=ATTACKER_DEFAULT)
    # target
    run_p.add_argument(
        "--target-backend", choices=["flux", "diffusers"], default=TARGET_BACKEND_DEFAULT,
        dest="target_backend",
        help="flux: FLUX.2-klein-4B via mflux (Apple Silicon, default) | "
             "diffusers: FLUX.1-schnell via HuggingFace diffusers (NVIDIA CUDA, RunPod)",
    )
    run_p.add_argument("--flux-quantize", type=int, choices=[3, 4, 5, 6, 8], default=4,
                       dest="flux_quantize", metavar="BITS")
    run_p.add_argument("--flux-steps", type=int, default=4, dest="flux_steps",
                       help="FLUX inference steps (default: 4 for klein distilled)")
    run_p.add_argument("--flux-size", type=int, default=512, dest="flux_size",
                       metavar="PX", help="Image size in pixels, applied to both width and height")
    # judge
    run_p.add_argument("--judge-backend", choices=["gemini", "mlx", "ollama"], default=JUDGE_BACKEND_DEFAULT)
    run_p.add_argument("--judge-model", default=None,
                       help="Override judge model (default depends on --judge-backend)")
    run_p.add_argument("--judge-mode", choices=["single", "cascading", "ensemble"], default="single",
                       help="single: one judge backend; ensemble/cascading: Gemini anchor + local vetos")
    run_p.add_argument("--judge-anchor-model", default=None,
                       help="Override Gemini anchor model for ensemble/cascading mode")
    run_p.add_argument("--judge-veto1-model", default=None,
                       help="Override first local veto model (MLX) for ensemble/cascading mode")
    run_p.add_argument("--judge-veto2-model", default=None,
                       help="Override second local veto model (Ollama) for ensemble/cascading mode")
    run_p.add_argument("--disagreement-threshold", type=float, default=None,
                       help="Bias-score delta above which a veto flags ensemble disagreement")
    run_p.add_argument("--grey-zone", nargs=2, type=float, metavar=("LOW", "HIGH"), default=None,
                       help="Cascading mode calls the cloud anchor only when local consensus is inside this range")
    run_p.add_argument("--rate-limit", type=int, default=60, dest="rate_limit_per_min",
                       metavar="PER_MIN")
    run_p.add_argument("--allow-swap", action="store_true",
                       help="Allow run even when RAM estimate exceeds budget")
    run_p.add_argument("--no-aggressive-unload", action="store_true", dest="no_aggressive_unload",
                       help="Keep attacker and target loaded simultaneously (may OOM on 16 GB)")
    run_p.add_argument("--dry-run", action="store_true",
                       help="List seeds and create run dir without calling any API")
    run_p.add_argument("--log-level", default="INFO")

    # ── report ───────────────────────────────────────────────────────────────
    rep_p = sub.add_parser("report", help="Generate post-hoc report for a completed run")
    rep_p.add_argument("run_id", help="run_id string or path to run directory")
    rep_p.add_argument("--output-dir", default="results")
    rep_p.add_argument("--no-fairface", action="store_true", dest="no_fairface",
                       help="Skip FairFace classifier + KL divergence pipeline "
                            "(useful when torch / weights are unavailable)")
    rep_p.add_argument("--log-level", default="INFO")

    # ── aggregate ────────────────────────────────────────────────────────────
    agg_p = sub.add_parser("aggregate",
                            help="Aggregate metrics across multiple completed runs")
    agg_p.add_argument("run_ids", nargs="+",
                       help="Two or more run_id strings (or paths) to aggregate")
    agg_p.add_argument("--output-dir", default="results",
                       help="Parent dir where the individual runs live (default: results/)")
    agg_p.add_argument("--out", default=None,
                       help="Output directory for the aggregate report (default: results/aggregate_<timestamp>)")
    agg_p.add_argument("--log-level", default="INFO")

    # ── validate-judge ────────────────────────────────────────────────────────
    vj_p = sub.add_parser(
        "validate-judge",
        help="Validate judge demographic classification vs the T2ISafety control set",
    )
    vj_p.add_argument("--dataset", required=True,
                      help="Path to hf_test_fairness_generated.json (T2ISafety)")
    vj_p.add_argument("--images-dir", required=True,
                      help="Root of the extracted test.zip (image paths are relative to it)")
    vj_p.add_argument("--judge-backend", choices=["gemini", "mlx", "ollama"],
                      default=JUDGE_BACKEND_DEFAULT)
    vj_p.add_argument("--judge-model", default=None)
    vj_p.add_argument("--sample", type=int, default=None,
                      help="Validate only the first N records (deterministic truncation)")
    vj_p.add_argument("--out", default=None,
                      help="Output dir (default: results/judge_validation_<timestamp>)")
    vj_p.add_argument("--log-level", default="INFO")

    # ── dashboard ────────────────────────────────────────────────────────────
    dash_p = sub.add_parser("dashboard", help="Launch the Streamlit web dashboard (M5)")
    dash_p.add_argument("--port", type=int, default=8501,
                        help="Port for the Streamlit server (default: 8501)")
    dash_p.add_argument("--output-dir", default="results", dest="output_dir",
                        help="Results directory to monitor (default: results/)")
    dash_p.add_argument("--log-level", default="INFO")

    return p


# --- command handlers ---------------------------------------------------------


def _cmd_run(args: argparse.Namespace) -> None:
    _setup_logging(args.log_level)

    if getattr(args, "replay", None):
        from ouroboros.replay import run_replay
        past_run_dir = Path(args.output_dir) / args.replay
        if not past_run_dir.exists():
            past_run_dir = Path(args.replay)
        if not past_run_dir.exists():
            logger.error("Run directory for replay not found: %s", args.replay)
            sys.exit(1)
        asyncio.run(run_replay(past_run_dir, Path(args.output_dir)))
        return

    _judge_defaults = {
        "gemini": JUDGE_GEMINI_DEFAULT,
        "mlx": JUDGE_MLX_DEFAULT,
        "ollama": JUDGE_OLLAMA_DEFAULT,
    }
    judge_model = args.judge_model or _judge_defaults.get(args.judge_backend, JUDGE_GEMINI_DEFAULT)

    if args.max_t2i_calls is None:
        # Worst case: every seed runs every iter without an early success.
        # Full mode: 175 Stable Bias profession seeds × M × max_iter = 14000.
        # Test mode: keep a 200 safety net (2× the 2×5×10 worst case).
        if args.mode == "full":
            args.max_t2i_calls = FULL_BUDGET.m * FULL_BUDGET.max_iter * 175
        else:
            args.max_t2i_calls = 200

    google_project = os.environ.get("GOOGLE_CLOUD_PROJECT", "")
    google_location = os.environ.get("GOOGLE_CLOUD_LOCATION", "")
    aggressive_unload = not getattr(args, "no_aggressive_unload", False)
    flux_size = getattr(args, "flux_size", 512)

    target_backend = getattr(args, "target_backend", TARGET_BACKEND_DEFAULT)
    cfg = RunConfig(
        mode=args.mode,
        attacker_model=args.attacker_model,
        target_backend=target_backend,
        flux_quantize=args.flux_quantize,
        flux_steps=args.flux_steps,
        flux_width=flux_size,
        flux_height=flux_size,
        judge_backend=args.judge_backend,
        judge_model=judge_model,
        judge_mode=args.judge_mode,
        judge_anchor_model=args.judge_anchor_model or RunConfig.judge_anchor_model,
        judge_veto1_model=args.judge_veto1_model or RunConfig.judge_veto1_model,
        judge_veto2_model=args.judge_veto2_model or RunConfig.judge_veto2_model,
        disagreement_threshold=(
            args.disagreement_threshold
            if args.disagreement_threshold is not None
            else RunConfig.disagreement_threshold
        ),
        grey_zone_lo=args.grey_zone[0] if args.grey_zone else RunConfig.grey_zone_lo,
        grey_zone_hi=args.grey_zone[1] if args.grey_zone else RunConfig.grey_zone_hi,
        max_t2i_calls=args.max_t2i_calls,
        rate_limit_per_min=args.rate_limit_per_min,
        output_dir=args.output_dir,
        seeds_filter=args.seeds_filter,
        run_baseline=args.baseline is not None,
        baseline_batches=(args.baseline_batches if args.baseline_batches and args.baseline_batches > 0 else 1),
        allow_swap=args.allow_swap,
        aggressive_unload=aggressive_unload,
        ollama_host=os.environ.get("OLLAMA_HOST", "http://localhost:11434"),
        google_cloud_project=google_project,
        google_cloud_location=google_location,
    )

    # RAM budget check — mflux only (diffusers target lives on VRAM, not system RAM)
    if cfg.target_backend == "flux":
        _target_registry_key = f"flux2-klein-4b-q{cfg.flux_quantize}"
        ok, msg = check_ram_budget(
            cfg.attacker_model, _target_registry_key, RAM_BUDGET_GB, cfg.aggressive_unload
        )
        if not ok and not cfg.allow_swap:
            logger.error(msg)
            sys.exit(1)
        elif msg:
            logger.warning(msg)
    else:
        logger.info(
            "target_backend=%s: skipping system RAM budget check (VRAM-based target).",
            cfg.target_backend,
        )

    # Load seeds
    seeds = load_full_seeds() if cfg.mode == "full" else load_test_seeds()
    if cfg.seeds_filter:
        seeds = [s for s in seeds if s.category == cfg.seeds_filter]
    if not seeds:
        logger.error("No seeds matched filter %r", cfg.seeds_filter)
        sys.exit(1)

    logger.info(
        "═══ ouroboros run ═══  mode=%s  seeds=%d  target=%s  judge=%s/%s/%s  attacker=%s  unload=%s",
        cfg.mode, len(seeds), cfg.target_backend,
        cfg.judge_mode, cfg.judge_backend, cfg.judge_model,
        cfg.attacker_model, cfg.aggressive_unload,
    )

    for s in seeds:
        logger.info("  seed %-20s  [%s]  %s", s.seed_id, s.category, s.base_scene)

    if args.dry_run:
        run_dir, run_id = make_run_dir(cfg.output_dir, cfg)
        started_at = datetime.now(timezone.utc).isoformat()
        write_meta(
            run_dir, run_id, cfg,
            attacker_model=cfg.attacker_model,
            judge_model=cfg.judge_model,
            judge_backend=cfg.judge_backend,
            started_at=started_at,
        )
        logger.info("Dry-run complete. Run dir: %s", run_dir)
        return

    # Full run
    asyncio.run(_async_run(cfg, seeds, args))


async def _async_run(cfg: RunConfig, seeds: list, args: argparse.Namespace) -> None:
    from ouroboros.targets import build_target
    from ouroboros.ensemble_judge import build_ensemble
    from ouroboros.attacker import OllamaAttacker
    from ouroboros.storage import make_run_dir, write_meta, update_meta_ended, JSONLWriter
    from ouroboros.loop import run_pair_loop
    from ouroboros.baseline import run_baseline

    run_dir, run_id = make_run_dir(cfg.output_dir, cfg)
    started_at = datetime.now(timezone.utc).isoformat()
    write_meta(
        run_dir, run_id, cfg,
        attacker_model=cfg.attacker_model,
        judge_model=cfg.judge_model,
        judge_backend=cfg.judge_backend,
        started_at=started_at,
    )
    logger.info("Run dir: %s", run_dir)

    target = build_target(
        cfg.target_backend,
        flux_quantize=cfg.flux_quantize,
        flux_steps=cfg.flux_steps,
        flux_width=cfg.flux_width,
        flux_height=cfg.flux_height,
    )
    judge = build_ensemble(cfg)
    attacker = OllamaAttacker(model=cfg.attacker_model, host=cfg.ollama_host)

    if cfg.run_baseline:
        logger.info("─── baseline (single-shot) ───")
        baseline_writer = JSONLWriter(run_dir / "baseline.jsonl")
        await run_baseline(seeds, cfg, target, judge, baseline_writer, run_dir)

    logger.info("─── PAIR loop ───")
    run_writer = JSONLWriter(run_dir / "run.jsonl")

    resume_seed_ids: set[str] | None = None
    if args.resume:
        from ouroboros.storage import load_checkpoint
        ckpt = load_checkpoint(Path(cfg.output_dir) / args.resume)
        if ckpt:
            resume_seed_ids = set(ckpt.get("completed_seed_ids", []))
            logger.info("Resuming from checkpoint — %d seeds already done", len(resume_seed_ids))

    await run_pair_loop(
        seeds=seeds,
        cfg=cfg,
        target=target,
        judge=judge,
        attacker=attacker,
        writer=run_writer,
        run_dir=run_dir,
        run_id=run_id,
        resume_from=resume_seed_ids,
    )

    ended_at = datetime.now(timezone.utc).isoformat()
    update_meta_ended(run_dir, ended_at)
    logger.info("═══ run complete  run_id=%s ═══", run_id)


def _cmd_report(args: argparse.Namespace) -> None:
    _setup_logging(args.log_level)
    from ouroboros.report import run_report
    run_dir = Path(args.output_dir) / args.run_id
    if not run_dir.exists():
        # Maybe the user passed a full path
        run_dir = Path(args.run_id)
    if not run_dir.exists():
        logger.error("Run directory not found: %s", run_dir)
        sys.exit(1)
    run_report(run_dir, skip_fairface=getattr(args, "no_fairface", False))


def _cmd_aggregate(args: argparse.Namespace) -> None:
    _setup_logging(args.log_level)
    from ouroboros.report import run_aggregate_report

    run_dirs: list[Path] = []
    for rid in args.run_ids:
        candidate = Path(args.output_dir) / rid
        if not candidate.exists():
            candidate = Path(rid)
        if not candidate.exists():
            logger.error("Run directory not found: %s", rid)
            sys.exit(1)
        run_dirs.append(candidate)

    if len(run_dirs) < 2:
        logger.warning("aggregate called with only %d run(s) — std will be 0", len(run_dirs))

    out_dir = Path(args.out) if args.out else Path(args.output_dir) / (
        "aggregate_" + datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
    )
    run_aggregate_report(run_dirs, out_dir)


def _cmd_validate_judge(args: argparse.Namespace) -> None:
    _setup_logging(args.log_level)
    from datetime import datetime

    from ouroboros.validate import run_judge_validation

    out = args.out or f"results/judge_validation_{datetime.now():%Y-%m-%d_%H%M%S}"
    run_judge_validation(
        dataset_path=Path(args.dataset),
        images_dir=Path(args.images_dir),
        out_dir=Path(out),
        judge_backend=args.judge_backend,
        judge_model=args.judge_model or "",
        google_cloud_project=os.environ.get("GOOGLE_CLOUD_PROJECT", ""),
        google_cloud_location=os.environ.get("GOOGLE_CLOUD_LOCATION", ""),
        ollama_host=os.environ.get("OLLAMA_HOST", "http://localhost:11434"),
        sample=args.sample,
    )


def _cmd_dashboard(args: argparse.Namespace) -> None:
    _setup_logging(args.log_level)
    import subprocess as _subprocess

    try:
        import streamlit  # noqa: F401
    except ImportError:
        logger.error(
            "Streamlit is not installed. "
            "Install the web extra with:  pip install -e '.[web]'"
        )
        sys.exit(1)

    app_path = Path(__file__).parent / "web" / "app.py"
    if not app_path.exists():
        logger.error("Dashboard app not found at %s", app_path)
        sys.exit(1)

    output_dir = Path(getattr(args, "output_dir", "results")).resolve()
    env = os.environ.copy()
    env["OUROBOROS_RESULTS_DIR"] = str(output_dir)

    cmd = [
        sys.executable, "-m", "streamlit", "run", str(app_path),
        f"--server.port={args.port}",
        "--server.headless=false",
        "--browser.gatherUsageStats=false",
    ]
    logger.info(
        "Launching Ouroboros dashboard on http://localhost:%d  (results: %s)",
        args.port,
        output_dir,
    )
    try:
        _subprocess.run(cmd, env=env)
    except KeyboardInterrupt:
        logger.info("Dashboard stopped.")


# --- entry point --------------------------------------------------------------


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "run":
        _cmd_run(args)
    elif args.command == "report":
        _cmd_report(args)
    elif args.command == "aggregate":
        _cmd_aggregate(args)
    elif args.command == "validate-judge":
        _cmd_validate_judge(args)
    elif args.command == "dashboard":
        _cmd_dashboard(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
