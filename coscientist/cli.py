#!/usr/bin/env python3
"""
Cosci CLI – lightweight CLI harness to drive post-goal steps.

Usage examples:
  - List goals:              python -m coscientist.cli goals
  - New goal dir only:       python -m coscientist.cli new --goal "..."
  - Start (LR+gen+tournament): python -m coscientist.cli start --goal "..." --n 4
  - Run single action:       python -m coscientist.cli step --goal "..." --action generate_new_hypotheses --n 2
  - Run full supervisor loop:python -m coscientist.cli run --goal "..." --max-iter 10
  - Checkpoints:             python -m coscientist.cli checkpoints --goal "..."
  - Clear goal dir:          python -m coscientist.cli clear --goal "..."

Notes:
  - This CLI is designed for debugging after goal confirmation.
  - It does not mock network: ensure provider API keys are set before running
    actions that invoke LLMs or GPTResearcher.
  - For quick smoke tests without long runs, use --pause-after-lr and small --n.
"""

import argparse
import asyncio
import os
import sys
from typing import Optional, Any

try:
    # Load .env if present but do not fail if package missing
    from dotenv import load_dotenv  # type: ignore
except Exception:  # pragma: no cover - optional
    load_dotenv = None  # type: ignore

# Lazy imports inside functions to avoid heavy deps on --help
CoscientistConfig = Any
CoscientistFramework = Any
CoscientistState = Any
CoscientistStateManager = Any


def _maybe_load_env():
    """Load .env from project root if python-dotenv is available."""
    if load_dotenv is not None:
        # Try workspace root and current working directory
        for candidate in [
            os.path.join(os.getcwd(), ".env"),
            os.path.join(os.path.dirname(__file__), "..", ".env"),
        ]:
            try:
                if os.path.exists(candidate):
                    load_dotenv(candidate)
                    break
            except Exception:
                pass


def _import_runtime_deps() -> None:
    global CoscientistConfig, CoscientistFramework, CoscientistState, CoscientistStateManager
    if CoscientistConfig is Any:
        from coscientist.framework import CoscientistConfig, CoscientistFramework  # type: ignore
        from coscientist.global_state import CoscientistState, CoscientistStateManager  # type: ignore


def _build_config(
    debug: bool,
    pause_after_lr: bool,
    save_on_error: bool,
    max_iter: int,
) -> CoscientistConfig:
    return CoscientistConfig(
        debug_mode=debug,
        pause_after_literature_review=pause_after_lr,
        save_on_error=save_on_error,
        max_supervisor_iterations=max_iter,
    )


def _ensure_state(goal: str, fresh: bool, checkpoint_path: Optional[str] = None) -> CoscientistState:
    """Create or load state directory for goal."""
    _import_runtime_deps()

    if checkpoint_path:
        # Load specific checkpoint
        return CoscientistState.load(checkpoint_path)
    if fresh:
        # Clear if exists to avoid FileExistsError from constructor
        try:
            CoscientistState.clear_goal_directory(goal)
        except Exception:
            pass
        return CoscientistState(goal=goal)

    # Try to resume from latest, else create new
    st = CoscientistState.load_latest(goal=goal)
    return st if st is not None else CoscientistState(goal=goal)


async def _run_start(framework: CoscientistFramework, n: int) -> None:
    await framework.start(n_hypotheses=n)


def _preflight_checks(fw: Any, action: str, n: Optional[int], k: Optional[int]) -> None:
    sm = fw.state_manager
    if action == "run_tournament":
        if sm.num_tournament_hypotheses < 2:
            raise ValueError(
                "Not enough hypotheses in tournament. Suggestion: run 'reflect' to move reviewed items, or 'generate_new_hypotheses' first."
            )
    if action == "evolve_hypotheses":
        try:
            eligible = sm.get_tournament_hypotheses_for_evolution()
        except Exception as e:
            raise ValueError(f"Tournament not ready for evolution: {e}")
        if len(eligible) < max(2, (n or 4)):
            raise ValueError(
                "Not enough qualified hypotheses for evolution. Run 'run_tournament' first and ensure matches were played."
            )


async def _run_step(framework: CoscientistFramework, action: str, n: Optional[int], k: Optional[int]) -> None:
    # Map optional numeric flags to method signatures
    if not hasattr(framework, action):
        raise ValueError(f"Unknown action: {action}. Available: {framework.available_actions()}")

    method = getattr(framework, action)

    _preflight_checks(framework, action, n, k)

    if action == "generate_new_hypotheses":
        await method(n_hypotheses=n or 2)
    elif action == "reflect":
        await method()
    elif action == "evolve_hypotheses":
        await method(n_hypotheses=n or 4)
    elif action == "expand_literature_review":
        await method()
    elif action == "run_tournament":
        await method(k_bracket=k or 8)
    elif action == "run_meta_review":
        await method(k_bracket=k or 8)
    elif action == "finish":
        await method()
    else:
        # Fallback for any new actions
        await method()


async def _run_full(framework: CoscientistFramework) -> None:
    await framework.run()


def _print_env_warnings():
    missing = []
    for key in ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY", "TAVILY_API_KEY"]:
        if not os.getenv(key):
            missing.append(key)
    if missing:
        print(
            f"[warn] Missing env vars for providers: {', '.join(missing)}\n"
            "       Some phases may fail if the selected LLM/tooling needs them.",
            file=sys.stderr,
        )


def cmd_goals(_: argparse.Namespace) -> int:
    goals = CoscientistState.list_all_goals()
    if not goals:
        print("No goals found under COSCIENTIST_DIR")
        return 0
    for original, h in goals:
        print(f"{original} -> {h}")
    return 0


def cmd_new(args: argparse.Namespace) -> int:
    goal = args.goal.strip()
    st = _ensure_state(goal, fresh=True)
    print(f"Created state directory: {st._output_dir}")
    return 0


def cmd_clear(args: argparse.Namespace) -> int:
    goal = args.goal.strip()
    result = CoscientistState.clear_goal_directory(goal)
    print(result)
    return 0


def cmd_checkpoints(args: argparse.Namespace) -> int:
    goal = args.goal.strip()
    ckpts = CoscientistState.list_checkpoints(goal=goal)
    if not ckpts:
        print("No checkpoints found")
        return 0
    for p in ckpts:
        print(p)
    return 0


def _build_framework(goal: str, fresh: bool, debug: bool, pause_after_lr: bool, save_on_error: bool, max_iter: int, checkpoint_path: Optional[str] = None) -> CoscientistFramework:
    _import_runtime_deps()
    st = _ensure_state(goal, fresh=fresh, checkpoint_path=checkpoint_path)
    sm = CoscientistStateManager(st)
    cfg = _build_config(debug=debug, pause_after_lr=pause_after_lr, save_on_error=save_on_error, max_iter=max_iter)
    return CoscientistFramework(cfg, sm)


def cmd_start(args: argparse.Namespace) -> int:
    _maybe_load_env()
    _print_env_warnings()
    fw = _build_framework(
        goal=args.goal.strip(),
        fresh=args.fresh,
        debug=args.debug,
        pause_after_lr=args.pause_after_lr,
        save_on_error=(not args.no_save_on_error),
        max_iter=args.max_iter,
        checkpoint_path=getattr(args, "checkpoint_path", None),
    )
    asyncio.run(_run_start(fw, n=args.n))
    return 0


def cmd_step(args: argparse.Namespace) -> int:
    _maybe_load_env()
    _print_env_warnings()
    fw = _build_framework(
        goal=args.goal.strip(),
        fresh=args.fresh,
        debug=args.debug,
        pause_after_lr=args.pause_after_lr,
        save_on_error=(not args.no_save_on_error),
        max_iter=args.max_iter,
        checkpoint_path=getattr(args, "checkpoint_path", None),
    )
    asyncio.run(_run_step(fw, action=args.action, n=args.n, k=args.k))
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    _maybe_load_env()
    _print_env_warnings()
    fw = _build_framework(
        goal=args.goal.strip(),
        fresh=args.fresh,
        debug=args.debug,
        pause_after_lr=args.pause_after_lr,
        save_on_error=(not args.no_save_on_error),
        max_iter=args.max_iter,
        checkpoint_path=getattr(args, "checkpoint_path", None),
    )
    asyncio.run(_run_full(fw))
    return 0


def cmd_actions(args: argparse.Namespace) -> int:
    # No heavy init to just show available actions
    fw = _build_framework(
        goal=args.goal.strip(),
        fresh=False,
        debug=False,
        pause_after_lr=False,
        save_on_error=True,
        max_iter=10,
    )
    print("Available actions:")
    for a in fw.available_actions():
        print(f"  - {a}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="cosci", description="Coscientist CLI test harness")
    sub = p.add_subparsers(dest="cmd", required=True)

    # goals
    s = sub.add_parser("goals", help="List all known goals")
    s.set_defaults(func=cmd_goals)

    # new
    s = sub.add_parser("new", help="Create a fresh goal directory")
    s.add_argument("--goal", required=True, help="Research goal string")
    s.set_defaults(func=cmd_new)

    # clear
    s = sub.add_parser("clear", help="Delete directory for goal")
    s.add_argument("--goal", required=True, help="Research goal string")
    s.set_defaults(func=cmd_clear)

    # checkpoints
    s = sub.add_parser("checkpoints", help="List checkpoints for a goal")
    s.add_argument("--goal", required=True, help="Research goal string")
    s.set_defaults(func=cmd_checkpoints)

    # start (LR + initial generation + tournament + meta-review)
    s = sub.add_parser("start", help="Run initial pipeline up to meta-review")
    s.add_argument("--goal", required=True, help="Research goal string")
    s.add_argument("-n", "--n", type=int, default=4, help="Initial hypotheses to target (>=2)")
    s.add_argument("--fresh", action="store_true", help="Start fresh (clear existing dir if any)")
    s.add_argument("--debug", action="store_true", help="Enable debug mode and verbose checkpoints")
    s.add_argument("--pause-after-lr", action="store_true", help="Pause after literature review")
    s.add_argument("--no-save-on-error", action="store_true", help="Disable auto save on errors")
    s.add_argument("--max-iter", type=int, default=50, help="Max supervisor iterations for run mode")
    s.add_argument("--checkpoint-path", help="Path to a specific .pkl checkpoint to resume from")
    s.set_defaults(func=cmd_start)

    # step (single action)
    s = sub.add_parser("step", help="Run a single framework action for targeted debugging")
    s.add_argument("--goal", required=True, help="Research goal string")
    s.add_argument("--action", required=True, choices=[
        "generate_new_hypotheses",
        "reflect",
        "evolve_hypotheses",
        "expand_literature_review",
        "run_tournament",
        "run_meta_review",
        "finish",
    ])
    s.add_argument("-n", "--n", type=int, default=None, help="Count argument for certain actions")
    s.add_argument("-k", "--k", type=int, default=None, help="K bracket for tournament/meta-review")
    s.add_argument("--fresh", action="store_true", help="Start from clean state directory")
    s.add_argument("--debug", action="store_true", help="Enable debug mode and verbose checkpoints")
    s.add_argument("--pause-after-lr", action="store_true", help="Pause after literature review")
    s.add_argument("--no-save-on-error", action="store_true", help="Disable auto save on errors")
    s.add_argument("--max-iter", type=int, default=50, help="Max supervisor iterations (if used)")
    s.add_argument("--checkpoint-path", help="Path to a specific .pkl checkpoint to resume from")
    s.set_defaults(func=cmd_step)

    # run (full loop controlled by supervisor)
    s = sub.add_parser("run", help="Run full supervisor-controlled loop")
    s.add_argument("--goal", required=True, help="Research goal string")
    s.add_argument("--fresh", action="store_true", help="Start fresh (clear existing dir if any)")
    s.add_argument("--debug", action="store_true", help="Enable debug mode and verbose checkpoints")
    s.add_argument("--pause-after-lr", action="store_true", help="Pause after literature review")
    s.add_argument("--no-save-on-error", action="store_true", help="Disable auto save on errors")
    s.add_argument("--max-iter", type=int, default=50, help="Max supervisor iterations")
    s.add_argument("--checkpoint-path", help="Path to a specific .pkl checkpoint to resume from")
    s.set_defaults(func=cmd_run)

    # actions (introspect)
    s = sub.add_parser("actions", help="Show available framework actions")
    s.add_argument("--goal", required=True, help="Research goal string")
    s.set_defaults(func=cmd_actions)

    return p


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)  # type: ignore[attr-defined]
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        return 130
    except Exception as e:
        print(f"[error] {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
