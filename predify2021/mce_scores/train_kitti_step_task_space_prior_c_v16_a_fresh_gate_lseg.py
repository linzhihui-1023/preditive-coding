"""C-V16-A: C-V15 frozen Proposal + fresh Gate + L_seg only."""

import argparse
import hashlib
import json
import pickle
import random
import time
from pathlib import Path

import torch

from predify2021.datasets.kitti_step import KITTISTEPSegmentationDataset
from predify2021.mce_scores.evaluate_kitti_step_dynamic_error_correction import sequence_groups
from predify2021.mce_scores.train_kitti_step_temporal_joint import FrozenRAFT
from predify2021.mce_scores import (
    c_v16_a_fresh_gate_lseg_training as training,
)
from predify2021.mce_scores import (
    train_kitti_step_task_space_prior_c_v15_proposal_conditioned_soft_acceptance as c_v15,
)
from predify2021.model_factory.deeplabv3plus_resnet50.task_space_c_v16_a_fresh_gate import (
    CV16AFreshGateCorrector,
)
from predify2021.model_factory.deeplabv3plus_resnet50.task_space_semantic_hysteresis import (
    EulerDynamicsError,
)


EXPERIMENT = "c_v16_a_fresh_gate_lseg"
C_V15_CHECKPOINT_DEFAULT = (
    "/home/lin/predify/experiments/"
    "kitti_step_task_space_prior_c_v15_proposal_conditioned_soft_acceptance/best.pt"
)
OUTPUT_DEFAULT = (
    "/home/lin/predify/experiments/"
    "kitti_step_task_space_prior_c_v16_a_fresh_gate_lseg"
)
RESULT_DEFAULT = "/home/lin/predify/results/kitti_step_task_space_prior_c_v16_a_fresh_gate_lseg"
EPOCHS = 3
GRADIENT_ACCUMULATION_STEPS = 8
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-2
SEED = 0


def _delta(candidate, reference):
    return {
        key: candidate[key] - reference[key]
        for key in ("mIoU", "mTC", "mVC8", "mVC16")
    }


def _selection_key(metrics):
    candidate = metrics["c_v16_a"]
    reference = metrics["c_v4_frozen"]
    floors_pass = (
        candidate["mIoU"] >= reference["mIoU"]
        and candidate["mTC"] >= reference["mTC"]
    )
    return (
        int(floors_pass),
        candidate["mIoU"],
        candidate["mTC"],
    )


def _load_c_v15_payload(path, wait, poll_seconds):
    path = Path(path)
    while True:
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
            if payload.get("experiment") != (
                "c_v15_proposal_conditioned_soft_acceptance"
            ):
                raise RuntimeError("C-V16-A requires a C-V15 checkpoint")
            if "corrector_state_dict" not in payload:
                raise RuntimeError("C-V15 checkpoint missing corrector_state_dict")
            if int(payload.get("epoch", 0)) <= 0:
                raise RuntimeError("C-V15 checkpoint has no completed epoch")
            return payload
        except FileNotFoundError:
            if not wait:
                raise RuntimeError(f"C-V15 checkpoint not found: {path}")
            print(
                json.dumps(
                    {
                        "status": "waiting_for_c_v15_checkpoint",
                        "checkpoint": str(path),
                        "poll_seconds": poll_seconds,
                    }
                ),
                flush=True,
            )
            time.sleep(poll_seconds)
        except (EOFError, RuntimeError, pickle.UnpicklingError) as exc:
            if not wait:
                raise
            print(
                json.dumps(
                    {
                        "status": "retrying_c_v15_checkpoint_load",
                        "checkpoint": str(path),
                        "error": str(exc),
                    }
                ),
                flush=True,
            )
            time.sleep(poll_seconds)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _c_v15_code_fingerprint():
    """Fingerprint only C-V15 source files; C-V15 output is live-owned by C-V15."""
    root = Path(__file__).resolve().parents[2]
    paths = (
        root / "mce_scores/c_v15_proposal_conditioned_soft_acceptance_training.py",
        root / "mce_scores/check_c_v15_dual_reference_acceptance_diagnostics.py",
        root / "mce_scores/check_c_v15_proposal_conditioned_soft_acceptance_contracts.py",
        root / "mce_scores/train_kitti_step_task_space_prior_c_v15_proposal_conditioned_soft_acceptance.py",
        root / "model_factory/deeplabv3plus_resnet50/task_space_proposal_conditioned_soft_acceptance.py",
    )
    return {str(path): _sha256(path) for path in paths if path.exists()}


def _architecture_metadata(corrector, cv15_path, cv15_payload, c_v4_epoch):
    return {
        "name": "C-V16-A Frozen C-V15 Proposal + Fresh Gate",
        "causal_path": "Frozen C-V15 Proposal -> Fresh Reliability Gate -> Corrected C4 -> Decoder",
        "proposal_source": "C-V15 proposal-conditioned soft acceptance checkpoint",
        "proposal_checkpoint": str(cv15_path),
        "proposal_checkpoint_epoch": int(cv15_payload["epoch"]),
        "proposal_generator_frozen": True,
        "proposal_modules": list(corrector.proposal_module_names()),
        "gate_modules_reset": list(corrector.gate_module_names()),
        "gate_semantics": "fresh gate jointly combines reset reliability prior + reset proposal-conditioned acceptance path",
        "gate_initialization": corrector.contract_state()["gate_initialization"],
        "training_objective": "L_seg only",
        "l_accept_backprop": False,
        "l_protect_backprop": False,
        "gt_beneficial_harmful_backprop": False,
        "rescue_ce": False,
        "temporal_loss": False,
        "raft_training": False,
        "frozen_c_v4_epoch": int(c_v4_epoch),
        "final_composition": "Decoder(c4 + g_t * frozen bounded C-V15 semantic Delta-c4)",
        "reporting_boundary": (
            "Tests whether a newly initialized Gate can learn spatial reliability "
            "from L_seg with the Proposal frozen; not joint from-scratch Proposal+Gate training."
        ),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="/home/lin/predify/kitti_step")
    parser.add_argument("--fast-b-checkpoint", default=c_v15.c_v5.FAST_B_CHECKPOINT_DEFAULT)
    parser.add_argument("--observer-checkpoint", default=c_v15.c_v5.OBSERVER_CHECKPOINT_DEFAULT)
    parser.add_argument("--residual-checkpoint", default=c_v15.c_v5.RESIDUAL_CHECKPOINT_DEFAULT)
    parser.add_argument("--base-checkpoint", default=c_v15.c_v5.BASE_CHECKPOINT_DEFAULT)
    parser.add_argument("--c-v3-checkpoint", default=c_v15.c_v5.C_V3_CHECKPOINT_DEFAULT)
    parser.add_argument("--c-v4-checkpoint", default=c_v15.training.C_V4_CHECKPOINT_DEFAULT)
    parser.add_argument("--c-v15-checkpoint", default=C_V15_CHECKPOINT_DEFAULT)
    parser.add_argument("--output", default=OUTPUT_DEFAULT)
    parser.add_argument("--result-output", default=RESULT_DEFAULT)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=GRADIENT_ACCUMULATION_STEPS)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--dynamics-tau-e", type=float, default=c_v15.c_v5.DYNAMICS_TAU_E)
    parser.add_argument("--dynamics-k-e", type=float, default=c_v15.c_v5.DYNAMICS_K_E)
    parser.add_argument("--dynamics-dt", type=float, default=c_v15.c_v5.DYNAMICS_DT)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--poll-seconds", type=float, default=15.0)
    parser.add_argument("--no-wait-for-c-v15", action="store_true")
    args = parser.parse_args(argv)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.epochs <= 0 or args.gradient_accumulation_steps <= 0:
        raise ValueError("epochs and gradient-accumulation-steps must be positive")
    if Path(args.output).resolve() == Path(c_v15.OUTPUT_DEFAULT).resolve():
        raise RuntimeError("C-V16-A output must be independent of C-V15 output")
    if Path(args.result_output).resolve() == Path(c_v15.RESULT_DEFAULT).resolve():
        raise RuntimeError("C-V16-A result output must be independent of C-V15 results")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    cv15_payload = _load_c_v15_payload(
        args.c_v15_checkpoint,
        wait=not args.no_wait_for_c_v15,
        poll_seconds=args.poll_seconds,
    )
    protected_before = _c_v15_code_fingerprint()

    c_v15.c_v5._validate_bounded_motion_checkpoint(args.residual_checkpoint)
    model = c_v15.c_v5.load_fast_b_model(args.fast_b_checkpoint).cuda().eval()
    model.requires_grad_(False)
    observer, _ = c_v15.c_v5._load_frozen_observer(args.observer_checkpoint)
    residual, residual_payload = c_v15.c_v5._load_frozen_residual(
        args.residual_checkpoint,
        observer,
    )
    correction, mask_predictor, base_payload = c_v15.c_v5._load_frozen_e1_base(
        args.base_checkpoint,
        observer,
    )
    refiner, cv3_payload = c_v15.c_v5._load_frozen_cv3_refiner(args.c_v3_checkpoint)
    c_v4_controller, cv4_payload = c_v15.c_v12_entry._load_frozen_c_v4_controller(
        args.c_v4_checkpoint
    )
    c_v4_epoch = int(cv4_payload["epoch"])

    corrector = CV16AFreshGateCorrector(
        num_classes=c_v15.c_v5.NUM_CLASSES,
        history_length=training.HISTORY_LENGTH,
        semantic_channels=training.SEMANTIC_CHANNELS,
        temporal_hidden_channels=c_v15.c_v5.CONTROLLER_HIDDEN_CHANNELS,
        host_channels=training.C4_CHANNELS,
        residual_scale=training.RESIDUAL_SCALE,
    ).cuda()
    # This call is deliberately one operation: load C-V15 first, then freeze
    # Proposal, then reset every deployed-Gate module.
    corrector.load_c_v15_state_dict(cv15_payload["corrector_state_dict"])
    initialization = corrector.contract_state()
    if not initialization["proposal_generator_frozen"]:
        raise RuntimeError("C-V16-A Proposal is not fully frozen")
    if initialization["proposal_parameters_receive_gradient"]:
        raise RuntimeError("C-V16-A Proposal parameters remain trainable")
    if not initialization["gate_initialized_after_checkpoint_loading"]:
        raise RuntimeError("C-V16-A Gate was not reset after checkpoint loading")

    dynamics = EulerDynamicsError(
        tau_e=args.dynamics_tau_e,
        k_e=args.dynamics_k_e,
        dt=args.dynamics_dt,
    )
    optimizer = torch.optim.AdamW(
        [parameter for parameter in corrector.parameters() if parameter.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    train_dataset = KITTISTEPSegmentationDataset.from_kitti_step_root(
        Path(args.root), "train"
    )
    val_dataset = KITTISTEPSegmentationDataset.from_kitti_step_root(
        Path(args.root), "val"
    )
    train_groups = c_v15.sequence_groups(train_dataset)
    all_val_groups = c_v15.sequence_groups(val_dataset)
    missing = [sequence for sequence in c_v15.c_v5.FULL9 if sequence not in all_val_groups]
    if missing:
        raise RuntimeError(f"Missing Full9 validation sequences: {missing}")
    val_groups = {sequence: all_val_groups[sequence] for sequence in c_v15.c_v5.FULL9}

    output = Path(args.output)
    result_output = Path(args.result_output)
    output.mkdir(parents=True, exist_ok=True)
    result_output.mkdir(parents=True, exist_ok=True)
    raft_metric = FrozenRAFT()
    history = []
    best = None

    for epoch in range(1, args.epochs + 1):
        train_stats = training.train_epoch(
            model,
            observer,
            residual,
            correction,
            mask_predictor,
            refiner,
            c_v4_controller,
            dynamics,
            corrector,
            train_groups,
            optimizer,
            args.gradient_accumulation_steps,
        )
        metrics, diagnostics = training.evaluate(
            model,
            observer,
            residual,
            correction,
            mask_predictor,
            refiner,
            c_v4_controller,
            dynamics,
            corrector,
            val_groups,
            raft_metric,
        )
        row = {
            "epoch": epoch,
            "train": train_stats,
            "metrics": metrics,
            "diagnostics": diagnostics,
            "delta_vs_host": _delta(metrics["c_v16_a"], metrics["host"]),
            "delta_vs_c_v3": _delta(metrics["c_v16_a"], metrics["c_v3_base"]),
            "delta_vs_c_v4": _delta(metrics["c_v16_a"], metrics["c_v4_frozen"]),
            "delta_vs_source_c_v15": _delta(
                metrics["c_v16_a"],
                cv15_payload["metrics"]["c_v15"],
            ),
        }
        row["selection_key"] = list(_selection_key(metrics))
        history.append(row)
        with (result_output / f"epoch_{epoch:03d}.json").open("w") as handle:
            json.dump(row, handle, indent=2)

        if best is None or tuple(row["selection_key"]) > tuple(best["selection_key"]):
            best = row
            torch.save(
                {
                    "experiment": EXPERIMENT,
                    "epoch": epoch,
                    "corrector_state_dict": corrector.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "metrics": metrics,
                    "diagnostics": diagnostics,
                    "train": train_stats,
                    "architecture": _architecture_metadata(
                        corrector,
                        args.c_v15_checkpoint,
                        cv15_payload,
                        c_v4_epoch,
                    ),
                    "training_supervision": _zero_supervision_contract(),
                    "initialization_contract": initialization,
                    "source_cv15_checkpoint": args.c_v15_checkpoint,
                    "source_cv15_epoch": int(cv15_payload["epoch"]),
                    "source_cv15_metrics": cv15_payload["metrics"]["c_v15"],
                    "dynamics": dynamics.config(),
                    "frozen_c_v4_epoch": c_v4_epoch,
                    "frozen_c_v4_metrics": metrics["c_v4_frozen"],
                },
                output / "best.pt",
            )
        print(json.dumps(row, indent=2), flush=True)

    if best is None:
        raise RuntimeError("C-V16-A produced no epoch result")

    protected_after = _c_v15_code_fingerprint()
    write_scope = {
        "c_v15_files_modified": protected_before != protected_after,
        "c_v15_checkpoint_modified": False,
        "c_v15_results_modified": False,
        "c_v15_write_scope": "read-only source checkpoint; no writes under C-V15 output/result paths",
        "c_v16_a_output_independent": True,
    }
    if write_scope["c_v15_files_modified"]:
        raise RuntimeError("C-V15 source files changed during C-V16-A")

    summary = {
        "experiment": "C-V16-A Frozen C-V15 Proposal + Fresh Gate + L_seg only",
        "research_question": (
            "Can a freshly initialized Gate learn effective spatial reliability "
            "from L_seg while the existing C-V15 Proposal generator is frozen?"
        ),
        "reporting_boundary": (
            "This does not establish joint from-scratch Proposal+Gate training; "
            "only the Gate is trained here."
        ),
        "best": best,
        "history": history,
        "architecture": _architecture_metadata(
            corrector,
            args.c_v15_checkpoint,
            cv15_payload,
            c_v4_epoch,
        ),
        "initialization_contract": initialization,
        "training_supervision": _zero_supervision_contract(),
        "write_scope_contract": write_scope,
        "source_cv15": {
            "checkpoint": args.c_v15_checkpoint,
            "epoch": int(cv15_payload["epoch"]),
            "metrics": cv15_payload["metrics"]["c_v15"],
        },
        "frozen_checkpoints": {
            "fast_b": args.fast_b_checkpoint,
            "observer": args.observer_checkpoint,
            "residual": args.residual_checkpoint,
            "e1_base": args.base_checkpoint,
            "c_v3_base": args.c_v3_checkpoint,
            "c_v4": args.c_v4_checkpoint,
            "c_v15": args.c_v15_checkpoint,
            "c_v3_checkpoint_epoch": cv3_payload.get("epoch"),
            "c_v4_checkpoint_epoch": c_v4_epoch,
            "residual_experiment": residual_payload.get("experiment"),
            "e1_experiment": base_payload.get("experiment"),
        },
        "epochs": args.epochs,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
    }
    with (result_output / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2), flush=True)


def _zero_supervision_contract():
    return training._zero_supervision_contract()


if __name__ == "__main__":
    main()
