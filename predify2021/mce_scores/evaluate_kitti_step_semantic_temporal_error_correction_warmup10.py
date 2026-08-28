from predify2021.mce_scores import evaluate_kitti_step_semantic_temporal_error_correction as experiment


experiment.WARMUP_FRACTION = 0.10
experiment.SKIP_WARMUP = True


if __name__ == "__main__":
    experiment.main()
