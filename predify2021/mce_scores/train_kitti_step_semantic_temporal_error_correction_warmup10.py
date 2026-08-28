from predify2021.mce_scores import train_kitti_step_semantic_temporal_error_correction as experiment


experiment.EPOCHS = 7
experiment.WARMUP_FRACTION = 0.10
experiment.SKIP_WARMUP = True


if __name__ == "__main__":
    experiment.main()
