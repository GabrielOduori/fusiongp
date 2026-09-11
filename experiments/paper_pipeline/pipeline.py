"""Orchestrates the FusionGP paper-reproduction experiment end to end."""

import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src.data import DataLoader, DataPreprocessor
from src.models import FusionSVGP, GridPriorMean
from src.training import Trainer
from src.training.callbacks import EarlyStopping
from src.inference import Predictor
from src.evaluation import r_squared

from . import config
from .data import (
    build_daily_dataset_from_sources,
    calibrate_sources_to_epa,
    downsample_source,
    filter_data_by_sources,
    flag_epa_outliers,
)
from .evaluation import evaluate_against_epa
from .inference import run_predictions as run_predictions_step
from .plotting import (
    create_diagnostic_plots,
    create_grid_scatter_plots,
    create_prediction_plots,
    create_spatial_maps_and_surfaces,
    plot_epa_timeseries_by_site,
    plot_training_curves,
)
from .reporting import (
    export_grid_predictions,
    save_experiment_summary,
    save_metrics_tables,
    save_model_checkpoint,
    save_training_history,
)


@dataclass
class PipelineState:
    """Accumulates state as it passes through each pipeline stage."""

    start_time: float = field(default_factory=time.time)
    main_pbar: Any = None

    data_path_for_run: Optional[Path] = None
    data: Any = None
    train_data: Any = None
    val_data: Any = None
    test_data: Any = None
    preprocessor: Any = None
    scalers: Any = None
    holdout_grid_ids: Any = None
    model_config: Optional[dict] = None
    training_sources: Optional[list] = None
    primary_source: str = config.PRIMARY_SOURCE

    prior_mean: Any = None
    model: Any = None

    trainer: Any = None
    history: Any = None

    learned_params: Optional[dict] = None

    prediction_task: Any = None
    predictor: Any = None
    predictions: Any = None
    analysis_grid: Any = None
    grid_predictions: Any = None
    noise_std_summary: Any = None
    lat_min: Any = None
    lat_max: Any = None
    lon_min: Any = None
    lon_max: Any = None
    timestamps: Any = None

    epa_eval_mask: Any = None
    epa_test_orig: Any = None
    pred_mean_at_epa: Any = None
    pred_std_at_epa: Any = None
    epa_metrics: Any = None
    epa_baseline_metrics: Any = None
    epa_bias_corrected_metrics: Any = None
    base_vs_epa_metrics: Any = None
    base_vs_fused_metrics: Any = None
    base_at_epa: Any = None
    per_source_metrics: Any = None
    y_true_orig: Any = None

    timestamp: Optional[str] = None
    experiment_dir: Optional[Path] = None
    figures_dir: Optional[Path] = None
    models_dir: Optional[Path] = None
    tables_dir: Optional[Path] = None


def load_data(state: PipelineState) -> PipelineState:
    """Step 1/8: Load and preprocess data."""
    state.main_pbar.set_description("Step 1/8: Loading data")
    print("\n[Step 1/8] Loading and preprocessing data...")

    if config.USE_INDIVIDUAL_SOURCES:
        if not config.USE_DATA_DIR.exists():
            raise FileNotFoundError(
                f"Data directory not found at {config.USE_DATA_DIR}. "
                "Please ensure data exists."
            )
        data_path_for_run = build_daily_dataset_from_sources(config.USE_DATA_DIR)
    else:
        if not config.DATA_PATH.exists():
            raise FileNotFoundError(
                f"Data file not found at {config.DATA_PATH}. "
                "Please ensure data exists."
            )
        data_path_for_run = config.DATA_PATH

    if config.EPA_OUTLIER_FILTER:
        df_outlier = pd.read_csv(data_path_for_run)
        if "epa_no2" in df_outlier.columns:
            outlier_mask, outlier_info = flag_epa_outliers(df_outlier)
            n_outliers = int(outlier_mask.sum())
            if n_outliers > 0:
                print(f"\n   Detected {n_outliers:,} EPA outliers in raw data ({outlier_info.get('method')}).")
                if config.EPA_OUTLIER_ACTION == "nan":
                    df_outlier.loc[outlier_mask, "epa_no2"] = np.nan
                elif config.EPA_OUTLIER_ACTION == "clip":
                    lower = outlier_info.get("lower")
                    upper = outlier_info.get("upper")
                    if lower is not None and upper is not None:
                        df_outlier["epa_no2"] = df_outlier["epa_no2"].clip(lower=lower, upper=upper)
                elif config.EPA_OUTLIER_ACTION == "drop":
                    df_outlier = df_outlier.loc[~outlier_mask].copy()
                else:
                    raise ValueError(f"Unknown EPA_OUTLIER_ACTION: {config.EPA_OUTLIER_ACTION}")

                cleaned_path = Path(data_path_for_run).with_name(
                    f"{Path(data_path_for_run).stem}{config.EPA_OUTLIER_CLEANED_SUFFIX}{Path(data_path_for_run).suffix}"
                )
                df_outlier.to_csv(cleaned_path, index=False)
                data_path_for_run = cleaned_path
                print(f"   ✓ Wrote cleaned EPA data to: {cleaned_path}")
        else:
            print("   Warning: epa_no2 column not found; skipping outlier filter.")

    # Fast schema check to ensure we are using the real dataset and not synthetic/low-cost fields.
    # This guards against accidental regressions if input files change.
    header_cols = pd.read_csv(data_path_for_run, nrows=0).columns
    required_cols = ["grid_id", "latitude", "longitude", "timestamp", config.GRID_PRIOR_COLUMN]
    epa_col = config.EPA_INTERP_COLUMN if config.USE_EPA_INTERPOLATED else "epa_no2"
    required_cols.append(epa_col)
    if config.HAS_SATELLITE:
        required_cols.append("satellite_no2")
    missing = [c for c in required_cols if c not in header_cols]
    if missing:
        raise ValueError(f"Missing required columns in {data_path_for_run}: {missing}")
    if any(c in header_cols for c in ["synthetic_no2", "base_syhthetic_no2", "low_cost_no2", "low_cost_data"]):
        print("   Note: synthetic/low-cost columns detected in source file but will be ignored.")

    # EPA is evaluation-only; skip interpolated EPA generation.

    loader = DataLoader(
        str(data_path_for_run),
        column_mapping={
            "grid_id": "grid_id",
            "latitude": "latitude",
            "longitude": "longitude",
            "timestamp": "timestamp",
            "satellite": "satellite_no2",
            "epa": config.EPA_INTERP_COLUMN if config.USE_EPA_INTERPOLATED else "epa_no2",
            "traffic": "__disabled__",
            "low_cost": "__disabled__",
        },
        covariate_columns=config.COVARIATE_COLUMNS,
    )
    data = loader.load()
    # Clean EPA values in-memory only (do not modify source CSV)
    epa_vals = data.observations.get("epa")
    if epa_vals is not None:
        invalid_mask = (epa_vals < 0) | (epa_vals > 100)
        if invalid_mask.any():
            epa_vals = epa_vals.copy()
            epa_vals[invalid_mask] = np.nan
            data.observations["epa"] = epa_vals
            data.source_masks["epa"] = ~np.isnan(epa_vals)
            print(f"   ✓ Dropped {invalid_mask.sum():,} EPA values (<0 or >100) for analysis")
    if getattr(data, "covariates", None) is not None:
        # Normalize covariates to [0,1]; invert distances so closer = higher pollutant.
        covariate_names = data.metadata.get("covariate_names", [])
        if covariate_names:
            covariates = data.covariates.astype(np.float64)
            for i, name in enumerate(covariate_names):
                col = covariates[:, i]
                col_min = np.nanmin(col)
                col_max = np.nanmax(col)
                if col_max == col_min:
                    norm = np.zeros_like(col, dtype=float)
                else:
                    norm = (col - col_min) / (col_max - col_min)
                if name in config.COVARIATE_DISTANCE_COLUMNS:
                    norm = 1.0 - norm
                covariates[:, i] = norm
            data.covariates = np.nan_to_num(covariates, nan=0.0)

    calibration_params = None
    if config.PRE_CALIBRATE_SOURCES and not config.USE_EPA_IN_TRAINING and config.HAS_SATELLITE:
        print("\n   Pre-calibrating satellite to EPA scale...")
        data, calibration_params = calibrate_sources_to_epa(
            data, sources_to_calibrate=('satellite',)
        )
        print("   ✓ Pre-calibration complete\n")

    # Per-source target normalization (~N(0,1)) is required regardless of
    # USE_EPA_IN_TRAINING because satellite now feeds in raw TROPOMI mol/m^2
    # column density (~1e-4 scale) rather than the converted ug/m3 value;
    # MODEL_CONFIG's noise/lengthscale priors are tuned for normalized scale.
    preprocessor = DataPreprocessor(normalize_targets=True)
    train_data, val_data, test_data = preprocessor.fit_transform(data)
    scalers = preprocessor.get_scalers()

    holdout_grid_ids = None
    if config.USE_EPA_HOLDOUT and "epa" in data.observations:
        rng = np.random.default_rng(config.HOLDOUT_SEED)
        if config.HOLDOUT_EPA_BY_GRID and data.grid_ids is not None:
            epa_grids = np.unique(data.grid_ids[data.source_masks["epa"]])
            n_holdout = max(1, int(len(epa_grids) * config.HOLDOUT_EPA_FRAC))
            holdout_grid_ids = rng.choice(epa_grids, size=n_holdout, replace=False)
        else:
            epa_idx = np.where(data.source_masks["epa"])[0]
            n_holdout = max(1, int(len(epa_idx) * config.HOLDOUT_EPA_FRAC))
            holdout_grid_ids = (
                np.unique(data.grid_ids[epa_idx]) if data.grid_ids is not None else None
            )
            holdout_indices = rng.choice(epa_idx, size=n_holdout, replace=False)
            if holdout_grid_ids is None:
                holdout_grid_ids = holdout_indices

        if holdout_grid_ids is not None and train_data.grid_ids is not None:
            def _mask_epa_by_grid(fdata):
                if "epa" not in fdata.source_masks:
                    return fdata
                mask = fdata.source_masks["epa"].copy()
                grid_mask = np.isin(fdata.grid_ids, holdout_grid_ids)
                mask[grid_mask] = False
                obs = fdata.observations["epa"].copy()
                obs[grid_mask] = np.nan
                new_obs = fdata.observations.copy()
                new_masks = fdata.source_masks.copy()
                new_obs["epa"] = obs
                new_masks["epa"] = mask
                from src.data.loader import FusionData
                return FusionData(
                    coords=fdata.coords,
                    timestamps=fdata.timestamps,
                    observations=new_obs,
                    source_masks=new_masks,
                    grid_ids=fdata.grid_ids,
                    raw_timestamps=fdata.raw_timestamps if hasattr(fdata, "raw_timestamps") else None,
                    covariates=fdata.covariates,
                    metadata=fdata.metadata,
                )

            train_data = _mask_epa_by_grid(train_data)
            val_data = _mask_epa_by_grid(val_data)

    # NOTE: the original script mutates the module-level PRIMARY_SOURCE here
    # (via globals()[...] = ...) depending on USE_EPA_IN_TRAINING/RUN_CASE0_EPA_ONLY,
    # and every downstream step (predictor noise_source, spatial map task,
    # grid-export task) reads that *mutated* value, not the config default.
    # state.primary_source replicates that effective value.
    primary_source = config.PRIMARY_SOURCE
    if config.RUN_CASE0_EPA_ONLY:
        training_sources = ["epa"]
        train_data = filter_data_by_sources(train_data, training_sources)
        val_data = filter_data_by_sources(val_data, training_sources)
        model_config = config.MODEL_CONFIG.copy()
        model_config["sources"] = training_sources
        primary_source = "epa"
    else:
        # Downsample satellite in train/val to reduce dominance
        if config.HAS_SATELLITE:
            train_data = downsample_source(train_data, "satellite", keep_frac=config.SATELLITE_KEEP_FRAC)
            val_data = downsample_source(val_data, "satellite", keep_frac=config.SATELLITE_KEEP_FRAC)

        if not config.USE_EPA_IN_TRAINING:
            training_sources = ["satellite"] if config.HAS_SATELLITE else []
            train_data = filter_data_by_sources(train_data, training_sources)
            val_data = filter_data_by_sources(val_data, training_sources)
            model_config = config.MODEL_CONFIG.copy()
            model_config["sources"] = training_sources
            primary_source = "satellite" if config.HAS_SATELLITE else "epa"
        else:
            model_config = config.MODEL_CONFIG.copy()

    print(f"   ✓ Loaded {len(data.coords)} total observations")
    print(f"   ✓ Train: {len(train_data.coords)}, Val: {len(val_data.coords)}, Test: {len(test_data.coords)}")
    state.main_pbar.update(1)

    state.data_path_for_run = data_path_for_run
    state.data = data
    state.train_data = train_data
    state.val_data = val_data
    state.test_data = test_data
    state.preprocessor = preprocessor
    state.scalers = scalers
    state.holdout_grid_ids = holdout_grid_ids
    state.model_config = model_config
    state.primary_source = primary_source
    return state


def build_model(state: PipelineState) -> PipelineState:
    """Step 2/8: Initialize the FusionSVGP model (with prior mean, if configured)."""
    state.main_pbar.set_description("Step 2/8: Initializing model")
    print("\n[Step 2/8] Initializing FusionSVGP model...")

    model_config = state.model_config
    if getattr(state.train_data, "covariates", None) is not None:
        model_config["n_covariates"] = state.train_data.covariates.shape[1]

    # Create prior mean if enabled
    prior_mean = None
    if config.USE_GRID_PRIOR:
        if config.ATMO_PLAN_AS_PRIOR and config.ATMO_PLAN_PATH.exists():
            print(f"   → Loading ATMO-Plan prior from {config.ATMO_PLAN_PATH.name} ({config.ATMO_PLAN_PRIOR_COLUMN})...")
            df_prior = pd.read_csv(config.ATMO_PLAN_PATH)
            prior_column = config.ATMO_PLAN_PRIOR_COLUMN
        else:
            print(f"   → Loading LUR prior from {config.GRID_PRIOR_COLUMN} column...")
            df_prior = pd.read_csv(state.data_path_for_run)
            prior_column = config.GRID_PRIOR_COLUMN
        grid_background = df_prior[df_prior[prior_column].notna()][
            ['latitude', 'longitude', prior_column]
        ].drop_duplicates(['latitude', 'longitude'])

        prior_mean = GridPriorMean(
            grid_coords=grid_background[['latitude', 'longitude']].values,
            grid_values=grid_background[prior_column].values,
            scalers=state.preprocessor.scalers,
            learnable_bias=config.GRID_PRIOR_LEARNABLE_BIAS,
        )
        print(f"   ✓ Grid prior loaded: {len(grid_background)} grid points")
        print(f"     Original value range: [{prior_mean.value_range_original[0]:.1f}, {prior_mean.value_range_original[1]:.1f}] µg/m³")
        if prior_mean._values_normalized:
            print(f"     Output value range (normalized): [{prior_mean.value_range[0]:.2f}, {prior_mean.value_range[1]:.2f}]")
        else:
            print(f"     Output value range (original): [{prior_mean.value_range[0]:.1f}, {prior_mean.value_range[1]:.1f}] µg/m³")
        del df_prior, grid_background  # Free memory

    model = FusionSVGP(prior_mean=prior_mean, **model_config)
    print(f"   ✓ Model initialized with {model.n_inducing} inducing points")
    print(model)
    state.main_pbar.update(1)

    state.prior_mean = prior_mean
    state.model = model
    return state


def train_model(state: PipelineState) -> PipelineState:
    """Step 3/8: Train the model."""
    state.main_pbar.set_description("Step 3/8: Training model")
    print("\n[Step 3/8] Training model...")

    trainer = Trainer(
        state.model,
        callbacks=[
            EarlyStopping(monitor="val_epa_rmse", patience=100, min_delta=1e-4, mode="min"),
        ],
        **config.TRAINING_CONFIG,
    )
    print(f"   ✓ Training config -> lr={trainer.learning_rate:.2e}, "
          f"epochs={trainer.n_epochs}, batch_size={trainer.batch_size}")

    history = trainer.fit(state.train_data, val_data=state.val_data)
    best_epoch_num = (history.best_epoch + 1) * trainer.val_interval
    print(f"   ✓ Training complete. Best validation loss: {history.best_val_loss:.4f} at epoch {best_epoch_num}")
    state.main_pbar.update(1)

    state.trainer = trainer
    state.history = history
    return state


def review_hyperparameters(state: PipelineState) -> PipelineState:
    """Step 4/8: Print learned hyperparameters and run data-scale sanity checks."""
    state.main_pbar.set_description("Step 4/8: Reviewing hyperparameters")
    print("\n[Step 4/8] Reviewing learned hyperparameters...")

    model = state.model
    train_data = state.train_data
    test_data = state.test_data
    scalers = state.scalers
    holdout_grid_ids = state.holdout_grid_ids

    learned_params = model.get_hyperparameters()
    print("\n" + "="*50)
    print("Learned Hyperparameters:")
    print("="*50)
    for key, val in learned_params.items():
        print(f"{key}: {val}")
    print("="*50 + "\n")

    # Print prior mean parameters if using GridPriorMean or similar
    if state.prior_mean is not None and hasattr(state.prior_mean, 'get_parameters'):
        prior_params = state.prior_mean.get_parameters()
        print("="*50)
        print("Prior Mean Parameters:")
        print("="*50)
        for key, val in prior_params.items():
            print(f"  {key}: {val}")
        print("="*50 + "\n")

    print("="*50)
    print("Data Statistics (All Sources):")
    print("="*50)

    # Print statistics for each source in training data
    print("\nTraining data sources:")
    for source in train_data.source_masks.keys():
        mask = train_data.source_masks[source]
        vals = train_data.observations[source][mask]
        if len(vals) == 0:
            print(f"  {source}: n=0")
            continue
        print(f"  {source}: n={mask.sum():,}, mean={vals.mean():.4f}, std={vals.std():.4f}, "
              f"range=[{vals.min():.4f}, {vals.max():.4f}]")

    # Print statistics for test data (includes EPA ground truth)
    print("\nTest data sources (for evaluation):")
    for source in test_data.source_masks.keys():
        mask = test_data.source_masks[source]
        vals = test_data.observations[source][mask]
        if len(vals) == 0:
            print(f"  {source}: n=0")
            continue
        print(f"  {source}: n={mask.sum():,}, mean={vals.mean():.4f}, std={vals.std():.4f}, "
              f"range=[{vals.min():.4f}, {vals.max():.4f}]")

    # Print scaler info
    print("\nScaler statistics:")
    if scalers.normalize_targets:
        for source in scalers.target_mean.keys():
            print(f"  {source}: mean={scalers.target_mean[source]:.4f}, std={scalers.target_std[source]:.4f}")
    else:
        print("  (no target normalization)")

    print("\n" + "-"*50)
    print("SCALE ALIGNMENT CHECK:")
    print("-"*50)
    epa_mask_stats = test_data.source_masks['epa']
    if holdout_grid_ids is not None and test_data.grid_ids is not None:
        epa_mask_stats = epa_mask_stats & np.isin(test_data.grid_ids, holdout_grid_ids)
    epa_vals = test_data.observations['epa'][epa_mask_stats]
    epa_mean = epa_vals.mean()
    for source in train_data.source_masks.keys():
        source_vals = train_data.observations[source][train_data.source_masks[source]]
        source_mean = source_vals.mean()
        diff = source_mean - epa_mean
        print(f"  {source} mean - EPA mean = {diff:.4f} ({source_mean:.4f} vs {epa_mean:.4f})")
    print("-"*50)
    print(f"Train coords range: lat=[{train_data.coords[:,0].min():.4f}, {train_data.coords[:,0].max():.4f}], "
          f"lon=[{train_data.coords[:,1].min():.4f}, {train_data.coords[:,1].max():.4f}]")
    print(f"Train time range: [{train_data.timestamps.min():.4f}, {train_data.timestamps.max():.4f}]")
    print("="*50 + "\n")
    state.main_pbar.update(1)

    state.learned_params = learned_params
    return state


def run_predictions(state: PipelineState) -> PipelineState:
    """Step 5/8: Make pointwise and gridded predictions."""
    state.main_pbar.set_description("Step 5/8: Making predictions")
    print("\n[Step 5/8] Making predictions...")

    prediction_task = state.primary_source if config.USE_EPA_IN_TRAINING else None
    # noise_source determines which target scaler inverse-transforms predictions
    # back to original units. FusionSVGP has a single shared latent function
    # anchored in EPA-normalized units (GridPriorMean uses target_source='epa'),
    # not whichever source happens to be PRIMARY_SOURCE — using a non-EPA
    # scaler here silently corrupts every downstream prediction's scale (was
    # "satellite", whose mean/std are ~1e-4, collapsing all predictions to ~0).
    predictor = Predictor(state.model, state.scalers, include_observation_noise=False, noise_source="epa")

    result = run_predictions_step(
        model=state.model,
        predictor=predictor,
        data=state.data,
        test_data=state.test_data,
        scalers=state.scalers,
        prediction_task=prediction_task,
        data_path_for_run=state.data_path_for_run,
    )
    state.main_pbar.update(1)

    state.prediction_task = prediction_task
    state.predictor = predictor
    state.predictions = result["predictions"]
    state.analysis_grid = result["analysis_grid"]
    state.grid_predictions = result["grid_predictions"]
    state.noise_std_summary = result["noise_std_summary"]
    state.lat_min = result["lat_min"]
    state.lat_max = result["lat_max"]
    state.lon_min = result["lon_min"]
    state.lon_max = result["lon_max"]
    state.timestamps = result["timestamps"]
    return state


def evaluate(state: PipelineState) -> PipelineState:
    """Step 6/8: Evaluate the fused model and prior baseline against EPA."""
    state.main_pbar.set_description("Step 6/8: Evaluating performance")
    print("\n[Step 6/8] Evaluating FusionGP performance...")

    result = evaluate_against_epa(
        model=state.model,
        predictor=state.predictor,
        train_data=state.train_data,
        test_data=state.test_data,
        scalers=state.scalers,
        holdout_grid_ids=state.holdout_grid_ids,
        prior_mean=state.prior_mean,
        use_epa_in_training=config.USE_EPA_IN_TRAINING,
        grid_prior_column=config.GRID_PRIOR_COLUMN,
        grid_prior_learnable_bias=config.GRID_PRIOR_LEARNABLE_BIAS,
    )
    state.main_pbar.update(1)

    state.training_sources = result["training_sources"]
    state.epa_eval_mask = result["epa_eval_mask"]
    state.epa_test_orig = result["epa_test_orig"]
    state.pred_mean_at_epa = result["pred_mean_at_epa"]
    state.pred_std_at_epa = result["pred_std_at_epa"]
    state.epa_metrics = result["epa_metrics"]
    state.epa_baseline_metrics = result["epa_baseline_metrics"]
    state.epa_bias_corrected_metrics = result["epa_bias_corrected_metrics"]
    state.base_vs_epa_metrics = result["base_vs_epa_metrics"]
    state.base_vs_fused_metrics = result["base_vs_fused_metrics"]
    state.base_at_epa = result["base_at_epa"]
    state.per_source_metrics = result["per_source_metrics"]
    state.y_true_orig = result["y_true_orig"]
    return state


def save_outputs(state: PipelineState) -> PipelineState:
    """Step 7/8: Create visualizations and save all experiment outputs."""
    state.main_pbar.set_description("Step 7/8: Saving results")
    print("\n[Step 7/8] Creating visualizations and saving results...")

    # Create timestamped experiment folder with organized subdirectories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_results_dir = Path(__file__).resolve().parent.parent / "results"
    experiment_dir = base_results_dir / f"experiment_{timestamp}"

    figures_dir = experiment_dir / "figures"
    models_dir = experiment_dir / "models"
    tables_dir = experiment_dir / "tables"

    figures_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"Saving results to: {experiment_dir}")
    print(f"  - Figures: {figures_dir}")
    print(f"  - Models: {models_dir}")
    print(f"  - Tables: {tables_dir}")
    print(f"{'='*70}\n")

    # TROPOMI-augmented prior export disabled.
    # If re-enabled, restore the block that builds model_no2_10m_tropomi
    # and writes prior_augmented_tropomi.csv.

    # Save experiment summary to tables directory (with current runtime)
    current_elapsed = time.time() - state.start_time
    summary_file = save_experiment_summary(
        experiment_dir=tables_dir,
        model=state.model,
        trainer=state.trainer,
        learned_params=state.learned_params,
        epa_metrics=state.epa_metrics,
        epa_test=state.epa_test_orig,
        scalers=state.scalers,
        timestamp=timestamp,
        elapsed_time=current_elapsed,
        training_sources=state.training_sources,
        base_vs_epa_metrics=state.base_vs_epa_metrics,
        base_vs_fused_metrics=state.base_vs_fused_metrics,
        epa_baseline_metrics=state.epa_baseline_metrics,
        epa_bias_corrected_metrics=state.epa_bias_corrected_metrics,
        predictor_noise_source=state.predictor.noise_source,
        prediction_task=state.prediction_task,
        noise_std_summary=state.noise_std_summary,
        epa_r2=r_squared(state.epa_test_orig, state.pred_mean_at_epa),
        base_r2=r_squared(state.epa_test_orig, state.base_at_epa) if state.base_vs_epa_metrics is not None else None,
    )
    print(f"   ✓ Experiment summary saved to: {summary_file}")

    plot_training_curves(
        history=state.history,
        save_path=str(figures_dir / "training_curves.png")
    )
    print(f"   ✓ Saved training curves plot")

    save_training_history(state.history, tables_dir)

    create_prediction_plots(
        grid_predictions=state.grid_predictions,
        predictions=state.predictions,
        epa_eval_mask=state.epa_eval_mask,
        test_data=state.test_data,
        scalers=state.scalers,
        figures_dir=figures_dir,
    )

    # EPA time-series diagnostics (top sites by EPA count)
    print("\n   Creating EPA time-series plots...")
    epa_ts_task = "epa" if config.USE_EPA_IN_TRAINING else None
    predictions_all = state.predictor.predict(state.data, task=epa_ts_task, verbose=False)
    plot_epa_timeseries_by_site(
        data=state.data,
        predictions=predictions_all,
        scalers=state.scalers,
        output_dir=figures_dir,
        n_sites=5,
    )
    print("   ✓ Saved EPA time-series plots")

    create_spatial_maps_and_surfaces(
        predictor=state.predictor,
        scalers=state.scalers,
        lat_min=state.lat_min,
        lat_max=state.lat_max,
        lon_min=state.lon_min,
        lon_max=state.lon_max,
        primary_source="epa",  # see noise_source note in run_predictions()
        use_analysis_grid=config.USE_ANALYSIS_GRID,
        analysis_grid=state.analysis_grid,
        grid_predictions=state.grid_predictions,
        figures_dir=figures_dir,
    )

    # -------------------------------------------------------------------------
    # Save Model and Additional Outputs
    # -------------------------------------------------------------------------
    print("\n   Saving model and metrics tables...")

    save_model_checkpoint(
        model=state.model,
        model_config=state.model_config,
        training_config=config.TRAINING_CONFIG,
        learned_params=state.learned_params,
        timestamp=timestamp,
        models_dir=models_dir,
    )

    save_metrics_tables(
        per_source_metrics=state.per_source_metrics,
        epa_metrics=state.epa_metrics,
        base_vs_epa_metrics=state.base_vs_epa_metrics,
        base_vs_fused_metrics=state.base_vs_fused_metrics,
        learned_params=state.learned_params,
        tables_dir=tables_dir,
    )

    grid_results_df, timestamps_for_export, _grid_locs = export_grid_predictions(
        data_path_for_run=state.data_path_for_run,
        scalers=state.scalers,
        predictor=state.predictor,
        tables_dir=tables_dir,
        primary_source="epa",  # see noise_source note in run_predictions()
    )

    print("\n   Creating grid-based visualizations...")
    create_grid_scatter_plots(grid_results_df, timestamps_for_export, figures_dir)

    print("\n   Creating additional evaluation visualizations...")
    create_diagnostic_plots(
        epa_eval_mask=state.epa_eval_mask,
        test_data=state.test_data,
        epa_test_orig=state.epa_test_orig,
        predictions=state.predictions,
        figures_dir=figures_dir,
    )

    state.main_pbar.update(1)
    state.main_pbar.close()

    state.timestamp = timestamp
    state.experiment_dir = experiment_dir
    state.figures_dir = figures_dir
    state.models_dir = models_dir
    state.tables_dir = tables_dir
    return state


def print_final_summary(state: PipelineState) -> None:
    """Print the closing "Paper Reproduction Experiment Complete!" summary."""
    end_time = time.time()
    elapsed_time = end_time - state.start_time
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)

    print("\n" + "="*70)
    print("Paper Reproduction Experiment Complete!")
    print("="*70)
    print(f"Results directory: {state.experiment_dir}")
    print(f"\nOrganized outputs:")
    print(f"  Figures: {state.figures_dir.relative_to(state.experiment_dir.parent.parent)}")
    print(f"  Models:  {state.models_dir.relative_to(state.experiment_dir.parent.parent)}")
    print(f"   Tables:  {state.tables_dir.relative_to(state.experiment_dir.parent.parent)}")
    print("\nKey Performance Metrics:")
    print(f"  • RMSE: {state.epa_metrics.to_dict()['rmse']:.4f} µg/m³")
    print(f"  • MAE:  {state.epa_metrics.to_dict()['mae']:.4f} µg/m³")
    print(f"  • R²:   {r_squared(state.epa_test_orig, state.pred_mean_at_epa):.4f}")
    print(f"  • Bias: {state.epa_metrics.to_dict()['bias']:.4f} µg/m³")
    print("\nTotal Runtime:")
    if hours > 0:
        print(f"  • {int(hours)}h {int(minutes)}m {seconds:.2f}s ({elapsed_time:.2f} seconds)")
    elif minutes > 0:
        print(f"  • {int(minutes)}m {seconds:.2f}s ({elapsed_time:.2f} seconds)")
    else:
        print(f"  • {seconds:.2f} seconds")
    print("="*70)


def run() -> None:
    """Run the full FusionGP paper-reproduction experiment end to end."""
    state = PipelineState()

    if config.USE_EPA_IN_TRAINING:
        raise RuntimeError(
            "EPA must be evaluation-only for this experiment. "
            "Set USE_EPA_IN_TRAINING = False."
        )

    # Set GPyTorch numerical stability settings
    import gpytorch
    # Set higher jitter and max tries globally for numerical stability
    import linear_operator
    linear_operator.settings.cholesky_jitter._global_double_value = 1e-4
    linear_operator.settings.cholesky_max_tries._global_value = 6

    print("="*70)
    print("FusionGP Paper Experiment")
    print("="*70)

    # Create main progress bar for experiment steps
    total_steps = 8
    state.main_pbar = tqdm(total=total_steps, desc="Experiment Progress", position=0, leave=True,
                     bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} steps [{elapsed}<{remaining}]')

    state = load_data(state)
    state = build_model(state)
    state = train_model(state)
    state = review_hyperparameters(state)
    state = run_predictions(state)
    state = evaluate(state)
    state = save_outputs(state)
    print_final_summary(state)
