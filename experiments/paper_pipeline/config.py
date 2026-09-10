"""Configuration constants for the FusionGP paper reproduction pipeline."""

from pathlib import Path

# Use the vetted real dataset (no synthetic/low-cost inputs) to avoid data leakage.
# Sources arrive as separate per-source files (EPA, satellite, traffic, ...) under
# USE_DATA_DIR; build_daily_dataset_from_sources() merges them by grid_id + date.
# Repo-relative so it resolves regardless of where the drive is mounted
# (/media/... vs /run/media/...).
USE_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "model_data"
DATA_PATH = USE_DATA_DIR / "FusionData_daily_merged.csv"
USE_INDIVIDUAL_SOURCES = True  # Use separate source files and merge by grid_id + day
TRAFFIC_PATH = DATA_PATH.with_name("traffic_timeseries.csv")
LUR_PATH = DATA_PATH.with_name("lur_predictions.csv")
WIND_SECTOR_PATH = USE_DATA_DIR / "wind_sector_features_era5land_2023-06_daily.csv"

# Overpass window: hours (inclusive) over which EPA and traffic readings are
# averaged before entering the model, matching the satellite overpass period.
# The actual UTC hour of a sun-synchronous satellite's overpass (e.g. TROPOMI)
# depends on longitude/orbit track and the timezone of the timestamp columns,
# so it is NOT a universal constant — it's inferred per run from the actual
# hour distribution of satellite retrievals (see _infer_overpass_window), with
# OVERPASS_WINDOW_PAD_HOURS as a margin for robustness over single-hour
# matching. Set OVERPASS_WINDOW_OVERRIDE to force a fixed (start, end) instead.
OVERPASS_WINDOW_PAD_HOURS = 1
OVERPASS_WINDOW_OVERRIDE = None  # e.g. (11, 14) to bypass auto-detection

COVARIATE_COLUMNS = []
COVARIATE_DISTANCE_COLUMNS = []

MODEL_CONFIG = {
    "n_inducing": 100,
    "spatial_kernel_type": "matern32",
    "temporal_kernel_type": "exponential",
    "learn_inducing_locations": True,
    "learn_kernel_hyperparams": True,
    "learn_noise": True,
    "learn_calibration": True,
    "initial_lengthscales": {"spatial_x": 0.1, "spatial_y": 0.1, "temporal": 0.1},
    "initial_noise": {"epa": 0.3, "satellite": 2.0},
}

TRAINING_CONFIG = {
    "learning_rate": 0.01, #0.01
    "n_epochs": 50,#150
    "batch_size": 2048,
    "val_interval": 5,
    "gradient_clip": 1.0,  # Prevent NaN gradients from corrupting parameters
}

GRID_RESOLUTION = 0.01
GRID_RESOLUTION_M = 100
USE_ANALYSIS_GRID = True
USE_PRIOR_GRID_FOR_PREDICTION = True
GEOJSON_GRID_PATH = Path("/media/gabriel-oduori/SERVER/dev_space/data-tools/osm/run_20260130_074609/grid.geojson")
MAX_ANALYSIS_GRID_POINTS = 5_000_000


USE_EPA_HOLDOUT = True
HOLDOUT_EPA_FRAC = 0.2
HOLDOUT_EPA_BY_GRID = True
HOLDOUT_SEED = 42

USE_EPA_IN_TRAINING = False  # EPA is evaluation-only
PRIMARY_SOURCE = "epa"
SATELLITE_KEEP_FRAC = 0.1
HAS_SATELLITE = True
RUN_CASE0_EPA_ONLY = False

AVAILABLE_SOURCES = ["epa"] + (["satellite"] if HAS_SATELLITE else [])

EPA_INTERP_COLUMN = "epa_interpolated"
USE_EPA_INTERPOLATED = False
PRE_CALIBRATE_SOURCES = False  # EPA evaluation-only; no calibration
EPA_OUTLIER_FILTER = True
EPA_OUTLIER_METHOD = "iqr"  # "iqr" or "zscore"
EPA_OUTLIER_IQR_MULT = 3.0
EPA_OUTLIER_Z_THRESH = 4.0
EPA_OUTLIER_ACTION = "nan"  # "nan", "clip", or "drop"
EPA_OUTLIER_CLEANED_SUFFIX = "_epa_cleaned"

# Prior mean configuration
# Option 1: Use GridPriorMean with CSV column (preferred for new datasets)
USE_GRID_PRIOR = True  # Set True to use grid-based prior from CSV column
GRID_PRIOR_COLUMN = "predicted_no2"  # LUR fallback column (merged daily dataset)
GRID_PRIOR_LEARNABLE_BIAS = False  # Fixed at 0 to preserve raw ATMOS-Plan baseline

# ATMO-Plan as prior (preferred): read directly from its own CSV rather than
# requiring model_no2_10m to be merged into the daily dataset.
ATMO_PLAN_PATH = USE_DATA_DIR / "atmos_plan_model_no2.csv"
ATMO_PLAN_AS_PRIOR = True
ATMO_PLAN_PRIOR_COLUMN = "model_no2_10m"
