"""Central configuration for the Lunar Foundation Model data pipeline."""

import math
from pathlib import Path

# === Project paths ===
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
ALIGNED_DIR = DATA_DIR / "aligned"
DERIVED_DIR = DATA_DIR / "derived"
OUTPUT_DIR = PROJECT_ROOT / "output"

# === Lunar constants ===
LUNAR_RADIUS_M = 1_737_400  # IAU mean radius (sphere, no ellipsoid)
LUNAR_CRS_PROJ4 = (
    "+proj=eqc +lat_ts=0 +lon_0=0 "
    f"+a={LUNAR_RADIUS_M} +b={LUNAR_RADIUS_M} +units=m"
)

# === Target grid: 128 pixels per degree ===
PPD = 128
GLOBAL_WIDTH = 360 * PPD    # 46080
GLOBAL_HEIGHT = 180 * PPD   # 23040
PIXEL_SIZE_DEG = 1.0 / PPD  # ~0.0078125 degrees
PIXEL_SIZE_M = (2 * math.pi * LUNAR_RADIUS_M) / (360 * PPD)  # ~236.9 m at equator

# Projected extent (meters) for the equirectangular grid
# Using -180..180 convention (lon_0=0) to match source datasets
X_MIN = -math.pi * LUNAR_RADIUS_M         # ~-5,459,237.9 m (= -180°)
X_MAX = math.pi * LUNAR_RADIUS_M          # ~+5,459,237.9 m (= +180°)
Y_MIN = -(math.pi / 2) * LUNAR_RADIUS_M   # ~-2,729,618.9 m (= -90°)
Y_MAX = (math.pi / 2) * LUNAR_RADIUS_M    # ~+2,729,618.9 m (= +90°)

# === Patch tiling ===
PATCH_SIZE = 256
N_PATCHES_X = GLOBAL_WIDTH // PATCH_SIZE   # 180
N_PATCHES_Y = GLOBAL_HEIGHT // PATCH_SIZE  # 90

# === Channel ordering ===
# Note: No daytime temperature gridded product exists at 128ppd.
# Channel 4 uses bolometric midnight temp from GHRM collection instead.
# Channel 5 uses nighttime surface temp from GDR L3.
CHANNELS = [
    "wac_morphology",       # 0: WAC panchromatic reflectance
    "elevation",            # 1: Blended SLDEM+LOLA elevation (m)
    "slope",                # 2: Surface slope (degrees)
    "roughness",            # 3: Surface roughness (std of elev in window)
    "diviner_tbol_midnight", # 4: Diviner bolometric temp at midnight (K, GHRM)
    "diviner_temp_night",   # 5: Diviner nighttime surface temp (K, GDR L3)
    "rock_abundance",       # 6: Rock abundance (fraction)
    "christiansen_feature", # 7: CF wavelength (μm)
]

# === Download URLs ===
URLS = {
    # Tier 0: GeoTIFF, direct download
    "lola_dem": (
        "https://planetarymaps.usgs.gov/mosaic/"
        "Lunar_LRO_LOLA_Global_LDEM_118m_Mar2014.tif"
    ),
    "wac_morphology": (
        "https://planetarymaps.usgs.gov/mosaic/"
        "Lunar_LRO_LROC-WAC_Mosaic_global_100m_June2013.tif"
    ),
    # Tier 1: PDS format (IMG + LBL pairs)
    "sldem2015_img": (
        "http://imbrium.mit.edu/DATA/SLDEM2015/GLOBAL/FLOAT_IMG/"
        "SLDEM2015_128_60S_60N_000_360_FLOAT.IMG"
    ),
    "sldem2015_lbl": (
        "http://imbrium.mit.edu/DATA/SLDEM2015/GLOBAL/FLOAT_IMG/"
        "SLDEM2015_128_60S_60N_000_360_FLOAT.LBL"
    ),
    # Geologic map
    "geologic_map": (
        "https://asc-astropedia.s3.us-west-2.amazonaws.com/"
        "Moon/Geology/Unified_Geologic_Map_of_the_Moon_GIS_v2.zip"
    ),
}

# Diviner product URLs (discovered from PDS archive)
DIVINER_GDR_L3_BASE = (
    "https://pds-geosciences.wustl.edu/lro/"
    "urn-nasa-pds-lro_diviner_derived1/"
    "data_derived_gdr_l3/cylindrical/img/"
)
DIVINER_GHRM_BASE = (
    "https://pds-geosciences.wustl.edu/lro/"
    "urn-nasa-pds-lro_diviner_derived1/"
    "data_derived_ghrm/img/"
)

# Exact Diviner product files (verified from PDS directory listing)
DIVINER_PRODUCTS = {
    "diviner_temp_night": {
        "img": DIVINER_GDR_L3_BASE + "dgdr_st_avg_cyl_128_img.img",
        "lbl": DIVINER_GDR_L3_BASE + "dgdr_st_avg_cyl_128_img.lbl",
        "description": "Nighttime surface temperature (K), 128ppd, 60S-60N",
        "size_gb": 1.76,
    },
    "rock_abundance": {
        "img": DIVINER_GDR_L3_BASE + "dgdr_ra_avg_cyl_128_img.img",
        "lbl": DIVINER_GDR_L3_BASE + "dgdr_ra_avg_cyl_128_img.lbl",
        "description": "Rock abundance (fraction), 128ppd, 60S-60N",
        "size_gb": 1.76,
    },
    "christiansen_feature": {
        "img": DIVINER_GDR_L3_BASE + "dgdr_std_cf_clc_cyl_128_img.img",
        "lbl": DIVINER_GDR_L3_BASE + "dgdr_std_cf_clc_cyl_128_img.lbl",
        "description": "Christiansen Feature wavelength (μm), 128ppd, 60S-60N",
        "size_gb": 1.32,
    },
    "diviner_tbol_midnight": {
        "img": DIVINER_GHRM_BASE + "dghrm_tbol_m_70s70n_img.img",
        "lbl": DIVINER_GHRM_BASE + "dghrm_tbol_m_70s70n_img.xml",
        "description": "Bolometric temp at midnight (K), 128ppd, 70S-70N (GHRM)",
        "size_gb": 3.08,
    },
}

# ODE REST API for WAC monthly mosaics
ODE_API_URL = (
    "http://oderest.rsl.wustl.edu/live2/"
    "?query=products&target=moon&ihid=lro&iid=lroc&pt=SDPWMG&output=JSON"
)

# === Raw file paths (after download) ===
RAW_FILES = {
    "lola_dem": RAW_DIR / "Lunar_LRO_LOLA_Global_LDEM_118m_Mar2014.tif",
    "wac_morphology": RAW_DIR / "Lunar_LRO_LROC-WAC_Mosaic_global_100m_June2013.tif",
    "sldem2015_img": RAW_DIR / "SLDEM2015_128_60S_60N_000_360_FLOAT.IMG",
    "sldem2015_lbl": RAW_DIR / "SLDEM2015_128_60S_60N_000_360_FLOAT.LBL",
    "geologic_map_zip": RAW_DIR / "Unified_Geologic_Map_of_the_Moon_GIS_v2.zip",
}

# === Aligned file paths (128 ppd GeoTIFFs) ===
ALIGNED_FILES = {
    "wac_morphology": ALIGNED_DIR / "wac_morphology.tif",
    "elevation_lola": ALIGNED_DIR / "elevation_lola.tif",
    "elevation_sldem": ALIGNED_DIR / "elevation_sldem.tif",
    "elevation_blended": ALIGNED_DIR / "elevation_blended.tif",
    "diviner_tbol_midnight": ALIGNED_DIR / "diviner_tbol_midnight.tif",
    "diviner_temp_night": ALIGNED_DIR / "diviner_temp_night.tif",
    "rock_abundance": ALIGNED_DIR / "diviner_rock_abundance.tif",
    "christiansen_feature": ALIGNED_DIR / "diviner_cf.tif",
    "geologic_units": ALIGNED_DIR / "geologic_units.tif",
    "mare_mask": ALIGNED_DIR / "mare_mask.tif",
}

# === Derived file paths ===
DERIVED_FILES = {
    "slope": DERIVED_DIR / "slope.tif",
    "roughness": DERIVED_DIR / "roughness.tif",
}

# === Output paths ===
HDF5_PATH = OUTPUT_DIR / "lunar_patches.h5"

# ======================================================================
# M3 (Moon Mineralogy Mapper) Configuration
# ======================================================================

M3_ODE_API_BASE = "https://oderest.rsl.wustl.edu/live2/"
M3_L2_PRODUCT_TYPE = "REFIMG"
M3_L1B_PRODUCT_TYPE = "CALIV3"
M3_INSTRUMENT_HOST = "CH1-ORB"
M3_INSTRUMENT_ID = "M3"

# 8 selected bands from M3's 85-band spectrometer (0-indexed)
M3_BAND_SELECTION = {
    "m3_750":  {"band_idx": 8,  "wavelength_nm": 750.44},   # continuum albedo
    "m3_950":  {"band_idx": 18, "wavelength_nm": 950.06},   # pyroxene/olivine
    "m3_1000": {"band_idx": 20, "wavelength_nm": 989.98},   # 1-um shoulder
    "m3_1250": {"band_idx": 33, "wavelength_nm": 1249.49},  # plagioclase
    "m3_1580": {"band_idx": 49, "wavelength_nm": 1578.86},  # SWIR continuum
    "m3_2000": {"band_idx": 60, "wavelength_nm": 2018.02},  # 2-um pyroxene
    "m3_2817": {"band_idx": 80, "wavelength_nm": 2816.50},  # OH/H2O
    "m3_2857": {"band_idx": 81, "wavelength_nm": 2856.43},  # OH/H2O shoulder
}
M3_BAND_NAMES = list(M3_BAND_SELECTION.keys())
M3_BAND_INDICES = [v["band_idx"] for v in M3_BAND_SELECTION.values()]

# Quality filter thresholds
M3_MAX_INCIDENCE_DEG = 80.0
M3_MAX_EMISSION_DEG = 60.0
M3_NODATA = -999.0

# M3 directories
M3_RAW_DIR = RAW_DIR / "m3"
M3_L2_DIR = M3_RAW_DIR / "l2_refimg"
M3_L1B_DIR = M3_RAW_DIR / "l1b_caliv3"
M3_CATALOG_PATH = M3_RAW_DIR / "m3_product_catalog.json"
M3_COMPACT_DIR = M3_RAW_DIR / "compact"

# Aligned M3 GeoTIFFs (one per selected band)
M3_ALIGNED_FILES = {
    name: ALIGNED_DIR / f"{name}.tif" for name in M3_BAND_SELECTION
}
M3_GEOMETRY_FILES = {
    "m3_n_observations": ALIGNED_DIR / "m3_n_observations.tif",
    "m3_mean_incidence": ALIGNED_DIR / "m3_mean_incidence.tif",
    "m3_mean_emission": ALIGNED_DIR / "m3_mean_emission.tif",
    "m3_mean_phase": ALIGNED_DIR / "m3_mean_phase.tif",
}

# Extended 16-channel list (V2)
CHANNELS_V2 = CHANNELS + M3_BAND_NAMES
HDF5_V2_PATH = OUTPUT_DIR / "lunar_patches_v2.h5"
M3_MULTIANGLE_PATH = OUTPUT_DIR / "m3_multiangle_index.h5"

# ======================================================================
# IIRS (Imaging Infrared Spectrometer, Chandrayaan-2) Configuration
# ======================================================================

# IIRS directories
IIRS_RAW_DIR = RAW_DIR / "iirs"
IIRS_COMPACT_DIR = IIRS_RAW_DIR / "compact"
IIRS_CATALOG_PATH = IIRS_RAW_DIR / "iirs_product_catalog.json"

# IIRS specs
IIRS_SPECTRAL_RANGE_NM = (800, 5000)
IIRS_N_BANDS = 256  # ~256 contiguous spectral channels
IIRS_SPATIAL_RES_M = 80  # 80 m/pixel at nadir
IIRS_SWATH_KM = 20
IIRS_NODATA = -999.0

# 6 selected bands from IIRS's ~256-band spectrometer
# Band indices TBD from actual ENVI headers — these are target wavelengths
IIRS_BAND_SELECTION = {
    "iirs_950":  {"target_nm": 950,  "purpose": "1-µm pyroxene (cross-cal with M3)"},
    "iirs_2000": {"target_nm": 2000, "purpose": "2-µm pyroxene (cross-cal with M3)"},
    "iirs_2800": {"target_nm": 2800, "purpose": "OH/H₂O (thermal-corrected)"},
    "iirs_3000": {"target_nm": 3000, "purpose": "3-µm water band center"},
    "iirs_3500": {"target_nm": 3500, "purpose": "Thermal/reflected transition"},
    "iirs_4500": {"target_nm": 4500, "purpose": "Thermal emission (surface temp)"},
}
IIRS_BAND_NAMES = list(IIRS_BAND_SELECTION.keys())

# Quality filter thresholds (same as M3)
IIRS_MAX_INCIDENCE_DEG = 80.0
IIRS_MAX_EMISSION_DEG = 60.0

# Aligned IIRS GeoTIFFs (one per selected band, 128 ppd)
IIRS_ALIGNED_FILES = {
    name: ALIGNED_DIR / f"{name}.tif" for name in IIRS_BAND_SELECTION
}
IIRS_GEOMETRY_FILES = {
    "iirs_n_observations": ALIGNED_DIR / "iirs_n_observations.tif",
    "iirs_mean_incidence": ALIGNED_DIR / "iirs_mean_incidence.tif",
    "iirs_mean_emission": ALIGNED_DIR / "iirs_mean_emission.tif",
    "iirs_mean_phase": ALIGNED_DIR / "iirs_mean_phase.tif",
}

# Extended 22-channel list (V3 = V2 + IIRS)
CHANNELS_V3 = CHANNELS + M3_BAND_NAMES + IIRS_BAND_NAMES
HDF5_V3_PATH = OUTPUT_DIR / "lunar_patches_v3.h5"

# IIRS reflectance value ranges for validation
IIRS_VALUE_RANGES = {
    "iirs_950":  (0.0, 0.45, "reflectance (950 nm)"),
    "iirs_2000": (0.0, 0.40, "reflectance (2000 nm)"),
    "iirs_2800": (0.0, 0.35, "reflectance (2800 nm)"),
    "iirs_3000": (0.0, 0.35, "reflectance (3000 nm)"),
    "iirs_3500": (0.0, 0.30, "radiance mix (3500 nm)"),
    "iirs_4500": (0.0, 0.25, "thermal radiance (4500 nm)"),
}

# M3 reflectance value ranges for validation
M3_VALUE_RANGES = {
    "m3_750":  (0.0, 0.5,  "reflectance (750 nm)"),
    "m3_950":  (0.0, 0.45, "reflectance (950 nm)"),
    "m3_1000": (0.0, 0.45, "reflectance (990 nm)"),
    "m3_1250": (0.0, 0.45, "reflectance (1250 nm)"),
    "m3_1580": (0.0, 0.40, "reflectance (1579 nm)"),
    "m3_2000": (0.0, 0.40, "reflectance (2018 nm)"),
    "m3_2817": (0.0, 0.35, "reflectance (2817 nm)"),
    "m3_2857": (0.0, 0.35, "reflectance (2857 nm)"),
}

# ======================================================================
# Full 28-channel dataset (V4) — superset of all instruments
# ======================================================================

# New channel names (12 additional beyond V2's 16)
NEW_CHANNEL_NAMES = [
    "grail_freeair",         # GRAIL free-air gravity anomaly (mGal)
    "grail_bouguer",         # GRAIL Bouguer gravity anomaly (mGal)
    "grail_uncertainty",     # GRAIL gravity uncertainty (mGal)
    "minirf_cpr",            # Mini-RF circular polarization ratio
    "minirf_s1",             # Mini-RF S-band S1 backscatter
    "wac_hapke_415nm",       # WAC Hapke reflectance 415nm
    "wac_hapke_566nm",       # WAC Hapke reflectance 566nm
    "wac_hapke_604nm",       # WAC Hapke reflectance 604nm
    "wac_hapke_689nm",       # WAC Hapke reflectance 689nm
    "clementine_uvvis_750nm",  # Clementine UVVIS 750nm reflectance
    "lpgrs_tio2",            # LP GRS TiO2 weight fraction
    "lpgrs_feo",             # LP GRS FeO weight fraction
]

CHANNELS_V4 = CHANNELS + M3_BAND_NAMES + NEW_CHANNEL_NAMES
HDF5_V4_PATH = OUTPUT_DIR / "lunar_patches_v4.h5"

# Aligned file paths for new channels
NEW_ALIGNED_FILES = {
    "grail_freeair":          ALIGNED_DIR / "grail_freeair.tif",
    "grail_bouguer":          ALIGNED_DIR / "grail_bouguer.tif",
    "grail_uncertainty":      ALIGNED_DIR / "grail_uncertainty.tif",
    "minirf_cpr":             ALIGNED_DIR / "minirf_cpr.tif",
    "minirf_s1":              ALIGNED_DIR / "minirf_s1.tif",
    "wac_hapke_415nm":        ALIGNED_DIR / "wac_hapke_415nm.tif",
    "wac_hapke_566nm":        ALIGNED_DIR / "wac_hapke_566nm.tif",
    "wac_hapke_604nm":        ALIGNED_DIR / "wac_hapke_604nm.tif",
    "wac_hapke_689nm":        ALIGNED_DIR / "wac_hapke_689nm.tif",
    "clementine_uvvis_750nm": ALIGNED_DIR / "clementine_uvvis_750nm.tif",
    "lpgrs_tio2":             ALIGNED_DIR / "lpgrs_tio2.tif",
    "lpgrs_feo":              ALIGNED_DIR / "lpgrs_feo.tif",
}

# Value ranges for new channels (for validation)
NEW_VALUE_RANGES = {
    "grail_freeair":          (-2000.0, 2000.0, "free-air gravity (mGal)"),
    "grail_bouguer":          (-1000.0, 1000.0, "Bouguer gravity (mGal)"),
    "grail_uncertainty":      (0.0, 50.0,       "gravity uncertainty (mGal)"),
    "minirf_cpr":             (0.0, 10.0,       "circular polarization ratio"),
    "minirf_s1":              (0.0, 1e6,        "S-band S1 backscatter"),
    "wac_hapke_415nm":        (0.0, 1.0,        "reflectance (415 nm)"),
    "wac_hapke_566nm":        (0.0, 1.0,        "reflectance (566 nm)"),
    "wac_hapke_604nm":        (0.0, 1.0,        "reflectance (604 nm)"),
    "wac_hapke_689nm":        (0.0, 1.0,        "reflectance (689 nm)"),
    "clementine_uvvis_750nm": (0.0, 0.5,        "reflectance (750 nm)"),
    "lpgrs_tio2":             (0.0, 0.2,        "TiO2 wt fraction"),
    "lpgrs_feo":              (0.0, 0.4,        "FeO wt fraction"),
}

# ======================================================================
# Modality Groups for V2 MAE architecture
# ======================================================================
# Channels grouped by physical modality. Each group shares a tokenizer.
# Order within each group matches CHANNELS_V4 ordering.

MODALITY_GROUPS = {
    "surface": {
        "channels": ["wac_morphology", "elevation", "slope", "roughness"],
        "description": "LRO surface morphology + topography",
    },
    "thermal": {
        "channels": ["diviner_tbol_midnight", "diviner_temp_night",
                      "rock_abundance", "christiansen_feature"],
        "description": "Diviner thermal/compositional products",
    },
    "spectral_m3": {
        "channels": ["m3_750", "m3_950", "m3_1000", "m3_1250",
                      "m3_1580", "m3_2000", "m3_2817", "m3_2857"],
        "description": "M3 hyperspectral reflectance (8 bands, 750-2857nm)",
    },
    "gravity": {
        "channels": ["grail_freeair", "grail_bouguer", "grail_uncertainty"],
        "description": "GRAIL gravity field products",
    },
    "radar": {
        "channels": ["minirf_cpr", "minirf_s1"],
        "description": "Mini-RF S-band SAR (log1p-transformed)",
    },
    "hapke": {
        "channels": ["wac_hapke_415nm", "wac_hapke_566nm",
                      "wac_hapke_604nm", "wac_hapke_689nm"],
        "description": "WAC Hapke photometric model (4 bands, 415-689nm)",
    },
    "composition": {
        "channels": ["clementine_uvvis_750nm", "lpgrs_tio2", "lpgrs_feo"],
        "description": "Surface composition (Clementine albedo + LP GRS elements)",
    },
}

# Build channel-to-group-index mapping
CHANNEL_TO_GROUP = {}
for group_name, group_info in MODALITY_GROUPS.items():
    for ch in group_info["channels"]:
        CHANNEL_TO_GROUP[ch] = group_name

MODALITY_GROUP_NAMES = list(MODALITY_GROUPS.keys())
