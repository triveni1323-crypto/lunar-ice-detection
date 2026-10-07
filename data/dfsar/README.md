# DFSAR Sample Data for Lunar Ice Detection

This folder contains a small synthetic sample dataset modeled to resemble DFSAR-style radar and terrain features used in lunar ice detection workflows.

Important:
- This is not the official ISRO/DFSAR mission product.
- It is a prototype dataset intended for project development, testing, and demonstration.
- Replace it with the official DFSAR data when available for real deployment.

The sample file `sample_dfsar_radar.csv` contains columns such as:
- `x`, `y`: spatial coordinates
- `backscatter_db`: radar backscatter intensity
- `slope_deg`: terrain slope in degrees
- `shadow_probability`: probability of permanent shadowing
- `roughness`: surface roughness measure
- `ice_label`: target label (1 = ice likely, 0 = non-ice)

This dataset is suitable for:
- exploratory analysis
- preprocessing and feature engineering
- ML model experiments
- landing site prioritization prototypes
