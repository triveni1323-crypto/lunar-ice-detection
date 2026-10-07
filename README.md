# Lunar Ice Detection

## Problem Statement
Detection and Characterization of Subsurface Ice in Lunar South Polar Regions Using Chandrayaan-2 Radar and Imagery Data for Landing Site and Rover Traverse Planning.

This project focuses on detecting and characterizing subsurface ice in the lunar south polar region using radar and imagery data from the Chandrayaan-2 mission. The goal is to identify ice-bearing regions that can support future lunar landings and rover traverses, while also aiding in resource planning for future lunar exploration missions.

## Objective
- Detect probable subsurface ice zones in lunar south polar regions
- Characterize ice-bearing regions using radar and imagery features
- Identify candidate landing sites with safer terrain and resource potential
- Recommend rover traverse routes for exploration and sampling

## Motivation
Water ice in permanently shadowed craters can be used for:
- life support systems
- oxygen and fuel production
- scientific exploration
- future lunar base planning

## Data Sources
- Chandrayaan-2 SAR data
- Chandrayaan-2 optical imagery (OHRC / available imagery subsets)
- Digital Elevation Models (DEM)
- Terrain and thermal datasets if available
- Lunar geospatial reference layers

## Methodology
1. Load and preprocess radar and optical datasets
2. Extract relevant texture, terrain, and radar-based features
3. Train an ML model to classify ice vs non-ice candidates
4. Generate ice-probability maps
5. Rank landing-site suitability
6. Plan rover traverse routes based on terrain and resource accessibility
7. Visualize results and prepare final report

## Technologies
- Python
- NumPy, Pandas, SciPy
- scikit-learn
- scikit-image
- OpenCV
- TensorFlow / PyTorch (optional)
- Rasterio / GDAL
- Matplotlib / Seaborn / Plotly
- Jupyter Notebook

## Repository Structure
```text
lunar-ice-detection/
├── README.md
├── requirements.txt
├── .gitignore
├── docs/
│   └── problem_statement.md
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── model.py
│   ├── landing_site_analysis.py
│   └── visualization.py
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   └── 03_model_training.ipynb
├── data/
│   ├── raw/
│   ├── processed/
│   └── metadata/
├── models/
│   └── trained/
├── outputs/
│   ├── maps/
│   ├── plots/
│   └── reports/
└── test/
```

## Expected Outputs
- Ice probability map
- Candidate landing sites
- Terrain suitability map
- Traverse path suggestions
- Final project report and visuals

## Reference
https://1drv.ms/p/c/E3467E30EAB586B9/IQC5vHlXJh-1Q71xytDBTvfNATFcl6aKzAk0ZektSIzpqaY?e=frxUHf
