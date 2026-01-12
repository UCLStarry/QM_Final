# QM_Final

CASA0007 – Quantitative Methods  
Night-Time Economy Analysis (QM_Project)

---

## Project Overview

This repository contains all scripts, data, and outputs used for the **CASA0007 Quantitative Methods coursework**, focusing on the quantitative analysis of **London’s night-time economy**.

The project applies spatial analysis, clustering, and regression methods to explore relationships between night-time transport accessibility, venue distribution and deprivation (IMD).  
The repository is organised to clearly reflect the full analytical workflow, from data preparation to final GIS outputs.

---

## Quick Navigation 

- **Final GIS maps and spatial outputs**: `Final/QM_Final.qgz`  
- **Main analysis scripts**: `Night_*.py`, `Regression_IMD_Night.py`  
- **Clustering analysis (PCA & K-means)**: `Clusters/`  
- **Intermediate processed data**: `outputs/`

---

## Repository Structure

```text
QM_Project/
├── Clusters/        # PCA and K-means clustering of night-time venues
├── outputs/         # Intermediate processed datasets
├── *.py             # Modular Python scripts for each analytical step
└──  *.csv / *.geojson / *.shp   # Input and derived datasets

Final/           # GIS outputs and QGIS project
└── outputs/         # Intermediate processed datasets



## Analytical Workflow

1. **Exploratory Data Analysis** (`EDA.py`)  
   Initial exploration and summary statistics of night-time economy and transport-related variables.

2. **Night-Time Venues and Transport Processing** (`Night_*.py`)  
   Extraction, cleaning, and spatial processing of night-time venues, bus stops, tube lines, and stations.

3. **Construction of Night-Time Accessibility Indicators** (`Night_Access.py`)  
   Aggregation of night-time transport and venue data to LSOA level to construct accessibility measures.

4. **Regression Analysis with IMD** (`Regression_IMD_Night.py`)  
   Statistical modelling to examine the relationship between night-time accessibility indicators and deprivation (IMD).

5. **Model Diagnostics and Robustness Checks** (`model_diagnostics.py`)  
   Assessment of model assumptions, residuals, and multicollinearity.

6. **Final Mapping and Visualisation** (`Final/QM_Final.qgz`)  
   Production of final GIS maps and spatial outputs used for interpretation and presentation.
