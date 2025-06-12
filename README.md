# lower-body-muscle-qMRI-pipeline
Pipeline for the analysis of quantitative MRI (qMRI) data from lumbar spine to ankle, focusing on muscle size and fat infiltration in the lower body.

This repository is designed to support studies in sarcopenia and muscle health in broad populations, and is currently used to study sarcopenia in transplant and dialysis patients.

## Overview

The pipeline includes:
- MRI protocol
- MRI data pre-processing
- Muscle segmentation using two previously trained CNNs to segment the lumbar spine and gluteal muscles.
- Quantitative map generation (fat fraction, T2 mapping )
- Feature extraction: muscle volumes, fat infiltration metrics
- Statistical analysis and visualization

## Pipeline Structure
- MRI Protocol
- Data pre-processing
- Muscle segmentation using deep learning and/or multi-atlas methods
- Quantitative map generation (Dixon-based fat fraction, optional T2 mapping)
- Feature extraction (volumes, normalized volumes, fat fraction per muscle)
- Statistical analysis and visualization
- TODO: segmentation of the thigh and calf muscles. Generation of T2 maps.

## Data Requirements
- MRI protocol covering lumbar spine (L1) to ankle
- Dixon MRI sequences (in-phase / out-phase) or water/fat images
- Optional T2 mapping sequences
- Supported formats: DICOM or NIfTI or MHDs

## Installation

Clone the repository and set up the environment:

```bash
git clone https://github.com/yourusername/lower-body-muscle-qMRI-pipeline.git
cd lower-body-muscle-qMRI-pipeline
conda env create -f environment.yml
conda activate lower-body-muscle-qMRI-pipeline
