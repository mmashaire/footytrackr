#!/usr/bin/env python3
"""
Footytrackr - Football Data Science Project
==========================================

This project demonstrates end-to-end data engineering and machine learning
for predicting football player market values using Transfermarkt data.

Quick Start:
    # Install dependencies
    pip install -e .[dev]

    # Run the full pipeline
    python -m footytrackr.cli build-features
    python -m footytrackr.cli train

    # Start the API
    python -m footytrackr.cli api

    # Run tests
    python -m footytrackr.cli test

Project Structure:
    data/           - Raw and processed datasets
    scripts/        - Data processing and training scripts
    notebooks/      - Exploratory analysis notebooks
    artifacts/      - Trained models and evaluation results
    footytrackr/    - Python package
    tests/          - Unit tests

Key Features:
    - Reproducible feature engineering pipeline
    - Ridge regression model for market value prediction
    - REST API for real-time predictions
    - Comprehensive test suite
    - CI/CD with GitHub Actions
    - Docker containerization

For more information, see README.md
"""

__version__ = "0.1.0"