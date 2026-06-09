# Car Diagnostics Audio Classification

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.21](https://img.shields.io/badge/TensorFlow-2.21-orange.svg)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Multi-class classification of car engine and mechanical sounds using 5 different ML approaches, hierarchical Mixture-of-Experts architectures, and ensemble voting.

## 📋 Overview

This project implements a comprehensive audio classification system for vehicle diagnostics. Using the Car Diagnostics Dataset from Kaggle, we classify audio recordings into 9 distinct mechanical fault categories across three vehicle states (braking, idle, startup). The system compares traditional machine learning, deep learning, transfer learning, hierarchical architectures, and ensemble methods.

**Key Results:**
- ✅ **Best single model:** XGBoost with MFCC features — **88.5% accuracy**
- ✅ **Best ensemble:** Majority vote of top-3 models — **91.5% accuracy**
- ✅ **Hierarchical MoE:** Interpretable state → fault architecture — **83–86% accuracy**

## 🎯 Problem Statement

Given an audio recording of a vehicle's mechanical sound, classify it into one of nine fault categories:

| State | Classes |
|-------|---------|
| **Braking** (2) | `normal_brakes`, `worn_out_brakes` |
| **Idle** (4) | `normal_engine_idle`, `low_oil`, `power_steering`, `serpentine_belt` |
| **Startup** (3) | `normal_engine_startup`, `dead_battery`, `bad_ignition` |

## 🗂️ Dataset

**Source:** [Car Diagnostics Dataset (Kaggle)](https://www.kaggle.com/datasets/malakragaie/car-diagnostics-dataset)

- **Original samples:** 949 audio files (WAV format)
- **After augmentation:** 1,967 samples (balanced)
- **Audio characteristics:** Variable duration (2-10 sec), real-world recordings
- **Train/Val/Test split:** 70% / 15% / 15% (stratified)

## 🧠 Models Implemented

| # | Model | Features | Accuracy |
|---|-------|----------|----------|
| 1 | **XGBoost (Baseline)** | MFCCs + deltas + chroma + ZCR + RMS | **88.5%** |
| 2 | CNN | Mel spectrograms (128×65) | 8.1% |
| 3 | CNN-LSTM | Mel spectrograms + temporal modeling | 14.5% |
| 4 | YAMNet Transfer Learning | YAMNet embeddings (1024-dim) | 79.1% |
| 5 | PANNs CNN14 Transfer Learning | PANNs embeddings (2048-dim) | 86.2% |
| 6 | **Mixture-of-Experts (PANNs)** | Hierarchical state → fault | 83.8% |
| 7 | **Mixture-of-Experts (XGB)** | Hierarchical with XGBoost | 86.5% |
| 8 | **Ensemble Vote (Top-3)** | Majority vote of Models 1,6,7 | **91.5%** |

## 🏗️ Architecture Highlights

### Mixture-of-Experts (Hierarchical)
