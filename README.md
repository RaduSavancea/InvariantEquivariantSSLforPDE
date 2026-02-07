# Equivariant and Invariant Self-Supervised Learning for PDEs

This repository contains the code for the project  
**Equivariant and Invariant Self-Supervised Learning for Partial Differential Equations**

We study how **symmetry-aware self-supervised learning (SSL)** can be used to learn meaningful representations from **spatio-temporal PDE simulations**, with a focus on **2D incompressible Navier–Stokes equations**.  
In particular, we compare **invariant**, **equivariant**, and **hybrid invariant–equivariant** SSL strategies and evaluate them on downstream physical tasks.



##  Motivation

Partial Differential Equations (PDEs) exhibit strong **physical symmetries**, such as translations, rotations, and Galilean invariance.  
Standard SSL methods often enforce strict invariance to data augmentations, which can discard **geometric and dynamical information** essential for modeling PDE evolution.

This project investigates how explicitly incorporating **invariance and equivariance** into SSL objectives affects representation quality and downstream performance.


## Methods Overview

We implement and compare three symmetry-aware SSL approaches:

### 1. Invariant SSL (Baseline)
- Enforces invariance to Lie symmetry transformations
- Uses **VICReg** to prevent collapse and encourage feature diversity
- Best suited for **global, symmetry-invariant targets**

### 2. Invariant–Equivariant Dual-Branch SSL
- Single shared encoder
- Two complementary objectives:
  - **Invariant objective** (VICReg)
  - **Equivariant objective** (CARE-style relational loss)
- Preserves structured transformations in latent space

### 3. Hybrid Invariant–Equivariant SSL
- Encoder output split into two subspaces:
  - invariant subspace (global physical properties)
  - equivariant subspace (structured dynamics)
- Different augmentations and losses applied to each subspace
- Balances robustness and expressivity



## Dataset

- **System:** 2D incompressible Navier–Stokes equations
- **Data:** Simulated spatio-temporal velocity fields on a regular grid
- **Trajectories:** Each trajectory corresponds to a full simulation from a distinct initial condition
- **SSL samples:** Short temporal windows extracted from trajectories

### Input Representation

Channels correspond to:
- spatial coordinates `(x, y)`
- time `t`
- velocity components `(u, v)`

Time steps are stacked along the channel dimension and processed jointly by the encoder.


## Model Architecture

### Encoder
- ResNet-18 backbone
- First convolution modified to accept `5 × T` input channels
- Final fully connected layer removed
- Output embedding dimension: **512**

### Projection Head
- Applied only for invariant objectives (VICReg)
- Two-layer MLP with batch normalization



## Physical Data Augmentations

Lie point symmetries of the Navier–Stokes equations are implemented via a Lie–Trotter expansion:
- spatial translations
- rotations
- scaling
- Galilean boosts

Augmentations are carefully selected to preserve physical validity.



## Downstream Evaluation Tasks

All evaluations use **frozen encoders**.

### 1. One-Step Time Prediction (Forecasting)

- **Task:** predict the velocity field at time `t+1`
- **Model:** lightweight conditioned U-Net
- **Conditioning:** encoder embedding injected via a learned conditioning MLP
- **Metric:** Mean Squared Error (MSE)
---

### 2. Parameter Regression (Buoyancy)

- **Task:** regress the global buoyancy parameter
- **Model:** linear probe on frozen encoder embeddings
- **Metric:** MSE / RMSE



## Repository Structure

```text
.
├── Baseline/                  # Invariant SSL (Lie-symmetry + VICReg)
├── Dual-Branch_EISSL/          # Invariant–Equivariant dual-objective SSL
├── Hybrid_ESSL/                # Hybrid latent-space split SSL
├── outputs/                    # Saved checkpoints and evaluation results
├── log/                        # Training logs
├── transformations.py          # Lie symmetry transformations
├── utils.py                    # Dataset loading and evaluation helpers
├── requirements.txt            # Python dependencies
└── README.md                   # This file
