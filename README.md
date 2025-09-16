# wyu
# Advanced UKF-Based Trajectory Prediction for UAVs

A comprehensive framework for UAV trajectory prediction combining adaptive Unscented Kalman Filters (UKF) with deep learning approaches including Transformer networks and Reservoir Computing.

## 🚀 Key Features

- **Adaptive Q/R Matrix Scaling**: Dynamic adjustment of process and measurement noise matrices using neural networks
- **Multi-Vehicle Support**: Optimized parameters for micro quadrotors, medium/large quadrotors, fixed-wing aircraft, and heavy multirotors
- **Hybrid Intelligence**: Combines physics-based models with data-driven approaches
- **Real-time Performance**: Optimized for real-time trajectory prediction with sub-10ms processing times
- **Comprehensive Evaluation**: Includes 7 baseline methods for thorough performance comparison

## 📋 Table of Contents

- [Architecture Overview](#architecture-overview)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [File Descriptions](#file-descriptions)
- [Experimental Results](#experimental-results)
- [Citation](#citation)
- [License](#license)

## 🏗️ Architecture Overview

### Core Components

1. **Smart Q/R Enhanced UKF**: Intelligent base Q/R matrix adjustment based on vehicle type and flight mode
2. **Transformer-based Scaling**: Real-time prediction of Q/R scaling factors using trajectory features
3. **Reservoir Computing**: Memory-enhanced temporal sequence learning
4. **Confidence-driven Fusion**: Adaptive strategy selection based on prediction confidence

### System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   Input Trajectory Data                     │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│              Feature Extraction Module                      │
│  • Position/Velocity Statistics  • Trajectory Characteristics│
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│              Q/R Scaling Transformer                        │
│  • Multi-head Attention  • Reservoir Memory                 │
│  • Confidence Estimation • Vehicle Type Classification      │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                Smart UKF Core                               │
│  • Base Q/R Matrix (Motion Pattern Analysis)               │
│  • Applied Scaling Factors                                 │
│  • Adaptive Covariance Regularization                      │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                Predicted Trajectory                         │
└─────────────────────────────────────────────────────────────┘
```

## 🛠️ Installation

### Requirements

```bash
Python >= 3.8
PyTorch >= 1.8.0
NumPy >= 1.19.0
Pandas >= 1.2.0
Scikit-learn >= 0.24.0
FilterPy >= 1.4.5
SciPy >= 1.6.0
```

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/advanced-ukf-trajectory-prediction.git
cd advanced-ukf-trajectory-prediction

# Install dependencies
pip install -r requirements.txt

# Optional: Install in development mode
pip install -e .
```

## 🚀 Quick Start

### 1. Training the Q/R Scaling Model

```bash
python reservoir_inspired_transformer_ukf.py \
    --data_path "your_trajectory_data.csv" \
    --output_dir "./models" \
    --device "cpu"
```

### 2. Running Baseline Comparisons

```bash
python simplified_run_experiments.py \
    --qr_model "models/BEST_qr_scaling.pth" \
    --vector_model "models/vector_best_model.pth" \
    --data_files "data1.csv,data2.csv,data3.csv" \
    --num_runs 5
```

### 3. Testing Individual Methods

```python
from final_working_baseline_methods import create_baseline_methods

# Create methods
methods = create_baseline_methods(
    qr_model_path="models/BEST_qr_scaling.pth",
    dt=0.1,
    device='cpu'
)

# Initialize and use
method = methods['Q/R缩放UKF']
method.initialize(initial_state)
prediction = method.predict_and_update(measurement)
```

## 📁 File Descriptions

### Core Files

| File | Description |
|------|-------------|
| `reservoir_inspired_transformer_ukf.py` | Main training script for Q/R scaling Transformer model |
| `final_working_baseline_methods.py` | Implementation of 5 core baseline methods |
| `fixed_dynamic_ukf.py` | Enhanced dynamic UKF with 24 flight modes |
| `simplified_run_experiments.py` | Comprehensive evaluation framework |

### Key Classes

- **`QRScalingTransformerNN`**: Multi-task Transformer for Q/R scaling prediction
- **`SmartQREnhancedUKF`**: Intelligent UKF with automatic Q/R adjustment
- **`QRScalingAdaptivePredictor`**: Confidence-driven prediction strategy
- **`DifferentialFlatnessTrajectoryPredictor`**: Physics-based trajectory predictor

### Supported Methods

1. **Fixed Parameter UKF**: Traditional UKF with constant parameters
2. **Physics-based Predictor**: Improved physics model prediction
3. **Q/R Scaling UKF**: Smart UKF + Transformer scaling (Main contribution)
4. **Kalman-RNN**: RNN-based sequential prediction
5. **Intelligent UKF**: 24 dynamic mode switching UKF
6. **VECTOR**: Neural velocity-based predictor
7. **Physics-DF**: Differential flatness physics predictor



## 🔧 Configuration

### Vehicle Types

The system supports four vehicle categories:

- `micro_quad`: Small quadrotors (< 0.5kg)
- `medium_large_quad`: Medium/large quadrotors (0.5-3kg)
- `fixed_wing`: Fixed-wing aircraft
- `heavy_multirotor`: Heavy multirotors (> 3kg)

### Training Parameters

```python
train_params = {
    'epochs': 50,
    'batch_size': 16,
    'learning_rate': 3e-5,
    'd_model': 128,
    'nlayers': 2,
    'nhead': 4,
    'reservoir_size': 32,
    'patience': 15
}
```

## 📈 Data Format

Expected CSV format for trajectory data:

```csv
time,x,y,z,vx,vy,vz
0.0,0.0,0.0,0.0,0.0,0.0,0.0
0.1,0.1,0.0,0.0,1.0,0.0,0.0
...
```

Required columns: `time`, position coordinates (`x`, `y`, `z`)



### Development Setup

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
python -m pytest tests/

# Format code
black .
isort .
```




## 🙏 Acknowledgments

- FilterPy library for UKF implementation
- PyTorch team for the deep learning framework
- Contributors to the trajectory prediction research community



⭐ If you find this project useful, please consider giving it a star!
