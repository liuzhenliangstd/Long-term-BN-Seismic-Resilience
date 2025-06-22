1. Dataset Description: The dataset used to train the artificial neural network (ANN) model was synthetically generated based on a large-scale parametric analysis framework. It consists of 83,680 samples, each representing a unique combination of bridge structural characteristics and seismic intensity measures. Specifically, each sample includes 31 input features—17 describing bridge geometry, material, and deterioration conditions, and 14 representing ground motion parameters derived from GMPEs. The corresponding outputs are six key seismic demand indicators (e.g., column ductility, bearing displacement) obtained from nonlinear time history analyses (NLTHA). This synthetic dataset comprehensively reflects the typical configurations and hazard scenarios of highway bridges in seismically active regions, providing a robust basis for data-driven seismic resilience modeling.

2. Scripts for Data Preprocessing and Model Training:
The provided scripts automate the full workflow from raw data preparation to ANN model training and evaluation. Key components include:
preprocess_data.m: This script handles data cleaning, normalization (e.g., min-max scaling), outlier filtering, and dataset splitting (e.g., into training, validation, and test sets). It ensures that all input features are scaled appropriately for efficient ANN learning and numerical stability.
train_ann_model.m Implements the artificial neural network using MATLAB or Python (e.g., TensorFlow/Keras). The script allows flexible definition of the model architecture (e.g., number of hidden layers and neurons), and includes hyperparameter configuration (learning rate, momentum, batch size, etc.).
cross_validation.m: Conducts K-fold cross-validation to evaluate model generalization performance and avoid overfitting. It reports metrics such as MSE, R², and standard deviation across folds.
evaluate_model.m: Compares prediction results against ground truth values, visualizes learning curves, and outputs performance metrics.
These scripts are modular and customizable, allowing users to easily replace the ANN model with other machine learning algorithms if needed.

3. Simulation Codes for Seismic Hazard Modeling and Resilience Computation:
This repository includes a set of modular simulation codes that support the end-to-end process of long-term regional bridge network (BN) resilience assessment. The workflow consists of the following major components:
(1). Seismic Hazard Modeling:
   * Implements regional-scale ground motion simulation using GMPEs (e.g., Boore-Atkinson model).
   * Inputs include earthquake magnitude, epicentral location, and local site conditions (e.g., VS30).
   * Outputs intensity measures (IMs) such as PGA, PGV, and spectral accelerations for each bridge site.
(2). **Bridge Seismic Damage Simulation (using ANN model):**
   * Loads the trained ANN model to rapidly predict seismic demands (e.g., curvature ductility, bearing displacement) for all bridge samples.
   * Simulated IMs serve as input; output is the predicted damage level or demand-to-capacity ratio per component.
   * Allows batch processing of tens of thousands of bridge scenarios.
(3). **Bridge Functionality and Recovery Process Modeling:**

   * Maps component-level damage to bridge-level functionality loss using fragility-based or empirical rules.
   * Simulates time-variant recovery using restoration functions (e.g., linear, exponential, or policy-driven recovery trajectories).
   * Allows incorporation of various repair durations and priorities.
(4). **Post-Earthquake Traffic Network Simulation:**
   * Updates link-level capacities based on damaged or repaired bridge functionality.
   * Employs modified Bureau of Public Roads (BPR) functions to simulate congestion and travel time under degraded conditions.
   * Supports both pre- and post-disaster traffic flow simulations.
(5). **Network Resilience Evaluation:**
   * Calculates time-variant system-level performance metrics such as emergency accessibility, average travel time, connectivity, and functional reachability.
   * Computes multi-dimensional resilience indicators (e.g., emergency response resilience, economic resilience) by integrating recovery curves.
   * Optional: includes functions for Monte Carlo simulation to assess uncertainty in hazard, vulnerability, and recovery.
These codes are organized into separate modules and can be run independently or integrated into a batch-processing pipeline. They allow users to test different hazard scenarios, repair strategies, and traffic patterns to evaluate BN resilience under varying assumptions.


