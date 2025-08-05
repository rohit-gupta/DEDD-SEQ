# DEDD-SEQ: Dual-Encoder Denoising-Decoder for Vid2GPS Sequence Prediction

A PyTorch implementation of **DEDD-SEQ**, a frame-to-GPS retrieval framework for global video geolocalization, introducing temporal alignment (TempGeo) and refinement (GeoRefiner) modules for precise, consistent trajectory prediction.

---

## Table of Contents

1. Features
2. Installation
3. Data Preparation
4. Usage

   * Training Phase I
   * Training Phase II
   * Inference
5. Model Architecture
6. Evaluation
7. Configuration
8. Citation
9. License

---

## Features

* **Dual Frame Encoder** combining CLIP and DINOv2 features for robust, semantically aligned frame representations.
* **TempGeo**, a lightweight transformer for temporal self-attention to stabilize jittery predictions.
* **GeoRefiner**, an encoder–decoder module that refines noisy GPS estimates through cross-attention with visual context.
* **Global-scale retrieval** via a uniformly generated GPS grid gallery.
* **End-to-end pipeline** with two-phase training and inference for temporally consistent video geolocalization.

---

## Installation

1. Clone the repository to your local machine.
2. Create and activate a Python virtual environment.
3. Upgrade pip and install the required packages listed in `requirements.txt`.
4. Ensure that PyTorch and Transformers are installed to load CLIP and DINOv2 weights automatically.

---

## Data Preparation

* **Mapillary Street-Level Sequences (MSLS)**: Register on the official dataset page, download the training splits, and extract frame images and GPS annotations.
* **Uniform Grid Gallery**: Use the `scripts/build_gallery.py` tool with the training frames directory to generate a global GPS-grid gallery file (e.g., `gallery_msls.pkl`).
* **Optional Datasets**: Follow the README files in `data/` for instructions on using GAMa, BDD100K, and CityGuessr68k datasets.

---

## Usage

### Training Phase I

Train the dual-encoder, TempGeo, and location encoder in a contrastive setup by running the phase I training script with the appropriate configuration file, training data root, gallery file, and output checkpoint directory.

### Training Phase II

With the components from Phase I frozen, train the GeoRefiner module using the Phase II training script and corresponding configuration, data root, gallery file, and output path.

### Inference

Perform end-to-end trajectory prediction on a directory of video frames by invoking the inference script, providing the video frames directory, Phase I and Phase II checkpoints, gallery file, and output destination for the trajectory CSV.

---

## Model Architecture

* **Dual Frame Encoder**: Extracts vision features from CLIP ViT and DINOv2 ViT backbones, concatenates them, projects via an MLP, and normalizes to unit length.
* **TempGeo**: A transformer encoder applying self-attention over sequential frame embeddings to enforce temporal consistency.
* **Location Encoder**: Transforms GPS coordinates using an equal-area projection combined with random Fourier feature mappings and projects via MLPs.
* **GeoRefiner**: A transformer encoder–decoder where the decoder attends over noisy GPS query embeddings and the encoder attends over the temporally refined visual embeddings.

Refer to the paper for detailed module schematics and equations.

---

## Evaluation

* **Metrics** include retrieval accuracy thresholds (e.g., within 500 m, 1 km, 5 km, 25 km), median distance error, Discrete Fréchet Distance, and Mean Range Difference.
* Use the provided evaluation script with predicted trajectory CSV, ground-truth annotations, and desired metrics to compute results.
* Visualize output trajectories on a map by running the visualization script, which produces a trajectory overlay image.

---

## Configuration

Experiment parameters are defined in YAML files under `configs/`. Key options include choice of visual backbone, number of transformer layers, gallery resolution, learning rates, batch sizes, and optimizer settings.

---

## Citation

When using this code, please cite the following:

DEDD-SEQ: Dual-Encoder Denoising-Decoder for Vid2GPS Sequence Prediction, CVPR 2025.

---

## License

This project is distributed under the MIT License. See the `LICENSE` file for details.
