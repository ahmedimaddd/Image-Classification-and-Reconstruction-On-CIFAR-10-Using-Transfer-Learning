# Advanced Image Classification and Reconstruction on CIFAR-10

This repository contains the code for a comprehensive deep learning project exploring both supervised and unsupervised learning on the CIFAR-10 dataset. The project is divided into two main parts:

1.  **Image Classification:** Fine-tuning state-of-the-art transfer learning models (MobileNetV3 and EfficientNet-B0) to achieve high-accuracy image classification.
2.  **Image Reconstruction:** Designing and training a Convolutional Autoencoder (CAE) from scratch to perform unsupervised image reconstruction.

## Key Achievements

* **High-Accuracy Classification:** Achieved a peak test accuracy of **95.84%** on CIFAR-10 with a fine-tuned EfficientNet-B0 model.
* **High-Fidelity Reconstruction:** The optimized Convolutional Autoencoder achieved an excellent test Mean Squared Error (MSE) of **0.000754**.
* **Systematic Hyperparameter Tuning:** Conducted extensive experiments to optimize learning rates, dropout, and layer unfreezing points for all models, with results tracked and visualized using TensorBoard.
* **Problem Solving:** Successfully navigated and resolved practical engineering challenges, including model underfitting, checkpoint loading issues, and cloud environment configuration on RunPod.

## Project Breakdown

### Part 1: Image Classification with Transfer Learning

This part of the project leverages pre-trained models to classify images from the CIFAR-10 dataset.

* **Models:** MobileNetV3-Small, EfficientNet-B0
* **Methodology:**
    * Applied model-specific preprocessing (resizing to 224x224, normalizing to `[-1, 1]`) to match ImageNet pre-training requirements.
    * Utilized data augmentation (random flips, rotations) to prevent overfitting and improve generalization.
    * Implemented a two-stage fine-tuning process: first training only the classification head, then unfreezing layers of the base model for end-to-end tuning.
    * Conducted a comparative analysis of model performance, confirming that EfficientNet-B0's compound scaling led to superior accuracy.

### Part 2: Image Reconstruction with a Convolutional Autoencoder (CAE)

This part of the project demonstrates unsupervised learning by training a CAE to compress and then reconstruct images.

* **Architecture:** A symmetric encoder-decoder model built with `Conv2D` and `Conv2DTranspose` layers. The encoder downsamples the input image into a low-dimensional latent space (the "bottleneck"), and the decoder upsamples from this space to reconstruct the original image.
* **Methodology:**
    * The model was trained to minimize the Mean Squared Error (MSE) between the original and reconstructed images.
    * Performed hyperparameter tuning on the learning rate and the number of filters in the bottleneck layer to find the optimal configuration.

## Technologies & Frameworks

* **Core Framework:** TensorFlow, Keras
* **Data Handling & Visualization:** NumPy, Pandas, Matplotlib, Seaborn, TensorBoard
* **Dimensionality Reduction:** Scikit-learn (for t-SNE visualization)
* **Platform:** RunPod

## Challenges & Learnings

* **Environment Management:** Encountered and resolved significant library and resource constraints when attempting to implement a Swin Transformer, leading to the strategic choice of the CAE. This highlighted the importance of matching model complexity to available hardware.
* **Checkpointing:** Addressed issues with saving model weights in `.h5` format by adapting the saving mechanism to use the more robust `.keras` format.
* **Hyperparameter Impact:** The experiments clearly demonstrated the critical impact of learning rate and regularization (dropout) on model performance and generalization.

## How to Run

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/ahmedimaddd/your-repo-name.git](https://github.com/ahmedimaddd/your-repo-name.git)
    ```
2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Run the notebook:**
    Open and run the Jupyter Notebook to train the models and see the evaluation results. The notebook is structured to handle both the classification and reconstruction tasks.

For a comprehensive analysis of the methodology and results, please see the full project report included in this repository.
