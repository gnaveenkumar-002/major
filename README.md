# Deep Autoencoder for Intrusion Detection

This repository contains the implementation of a **Deep Autoencoder-based Intrusion Detection System (IDS)** using the **NSL-KDD dataset**. The project leverages deep learning techniques to identify and respond to network anomalies and threats effectively.

## Project Overview

Network security is a critical concern in the modern digital era, with increasing threats ranging from data breaches to complex cyberattacks. This project addresses these challenges by developing an IDS based on deep autoencoder architectures. The system detects anomalies in network traffic by analyzing reconstruction errors, making it robust against both known and unknown threats.

### Key Features
- Utilizes **deep autoencoders** for unsupervised anomaly detection.
- Implements preprocessing techniques such as normalization and feature scaling.
- Evaluates performance using metrics like accuracy, precision, recall, and F1-score.
- Uses the **NSL-KDD dataset**, a benchmark dataset for intrusion detection research.

## Dataset

The **NSL-KDD dataset** is an improved version of the KDD Cup 99 dataset, addressing issues such as redundancy and class imbalance. The dataset consists of 41 features, divided into:
- **Categorical features**: Protocol type, service, and flag.
- **Continuous features**: Connection duration, byte count, etc.

Classes:
- **Normal traffic**
- **Attack types**: Denial of Service (DoS), Probing, User-to-Root (U2R), and Remote-to-Local (R2L).

For more information, refer to the official dataset documentation: [NSL-KDD Dataset](https://www.unb.ca/cic/datasets/nsl.html).

## Methodology

1. **Preprocessing**:
   - Handle missing values and outliers.
   - Normalize or standardize features.
   - Address data imbalance using resampling or class weights.

2. **Model Architecture**:
   - **Encoder**: Compresses input data to a lower-dimensional latent space.
   - **Decoder**: Reconstructs data from the latent space.
   - Reconstruction errors above a threshold are flagged as anomalies.

3. **Training and Evaluation**:
   - Early stopping to prevent overfitting.
   - Metrics: Accuracy, precision, recall, F1-score, and ROC-AUC.

## Results

The model achieved high accuracy and robustness in detecting network anomalies, surpassing traditional methods like decision trees and SVMs. Performance highlights:
- **Accuracy**: 98.50%
- **F1-Score**: High across multiple attack types.

## Dependencies

The project uses the following libraries:
- Python 3.7+
- TensorFlow
- NumPy
- Pandas
- Matplotlib
- Scikit-learn

Install dependencies using:
```bash
pip install -r requirements.txt
```

## How to Use

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/deep-autoencoder-ids.git
   ```
2. Navigate to the project directory:
   ```bash
   cd deep-autoencoder-ids
   ```
3. Run the training script:
   ```bash
   python train_autoencoder.py
   ```
4. Evaluate the model:
   ```bash
   python evaluate_model.py
   ```

## Future Work

- Integrate hybrid models combining autoencoders with supervised classifiers.
- Test the model on encrypted traffic patterns.
- Optimize computational efficiency for real-time deployment.

## Contributing

Contributions are welcome! Feel free to fork the repository and submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Researchers and contributors of the NSL-KDD dataset.
- Inspiration from various deep learning studies in cybersecurity.
