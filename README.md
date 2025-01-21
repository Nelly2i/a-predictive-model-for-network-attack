# A Predictive Model for Network Attacks Using Machine Learning Algorithms

## Project Overview
This project involves the development of a predictive model for network attacks using a Convolutional Neural Network (CNN). The aim is to address the growing challenge of detecting and mitigating cyber threats by leveraging machine learning techniques. The project includes model development, evaluation, and the creation of an API endpoint for potential integration into real-world applications.

---

## Objectives
1.	To collect and preprocess network traffic data for analysis.
2.	To identify relevant features that contribute to network attacks.
3.	To evaluate various machine learning algorithms for predicting network attacks.
4.	To develop and train a predictive model using the most effective algorithm.
5.	To assess the performance of the model using standard evaluation metrics.
6.	To provide recommendations for implementing the model in real-world network security systems.

---

## Features
- **Data Preprocessing:** Cleaning and normalizing raw dataset to ensure quality.
- **Machine Learning Model:** CNN-based model trained to detect network attacks.
- **API Endpoint:** Enables organizations to send data and receive predictions.
- **Performance Metrics:** Comprehensive evaluation to ensure the reliability of the system.

---

## Repository Structure
```plaintext
predictive-model-network-attacks/
├── README.md
├── data/
│   ├── raw/                # Raw datasets
│   ├── processed/          # Processed datasets
├── src/
│   ├── models/             # CNN model implementation
│   ├── utils/              # Utility functions
├── notebooks/
│   ├── exploratory_data_analysis.ipynb
│   ├── model_training.ipynb
├── api/
│   ├── app.py              # Flask/FastAPI application
│   ├── requirements.txt    # Python dependencies
├── tests/
│   ├── test_model.py       # Unit tests for the model
│   ├── test_api.py         # Unit tests for the API
├── docs/
│   ├── literature_review.md
│   ├── system_design.md
│   ├── methodology.md
│   ├── evaluation_results.md
│   ├── future_work.md
├── LICENSE
└── .gitignore
```

---

## Getting Started
### Prerequisites
1. Python 3.8 or later.
2. Libraries: TensorFlow, NumPy, pandas, Flask/FastAPI, and others listed in `requirements.txt`.

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/Nelly2i/a-predictive-model-for-network-attack.git
   ```
2. Navigate to the project directory:
   ```bash
   cd predictive-model-network-attacks
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Usage
1. Run the API server:
   ```bash
   python flask_network_attack.py
   ```
2. Send a POST request to the endpoint:
   ```
   /predict
   ```
   - Input: JSON payload with network traffic data.
   - Output: Predicted network attack type.

---

## Model Description
The CNN model was chosen for its effectiveness in feature extraction and pattern recognition. Key components include:
- **Input Layer:** Processes raw network traffic data.
- **Convolutional Layers:** Extract hierarchical features.
- **Fully Connected Layers:** Perform classification.
- **Output Layer:** Predicts the type of network attack.

---

## Evaluation Metrics
- **Accuracy:** Measures the proportion of correctly identified attacks.
- **Precision:** Assesses the correctness of positive predictions.
- **Recall:** Evaluates the model’s ability to detect all relevant instances.
- **F1-Score:** Balances precision and recall for an overall performance metric.

---

## Results
The CNN model achieved the following metrics during evaluation:
- Accuracy: **99.76%**
- ROC-AUC Score: **99.99%**
- Mean Cross Validation Accuracy: **99.76%**
Classification Report
```
              precision    recall  f1-score   support
  
     ipsweep       1.00      1.00      1.00      1088
       satan       1.00      0.99      1.00       710
   portsweep       0.99      1.00      1.00       410
        back       1.00      1.00      1.00       268
      normal       0.98      1.00      0.99       122

    accuracy                           1.00      2598
   macro avg       0.99      1.00      1.00      2598
weighted avg       1.00      1.00      1.00      2598
```
---

## Future Work
1. Enhance the model for real-time data processing.
2. Improve accuracy by experimenting with hybrid models.
3. Expand datasets to include more attack types.
4. Integrate the API into a comprehensive security information and event management (SIEM) system.

---

## Acknowledgments
Special thanks to contributors and the open-source community for providing datasets and libraries that made this project possible.

