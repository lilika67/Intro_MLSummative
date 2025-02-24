# Crop Yield Classifier

## Overview
This project explores the impact of various optimization techniques on the performance of a crop yield classification model that helps farmers make informed decisions by classifying agricultural yields into three categories: high yield, medium yield, or low yield. The classification assists farmers in:
- Understanding their land's production potential
- Planning resource allocation effectively
- Making data-driven farming decisions
- Optimizing agricultural practices based on predicted outcomes

The model implements different machine learning techniques, comparing their performance using key error analysis metrics to ensure reliable yield predictions.

## Classification Categories
- **High Yield**: Indicates optimal production conditions and farming practices
- **Medium Yield**: Suggests average production with potential for improvement
- **Low Yield**: Signals need for intervention in farming practices or conditions

# Project Resources

## Video Presentation
[Watch the video presentation on YouTube](https://www.youtube.com/watch?v=-CjEXIuyB2E)

## Project Notebook
[View the notebook on Google Colab](https://colab.research.google.com/github/lilika67/Intro_MLSummative/blob/main/Summative_Intro_to_ml_%5BKayitesi_Liliane%5D_assignment.ipynb)

## Training Instances and Optimization Techniques

| Instance | Optimizer | Regularizer | Epochs | Early Stopping | Layers | Learning Rate | Accuracy | F1-Score | Recall | Precision |
|----------|-----------|-------------|--------|----------------|---------|---------------|----------|-----------|----------|------------|
| 1 | Adam | None | 50 | No | 3 | 0.001 | 0.5502 | 0.486 | 0.547 | 0.520 |
| 2 | Nadam | L2+Dropout | 50 | Yes | 4 | 0.0005 | 0.5270 | 0.044 | N/A | N/A |
| 3 | RMSprop | L2/L1 | 100 | Yes | 5 | 0.0001 | 0.5270 | 0.496 | 0.541 | 0.509 |
| 4 | Adam | Experimental | 75 | Yes | 4 | 0.0003 | 0.5270 | 0.493 | 0.552 | 0.532 |

## Getting Started


### Prerequisites
* Python 3.8+
* Jupyter Notebook
* Required packages:
```bash
pip install -r requirements.txt
```

### Running the Notebook
1. Clone the repository:
```bash
git clone https://github.com/lilika67/Intro_MLSummative.git
cd Intro_MLSummative
```

2. Start Jupyter Notebook:
```bash
jupyter notebook
```

3. Open `Summative_Intro_to_ml_[Kayitesi_Liliane]_assignment.ipynb` and run all cells

### Loading the Best Model
The best performing model (Instance 1: Adam optimizer, 3 layers) can be loaded using:
```python
from tensorflow.keras.models import load_model
# Load the saved model
model = load_model('saved_models/agriculture_yield_bestmodel.keras')

# For predictions
# Returns predictions as: 0 (Low Yield), 1 (Medium Yield), or 2 (High Yield)
predictions = model.predict(X_test)
```

## Summary of Results

### Best Performing Combination
From the results, the combination of **Adam optimizer without regularization** (Instance 1) yielded the highest accuracy of 0.5502. While Instance 3 with **RMSprop and L1/L2 regularization** achieved the highest F1-score of 0.496, showing good balance between precision and recall. The use of a simpler model architecture (3 layers) with a moderate learning rate of 0.001 contributed to better model performance in Instance 1.

### Comparison: Machine Learning Algorithm vs Neural Network
Looking at the neural network configurations, models with **3 layers** (Instances 1 and 4) generally performed better in terms of accuracy compared to deeper architectures. However, the **5-layer network** with RMSprop and L1/L2 regularization showed strengths in achieving a balanced F1-score, suggesting better handling of class imbalance.

## Key Findings

* **Regularization**: L1/L2 regularization (Instance 3) helped achieve a high F1-score (0.496), while the L2+Dropout combination (Instance 2) led to a significant drop in F1-score (0.044), suggesting that dropout might be too aggressive for this particular problem
* **Optimizers**: Adam optimizer demonstrated the best performance in terms of accuracy, while RMSprop showed promise in balancing precision and recall as evidenced by the F1-score
* **Early Stopping**: Results show mixed effectiveness - models with early stopping (Instances 2, 3, and 5) maintained consistent accuracy (0.5270) but varied significantly in F1-scores, suggesting that early stopping helped prevent accuracy degradation but didn't necessarily improve overall model performance

## Practical Applications
The model's classifications can help farmers:
- **High Yield Prediction**: Identify and replicate successful farming practices
- **Medium Yield Prediction**: Find opportunities for yield improvement
- **Low Yield Prediction**: Take early corrective actions to improve productivity

## Conclusion
This project highlights how different optimization techniques impact crop yield classification. The results indicate that simpler architectures with Adam optimizer and careful learning rate selection (0.001) perform best for accuracy, while RMSprop with L1/L2 regularization is more suitable when balanced precision and recall are priority. Early stopping should be carefully considered as its benefits may vary depending on other hyperparameters.
