# Crop Yield Classifier

## Overview
This project explores the impact of various optimization techniques on the performance of a crop yield classification model. It implements different machine learning models with and without optimization techniques, comparing their performance using key error analysis metrics.

## Training Instances and Optimization Techniques

| Instance | Optimizer | Regularizer | Epochs | Early Stopping | Layers | Learning Rate | Accuracy | F1-Score |
|----------|-----------|-------------|--------|----------------|---------|---------------|----------|-----------|
| 1 | Adam | None | 50 | No | 3 | 0.001 | 0.5502 | 0.486 |
| 2 | Nadam | L2+Dropout | 50 | Yes | 4 | 0.0005 | 0.5270 | 0.044 |
| 3 | RMSprop | L1/L2 | 100 | Yes | 5 | 0.0001 | 0.5270 | 0.496 |
| 4 | Nadam | None | 50 | No | 3 | 0.01 | 0.5270 | - |
| 5 | Adam | Experimental | 75 | Yes | 4 | 0.0003 | 0.5270 | 0.493 |

## Summary of Results

### Best Performing Combination
From the experiments, the combination of **Adam optimizer without regularization** (Instance 1) yielded the highest accuracy of 0.5502. While Instance 3 with **RMSprop and L1/L2 regularization** achieved the highest F1-score of 0.496, showing good balance between precision and recall. The use of a simpler model architecture (3 layers) with a moderate learning rate of 0.001 contributed to better model performance in Instance 1.

### Comparison: Machine Learning Algorithm vs Neural Network
Looking at the neural network configurations, models with **3 layers** (Instances 1 and 4) generally performed better in terms of accuracy compared to deeper architectures. However, the **5-layer network** with RMSprop and L1/L2 regularization showed strengths in achieving a balanced F1-score, suggesting better handling of class imbalance.

## Key Findings

* **Regularization**: L1/L2 regularization (Instance 3) helped achieve a high F1-score (0.496), while the L2+Dropout combination (Instance 2) led to a significant drop in F1-score (0.044), suggesting that dropout might be too aggressive for this particular problem
* **Optimizers**: Adam optimizer demonstrated the best performance in terms of accuracy, while RMSprop showed promise in balancing precision and recall as evidenced by the F1-score
* **Early Stopping**: Results show mixed effectiveness - models with early stopping (Instances 2, 3, and 5) maintained consistent accuracy (0.5270) but varied significantly in F1-scores, suggesting that early stopping helped prevent accuracy degradation but didn't necessarily improve overall model performance

## Conclusion
This project highlights how different optimization techniques impact crop yield classification. The results indicate that simpler architectures with Adam optimizer and careful learning rate selection (0.001) perform best for accuracy, while RMSprop with L1/L2 regularization is more suitable when balanced precision and recall are priority. Early stopping should be carefully considered as its benefits may vary depending on other hyperparameters.
