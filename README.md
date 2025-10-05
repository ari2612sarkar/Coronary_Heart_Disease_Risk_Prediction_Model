# Coronary Heart Disease (CHD) Risk Prediction

## ES60011 - Application of Machine Learning in Biological Systems  
**Project 2: Logistic Regression from Scratch**

## Overview
This project implements a logistic regression model from scratch (without using scikit-learn) to predict the 10-year risk of coronary heart disease (CHD) using the Framingham Heart Study dataset.

## Dataset
The dataset contains patient information including:
- **Demographics**: sex, age, education
- **Behavioral factors**: smoking status, cigarettes per day
- **Medical history**: blood pressure medication, stroke history, hypertension, diabetes
- **Current medical condition**: cholesterol, blood pressure, BMI, heart rate, glucose
- **Target variable**: `TenYearCHD` (1 = CHD risk, 0 = no risk)

**Source**: Framingham Heart Study  
**Samples**: 4,240 patients  
**Features**: 15 clinical features

## Project Structure
```
├── README.md
├── requirements.txt
├── data/
│   └── framingham.csv
├── notebooks/
│   └── 23IE10006_CHR_Prediction_notebook.ipynb
├── models/
│   └── 23IE10006_CHD_logistic_model.pkl
├── results/
   └── predictions.csv
```

## Implementation Details

### Model Architecture
- **Algorithm**: Logistic Regression with L2 Regularization
- **Optimization**: Gradient Descent
- **Features**: Standardized using z-score normalization
- **Missing Data**: Median imputation for continuous features, mode for categorical

### Hyperparameters
- Learning rate: 0.5
- Regularization parameter (λ): 1.0
- Max iterations: 20,000
- Convergence tolerance: 1e-7

## Results

### Performance Metrics
| Metric | Value |
|--------|-------|
| Accuracy | 86.79% |
| Precision | 90.00% |
| Recall | 7.50% |
| F1-Score | 0.138 |
| AUC-ROC | ~0.65 |

### Confusion Matrix
```
                Predicted
              No Risk  Risk
Actual No      727      1
Actual Yes     111      9
```

## Key Findings

**Strengths:**
- High precision (90%) - predictions of CHD risk are reliable
- Successfully implemented logistic regression from scratch
- Proper data preprocessing and standardization

**Limitations:**
- **Low recall (7.5%)** - model misses most patients with actual CHD risk
- **Class imbalance issue** - dataset heavily skewed toward "no risk" cases
- Model essentially predicts "no risk" for most patients

### Why Low Recall?
The dataset has severe class imbalance (~85% no risk vs ~15% risk). The model optimizes for overall accuracy, which is achieved by predicting the majority class.

## Future Improvements

1. **Address class imbalance:**
   - Implement SMOTE (Synthetic Minority Over-sampling)
   - Add class weights to loss function
   - Adjust decision threshold (< 0.5)

2. **Model optimization:**
   - Cross-validation for hyperparameter tuning
   - Grid search for optimal λ and learning rate
   - Feature importance analysis


## Dependencies
- Python 3.8+
- NumPy
- Pandas
- Matplotlib
- Jupyter Notebook


## File Descriptions

- `23IE10006_CHR_Prediction_notebook.ipynb`: Main analysis notebook
- `framingham.csv`: Raw dataset
- `predictions.csv`: Model predictions on test set
- `23IE10006_CHD_logistic_model.pkl`: Trained model weights and parameters

## Author
**Roll Number**: 23IE10006  
**Course**: ES60011 - Application of Machine Learning in Biological Systems  
**Institution**: IIT Kharagpur

## License
This project is for educational purposes as part of coursework.

## Acknowledgments
- Framingham Heart Study for the dataset
