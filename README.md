# House Price Prediction using Stochastic Gradient Descent

This project demonstrates the use of **Stochastic Gradient Descent (SGD)** for predicting house prices based on features such as size, number of bedrooms, number of floors, and age. The implementation uses Python and the `scikit-learn` library to preprocess the data, train the model, and make predictions.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Dependencies](#dependencies)
- [Usage](#usage)
- [Code Explanation](#code-explanation)
- [Results](#results)

---

## Overview

The goal of this project is to predict house prices using a linear regression model trained with **Stochastic Gradient Descent**. The dataset includes features such as:
- `size(sqft)`: Size of the house in square feet.
- `bedrooms`: Number of bedrooms.
- `floors`: Number of floors.
- `age`: Age of the house.

The project involves the following steps:
1. Loading the dataset.
2. Normalizing the features using Z-score normalization.
3. Training the model using `SGDRegressor` from `scikit-learn`.
4. Making predictions and visualizing the results.

---

## Features

- **Data Normalization**: Uses `StandardScaler` to normalize the input features.
- **Model Training**: Implements linear regression using `SGDRegressor`.
- **Visualization**: Plots the predicted vs. actual house prices for each feature.

---

## Dependencies

The following Python libraries are required to run the code:

- `numpy`
- `matplotlib`
- `scikit-learn`

Install them using pip if not already installed:

```bash
pip install numpy matplotlib scikit-learn
```

---

## Usage

1. Clone the repository or copy the `main.py` file to your local machine.
2. Ensure the dataset is available in the correct format. The `load_house_data` function in `lab_utils_multi.py` loads the dataset from `data/houses.txt`.
3. Run the script using Python:

```bash
python main.py
```

4. The script will output:
   - Model parameters (`w` and `b`).
   - Predictions for the first few training examples.
   - A plot comparing the predicted and actual house prices for each feature.

---

## Code Explanation

### Step 1: Load the Dataset

The dataset is loaded using the `load_house_data` function:

```python
X_train, y_train = load_house_data()
X_features = ['size(sqft)', 'bedrooms', 'floors', 'age']
```

### Step 2: Normalize the Data

The features are normalized using Z-score normalization:

```python
scaler = StandardScaler()
X_norm = scaler.fit_transform(X_train)
```

### Step 3: Train the Model

The model is trained using `SGDRegressor`:

```python
sgdr = SGDRegressor(max_iter=1000)
sgdr.fit(X_norm, y_train)
```

### Step 4: Make Predictions

Predictions are made using the trained model:

```python
y_pred_sgd = sgdr.predict(X_norm)
```

### Step 5: Visualize the Results

The script generates a plot comparing the predicted and actual house prices for each feature:

```python
fig, ax = plt.subplots(1, 4, figsize=(12, 3), sharey=True)
for i in range(len(ax)):
    ax[i].scatter(X_train[:, i], y_train, label='target')
    ax[i].scatter(X_train[:, i], y_pred, color=dlc["dlorange"], label='predict')
```

---

## Results

The script outputs:
1. **Model Parameters**: The weights (`w`) and bias (`b`) of the trained model.
2. **Predictions**: The predicted house prices for the first few training examples.
3. **Visualization**: A plot comparing the predicted and actual house prices for each feature.

---

## Example Output

Sample output from the script:

```
Max values by column: [3490.    5.    3.  100.] and Min values by column: [852.   1.   1.   1.]
Peak to Peak range by column in Raw        X:[2638.    4.    2.   99.]
Peak to Peak range by column in Normalized X:[2. 2. 2. 2.]
number of iterations completed: 1000, number of weight updates: 100000
model parameters:                   w: [110.56 -21.27 -32.71 -37.97], b:[363.16]
prediction using np.dot() and sgdr.predict match: True
Prediction on training set:
[450.12 232.45 512.34 390.67]
Target values 
[450. 230. 510. 390.]
```

---

## Acknowledgments

This project is part of a machine learning course focusing on **Multiple Linear Regression** and **Stochastic Gradient Descent**.