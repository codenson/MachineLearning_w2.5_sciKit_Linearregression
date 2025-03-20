# Polynomial Regression with Feature Engineering

This project demonstrates the implementation of **Polynomial Regression** using **Gradient Descent** in Python. The code explores how feature engineering can improve the performance of regression models by adding polynomial features.

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

The goal of this project is to predict a target variable `y` based on input features `x` by applying polynomial regression. The code includes examples of:
1. Linear regression without feature engineering.
2. Quadratic regression by adding a squared feature (`x**2`).
3. Polynomial regression by adding multiple features (`x`, `x**2`, `x**3`).

---

## Features

- **Feature Engineering**: Adds polynomial features (`x**2`, `x**3`) to improve model accuracy.
- **Gradient Descent Optimization**: Uses the `run_gradient_descent_feng` function to optimize weights and bias.
- **Visualization**: Plots the actual vs. predicted values for each regression model.

---

## Dependencies

The following Python libraries are required to run the code:

- `numpy`
- `matplotlib`

Install them using pip if not already installed:

```bash
pip install numpy matplotlib
```

---

## Usage

1. Clone the repository or copy the `main.py` file to your local machine.
2. Run the script using Python:

```bash
python main.py
```

3. The script will generate three plots:
   - Linear regression without feature engineering.
   - Quadratic regression with the `x**2` feature.
   - Polynomial regression with `x`, `x**2`, and `x**3` features.

---

## Code Explanation

### Linear Regression

The first part of the code performs linear regression without any feature engineering:

```python
x = np.arange(0, 20, 1)
y = 1 + x**2
X = x.reshape(-1, 1)

model_w, model_b = run_gradient_descent_feng(X, y, iterations=1000, alpha=1e-2)

def print_linear_graph():
    plt.scatter(x, y, marker='x', c='r', label="Actual Value")
    plt.plot(x, X @ model_w + model_b, label="Predicted Value")
    plt.title("no feature engineering")
    plt.xlabel("X")
    plt.ylabel("y")
    plt.legend()
    plt.show()

print_linear_graph()
```

### Quadratic Regression

The second part adds a squared feature (`x**2`) to improve the model:

```python
X = x**2
X = X.reshape(-1, 1)

def print_quadratic_graph_1():
    model_w, model_b = run_gradient_descent_feng(X, y, iterations=10000, alpha=1e-5)
    plt.scatter(x, y, marker='x', c='r', label="Actual Value")
    plt.plot(x, np.dot(X, model_w) + model_b, label="Predicted Value")
    plt.title("Added x**2 feature")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.show()

print_quadratic_graph_1()
```

### Polynomial Regression

The final part adds multiple polynomial features (`x`, `x**2`, `x**3`):

```python
X = np.c_[x, x**2, x**3]

def print_polynomial_graph():
    model_w, model_b = run_gradient_descent_feng(X, y, iterations=10000, alpha=1e-7)
    plt.scatter(x, y, marker='x', c='r', label="Actual Value")
    plt.plot(x, X @ model_w + model_b, label="Predicted Value")
    plt.title("x, x**2, x**3 features")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.show()

print_polynomial_graph()
```

---

## Results

The script generates three plots:
1. **Linear Regression**: The model struggles to fit the data due to the lack of polynomial features.
2. **Quadratic Regression**: Adding the `x**2` feature significantly improves the fit.
3. **Polynomial Regression**: Adding `x`, `x**2`, and `x**3` features further enhances the model's ability to capture the data's complexity.

---

Enjoy experimenting with polynomial regression and feature engineering!# MachineLearning_w2.5_sciKit_Linearregression
