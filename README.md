# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm:
1.Import necessary libraries (numpy, matplotlib)

2.Generate synthetic data with a linear relationship and noise

3.Add bias term to the input features

4.Initialize model parameters randomly

5.Define gradient descent function to update parameters iteratively

6.Train the model using the gradient descent function

7.Print learned parameters (intercept and slope)

8.Visualize the results with a scatter plot and regression line
## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by:Mahalakshmi S
RegisterNumber: 25018377
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv("C:/Users/acer/Downloads/50_Startups.csv")
x = data["R&D Spend"].values
y = data["Profit"].values

# -------- Feature Scaling --------
x_mean = np.mean(x)
x_std = np.std(x)
x =(x-x_mean) / x_std
# Parameters
w = 0.0
b = 0.0
alpha = 0.01
epochs = 100
n = len(x)

losses = []
# Gradient Descent
for _ in range(epochs):
    y_hat = w*x+b
    loss = np.mean((y_hat - y)**2)
    losses.append(loss)
    
    dw = (2/n) * np.sum((y_hat - y)*x)
    db = (2/n) * np.sum(y_hat -y)
    
    w -= alpha * dw
    b -= alpha * db
# Plot
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(losses)
plt.xlabel("Iterations")
plt.ylabel("Loss(MSE)")
plt.title("Loss vs Interations")

plt.subplot(1,2,2)
plt.scatter(x,y)
x_sorted = np.argsort(x)
plt.plot(x[x_sorted],(w * x + b)[x_sorted], color='red')
plt.xlabel("R&D Spend (scaled)")
plt.ylabel("Profit")
plt.title("Linear Regression Fit")

plt.tight_layout()
plt.show()

print("Final weight (w):",w)
print("Final bias (b):",b)
*/

```

## Output:
<img width="1379" height="581" alt="Screenshot 2026-01-31 081202" src="https://github.com/user-attachments/assets/35523f40-13ff-4181-a8cb-e8975c366065" />


## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
