import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor # Stochastic Gradient Descent
from sklearn.preprocessing import StandardScaler# Standard Scaler for normalizing the data
from lab_utils_multi import  load_house_data
from lab_utils_common import dlc
np.set_printoptions(precision=2)
plt.style.use('./deeplearning.mplstyle')

#Step 1 : Load the data set/ Train the data
X_train, y_train = load_house_data()
X_features = ['size(sqft)','bedrooms','floors','age']

X_norm = 0


print(X_train.shape , y_train.shape) ## X= (99, 4) Y =(99,)

max  = np.max(X_train, axis=0)
min  = np.min(X_train,axis=0)
print(f"Max values by column: {max} and Min values by column: {min}")




"""Step 2: Scale the data: Scale/normalize the training data"""

scaler = StandardScaler()
X_norm = scaler.fit_transform(X_train)
print(f"Peak to Peak range by column in Raw        X:{np.ptp(X_train,axis=0)}")   
print(f"Peak to Peak range by column in Normalized X:{np.ptp(X_norm,axis=0)}")




"""Step 3: Fit the data: Fit the model using the scaled data"""
sgdr = SGDRegressor(max_iter=1000)
sgdr.fit(X_norm, y_train)
print(sgdr)
print(f"number of iterations completed: {sgdr.n_iter_}, number of weight updates: {sgdr.t_}")


"""get the e parameters of the model w and b"""
b_norm = sgdr.intercept_
w_norm = sgdr.coef_
print(f"model parameters:                   w: {w_norm}, b:{b_norm}")
print( "model parameters from previous lab: w: [110.56 -21.27 -32.71 -37.97], b: 363.16")

#Step 4: Predict the data: Predict the target value for the test data
# make a prediction using sgdr.predict()
y_pred_sgd = sgdr.predict(X_norm)
# make a prediction using w,b. 
y_pred = np.dot(X_norm, w_norm) + b_norm  
print(f"prediction using np.dot() and sgdr.predict match: {(y_pred == y_pred_sgd).all()}")

print(f"Prediction on training set:\n{y_pred[:4]}" )
print(f"Target values \n{y_train[:4]}")

# plot predictions and targets vs original features    
fig,ax=plt.subplots(1,4,figsize=(12,3),sharey=True)
for i in range(len(ax)):
    ax[i].scatter(X_train[:,i],y_train, label = 'target')
    ax[i].set_xlabel(X_features[i])
    ax[i].scatter(X_train[:,i],y_pred,color=dlc["dlorange"], label = 'predict')
ax[0].set_ylabel("Price"); ax[0].legend();
fig.suptitle("target versus prediction using z-score normalized model")
plt.show()
