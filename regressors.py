import pandas as pd
import numpy as np
from sklearn.linear_model import ElasticNet, Ridge, LinearRegression, Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import KFold, cross_val_score
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# This code conducts regression analysis on the Air Quality dataset from the UCI Machine Learning Repository

df = pd.read_csv('data/AirI.csv', header=None)
X = df.iloc[:, 1:].to_numpy()
y = df.iloc[:, 0].to_numpy()

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42, shuffle=True)

model = ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=20000, tol=0.000001, fit_intercept=True)

if len(y_train.shape) == 1:
    y_train = y_train.reshape(-1, 1)
if len(y_test.shape) == 1:
    y_test = y_test.reshape(-1, 1)

model.fit(X_train, y_train)

score = model.score(X_test, y_test)

y_pred = model.predict(X_test)

# Print Mean Squared Error (Linear Regression)
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))

np.transpose(X)

# Check if the matrix X^T * X is singular
M = np.transpose(X).dot(X)
X = df.iloc[:, 1:].to_numpy()
y = df.iloc[:, 0].to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42, shuffle=True)
np.linalg.det(M)

# Make the matrix non-singular by adding a small value to the diagonal
np.linalg.det(M+0.000001*np.eye(len(M)))

# Calculate X^T * X
X_transpose = X.T
X_product = np.dot(X_transpose, X)

# Calculate the determinant of X^T * X
determinant = np.linalg.det(X_product)
print("Matrix Determinant:", int(determinant))

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Apply polynomial features if needed
poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train_scaled)
X_test_poly = poly.transform(X_test_scaled)

# Initialize and train Ridge regression model
alpha = 0.01  # Regularization strength
model = Ridge(alpha=alpha, fit_intercept=True)
model.fit(X_train_scaled, y_train)

# Make predictions and evaluate the model
y_pred = model.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)

print("Mean Squared Error (Ridge Regression):", mse)

lasso_regressor = Lasso(max_iter=50000)
ridge_regressor = Ridge(max_iter=50000)
elasticnet_regressor = ElasticNet(max_iter=50000)

# Set alpha values for Lasso, Ridge, and ElasticNet
alphas_lasso = np.arange(0.001, 2.001, 0.001)
alphas_ridge = np.arange(0.1, 200.1, 0.1)
alphas_elasticnet = np.arange(0.001, 2.001, 0.001)

# Initialize lists to store MSE values
mse_lasso = []
mse_ridge = []
mse_elasticnet = []

# Perform 10-fold cross-validation with MSE as the scoring metric
for alpha in alphas_lasso:
    lasso_regressor.alpha = alpha
    mse_scores = cross_val_score(lasso_regressor, X, y, cv=10, scoring='neg_mean_squared_error')
    mse_lasso.append(-np.mean(mse_scores))

for alpha in alphas_ridge:
    ridge_regressor.alpha = alpha
    mse_scores = cross_val_score(ridge_regressor, X, y, cv=10, scoring='neg_mean_squared_error')
    mse_ridge.append(-np.mean(mse_scores))

for alpha in alphas_elasticnet:
    elasticnet_regressor.alpha = alpha
    mse_scores = cross_val_score(elasticnet_regressor, X, y, cv=10, scoring='neg_mean_squared_error')
    mse_elasticnet.append(-np.mean(mse_scores))


#print("\nLasso MSE:", mse_lasso)
#print("\nRidge MSE:", mse_ridge)
#print("\nElastic Net MSE:", mse_elasticnet)


# Load your dataset and split it into features (X) and target (y)

# Create Lasso, Ridge, and ElasticNet regressors
lasso_regressor = Lasso(max_iter=50000)
ridge_regressor = Ridge(max_iter=50000)
elasticnet_regressor = ElasticNet(max_iter=50000)

# Set alpha values for Lasso, Ridge, and ElasticNet
alphas_lasso = np.arange(0.001, 2.001, 0.001)
alphas_ridge = np.arange(0.1, 200.1, 0.1)
alphas_elasticnet = np.arange(0.001, 2.001, 0.001)

# Initialize lists to store MSE values
mse_lasso = []
mse_ridge = []
mse_elasticnet = []

# Perform 10-fold cross-validation with MSE as the scoring metric
for alpha in alphas_lasso:
    lasso_regressor.alpha = alpha
    mse_scores = cross_val_score(lasso_regressor, X, y, cv=10, scoring='neg_mean_squared_error')
    mse_lasso.append(-np.mean(mse_scores))

for alpha in alphas_ridge:
    ridge_regressor.alpha = alpha
    mse_scores = cross_val_score(ridge_regressor, X, y, cv=10, scoring='neg_mean_squared_error')
    mse_ridge.append(-np.mean(mse_scores))

for alpha in alphas_elasticnet:
    elasticnet_regressor.alpha = alpha
    mse_scores = cross_val_score(elasticnet_regressor, X, y, cv=10, scoring='neg_mean_squared_error')
    mse_elasticnet.append(-np.mean(mse_scores))

# Find the alpha value that resulted in the lowest MSE for each regressor
best_alpha_lasso = alphas_lasso[np.argmin(mse_lasso)]
best_alpha_ridge = alphas_ridge[np.argmin(mse_ridge)]
best_alpha_elasticnet = alphas_elasticnet[np.argmin(mse_elasticnet)]

# Print the results
print("\nLasso:")
print("Best Alpha:", best_alpha_lasso)
print("Average MSE:", min(mse_lasso))

print("\nRidge:")
print("Best Alpha:", best_alpha_ridge)
print("Average MSE:", min(mse_ridge))

print("\nElasticNet:")
print("Best Alpha:", best_alpha_elasticnet)
print("Average MSE:", min(mse_elasticnet))