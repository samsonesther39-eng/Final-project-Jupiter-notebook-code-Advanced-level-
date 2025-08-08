# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Generate dataset
np.random.seed(0)

# Create a dictionary to store the data
data = {
    'Bedrooms': np.random.randint(1, 6, 100),
    'Bathrooms': np.random.randint(1, 4, 100),
    'SqFt': np.random.randint(1000, 3001, 100),
    'Price': np.random.randint(100000, 500001, 100)
}

# Create a DataFrame from the data
df = pd.DataFrame(data)

# Display the first few rows of the dataset
print("Dataset:")
print(df.head())

# Split the data into features (X) and target (y)
X = df[['Bedrooms', 'Bathrooms', 'SqFt']]
y = df['Price']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train a Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Display model coefficients
print("\nModel Coefficients:")
print(f"Intercept: {model.intercept_}")
print(f"Coefficients: {model.coef_}")

# Make predictions and evaluate the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

# Display predictions and evaluation metrics
print("\nPredictions:")
print(y_pred)
print("\nActual Prices:")
print(y_test.values)
print(f'\nMean Squared Error: {mse:.2f}')
print(f'Root Mean Squared Error: {rmse:.2f}')