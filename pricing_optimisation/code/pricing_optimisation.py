import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression

# Generate synthetic data
np.random.seed(42)

# =============================================================================
# random: This is a submodule in NumPy that provides functions for generating
# pseudo-random numbers.
#
# seed(42): The number 42 is an arbitrary choice for the seed.it means that every
# time you run the program, you will get the same sequence of random numbers.
# This is useful for debugging and ensuring that your results are reproducible
# =============================================================================

# Generate random customer data
# =============================================================================
# I will be using backslash (\) quite a few times to specify that the code is
# continuing in the next line
# =============================================================================

customer_data = pd.DataFrame({
    'customer_id': np.arange(1, 501),
    'customer_age': np.random.randint(18, 65, size=500),
    'customer_gender': np.random.choice(['Male', 'Female'], size=500),
    'customer_location': np.random.choice(['Urban', 'Suburban', 'Rural'], \
                                          size=500)
})

# Generate random product data
product_data = pd.DataFrame({
    'product_id': np.arange(1, 11),
    'product_category': np.random.choice(['Electronics', 'Clothing', 'Home', \
                                          'Sports'], size=10),
    'product_brand': np.random.choice(['Brand_A', 'Brand_B', 'Brand_C'], size=10),
    'product_price': np.random.uniform(50, 500, size=10),
    'min_price': np.random.uniform(40,4900, size=10),
    'cm': np.random.uniform(50,500, size = 10)/1000
})

# Generate random sales data
sales_data = pd.DataFrame({
    'customer_id': np.random.choice(np.arange(1, 501), size=1000),
    'product_id': np.random.choice(np.arange(1, 11), size=1000),
    'sales_quantity': np.random.randint(1, 10, size=1000)
})

# Merge datasets to consolidate all the data and join on primary keys
all_data = pd.merge(customer_data, sales_data, on="customer_id")
all_data = pd.merge(all_data, product_data, on="product_id")

# =============================================================================
# Define features such as what you want to evaluate the model on and target variable
# Apart from internal factors (features) such as customer age,CLV, marketing campaign,
# you can also include external factors such as GDP, Inflation, Unemployment rate,
# salary levels, competition price, market trend index, time of the day, seasonality,
# weather conditions etc. It will make the model's predictions better.
# In this example, I am using only customer age and product price as features
# =============================================================================

features = ['customer_age', 'product_price']

# =============================================================================
# As we want to optimize the price, it is essential for us to know the demand
# of each of our products. Therefore, our target variable would be sales_quantity
# =============================================================================

target_variable = 'sales_quantity'

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(all_data[features], \
                    all_data[target_variable], test_size=0.2, random_state=42)

# Standardize numerical features
scaler = StandardScaler()

# =============================================================================
# Applying the scale using fit_transorm for X_train
# The fit method analyzes the data and learns the transformation that needs to
# be applied. After fitting, the transformer is ready to be used to transform
# or preprocess new data.
# Basically When you call fit_transform on the training set, the model learns
# the parameters needed for transformation (e.g., mean and standard deviation
# for standardization based on the training data.
# It scales or transforms the training data accordingly.
# =============================================================================

X_train_scaled = scaler.fit_transform(X_train)

# =============================================================================
# Applying the scale using only transform for X_test. Once the model has been
# fitted to the training data, it has learned the transformation parameters.
# When you want to apply the same transformation to the testing set, you use
# transform without fitting the model again. This ensures that the testing data
# is scaled or transformed using the parameters learned from the training data.
# This is important because in real-world scenarios, your model will encounter
# new, unseen data, and you want to evaluate its performance on such data.
# Using fit_transform on the testing set could lead to data leakage.If you were
# to fit the scaler again on the testing set, it might learn different parameters
# , and your evaluation would not accurately represent how the model would
# perform on new, unseen data.
# =============================================================================

X_test_scaled = scaler.transform(X_test)

# Train a linear regression model (you can replace it with your preferred model)
model = LinearRegression()

# =============================================================================
# You can also use the neural network model using TensorFlow (Keras) as follows
# model = tf.keras.Sequential([
#      tf.keras.layers.Dense(64, activation='relu', \
#                            input_shape=(X_train_scaled.shape[1],)),
#     tf.keras.layers.Dense(32, activation='relu'),
#      tf.keras.layers.Dense(1)
#  ])
#
# # Compile the model using adam optimizer and MSE as a loss function
# model.compile(optimizer='adam', loss='mean_squared_error')
# model.fit(X_train_scaled, y_train, epochs=10, batch_size=32, validation_split=0.2)
# =============================================================================

# =============================================================================
# Now all the preprocessing has been done. It is time to train the model
# You can tune your hyperparameters in deep learning models as well
# such as epochs, batch_size, validation_split, optimizer, loss function,
# learning rate, the depth of decision trees activation function, # of hidden
# layers etc.
# Hyperparameters are those parameters over which you have control.
# The other parameters such as weights, biases, coefficients, split points and
# leaf values are something that model learns by itself
# =============================================================================

model.fit(X_train_scaled, y_train)

# Predict sales quantity for the test set
y_pred = model.predict(X_test_scaled)

# Evaluate the model on the test set
y_pred = model.predict(X_test_scaled)
test_rmse = mean_squared_error(y_test, y_pred, squared=False)
print(f"Test RMSE: {test_rmse:.2f}")

# Make predictions on the entire dataset
all_data["predicted_sales"] = model.predict(scaler.transform(all_data[features]))

# =============================================================================
# Now all the model parameters have been defined. It is time to train the model
# You can tune your hyperparameters such as epochs, batch_size, validation_split,
# optimizer, loss function, learning rate, the depth of decision trees
# activation function, number of hidden layers etc.
# Hyperparameters are those parameters over which you have control.
# The other parameters such as weights, biases, coefficients, split points and
# leaf values are something that model learns by itself
# =============================================================================

# =============================================================================
# You can also choose to display sample predictions using the following code
# print(all_data[['customer_id', 'product_id', 'sales_quantity', 'predicted_sales']]\
#       .head(10))
# =============================================================================

# =============================================================================
# Now our predicted sales information is ready. It is time to optimize the
# pricing of our products. However before that we should also calculate
# mean centering, standard deviation, and scaling.
#
# Here we are applying scaling techniques to obtain the scaling factor.This factor
# is calculated based on how far each predicted sale value deviates from the mean,
# normalized by the standard deviation. It introduces a form of normalization or
# scaling to the predicted sales. By applying scaling, we can standardize the sales
# for individual products. The use of mean and standard deviation in the scaling
# factor calculation implies that the process is influenced by the distribution of
# the predicted sales. If there are outliers in the predicted sales, the scaling
# factor will be sensitive to them. Values that are far off the mean will have a
# higher scaling factor, that will influence the final outcome i.e. optimized price.
# As you will observe that I have used 0.1 in the calculation of the scaling factor.
# You can also adjust it if you want a larger or smaller impact of the factor.
# In summary, this step aims to adjust the predicted sales values based on their
# distribution, making them comparable and potentially more suitable for downstream
# processes, analyses, or decision-making. It's a common preprocessing step to
# ensure that the data behaves in a desirable way
# =============================================================================

mean_sales = all_data["predicted_sales"].mean()
std_sales = all_data["predicted_sales"].std()
scaling_factor = 1 +  (0.1 * (all_data["predicted_sales"] - mean_sales) / std_sales)
all_data['scf'] = scaling_factor.astype(float)

# Calculate adjusted price based on the above variables
all_data['adj_psc'] = all_data['product_price'] * all_data['scf']

# =============================================================================
# Apply custom adjustment based on margin and minimum price.
# In this line of code, we are calculating another price based on either the
# margin that we want to maintain on the price or the minimum price below which
# we cannot or do not want to go
# apbm = adjusted price based on the margin. The margins that we defined in our
# product data set that we generated above.
# =============================================================================

all_data["apbm"] = all_data["product_price"] * (1 + all_data["cm"])

# =============================================================================
# In the following line of code we are taking the maximum of all the prices. This code
# can be modified based on your business case of optimizing the price of the products.
# I am taking maximum of all the prices. However, it can also be adjusted to customize
# based on your requirements
# =============================================================================

all_data['adj_price'] = np.maximum(\
                                   np.maximum(all_data['min_price'], \
                                              all_data['adj_psc']) , \
                                       all_data['apbm'])
# =============================================================================
# One can also use relatively less memory way of adjusting the logic. Note that
# using a combination of loop and if-else statements is less efficient than vectorization
#
# for index, row in all_data.iterrows():
#     if row['apbm'] <= row['adj_psc']:
#         all_data.at[index, 'adj_price'] = row['adj_psc']
#     else:
#         all_data.at[index, 'adj_price'] = row['min_p']
# =============================================================================

# Display the DataFrame with adjusted prices

print(all_data[['product_price','scf','min_price',\
               'adj_psc','apbm','adj_price']].sample(n=10))
