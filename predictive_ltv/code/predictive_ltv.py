import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import RandomizedSearchCV

# Set the random seed for reproducibility
np.random.seed(42)

def generate_user_data(n_samples):
    # Engagement metrics
    session_length = np.random.normal(loc=15, scale=5, size=n_samples)
    session_frequency = np.random.poisson(lam=3, size=n_samples)
    time_spent_core = np.random.normal(loc=10, scale=3, size=n_samples)
    time_spent_social = np.random.normal(loc=5, scale=2, size=n_samples)
    levels_completed = np.random.randint(0, 100, size=n_samples)
    achievements = np.random.randint(0, 50, size=n_samples)
    social_interactions = np.random.poisson(lam=5, size=n_samples)

    # Retention metrics
    days_since_first_play = np.random.randint(0, 365, size=n_samples)
    days_since_last_play = np.random.randint(0, 30, size=n_samples)
    days_played_first_7 = np.random.randint(0, 8, size=n_samples)
    days_played_first_30 = np.random.randint(0, 31, size=n_samples)
    retention_day_1 = np.random.choice([0, 1], size=n_samples, p=[0.4, 0.6])
    retention_day_7 = np.random.choice([0, 1], size=n_samples, p=[0.6, 0.4])
    retention_day_30 = np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3])

    # Monetization metrics
    number_of_purchases = np.random.poisson(lam=2, size=n_samples)
    time_to_first_purchase = np.random.exponential(scale=7, size=n_samples)
    purchase_frequency = np.random.exponential(scale=0.1, size=n_samples)
    avg_purchase_value = np.random.lognormal(mean=2, sigma=1, size=n_samples)
    total_spending = number_of_purchases * avg_purchase_value
    items_purchased = np.random.choice(['Currency', 'Boost', 'Cosmetic', 'Bundle'], size=n_samples)

    # Player progression
    current_level = np.random.randint(1, 101, size=n_samples)
    progress_speed = np.random.uniform(0.1, 1.0, size=n_samples)
    skill_level = np.random.choice(['Beginner', 'Intermediate', 'Advanced', 'Expert'], size=n_samples)

    # In-game economy metrics
    virtual_currency_balance = np.random.randint(0, 10000, size=n_samples)
    resource_accumulation_rate = np.random.uniform(0, 10, size=n_samples)
    resource_spending_rate = np.random.uniform(0, 8, size=n_samples)

    # User acquisition data
    acquisition_channel = np.random.choice(['Social Media', 'Search Engine', 'App Store', 'Referral'], size=n_samples)
    acquisition_cost = np.random.lognormal(mean=1, sigma=0.5, size=n_samples)
    install_date = pd.date_range(end=datetime.now(), periods=n_samples)
    country = np.random.choice(['USA', 'UK', 'Canada', 'Germany', 'India', 'Brazil', 'Japan'], size=n_samples)

    # Device and technical data
    device_type = np.random.choice(['Mobile', 'Tablet', 'Desktop'], size=n_samples)
    operating_system = np.random.choice(['iOS', 'Android', 'Windows', 'macOS'], size=n_samples)
    app_version = np.random.choice(['1.0', '1.1', '1.2', '2.0'], size=n_samples)

    # Behavioral segments
    player_type = np.random.choice(['Casual', 'Competitive', 'Social'], size=n_samples)
    spending_category = np.random.choice(['Non-spender', 'Minnow', 'Dolphin', 'Whale'], size=n_samples, p=[0.7, 0.2, 0.08, 0.02])

    # Seasonality and time-based features
    day_of_week = np.random.choice(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'], size=n_samples)
    time_of_day = np.random.choice(['Morning', 'Afternoon', 'Evening', 'Night'], size=n_samples)
    season = np.random.choice(['Spring', 'Summer', 'Fall', 'Winter'], size=n_samples)
    is_holiday = np.random.choice([0, 1], size=n_samples, p=[0.9, 0.1])

    # Game-specific metrics
    core_loops_completed = np.random.poisson(lam=10, size=n_samples)
    event_participation = np.random.choice([0, 1], size=n_samples, p=[0.3, 0.7])
    feature_usage = np.random.uniform(0, 1, size=n_samples)

    # LTV calculations
    initial_spend = avg_purchase_value * np.random.random(size=n_samples)
    ltv_day_0 = initial_spend
    ltv_day_1 = initial_spend * retention_day_1
    ltv_day_7 = ltv_day_1 * retention_day_7 + np.random.normal(loc=5, scale=2, size=n_samples) * retention_day_7

    # Construct the DataFrame
    user_data = pd.DataFrame({
        'Session Length': session_length,
        'Session Frequency': session_frequency,
        'Time Spent Core': time_spent_core,
        'Time Spent Social': time_spent_social,
        'Levels Completed': levels_completed,
        'Achievements': achievements,
        'Social Interactions': social_interactions,
        'Days Since First Play': days_since_first_play,
        'Days Since Last Play': days_since_last_play,
        'Days Played First 7': days_played_first_7,
        'Days Played First 30': days_played_first_30,
        'Retention Day 1': retention_day_1,
        'Retention Day 7': retention_day_7,
        'Retention Day 30': retention_day_30,
        'Number of Purchases': number_of_purchases,
        'Time to First Purchase': time_to_first_purchase,
        'Purchase Frequency': purchase_frequency,
        'Avg Purchase Value': avg_purchase_value,
        'Total Spending': total_spending,
        'Items Purchased': items_purchased,
        'Current Level': current_level,
        'Progress Speed': progress_speed,
        'Skill Level': skill_level,
        'Virtual Currency Balance': virtual_currency_balance,
        'Resource Accumulation Rate': resource_accumulation_rate,
        'Resource Spending Rate': resource_spending_rate,
        'Acquisition Channel': acquisition_channel,
        'Acquisition Cost': acquisition_cost,
        'Install Date': install_date,
        'Country': country,
        'Device Type': device_type,
        'Operating System': operating_system,
        'App Version': app_version,
        'Player Type': player_type,
        'Spending Category': spending_category,
        'Day of Week': day_of_week,
        'Time of Day': time_of_day,
        'Season': season,
        'Is Holiday': is_holiday,
        'Core Loops Completed': core_loops_completed,
        'Event Participation': event_participation,
        'Feature Usage': feature_usage,
        'LTV Day 0': ltv_day_0,
        'LTV Day 1': ltv_day_1,
        'LTV Day 7': ltv_day_7,
    })
    return user_data

# Generate synthetic user data
n_samples = 10000
user_data = generate_user_data(n_samples)

# Calculate LTV for longer periods
def calculate_ltv(data, days):
    return data['Total Spending'] * (1 + data['Retention Day 30']) * (days / 30)

user_data['LTV_30'] = calculate_ltv(user_data, 30)
user_data['LTV_90'] = calculate_ltv(user_data, 90)
user_data['LTV_365'] = calculate_ltv(user_data, 365)

# Prepare features and targets
feature_columns = [col for col in user_data.columns if col not in ['LTV Day 0', 'LTV Day 1', 'LTV Day 7', 'LTV_30', 'LTV_90', 'LTV_365', 'Install Date']]
target_columns = ['LTV Day 0', 'LTV Day 1', 'LTV Day 7', 'LTV_30', 'LTV_90', 'LTV_365']

# Identify numeric and categorical columns
numeric_features = user_data[feature_columns].select_dtypes(include=['int64', 'float64']).columns
categorical_features = user_data[feature_columns].select_dtypes(include=['object']).columns

# Create preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# Create a function to train and evaluate the model
def train_and_evaluate_model(X, y, target_name):
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create the pipeline
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', GradientBoostingRegressor(random_state=42))
    ])

    # Define the parameter grid for RandomizedSearchCV
    param_grid = {
        'regressor__n_estimators': [100, 200, 300],
        'regressor__max_depth': [3, 4, 5],
        'regressor__learning_rate': [0.01, 0.1, 0.2],
        'regressor__min_samples_split': [2, 5, 10],
        'regressor__min_samples_leaf': [1, 2, 4],
        'regressor__subsample': [0.8, 0.9, 1.0]
    }

    # Perform RandomizedSearchCV
    random_search = RandomizedSearchCV(model, param_distributions=param_grid, n_iter=20, cv=3, random_state=42, n_jobs=-1)
    random_search.fit(X_train, y_train)

    # Get the best model
    best_model = random_search.best_estimator_

    # Make predictions
    y_pred = best_model.predict(X_test)

    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"\nResults for {target_name}:")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"R-squared: {r2:.2f}")

    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': best_model.named_steps['preprocessor'].get_feature_names_out(), #Call get_feature_names_out on the fitted preprocessor in the pipeline
        'importance': best_model.named_steps['regressor'].feature_importances_
    })
    print("\nTop 10 Feature Importance:")
    print(feature_importance.sort_values('importance', ascending=False).head(10))

    return best_model

# Train models for each LTV period
models = {}
for target in target_columns:
    print(f"\nTraining model for {target}")
    X = user_data[feature_columns]
    y = user_data[target]
    models[target] = train_and_evaluate_model(X, y, target)

# Function to predict LTV for a new user
def predict_ltv(new_user_data):
    ltv_predictions = {}
    for target in target_columns:
        ltv_predictions[target] = models[target].predict(new_user_data)[0]
    return ltv_predictions

# Example: Predict LTV for a new user
new_user = generate_user_data(1)  # Generate data for one new user
new_user_ltv = predict_ltv(new_user[feature_columns])

print("\nPredicted LTV for new user:")
for period, ltv in new_user_ltv.items():
    print(f"{period}: ${ltv:.2f}")
