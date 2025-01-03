import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import joblib

def prepare_data(df):
    data = df.copy()
    data['Date'] = pd.to_datetime(data['Date'])
    le = LabelEncoder()
    data['Index_Encoded'] = le.fit_transform(data['Index'])
    data['Prev_Close'] = data.groupby('Index')['Close'].shift(1)
    data['Price_Range'] = data['High'] - data['Low']
    data['Price_Change'] = data['Close'] - data['Open']
    data = data.dropna()
    features = ['Index_Encoded', 'Open', 'High', 'Low', 'Prev_Close', 
                'Price_Range', 'Price_Change']
    X = data[features]
    y = data['Close']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    
    return X_scaled, y, le, scaler

def train_and_evaluate_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                        random_state=42, shuffle=False)
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'max_features': ['auto', 'sqrt']
    }
    base_model = RandomForestRegressor(random_state=42)
    grid_search = GridSearchCV(estimator=base_model, param_grid=param_grid,
                             cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    model = grid_search.best_estimator_
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    cv_scores = cross_val_score(model, X, y, cv=5)
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    joblib.dump(model, "random_forest_model.joblib")

    return {
        'model': model,
        'metrics': {
            'r2_score': r2,
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
            'cv_scores_mean': cv_scores.mean(),
            'cv_scores_std': cv_scores.std()
        },
        'feature_importance': feature_importance,
        'test_actual': y_test,
        'test_predicted': y_pred,
        'best_params': grid_search.best_params_
    }

def predict_close_price(model, le, scaler, index, open_price, high, low, prev_close):
    price_range = high - low
    price_change = open_price - prev_close
    index_encoded = le.transform([index])
    input_data = pd.DataFrame({
        'Index_Encoded': index_encoded,
        'Open': [open_price],
        'High': [high],
        'Low': [low],
        'Prev_Close': [prev_close],
        'Price_Range': [price_range],
        'Price_Change': [price_change]
    })
    input_scaled = scaler.transform(input_data)
    
    return model.predict(input_scaled)[0]

# Load and prepare data + models...
df = pd.read_csv('Data.csv')
X, y, le, scaler = prepare_data(df)
joblib.dump(le, "label_encoder.joblib")
joblib.dump(scaler, "scaler.joblib")

results = train_and_evaluate_model(X, y)

print("\nModel Performance Metrics:")
print(f"R² Score: {results['metrics']['r2_score']:.4f}")
print(f"Root Mean Squared Error: {results['metrics']['rmse']:.4f}")
print(f"Mean Absolute Error: {results['metrics']['mae']:.4f}")
print(f"Mean Absolute Percentage Error: {results['metrics']['mape']:.2f}%")
print(f"Cross-validation Score: {results['metrics']['cv_scores_mean']:.4f} (+/- {results['metrics']['cv_scores_std']*2:.4f})")
print("\nBest Parameters:", results['best_params'])

prediction = predict_close_price(results['model'], le, scaler, 'NYA', 532.070007, 532.070007, 532.070007, 531.119995)
print(f"\nPredicted Close Price: {prediction:.2f}")