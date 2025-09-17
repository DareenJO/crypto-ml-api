from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import requests
from datetime import datetime
import os

app = Flask(__name__)
CORS(app)

class CryptoMLPredictor:
    def __init__(self):
        self.price_model = None
        self.direction_model = None
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.model_accuracy = {}
        self.last_training = None
        
    def fetch_historical_data(self, symbol, days=200):
        """Fetch real data from CoinGecko"""
        try:
            url = f"https://api.coingecko.com/api/v3/coins/{symbol}/market_chart"
            params = {'vs_currency': 'usd', 'days': days, 'interval': 'daily'}
            
            response = requests.get(url, params=params, timeout=10)
            if response.status_code != 200:
                return None
                
            data = response.json()
            if 'prices' not in data:
                return None
                
            prices = data['prices']
            df = pd.DataFrame(prices, columns=['timestamp', 'price'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Add synthetic volume for calculations
            df['volume'] = df['price'] * np.random.uniform(0.8, 1.2, len(df)) * 1000000
            
            return df
            
        except Exception as e:
            print(f"Data fetch error: {e}")
            return None
    
    def calculate_rsi(self, prices, window=14):
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / (loss + 0.0001)  # Avoid division by zero
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)
    
    def engineer_features(self, df):
        """Create technical features"""
        try:
            if df is None or len(df) < 30:
                return None
                
            # Price features
            df['returns'] = df['price'].pct_change().fillna(0)
            df['price_ma_7'] = df['price'].rolling(7).mean().fillna(df['price'])
            df['price_ma_21'] = df['price'].rolling(21).mean().fillna(df['price'])
            
            # Technical indicators
            df['rsi'] = self.calculate_rsi(df['price'])
            
            # MACD
            ema12 = df['price'].ewm(span=12).mean()
            ema26 = df['price'].ewm(span=26).mean()
            df['macd'] = (ema12 - ema26).fillna(0)
            df['macd_signal'] = df['macd'].ewm(span=9).mean().fillna(0)
            
            # Bollinger Bands
            rolling_mean = df['price'].rolling(20).mean()
            rolling_std = df['price'].rolling(20).std()
            df['bb_width'] = (rolling_std * 2).fillna(df['price'].std())
            
            # Volume and momentum
            df['volume_ma'] = df['volume'].rolling(21).mean().fillna(df['volume'])
            df['volume_ratio'] = (df['volume'] / df['volume_ma']).fillna(1)
            df['volatility'] = df['returns'].rolling(21).std().fillna(0.01)
            df['price_position'] = ((df['price'] - df['price_ma_21']) / df['price_ma_21']).fillna(0)
            df['momentum_3'] = df['price'].pct_change(3).fillna(0)
            
            # Target variables
            df['future_return_1d'] = df['price'].pct_change(1).shift(-1)
            df['future_direction'] = (df['future_return_1d'] > 0).astype(int)
            
            self.feature_columns = [
                'returns', 'rsi', 'macd', 'macd_signal', 'bb_width', 
                'volume_ratio', 'volatility', 'price_position', 'momentum_3'
            ]
            
            # Fill any NaN values
            for col in self.feature_columns:
                df[col] = df[col].fillna(0)
            
            return df
            
        except Exception as e:
            print(f"Feature engineering error: {e}")
            return None
    
    def train_models(self, df):
        """Train ML models"""
        try:
            if df is None:
                return False
                
            required_cols = self.feature_columns + ['future_return_1d', 'future_direction']
            df_clean = df[required_cols].dropna()
            
            if len(df_clean) < 50:
                return False
            
            X = df_clean[self.feature_columns].values
            y_price = df_clean['future_return_1d'].values
            y_direction = df_clean['future_direction'].values
            
            # Train/test split
            X_train, X_test, y_price_train, y_price_test, y_dir_train, y_dir_test = train_test_split(
                X, y_price, y_direction, test_size=0.3, random_state=42
            )
            
            # Scale features
            self.scaler.fit(X_train)
            X_train_scaled = self.scaler.transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train models (smaller for Vercel limits)
            self.price_model = RandomForestRegressor(n_estimators=20, max_depth=5, random_state=42)
            self.direction_model = RandomForestClassifier(n_estimators=20, max_depth=5, random_state=42)
            
            self.price_model.fit(X_train_scaled, y_price_train)
            self.direction_model.fit(X_train_scaled, y_dir_train)
            
            # Evaluate
            price_pred = self.price_model.predict(X_test_scaled)
            dir_pred = self.direction_model.predict(X_test_scaled)
            
            price_mae = np.mean(np.abs(price_pred - y_price_test))
            dir_accuracy = np.mean(dir_pred == y_dir_test)
            
            self.model_accuracy = {
                'price_mae': float(price_mae),
                'direction_accuracy': float(dir_accuracy),
                'training_samples': len(df_clean),
                'cv_folds': 1
            }
            
            self.last_training = datetime.now()
            return True
            
        except Exception as e:
            print(f"Training error: {e}")
            return False
    
    def predict(self, current_data):
        """Make predictions"""
        try:
            if self.price_model is None:
                return None
            
            features = [float(current_data.get(col, 0)) for col in self.feature_columns]
            features_array = np.array(features).reshape(1, -1)
            features_scaled = self.scaler.transform(features_array)
            
            price_pred = float(self.price_model.predict(features_scaled)[0])
            direction_pred = int(self.direction_model.predict(features_scaled)[0])
            
            try:
                direction_proba = self.direction_model.predict_proba(features_scaled)[0]
                confidence = float(max(direction_proba) * 100)
            except:
                confidence = 65.0
            
            # Generate recommendation
            if direction_pred == 1 and price_pred > 0.02:
                recommendation = 'BUY'
            elif direction_pred == 0 and price_pred < -0.02:
                recommendation = 'SELL'
            else:
                recommendation = 'HOLD'
            
            return {
                'prediction': {
                    'direction': 'UP' if direction_pred == 1 else 'DOWN',
                    'expected_return': price_pred,
                    'confidence': confidence,
                    'recommendation': recommendation
                },
                'model_info': {
                    'last_training': self.last_training.isoformat() if self.last_training else None,
                    'accuracy_metrics': self.model_accuracy
                }
            }
            
        except Exception as e:
            print(f"Prediction error: {e}")
            return None

# Global predictor instance
predictor = CryptoMLPredictor()

@app.route('/', methods=['GET'])
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'models_trained': predictor.price_model is not None,
        'platform': 'Vercel Serverless'
    })

@app.route('/train/<symbol>', methods=['POST'])
def train_model(symbol):
    try:
        df = predictor.fetch_historical_data(symbol, days=150)  # Reduced for speed
        if df is None:
            return jsonify({'error': 'Failed to fetch data'}), 400
        
        df = predictor.engineer_features(df)
        if df is None:
            return jsonify({'error': 'Feature engineering failed'}), 400
        
        success = predictor.train_models(df)
        if not success:
            return jsonify({'error': 'Training failed'}), 500
        
        return jsonify({
            'message': f'Models trained for {symbol}',
            'accuracy': predictor.model_accuracy,
            'last_training': predictor.last_training.isoformat(),
            'data_points': len(df)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict/<symbol>', methods=['GET'])
def predict_crypto(symbol):
    try:
        df = predictor.fetch_historical_data(symbol, days=60)
        if df is None:
            return jsonify({'error': 'Failed to fetch data'}), 400
        
        df = predictor.engineer_features(df)
        if df is None:
            return jsonify({'error': 'Failed to process data'}), 400
        
        current_features = {col: float(df[col].iloc[-1]) for col in predictor.feature_columns}
        
        prediction = predictor.predict(current_features)
        if prediction is None:
            return jsonify({'error': 'Prediction failed'}), 500
        
        return jsonify({
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            **prediction
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/model_info', methods=['GET'])
def model_info():
    return jsonify({
        'models_trained': predictor.price_model is not None,
        'last_training': predictor.last_training.isoformat() if predictor.last_training else None,
        'accuracy_metrics': predictor.model_accuracy,
        'feature_columns': predictor.feature_columns
    })

# Vercel serverless function handler
def handler(event, context):
    return app

# For local development
if __name__ == '__main__':
    app.run(debug=True)