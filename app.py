"""
Real Machine Learning Cryptocurrency Trading Prediction API
"""
from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import requests
from datetime import datetime
import logging
import warnings
import os

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)

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
        
    def fetch_historical_data(self, symbol, days=365):
        """Fetch real historical data from CoinGecko"""
        try:
            url = f"https://api.coingecko.com/api/v3/coins/{symbol}/market_chart"
            params = {
                'vs_currency': 'usd', 
                'days': days, 
                'interval': 'daily'
            }
            
            response = requests.get(url, params=params, timeout=20)
            if response.status_code != 200:
                return None
                
            data = response.json()
            if 'prices' not in data:
                return None
                
            # Create DataFrame
            prices = data['prices']
            df = pd.DataFrame(prices, columns=['timestamp', 'price'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Add volume (use prices if volume not available)
            if 'total_volumes' in data and len(data['total_volumes']) == len(prices):
                volumes = data['total_volumes']
                volume_data = [v[1] for v in volumes]
                df['volume'] = volume_data
            else:
                df['volume'] = df['price'] * 1000000  # Synthetic volume
            
            return df
            
        except Exception as e:
            print(f"Error fetching data: {e}")
            return None
    
    def calculate_rsi(self, prices, window=14):
        """Simple RSI calculation without TA-Lib"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)
    
    def calculate_macd(self, prices, fast=12, slow=26):
        """Simple MACD calculation"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        return macd.fillna(0)
    
    def engineer_features(self, df):
        """Create technical features without TA-Lib"""
        try:
            if df is None or len(df) < 50:
                return None
                
            # Basic price features
            df['returns'] = df['price'].pct_change().fillna(0)
            df['price_ma_7'] = df['price'].rolling(window=7).mean().fillna(df['price'])
            df['price_ma_21'] = df['price'].rolling(window=21).mean().fillna(df['price'])
            
            # Technical indicators (simplified)
            df['rsi'] = self.calculate_rsi(df['price'])
            df['macd'] = self.calculate_macd(df['price'])
            df['macd_signal'] = df['macd'].ewm(span=9).mean().fillna(0)
            
            # Bollinger Bands (simplified)
            rolling_mean = df['price'].rolling(window=20).mean()
            rolling_std = df['price'].rolling(window=20).std()
            df['bb_high'] = rolling_mean + (rolling_std * 2)
            df['bb_low'] = rolling_mean - (rolling_std * 2)
            df['bb_width'] = (df['bb_high'] - df['bb_low']).fillna(df['price'].std())
            
            # Volume indicators
            df['volume_ma'] = df['volume'].rolling(window=21).mean().fillna(df['volume'])
            df['volume_ratio'] = (df['volume'] / df['volume_ma']).fillna(1)
            
            # Additional features
            df['volatility'] = df['returns'].rolling(window=21).std().fillna(0.01)
            df['price_position'] = ((df['price'] - df['price_ma_21']) / df['price_ma_21']).fillna(0)
            df['momentum_3'] = df['price'].pct_change(periods=3).fillna(0)
            
            # Target variables
            df['future_return_1d'] = df['price'].pct_change(periods=1).shift(-1)
            df['future_direction'] = (df['future_return_1d'] > 0).astype(int)
            
            # Feature list
            self.feature_columns = [
                'returns', 'rsi', 'macd', 'macd_signal', 'bb_width', 
                'volume_ratio', 'volatility', 'price_position', 'momentum_3'
            ]
            
            # Fill any remaining NaN values
            for col in self.feature_columns:
                df[col] = df[col].fillna(0)
            
            return df
            
        except Exception as e:
            print(f"Feature engineering error: {e}")
            return None
    
    def train_models(self, df):
        """Train models with simple train/test split"""
        try:
            if df is None:
                return False
                
            # Prepare data
            required_cols = self.feature_columns + ['future_return_1d', 'future_direction']
            df_clean = df[required_cols].dropna()
            
            if len(df_clean) < 100:
                return False
            
            X = df_clean[self.feature_columns].values
            y_price = df_clean['future_return_1d'].values
            y_direction = df_clean['future_direction'].values
            
            # Simple train/test split
            X_train, X_test, y_price_train, y_price_test, y_dir_train, y_dir_test = train_test_split(
                X, y_price, y_direction, test_size=0.2, random_state=42
            )
            
            # Scale features
            self.scaler.fit(X_train)
            X_train_scaled = self.scaler.transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train models
            self.price_model = RandomForestRegressor(
                n_estimators=50, 
                max_depth=8, 
                random_state=42
            )
            
            self.direction_model = RandomForestClassifier(
                n_estimators=50, 
                max_depth=8, 
                random_state=42
            )
            
            # Fit models
            self.price_model.fit(X_train_scaled, y_price_train)
            self.direction_model.fit(X_train_scaled, y_dir_train)
            
            # Evaluate
            price_pred = self.price_model.predict(X_test_scaled)
            dir_pred = self.direction_model.predict(X_test_scaled)
            
            # Calculate metrics
            price_mae = np.mean(np.abs(price_pred - y_price_test))
            dir_accuracy = np.mean(dir_pred == y_dir_test)
            
            self.model_accuracy = {
                'price_mae': float(price_mae),
                'price_mae_std': 0.0,
                'direction_accuracy': float(dir_accuracy),
                'direction_accuracy_std': 0.0,
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
            if self.price_model is None or self.direction_model is None:
                return None
            
            # Prepare features
            features = []
            for col in self.feature_columns:
                value = current_data.get(col, 0)
                if pd.isna(value) or np.isinf(value):
                    value = 0
                features.append(float(value))
            
            features_array = np.array(features).reshape(1, -1)
            features_scaled = self.scaler.transform(features_array)
            
            # Get predictions
            price_pred = float(self.price_model.predict(features_scaled)[0])
            direction_pred = int(self.direction_model.predict(features_scaled)[0])
            
            try:
                direction_proba = self.direction_model.predict_proba(features_scaled)[0]
                confidence = float(max(direction_proba) * 100)
            except:
                confidence = 60.0
            
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

# Initialize predictor
predictor = CryptoMLPredictor()

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'models_trained': predictor.price_model is not None
    })

@app.route('/train/<symbol>', methods=['POST'])
def train_model(symbol):
    try:
        # Fetch data
        df = predictor.fetch_historical_data(symbol, days=365)
        if df is None:
            return jsonify({'error': 'Failed to fetch data'}), 400
        
        # Engineer features
        df = predictor.engineer_features(df)
        if df is None:
            return jsonify({'error': 'Feature engineering failed'}), 400
        
        # Train models
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
        # Fetch recent data
        df = predictor.fetch_historical_data(symbol, days=60)
        if df is None:
            return jsonify({'error': 'Failed to fetch data'}), 400
        
        # Engineer features
        df = predictor.engineer_features(df)
        if df is None:
            return jsonify({'error': 'Failed to process data'}), 400
        
        # Get current features
        current_features = {}
        for col in predictor.feature_columns:
            current_features[col] = float(df[col].iloc[-1])
        
        # Make prediction
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

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)
