"""
Real Machine Learning Cryptocurrency Trading Prediction API
"""
from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
import requests
import ta
from datetime import datetime
import logging
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
        try:
            url = f"https://api.coingecko.com/api/v3/coins/{symbol}/market_chart"
            params = {'vs_currency': 'usd', 'days': days, 'interval': 'daily'}
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            prices = data['prices']
            volumes = data['total_volumes']
            
            df = pd.DataFrame(prices, columns=['timestamp', 'price'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            volume_df = pd.DataFrame(volumes, columns=['timestamp', 'volume'])
            volume_df['timestamp'] = pd.to_datetime(volume_df['timestamp'], unit='ms')
            volume_df.set_index('timestamp', inplace=True)
            
            df = df.join(volume_df)
            
            logger.info(f"Fetched {len(df)} days of historical data for {symbol}")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {str(e)}")
            return None
    
    def engineer_features(self, df):
        try:
            df['returns'] = df['price'].pct_change()
            df['price_ma_7'] = df['price'].rolling(7).mean()
            df['price_ma_21'] = df['price'].rolling(21).mean()
            
            df['rsi'] = ta.momentum.RSIIndicator(df['price']).rsi()
            df['macd'] = ta.trend.MACD(df['price']).macd()
            df['macd_signal'] = ta.trend.MACD(df['price']).macd_signal()
            df['bb_high'] = ta.volatility.BollingerBands(df['price']).bollinger_hband()
            df['bb_low'] = ta.volatility.BollingerBands(df['price']).bollinger_lband()
            df['bb_width'] = df['bb_high'] - df['bb_low']
            
            df['volume_ma'] = df['volume'].rolling(21).mean()
            df['volume_ratio'] = df['volume'] / df['volume_ma']
            
            df['volatility'] = df['returns'].rolling(21).std()
            df['price_position'] = (df['price'] - df['price_ma_21']) / df['price_ma_21']
            df['momentum_3'] = df['price'].pct_change(3)
            
            df['future_return_1d'] = df['price'].pct_change(1).shift(-1)
            df['future_direction'] = (df['future_return_1d'] > 0).astype(int)
            
            self.feature_columns = [
                'returns', 'rsi', 'macd', 'macd_signal', 'bb_width', 'volume_ratio',
                'volatility', 'price_position', 'momentum_3'
            ]
            
            return df
            
        except Exception as e:
            logger.error(f"Error in feature engineering: {str(e)}")
            return df
    
    def train_models(self, df):
        try:
            df_clean = df[self.feature_columns + ['future_return_1d', 'future_direction']].dropna()
            
            if len(df_clean) < 100:
                raise ValueError("Insufficient data for training")
            
            X = df_clean[self.feature_columns]
            y_price = df_clean['future_return_1d']
            y_direction = df_clean['future_direction']
            
            tss = TimeSeriesSplit(n_splits=5)
            X_scaled = self.scaler.fit_transform(X)
            
            self.price_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
            self.direction_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
            
            price_cv_scores = cross_val_score(self.price_model, X_scaled, y_price, cv=tss, scoring='neg_mean_absolute_error')
            direction_cv_scores = cross_val_score(self.direction_model, X_scaled, y_direction, cv=tss, scoring='accuracy')
            
            self.price_model.fit(X_scaled, y_price)
            self.direction_model.fit(X_scaled, y_direction)
            
            self.model_accuracy = {
                'price_mae': abs(price_cv_scores.mean()),
                'price_mae_std': price_cv_scores.std(),
                'direction_accuracy': direction_cv_scores.mean(),
                'direction_accuracy_std': direction_cv_scores.std(),
                'training_samples': len(df_clean),
                'cv_folds': len(direction_cv_scores)
            }
            
            self.last_training = datetime.now()
            logger.info(f"Models trained - Accuracy: {self.model_accuracy['direction_accuracy']:.3f}")
            return True
            
        except Exception as e:
            logger.error(f"Error training models: {str(e)}")
            return False
    
    def predict(self, current_data):
        try:
            if self.price_model is None:
                raise ValueError("Models not trained yet")
            
            features = [current_data.get(col, 0) for col in self.feature_columns]
            features_array = np.array(features).reshape(1, -1)
            features_scaled = self.scaler.transform(features_array)
            
            price_pred = self.price_model.predict(features_scaled)[0]
            direction_pred = self.direction_model.predict(features_scaled)[0]
            direction_proba = self.direction_model.predict_proba(features_scaled)[0]
            
            confidence = max(direction_proba) * 100
            
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
            logger.error(f"Error making prediction: {str(e)}")
            return None

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
        df = predictor.fetch_historical_data(symbol, days=365)
        if df is None:
            return jsonify({'error': 'Failed to fetch data'}), 400
        
        df = predictor.engineer_features(df)
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
        current_features = df.iloc[-1][predictor.feature_columns].to_dict()
        
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
    import os
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port, debug=False)
