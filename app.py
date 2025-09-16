"""
Real Machine Learning Cryptocurrency Trading Prediction API
Uses actual scikit-learn algorithms with historical data training
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
import os

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
        """Fetch real historical cryptocurrency data from CoinGecko API"""
        try:
            url = f"https://api.coingecko.com/api/v3/coins/{symbol}/market_chart"
            params = {
                'vs_currency': 'usd', 
                'days': days, 
                'interval': 'daily'
            }
            
            response = requests.get(url, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()
            
            # Handle API response structure
            if 'prices' not in data:
                logger.error(f"No price data in response for {symbol}")
                return None
                
            prices = data['prices']
            volumes = data.get('total_volumes', [])
            
            # Create DataFrame from prices
            df = pd.DataFrame(prices, columns=['timestamp', 'price'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Add volume data if available
            if volumes and len(volumes) == len(prices):
                volume_df = pd.DataFrame(volumes, columns=['timestamp', 'volume'])
                volume_df['timestamp'] = pd.to_datetime(volume_df['timestamp'], unit='ms')
                volume_df.set_index('timestamp', inplace=True)
                df = df.join(volume_df)
            else:
                # Create dummy volume data if not available
                df['volume'] = df['price'] * 1000000  # Fake volume for calculation
            
            logger.info(f"Successfully fetched {len(df)} days of data for {symbol}")
            return df
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Network error fetching data for {symbol}: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Error processing data for {symbol}: {str(e)}")
            return None
    
    def engineer_features(self, df):
        """Create real technical analysis features using TA-Lib"""
        try:
            if df is None or len(df) < 50:
                logger.error("Insufficient data for feature engineering")
                return None
                
            # Basic price features
            df['returns'] = df['price'].pct_change()
            df['price_ma_7'] = df['price'].rolling(window=7, min_periods=1).mean()
            df['price_ma_21'] = df['price'].rolling(window=21, min_periods=1).mean()
            
            # Technical indicators with error handling
            try:
                df['rsi'] = ta.momentum.RSIIndicator(close=df['price'], window=14).rsi()
            except:
                df['rsi'] = 50.0  # Default neutral RSI
                
            try:
                macd_indicator = ta.trend.MACD(close=df['price'])
                df['macd'] = macd_indicator.macd()
                df['macd_signal'] = macd_indicator.macd_signal()
            except:
                df['macd'] = 0.0
                df['macd_signal'] = 0.0
                
            try:
                bb_indicator = ta.volatility.BollingerBands(close=df['price'])
                df['bb_high'] = bb_indicator.bollinger_hband()
                df['bb_low'] = bb_indicator.bollinger_lband()
                df['bb_width'] = df['bb_high'] - df['bb_low']
            except:
                df['bb_width'] = df['price'].std()
            
            # Volume indicators
            df['volume_ma'] = df['volume'].rolling(window=21, min_periods=1).mean()
            df['volume_ratio'] = df['volume'] / df['volume_ma'].replace(0, 1)
            
            # Additional features
            df['volatility'] = df['returns'].rolling(window=21, min_periods=1).std()
            df['price_position'] = (df['price'] - df['price_ma_21']) / df['price_ma_21'].replace(0, 1)
            df['momentum_3'] = df['price'].pct_change(periods=3)
            
            # Target variables for ML
            df['future_return_1d'] = df['price'].pct_change(periods=1).shift(-1)
            df['future_direction'] = (df['future_return_1d'] > 0).astype(int)
            
            # Define feature columns
            self.feature_columns = [
                'returns', 'rsi', 'macd', 'macd_signal', 'bb_width', 
                'volume_ratio', 'volatility', 'price_position', 'momentum_3'
            ]
            
            # Fill NaN values
            for col in self.feature_columns:
                df[col] = df[col].fillna(method='ffill').fillna(0)
            
            logger.info(f"Successfully engineered {len(self.feature_columns)} features")
            return df
            
        except Exception as e:
            logger.error(f"Error in feature engineering: {str(e)}")
            return None
    
    def train_models(self, df):
        """Train Random Forest models with proper validation"""
        try:
            if df is None:
                raise ValueError("No data provided for training")
                
            # Prepare training data
            required_cols = self.feature_columns + ['future_return_1d', 'future_direction']
            df_clean = df[required_cols].dropna()
            
            if len(df_clean) < 100:
                raise ValueError(f"Insufficient clean data: {len(df_clean)} samples (need 100+)")
            
            X = df_clean[self.feature_columns]
            y_price = df_clean['future_return_1d']
            y_direction = df_clean['future_direction']
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Initialize models
            self.price_model = RandomForestRegressor(
                n_estimators=100, 
                max_depth=10, 
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=1  # Single core for stability
            )
            
            self.direction_model = RandomForestClassifier(
                n_estimators=100, 
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2, 
                random_state=42,
                n_jobs=1  # Single core for stability
            )
            
            # Time series cross-validation
            tss = TimeSeriesSplit(n_splits=5)
            
            # Validate models
            try:
                price_cv_scores = cross_val_score(
                    self.price_model, X_scaled, y_price, 
                    cv=tss, scoring='neg_mean_absolute_error',
                    n_jobs=1
                )
                direction_cv_scores = cross_val_score(
                    self.direction_model, X_scaled, y_direction,
                    cv=tss, scoring='accuracy',
                    n_jobs=1
                )
            except Exception as cv_error:
                logger.warning(f"Cross-validation error: {cv_error}, using simple split")
                # Fallback to simple train-test split
                split_idx = int(len(X_scaled) * 0.8)
                X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
                y_price_train, y_price_test = y_price[:split_idx], y_price[split_idx:]
                y_dir_train, y_dir_test = y_direction[:split_idx], y_direction[split_idx:]
                
                self.price_model.fit(X_train, y_price_train)
                self.direction_model.fit(X_train, y_dir_train)
                
                price_pred = self.price_model.predict(X_test)
                dir_pred = self.direction_model.predict(X_test)
                
                price_mae = np.mean(np.abs(price_pred - y_price_test))
                dir_accuracy = np.mean(dir_pred == y_dir_test)
                
                price_cv_scores = np.array([-price_mae])
                direction_cv_scores = np.array([dir_accuracy])
            
            # Final training on all data
            self.price_model.fit(X_scaled, y_price)
            self.direction_model.fit(X_scaled, y_direction)
            
            # Store accuracy metrics
            self.model_accuracy = {
                'price_mae': abs(price_cv_scores.mean()),
                'price_mae_std': price_cv_scores.std(),
                'direction_accuracy': direction_cv_scores.mean(),
                'direction_accuracy_std': direction_cv_scores.std(),
                'training_samples': len(df_clean),
                'cv_folds': len(direction_cv_scores)
            }
            
            self.last_training = datetime.now()
            
            logger.info(f"Models trained successfully:")
            logger.info(f"Direction accuracy: {self.model_accuracy['direction_accuracy']:.3f} ± {self.model_accuracy['direction_accuracy_std']:.3f}")
            logger.info(f"Price MAE: {self.model_accuracy['price_mae']:.4f}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error training models: {str(e)}")
            return False
    
    def predict(self, current_data):
        """Make ML predictions using trained models"""
        try:
            if self.price_model is None or self.direction_model is None:
                raise ValueError("Models not trained yet")
            
            # Prepare features
            features = []
            for col in self.feature_columns:
                value = current_data.get(col, 0)
                # Handle NaN/inf values
                if pd.isna(value) or np.isinf(value):
                    value = 0
                features.append(float(value))
            
            features_array = np.array(features).reshape(1, -1)
            features_scaled = self.scaler.transform(features_array)
            
            # Get predictions
            price_pred = float(self.price_model.predict(features_scaled)[0])
            direction_pred = int(self.direction_model.predict(features_scaled)[0])
            direction_proba = self.direction_model.predict_proba(features_scaled)[0]
            
            # Calculate confidence
            confidence = float(max(direction_proba) * 100)
            
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
            logger.error(f"Error making prediction: {str(e)}")
            return None

# Initialize predictor
predictor = CryptoMLPredictor()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'models_trained': predictor.price_model is not None
    })

@app.route('/train/<symbol>', methods=['POST'])
def train_model(symbol):
    """Train ML models for a cryptocurrency"""
    try:
        logger.info(f"Starting training for {symbol}")
        
        # Fetch historical data
        df = predictor.fetch_historical_data(symbol, days=365)
        if df is None:
            return jsonify({'error': f'Failed to fetch historical data for {symbol}'}), 400
        
        # Engineer features
        df = predictor.engineer_features(df)
        if df is None:
            return jsonify({'error': 'Failed to engineer features'}), 400
        
        # Train models
        success = predictor.train_models(df)
        if not success:
            return jsonify({'error': 'Model training failed'}), 500
        
        return jsonify({
            'message': f'Models trained successfully for {symbol}',
            'accuracy': predictor.model_accuracy,
            'last_training': predictor.last_training.isoformat(),
            'data_points': len(df)
        })
        
    except Exception as e:
        logger.error(f"Error in train_model: {str(e)}")
        return jsonify({'error': f'Training error: {str(e)}'}), 500

@app.route('/predict/<symbol>', methods=['GET'])
def predict_crypto(symbol):
    """Get ML prediction for a cryptocurrency"""
    try:
        # Fetch recent data
        df = predictor.fetch_historical_data(symbol, days=60)
        if df is None:
            return jsonify({'error': f'Failed to fetch current data for {symbol}'}), 400
        
        # Engineer features
        df = predictor.engineer_features(df)
        if df is None:
            return jsonify({'error': 'Failed to process current data'}), 400
        
        # Get current features
        current_features = df.iloc[-1][predictor.feature_columns].to_dict()
        
        # Make prediction
        prediction = predictor.predict(current_features)
        if prediction is None:
            return jsonify({'error': 'Prediction generation failed'}), 500
        
        return jsonify({
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            **prediction
        })
        
    except Exception as e:
        logger.error(f"Error in predict_crypto: {str(e)}")
        return jsonify({'error': f'Prediction error: {str(e)}'}), 500

@app.route('/model_info', methods=['GET'])
def model_info():
    """Get information about trained models"""
    return jsonify({
        'models_trained': predictor.price_model is not None,
        'last_training': predictor.last_training.isoformat() if predictor.last_training else None,
        'accuracy_metrics': predictor.model_accuracy,
        'feature_columns': predictor.feature_columns
    })

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """Get predictions for multiple cryptocurrencies"""
    try:
        data = request.json
        symbols = data.get('symbols', [])
        
        if not symbols:
            return jsonify({'error': 'No symbols provided'}), 400
        
        results = {}
        
        for symbol in symbols:
            try:
                df = predictor.fetch_historical_data(symbol, days=60)
                if df is not None:
                    df = predictor.engineer_features(df)
                    if df is not None:
                        current_features = df.iloc[-1][predictor.feature_columns].to_dict()
                        prediction = predictor.predict(current_features)
                        
                        if prediction:
                            results[symbol] = prediction
                        else:
                            results[symbol] = {'error': 'Prediction failed'}
                    else:
                        results[symbol] = {'error': 'Feature engineering failed'}
                else:
                    results[symbol] = {'error': 'Data fetch failed'}
                    
            except Exception as e:
                results[symbol] = {'error': str(e)}
        
        return jsonify({
            'timestamp': datetime.now().isoformat(),
            'predictions': results,
            'model_info': predictor.model_accuracy
        })
        
    except Exception as e:
        logger.error(f"Error in batch prediction: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port, debug=False)
