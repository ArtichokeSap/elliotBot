"""
Machine Learning Training Framework for Elliott Wave System
Provides training infrastructure for wave count confidence, scenario ranking, and zone scoring
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import logging
import json
import pickle
from pathlib import Path

# ML Libraries
try:
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
    from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.metrics import accuracy_score, precision_score, recall_score, mean_squared_error
    import joblib
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("Scikit-learn not available. ML training features disabled.")

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logging.warning("XGBoost not available. Using scikit-learn models only.")

@dataclass
class TrainingData:
    """Single training data point for ML models"""
    # Identifiers
    symbol: str
    timestamp: str
    timeframe: str
    
    # Elliott Wave Features
    wave_count: str  # e.g., "1-2-3", "A-B-C"
    projected_wave: str  # e.g., "4", "C"
    target_zone: List[float]  # [low, high] price range
    confluence_score: int
    confluence_methods: List[str]
    
    # Market Context Features
    current_price: float
    trend_direction: str
    rsi: float
    macd_signal: str
    volume_ratio: float
    volatility: float
    
    # Elliott Wave Quality Features
    wave_structure_quality: float  # 0-1 score
    fibonacci_alignment: float  # 0-1 score
    rule_compliance: float  # 0-1 score
    pattern_clarity: float  # 0-1 score
    
    # Historical Performance Features
    similar_pattern_success_rate: float
    timeframe_success_rate: float
    symbol_success_rate: float
    
    # Labels (ground truth)
    hit: bool  # Whether target was hit
    hit_accuracy: float  # How close to target center
    time_to_hit: Optional[int]  # Periods to reach target
    max_adverse_move: float  # Maximum drawdown before hitting target

@dataclass
class ModelPerformance:
    """Model performance metrics"""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    feature_importance: Dict[str, float]
    cross_val_scores: List[float]
    training_samples: int

class MLTrainingFramework:
    """
    Machine Learning Training Framework for Elliott Wave System
    Supports training of multiple model types for different prediction tasks
    """
    
    def __init__(self, data_dir: str = "ml_data", models_dir: str = "models"):
        self.logger = logging.getLogger(__name__)
        self.data_dir = Path(data_dir)
        self.models_dir = Path(models_dir)
        
        # Create directories if they don't exist
        self.data_dir.mkdir(exist_ok=True)
        self.models_dir.mkdir(exist_ok=True)
        
        # Initialize models
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        
        # Training configuration
        self.test_size = 0.2
        self.cv_folds = 5
        self.random_state = 42
        
        self.logger.info("🤖 ML Training Framework initialized")
    
    def collect_training_data(self, market_data: pd.DataFrame, symbol: str, 
                            start_date: str, end_date: str) -> List[TrainingData]:
        """
        Generate training data by backtesting the full Elliott Wave system
        
        Args:
            market_data: Historical OHLCV data
            symbol: Trading symbol
            start_date: Start date for backtest
            end_date: End date for backtest
            
        Returns:
            List of TrainingData objects
        """
        try:
            from ..analysis.enhanced_wave_detector import EnhancedWaveDetector
            from ..analysis.technical_confluence import TechnicalConfluenceAnalyzer
            
            training_samples = []
            detector = EnhancedWaveDetector()
            confluence_analyzer = TechnicalConfluenceAnalyzer()
            
            self.logger.info(f"📊 Collecting training data for {symbol} from {start_date} to {end_date}")
            
            # Convert date strings to datetime
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            
            # Filter data to date range
            data = market_data[(market_data.index >= start_dt) & (market_data.index <= end_dt)]
            
            # Rolling window analysis
            window_size = 100  # Minimum data for analysis
            step_size = 10     # Days between analysis points
            
            for i in range(window_size, len(data), step_size):
                analysis_data = data.iloc[:i]  # Data up to this point
                future_data = data.iloc[i:i+50]  # Next 50 periods for validation
                
                if len(future_data) < 10:  # Need enough future data
                    continue
                
                current_timestamp = data.index[i]
                current_price = analysis_data['close'].iloc[-1]
                
                # Run Elliott Wave analysis
                elliott_result = detector.detect_elliott_waves(analysis_data, symbol)
                
                if not elliott_result or elliott_result.get('validation_score', 0) < 0.3:
                    continue
                
                # Run confluence analysis
                target_zones = confluence_analyzer.analyze_target_zones(
                    analysis_data, elliott_result, '1d'
                )
                
                if not target_zones:
                    continue
                
                # Create training samples for each target zone
                for target_zone in target_zones[:3]:  # Top 3 targets
                    # Calculate features
                    features = self._extract_features(
                        analysis_data, elliott_result, target_zone, symbol
                    )
                    
                    # Calculate labels (ground truth)
                    labels = self._calculate_labels(
                        target_zone, current_price, future_data
                    )
                    
                    # Create training sample
                    sample = TrainingData(
                        symbol=symbol,
                        timestamp=current_timestamp.isoformat(),
                        timeframe='1d',
                        wave_count=elliott_result.get('wave_structure', 'unknown'),
                        projected_wave=target_zone.wave_target,
                        target_zone=[target_zone.price_level * 0.99, target_zone.price_level * 1.01],
                        confluence_score=target_zone.confluence_score,
                        confluence_methods=target_zone.confluences[:5],
                        current_price=current_price,
                        **features,
                        **labels
                    )
                    
                    training_samples.append(sample)
            
            self.logger.info(f"✅ Collected {len(training_samples)} training samples")
            return training_samples
            
        except Exception as e:
            self.logger.error(f"Error collecting training data: {e}")
            return []
    
    def _extract_features(self, data: pd.DataFrame, elliott_result: Dict, 
                         target_zone, symbol: str) -> Dict[str, float]:
        """Extract features for ML training"""
        try:
            features = {}
            
            # Technical indicator features
            closes = data['close']
            highs = data['high']
            lows = data['low']
            volumes = data['volume']
            
            # Trend and momentum features
            features['rsi'] = self._calculate_rsi(closes).iloc[-1]
            features['macd_signal'] = self._get_macd_signal(closes)
            features['volume_ratio'] = volumes.iloc[-1] / volumes.rolling(20).mean().iloc[-1]
            features['volatility'] = closes.pct_change().rolling(20).std().iloc[-1]
            features['trend_direction'] = 1 if closes.iloc[-1] > closes.rolling(20).mean().iloc[-1] else 0
            
            # Elliott Wave quality features
            features['wave_structure_quality'] = elliott_result.get('validation_score', 0)
            features['fibonacci_alignment'] = self._calculate_fibonacci_alignment(elliott_result)
            features['rule_compliance'] = self._calculate_rule_compliance(elliott_result)
            features['pattern_clarity'] = self._calculate_pattern_clarity(elliott_result)
            
            # Historical performance features (simplified)
            features['similar_pattern_success_rate'] = 0.6  # Would be calculated from historical data
            features['timeframe_success_rate'] = 0.65
            features['symbol_success_rate'] = 0.55
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error extracting features: {e}")
            return {}
    
    def _calculate_labels(self, target_zone, current_price: float, 
                         future_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate ground truth labels"""
        try:
            target_price = target_zone.price_level
            tolerance = abs(target_price - current_price) * 0.02  # 2% tolerance
            
            # Check if target was hit
            hit = False
            time_to_hit = None
            hit_accuracy = 0.0
            max_adverse_move = 0.0
            
            for i, (_, row) in enumerate(future_data.iterrows()):
                # Check adverse move
                current_adverse = abs(row['close'] - current_price) / current_price
                max_adverse_move = max(max_adverse_move, current_adverse)
                
                # Check if target hit
                if not hit and abs(row['close'] - target_price) <= tolerance:
                    hit = True
                    time_to_hit = i + 1
                    hit_accuracy = 1.0 - (abs(row['close'] - target_price) / tolerance)
                    break
                elif not hit and (
                    (target_price > current_price and row['high'] >= target_price - tolerance) or
                    (target_price < current_price and row['low'] <= target_price + tolerance)
                ):
                    hit = True
                    time_to_hit = i + 1
                    # Calculate accuracy based on how close we got
                    closest_price = target_price  # Assume we hit exactly for wick touches
                    hit_accuracy = 1.0 - (abs(closest_price - target_price) / tolerance)
                    break
            
            return {
                'hit': hit,
                'hit_accuracy': hit_accuracy,
                'time_to_hit': time_to_hit,
                'max_adverse_move': max_adverse_move
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating labels: {e}")
            return {
                'hit': False,
                'hit_accuracy': 0.0,
                'time_to_hit': None,
                'max_adverse_move': 0.0
            }
    
    def prepare_datasets(self, training_data: List[TrainingData]) -> Dict[str, Any]:
        """Prepare datasets for different ML tasks"""
        try:
            if not training_data:
                raise ValueError("No training data provided")
            
            # Convert to DataFrame
            df = pd.DataFrame([asdict(sample) for sample in training_data])
            
            # Define feature columns
            feature_columns = [
                'confluence_score', 'current_price', 'rsi', 'volume_ratio', 'volatility',
                'wave_structure_quality', 'fibonacci_alignment', 'rule_compliance', 
                'pattern_clarity', 'similar_pattern_success_rate', 'timeframe_success_rate',
                'symbol_success_rate'
            ]
            
            # Encode categorical features
            categorical_features = ['symbol', 'wave_count', 'projected_wave', 'timeframe']
            for feature in categorical_features:
                if feature not in self.encoders:
                    self.encoders[feature] = LabelEncoder()
                
                df[f'{feature}_encoded'] = self.encoders[feature].fit_transform(df[feature].astype(str))
                feature_columns.append(f'{feature}_encoded')
            
            # Prepare feature matrix
            X = df[feature_columns].fillna(0)
            
            # Scale features
            if 'main' not in self.scalers:
                self.scalers['main'] = StandardScaler()
            
            X_scaled = pd.DataFrame(
                self.scalers['main'].fit_transform(X),
                columns=X.columns,
                index=X.index
            )
            
            # Prepare different target variables
            datasets = {
                'wave_confidence': {
                    'X': X_scaled,
                    'y': df['hit'].astype(int),  # Binary classification
                    'task': 'classification'
                },
                'zone_scoring': {
                    'X': X_scaled,
                    'y': df['hit_accuracy'],  # Regression
                    'task': 'regression'
                },
                'scenario_ranking': {
                    'X': X_scaled,
                    'y': (df['hit'].astype(int) * df['hit_accuracy']).fillna(0),  # Combined score
                    'task': 'regression'
                }
            }
            
            self.logger.info(f"📊 Prepared datasets with {len(X)} samples and {len(feature_columns)} features")
            return datasets
            
        except Exception as e:
            self.logger.error(f"Error preparing datasets: {e}")
            return {}
    
    def train_wave_confidence_classifier(self, datasets: Dict[str, Any]) -> ModelPerformance:
        """Train classifier for wave count confidence"""
        try:
            if not SKLEARN_AVAILABLE:
                raise ImportError("Scikit-learn not available")
            
            dataset = datasets['wave_confidence']
            X, y = dataset['X'], dataset['y']
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.test_size, random_state=self.random_state,
                stratify=y if len(y.unique()) > 1 else None
            )
            
            # Train model
            if XGBOOST_AVAILABLE:
                model = xgb.XGBClassifier(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=self.random_state
                )
            else:
                model = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    random_state=self.random_state
                )
            
            model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = model.predict(X_test)
            
            # Cross-validation
            cv_scores = cross_val_score(model, X, y, cv=self.cv_folds)
            
            # Feature importance
            if hasattr(model, 'feature_importances_'):
                feature_importance = dict(zip(X.columns, model.feature_importances_))
            else:
                feature_importance = {}
            
            performance = ModelPerformance(
                accuracy=accuracy_score(y_test, y_pred),
                precision=precision_score(y_test, y_pred, average='weighted'),
                recall=recall_score(y_test, y_pred, average='weighted'),
                f1_score=2 * precision_score(y_test, y_pred, average='weighted') * 
                        recall_score(y_test, y_pred, average='weighted') / 
                        (precision_score(y_test, y_pred, average='weighted') + 
                         recall_score(y_test, y_pred, average='weighted')),
                feature_importance=feature_importance,
                cross_val_scores=cv_scores.tolist(),
                training_samples=len(X)
            )
            
            # Save model
            self.models['wave_confidence'] = model
            self._save_model('wave_confidence', model)
            
            self.logger.info(f"✅ Wave confidence classifier trained: {performance.accuracy:.3f} accuracy")
            return performance
            
        except Exception as e:
            self.logger.error(f"Error training wave confidence classifier: {e}")
            return ModelPerformance(0, 0, 0, 0, {}, [], 0)
    
    def train_scenario_ranking_model(self, datasets: Dict[str, Any]) -> ModelPerformance:
        """Train model for ranking Elliott Wave scenarios"""
        try:
            if not SKLEARN_AVAILABLE:
                raise ImportError("Scikit-learn not available")
            
            dataset = datasets['scenario_ranking']
            X, y = dataset['X'], dataset['y']
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.test_size, random_state=self.random_state
            )
            
            # Train model
            if XGBOOST_AVAILABLE:
                model = xgb.XGBRegressor(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=self.random_state
                )
            else:
                model = RandomForestRegressor(
                    n_estimators=100,
                    max_depth=10,
                    random_state=self.random_state
                )
            
            model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            
            # Cross-validation for regression
            cv_scores = cross_val_score(model, X, y, cv=self.cv_folds, 
                                      scoring='neg_mean_squared_error')
            
            # Feature importance
            if hasattr(model, 'feature_importances_'):
                feature_importance = dict(zip(X.columns, model.feature_importances_))
            else:
                feature_importance = {}
            
            # Convert MSE to pseudo-accuracy for consistent reporting
            accuracy = max(0, 1 - mse)
            
            performance = ModelPerformance(
                accuracy=accuracy,
                precision=accuracy,  # For regression, use same metric
                recall=accuracy,
                f1_score=accuracy,
                feature_importance=feature_importance,
                cross_val_scores=(-cv_scores).tolist(),  # Convert back to positive
                training_samples=len(X)
            )
            
            # Save model
            self.models['scenario_ranking'] = model
            self._save_model('scenario_ranking', model)
            
            self.logger.info(f"✅ Scenario ranking model trained: {accuracy:.3f} score")
            return performance
            
        except Exception as e:
            self.logger.error(f"Error training scenario ranking model: {e}")
            return ModelPerformance(0, 0, 0, 0, {}, [], 0)
    
    def train_zone_confidence_model(self, datasets: Dict[str, Any]) -> ModelPerformance:
        """Train regression model for zone confidence scoring"""
        try:
            if not SKLEARN_AVAILABLE:
                raise ImportError("Scikit-learn not available")
            
            dataset = datasets['zone_scoring']
            X, y = dataset['X'], dataset['y']
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.test_size, random_state=self.random_state
            )
            
            # Train model
            if XGBOOST_AVAILABLE:
                model = xgb.XGBRegressor(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=self.random_state
                )
            else:
                model = RandomForestRegressor(
                    n_estimators=100,
                    max_depth=10,
                    random_state=self.random_state
                )
            
            model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            
            # Cross-validation
            cv_scores = cross_val_score(model, X, y, cv=self.cv_folds, 
                                      scoring='neg_mean_squared_error')
            
            # Feature importance
            if hasattr(model, 'feature_importances_'):
                feature_importance = dict(zip(X.columns, model.feature_importances_))
            else:
                feature_importance = {}
            
            # Convert MSE to accuracy metric
            accuracy = max(0, 1 - mse)
            
            performance = ModelPerformance(
                accuracy=accuracy,
                precision=accuracy,
                recall=accuracy,
                f1_score=accuracy,
                feature_importance=feature_importance,
                cross_val_scores=(-cv_scores).tolist(),
                training_samples=len(X)
            )
            
            # Save model
            self.models['zone_confidence'] = model
            self._save_model('zone_confidence', model)
            
            self.logger.info(f"✅ Zone confidence model trained: {accuracy:.3f} score")
            return performance
            
        except Exception as e:
            self.logger.error(f"Error training zone confidence model: {e}")
            return ModelPerformance(0, 0, 0, 0, {}, [], 0)
    
    def run_full_training_pipeline(self, symbols: List[str], start_date: str, 
                                 end_date: str) -> Dict[str, ModelPerformance]:
        """Run complete training pipeline for all models"""
        try:
            self.logger.info("🚀 Starting full ML training pipeline...")
            
            # Import data loader here to avoid circular imports
            from ..data.data_loader import DataLoader
            data_loader = DataLoader()
            
            all_training_data = []
            
            # Collect training data for all symbols
            for symbol in symbols:
                self.logger.info(f"📈 Processing {symbol}...")
                try:
                    market_data = data_loader.get_yahoo_data(symbol, period='5y', interval='1d')
                    if market_data.empty:
                        self.logger.warning(f"No data for {symbol}")
                        continue
                    
                    symbol_data = self.collect_training_data(
                        market_data, symbol, start_date, end_date
                    )
                    all_training_data.extend(symbol_data)
                    
                except Exception as e:
                    self.logger.error(f"Error processing {symbol}: {e}")
                    continue
            
            if not all_training_data:
                raise ValueError("No training data collected")
            
            self.logger.info(f"📊 Total training samples: {len(all_training_data)}")
            
            # Prepare datasets
            datasets = self.prepare_datasets(all_training_data)
            
            # Train all models
            results = {}
            
            if datasets:
                results['wave_confidence'] = self.train_wave_confidence_classifier(datasets)
                results['scenario_ranking'] = self.train_scenario_ranking_model(datasets)
                results['zone_confidence'] = self.train_zone_confidence_model(datasets)
            
            # Save training data for future use
            self._save_training_data(all_training_data)
            
            self.logger.info("✅ Full training pipeline completed!")
            return results
            
        except Exception as e:
            self.logger.error(f"Error in training pipeline: {e}")
            return {}
    
    def predict_wave_confidence(self, features: Dict[str, float]) -> float:
        """Predict wave count confidence using trained model"""
        try:
            if 'wave_confidence' not in self.models:
                self.logger.warning("Wave confidence model not trained")
                return 0.5
            
            # Prepare features
            feature_vector = self._prepare_prediction_features(features)
            
            # Make prediction
            prediction = self.models['wave_confidence'].predict_proba([feature_vector])[0]
            
            # Return probability of positive class (hit = True)
            return prediction[1] if len(prediction) > 1 else prediction[0]
            
        except Exception as e:
            self.logger.error(f"Error predicting wave confidence: {e}")
            return 0.5
    
    def rank_scenarios(self, scenarios: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Rank Elliott Wave scenarios using trained model"""
        try:
            if 'scenario_ranking' not in self.models:
                self.logger.warning("Scenario ranking model not trained")
                return scenarios
            
            ranked_scenarios = []
            
            for scenario in scenarios:
                # Extract features
                features = self._extract_scenario_features(scenario)
                feature_vector = self._prepare_prediction_features(features)
                
                # Predict ranking score
                score = self.models['scenario_ranking'].predict([feature_vector])[0]
                
                scenario_copy = scenario.copy()
                scenario_copy['ml_score'] = score
                ranked_scenarios.append(scenario_copy)
            
            # Sort by ML score (highest first)
            ranked_scenarios.sort(key=lambda x: x.get('ml_score', 0), reverse=True)
            
            return ranked_scenarios
            
        except Exception as e:
            self.logger.error(f"Error ranking scenarios: {e}")
            return scenarios
    
    def predict_zone_confidence(self, zone_features: Dict[str, Any]) -> float:
        """Predict confidence score for a target zone"""
        try:
            if 'zone_confidence' not in self.models:
                self.logger.warning("Zone confidence model not trained")
                return 0.5
            
            # Prepare features
            feature_vector = self._prepare_prediction_features(zone_features)
            
            # Make prediction
            confidence = self.models['zone_confidence'].predict([feature_vector])[0]
            
            # Ensure confidence is between 0 and 1
            return max(0, min(1, confidence))
            
        except Exception as e:
            self.logger.error(f"Error predicting zone confidence: {e}")
            return 0.5
    
    def _prepare_prediction_features(self, features: Dict[str, float]) -> List[float]:
        """Prepare features for model prediction"""
        # This would need to match the exact feature order used in training
        # For now, return a simple feature vector
        feature_order = [
            'confluence_score', 'current_price', 'rsi', 'volume_ratio', 'volatility',
            'wave_structure_quality', 'fibonacci_alignment', 'rule_compliance',
            'pattern_clarity', 'similar_pattern_success_rate', 'timeframe_success_rate',
            'symbol_success_rate'
        ]
        
        feature_vector = []
        for feature_name in feature_order:
            value = features.get(feature_name, 0.0)
            feature_vector.append(float(value))
        
        return feature_vector
    
    def _extract_scenario_features(self, scenario: Dict[str, Any]) -> Dict[str, float]:
        """Extract features from Elliott Wave scenario for ranking"""
        # This would extract relevant features from the scenario
        # For now, return basic features
        return {
            'confluence_score': scenario.get('confluence_score', 0),
            'wave_structure_quality': scenario.get('validation_score', 0),
            'fibonacci_alignment': 0.7,  # Would be calculated
            'rule_compliance': 0.8,
            'pattern_clarity': 0.6
        }
    
    def _save_model(self, model_name: str, model) -> None:
        """Save trained model to disk"""
        try:
            model_path = self.models_dir / f"{model_name}.pkl"
            joblib.dump(model, model_path)
            self.logger.info(f"💾 Model saved: {model_path}")
        except Exception as e:
            self.logger.error(f"Error saving model {model_name}: {e}")
    
    def _load_model(self, model_name: str):
        """Load trained model from disk"""
        try:
            model_path = self.models_dir / f"{model_name}.pkl"
            if model_path.exists():
                model = joblib.load(model_path)
                self.models[model_name] = model
                self.logger.info(f"📂 Model loaded: {model_path}")
                return model
            else:
                self.logger.warning(f"Model file not found: {model_path}")
                return None
        except Exception as e:
            self.logger.error(f"Error loading model {model_name}: {e}")
            return None
    
    def _save_training_data(self, training_data: List[TrainingData]) -> None:
        """Save training data for future analysis"""
        try:
            data_path = self.data_dir / f"training_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            # Convert to serializable format
            serializable_data = [asdict(sample) for sample in training_data]
            
            with open(data_path, 'w') as f:
                json.dump(serializable_data, f, indent=2, default=str)
            
            self.logger.info(f"💾 Training data saved: {data_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving training data: {e}")
    
    def load_all_models(self) -> bool:
        """Load all available trained models"""
        try:
            model_names = ['wave_confidence', 'scenario_ranking', 'zone_confidence']
            loaded_count = 0
            
            for model_name in model_names:
                if self._load_model(model_name):
                    loaded_count += 1
            
            self.logger.info(f"📂 Loaded {loaded_count}/{len(model_names)} models")
            return loaded_count > 0
            
        except Exception as e:
            self.logger.error(f"Error loading models: {e}")
            return False
    
    # Helper methods for feature extraction
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _get_macd_signal(self, prices: pd.Series) -> float:
        """Get MACD signal as numeric value"""
        ema_12 = prices.ewm(span=12).mean()
        ema_26 = prices.ewm(span=26).mean()
        macd = ema_12 - ema_26
        signal = macd.ewm(span=9).mean()
        return 1.0 if macd.iloc[-1] > signal.iloc[-1] else 0.0
    
    def _calculate_fibonacci_alignment(self, elliott_result: Dict) -> float:
        """Calculate how well waves align with Fibonacci ratios"""
        # Simplified calculation
        return elliott_result.get('fibonacci_score', 0.6)
    
    def _calculate_rule_compliance(self, elliott_result: Dict) -> float:
        """Calculate Elliott Wave rule compliance score"""
        rule_compliance = elliott_result.get('rule_compliance', {})
        if not rule_compliance:
            return 0.6
        
        # Calculate average compliance
        compliance_values = [v for v in rule_compliance.values() if isinstance(v, (int, float))]
        return np.mean(compliance_values) if compliance_values else 0.6
    
    def _calculate_pattern_clarity(self, elliott_result: Dict) -> float:
        """Calculate pattern clarity score"""
        return elliott_result.get('pattern_clarity', 0.6)
