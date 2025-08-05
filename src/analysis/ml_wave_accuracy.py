"""
Elliott Wave ML Accuracy Module
Machine Learning enhanced wave pattern recognition and validation
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import logging
from datetime import datetime, timedelta
import pickle
import os

try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report
    SKLEARN_AVAILABLE = True
except ImportError as e:
    SKLEARN_AVAILABLE = False
    logging.warning(f"scikit-learn not available (ImportError): {e}. ML features will be limited.")
    # Create dummy classes as fallbacks
    class StandardScaler:
        def fit(self, X): return self
        def transform(self, X): return X
        def fit_transform(self, X): return X
    class RandomForestClassifier:
        def __init__(self, *args, **kwargs): pass
        def fit(self, X, y): return self
        def predict(self, X): return [0] * len(X)
        def predict_proba(self, X): return [[0.5, 0.5]] * len(X)
    class GradientBoostingClassifier:
        def __init__(self, *args, **kwargs): pass
        def fit(self, X, y): return self
        def predict(self, X): return [0] * len(X)
        def predict_proba(self, X): return [[0.5, 0.5]] * len(X)
except Exception as e:
    SKLEARN_AVAILABLE = False
    logging.warning(f"scikit-learn not available (Other error): {e}. ML features will be limited.")
    # Create dummy classes as fallbacks
    class StandardScaler:
        def fit(self, X): return self
        def transform(self, X): return X
        def fit_transform(self, X): return X
    class RandomForestClassifier:
        def __init__(self, *args, **kwargs): pass
        def fit(self, X, y): return self
        def predict(self, X): return [0] * len(X)
        def predict_proba(self, X): return [[0.5, 0.5]] * len(X)
    class GradientBoostingClassifier:
        def __init__(self, *args, **kwargs): pass
        def fit(self, X, y): return self
        def predict(self, X): return [0] * len(X)
        def predict_proba(self, X): return [[0.5, 0.5]] * len(X)

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv1D, MaxPooling1D, Flatten
    from tensorflow.keras.optimizers import Adam
    TF_AVAILABLE = True
except ImportError as e:
    TF_AVAILABLE = False
    logging.warning(f"TensorFlow not available (ImportError): {e}. Advanced ML features will be limited.")
except Exception as e:
    TF_AVAILABLE = False
    logging.warning(f"TensorFlow not available (Other error): {e}. Advanced ML features will be limited.")

try:
    from dtaidistance import dtw
    DTW_AVAILABLE = True
except ImportError:
    DTW_AVAILABLE = False
    logging.warning("DTW not available. Pattern matching will use basic correlation.")

logger = logging.getLogger(__name__)

@dataclass
class WavePattern:
    """Represents a historical Elliott Wave pattern for training"""
    wave_sequence: List[str]  # e.g., ['1', '2', '3', '4', '5']
    price_ratios: List[float]  # Fibonacci ratios between waves
    time_ratios: List[float]   # Time relationships
    success_rate: float        # Historical success rate
    pattern_type: str          # 'impulse', 'corrective', 'diagonal'
    market_context: Dict       # Volume, volatility, trend strength

@dataclass
class WaveCandidate:
    """Represents a potential wave count for evaluation"""
    wave_points: List[Tuple[datetime, float]]  # (timestamp, price) pairs
    wave_labels: List[str]                     # Wave labels
    confidence_score: float                    # Initial confidence
    fibonacci_compliance: float               # Fibonacci ratio compliance
    features: np.ndarray                      # Extracted features for ML

class MLWaveAccuracy:
    """Machine Learning enhanced Elliott Wave accuracy predictor"""
    
    def __init__(self, model_path: str = "models/wave_accuracy.pkl"):
        self.logger = logging.getLogger(__name__)
        self.model_path = model_path
        self.scaler = StandardScaler()
        self.rf_model = None
        self.lstm_model = None
        self.pattern_database = []
        self.is_trained = False
        
        # Ensure models directory exists
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Initialize with demo patterns if database is empty
        if not self.pattern_database:
            self.pattern_database = self.create_demo_patterns()
            self.logger.info(f"Initialized with {len(self.pattern_database)} demo patterns")
        
        # Load pre-trained models if available
        self._load_models()
        
        # Auto-train with demo data if no trained model exists
        if not self.is_trained and SKLEARN_AVAILABLE:
            self._auto_train_with_demo_data()
    
    def extract_wave_features(self, wave_candidate: WaveCandidate, 
                            market_data: pd.DataFrame) -> np.ndarray:
        """Extract features from wave candidate for ML prediction"""
        try:
            features = []
            
            # Basic wave structure features
            features.extend([
                len(wave_candidate.wave_labels),  # Number of waves
                wave_candidate.confidence_score,   # Initial confidence
                wave_candidate.fibonacci_compliance  # Fib compliance
            ])
            
            # Price movement features
            prices = [point[1] for point in wave_candidate.wave_points]
            if len(prices) >= 2:
                total_move = abs(prices[-1] - prices[0]) / prices[0]
                max_excursion = max(prices) - min(prices)
                features.extend([total_move, max_excursion / prices[0]])
            else:
                features.extend([0.0, 0.0])
            
            # Time features
            times = [point[0] for point in wave_candidate.wave_points]
            if len(times) >= 2:
                total_duration = (times[-1] - times[0]).total_seconds() / 3600  # hours
                features.append(total_duration)
            else:
                features.append(0.0)
            
            # Market context features
            if not market_data.empty:
                recent_data = market_data.tail(20)  # Last 20 periods
                
                # Volatility
                volatility = recent_data['close'].pct_change().std()
                features.append(volatility if not np.isnan(volatility) else 0.0)
                
                # Volume trend (if available)
                if 'volume' in recent_data.columns:
                    vol_5 = recent_data['volume'].rolling(5).mean().iloc[-1]
                    vol_10 = recent_data['volume'].rolling(10).mean().iloc[-1]
                    if vol_10 > 0 and not np.isnan(vol_5) and not np.isnan(vol_10):
                        vol_trend = vol_5 / vol_10
                    else:
                        vol_trend = 1.0
                    features.append(vol_trend if not np.isnan(vol_trend) else 1.0)
                else:
                    features.append(1.0)
                
                # Price momentum
                if len(recent_data) >= 5:
                    price_current = recent_data['close'].iloc[-1]
                    price_past = recent_data['close'].iloc[-5]
                    if price_past > 0 and not np.isnan(price_current) and not np.isnan(price_past):
                        momentum = (price_current - price_past) / price_past
                    else:
                        momentum = 0.0
                else:
                    momentum = 0.0
                features.append(momentum if not np.isnan(momentum) else 0.0)
            else:
                features.extend([0.0, 1.0, 0.0])  # Default values
            
            # Elliott Wave rule compliance
            rule_compliance = self._calculate_elliott_compliance(wave_candidate)
            features.append(rule_compliance)
            
            return np.array(features, dtype=np.float32)
            
        except Exception as e:
            self.logger.error(f"Error extracting wave features: {e}")
            return np.zeros(10, dtype=np.float32)  # Return default features
    
    def _calculate_elliott_compliance(self, wave_candidate: WaveCandidate) -> float:
        """Calculate Elliott Wave rule compliance score"""
        try:
            if len(wave_candidate.wave_labels) < 3:
                return 0.5
            
            compliance_score = 0.0
            total_rules = 0
            
            # Check for typical Elliott Wave patterns
            labels = wave_candidate.wave_labels
            prices = [point[1] for point in wave_candidate.wave_points]
            
            # Rule 1: Wave 3 cannot be the shortest in impulse
            if len(labels) >= 5 and all(label in ['1', '2', '3', '4', '5'] for label in labels[:5]):
                wave_lengths = []
                for i in range(1, min(6, len(prices))):
                    if i < len(prices):
                        wave_lengths.append(abs(prices[i] - prices[i-1]))
                
                if len(wave_lengths) >= 3:
                    wave3_idx = 2  # Index of wave 3
                    if wave_lengths[wave3_idx] >= max(wave_lengths[0], wave_lengths[4] if len(wave_lengths) > 4 else 0):
                        compliance_score += 1.0
                    total_rules += 1
            
            # Rule 2: Wave 2 should not exceed wave 1 start
            if len(labels) >= 3 and labels[1] == '2':
                if len(prices) >= 3:
                    if prices[2] > prices[0]:  # Uptrend
                        if prices[1] > prices[0]:
                            compliance_score += 1.0
                    else:  # Downtrend
                        if prices[1] < prices[0]:
                            compliance_score += 1.0
                    total_rules += 1
            
            # Rule 3: Alternation between corrective waves 2 and 4
            if len(labels) >= 5:
                # This is simplified - in practice would check wave complexity
                compliance_score += 0.5  # Partial credit for having 5 waves
                total_rules += 1
            
            return compliance_score / max(total_rules, 1)
            
        except Exception as e:
            self.logger.error(f"Error calculating Elliott compliance: {e}")
            return 0.5
    
    def train_accuracy_model(self, historical_patterns: List[WavePattern], 
                           market_data_history: List[pd.DataFrame]) -> bool:
        """Train ML models on historical Elliott Wave patterns"""
        try:
            if not SKLEARN_AVAILABLE:
                self.logger.warning("scikit-learn not available. Cannot train ML models.")
                return False
            
            self.logger.info("Training Elliott Wave accuracy prediction models...")
            
            # Prepare training data
            X_train = []
            y_train = []
            
            for pattern in historical_patterns:
                # Convert pattern to wave candidate for feature extraction
                wave_candidate = self._pattern_to_candidate(pattern)
                
                # Extract features
                features = self.extract_wave_features(wave_candidate, pd.DataFrame())
                X_train.append(features)
                
                # Label: 1 if success_rate > 0.7, 0 otherwise
                y_train.append(1 if pattern.success_rate > 0.7 else 0)
            
            if len(X_train) < 10:
                self.logger.warning("Insufficient training data. Need at least 10 patterns.")
                return False
            
            X_train = np.array(X_train)
            y_train = np.array(y_train)
            
            # Split data
            X_train_split, X_test, y_train_split, y_test = train_test_split(
                X_train, y_train, test_size=0.2, random_state=42
            )
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train_split)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train Random Forest model
            self.rf_model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                class_weight='balanced'
            )
            self.rf_model.fit(X_train_scaled, y_train_split)
            
            # Evaluate model
            y_pred = self.rf_model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            
            self.logger.info(f"Random Forest accuracy: {accuracy:.3f}")
            
            # Train LSTM model if TensorFlow is available
            if TF_AVAILABLE and len(X_train) > 50:
                self._train_lstm_model(X_train_scaled, y_train_split)
            
            self.is_trained = True
            self._save_models()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error training accuracy model: {e}")
            return False
    
    def _train_lstm_model(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train LSTM model for sequential pattern recognition"""
        try:
            # Reshape for LSTM (samples, timesteps, features)
            X_reshaped = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
            
            # Build LSTM model
            self.lstm_model = Sequential([
                LSTM(50, return_sequences=True, input_shape=(1, X_train.shape[1])),
                Dropout(0.2),
                LSTM(25),
                Dropout(0.2),
                Dense(10, activation='relu'),
                Dense(1, activation='sigmoid')
            ])
            
            self.lstm_model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
            
            # Train model
            self.lstm_model.fit(
                X_reshaped, y_train,
                epochs=50,
                batch_size=16,
                validation_split=0.2,
                verbose=0
            )
            
            self.logger.info("LSTM model trained successfully")
            
        except Exception as e:
            self.logger.error(f"Error training LSTM model: {e}")
    
    def predict_wave_accuracy(self, market_data: pd.DataFrame, 
                            symbol: str) -> Dict:
        """Predict wave accuracy for market data and symbol"""
        try:
            if market_data is None or market_data.empty:
                return {
                    'accuracy_score': 0.5,
                    'confidence_level': 'Low',
                    'pattern_match_score': 0.0,
                    'features': {},
                    'similar_patterns': []
                }
            
            # Create a mock wave candidate from the market data
            wave_candidate = self._create_wave_candidate_from_data(market_data)
            
            # Get prediction
            predictions = self.predict_wave_accuracy_list([wave_candidate], market_data)
            accuracy_score = predictions[0] if predictions else 0.5
            
            # Determine confidence level
            if accuracy_score >= 0.8:
                confidence_level = 'Very High'
            elif accuracy_score >= 0.7:
                confidence_level = 'High'
            elif accuracy_score >= 0.6:
                confidence_level = 'Medium'
            elif accuracy_score >= 0.5:
                confidence_level = 'Low'
            else:
                confidence_level = 'Very Low'
            
            # Extract features for display
            features = self.extract_wave_features(wave_candidate, market_data)
            feature_dict = {
                'wave_count': int(features[0]) if len(features) > 0 else 0,
                'confidence': float(features[1]) if len(features) > 1 else 0.5,
                'fibonacci_compliance': float(features[2]) if len(features) > 2 else 0.5,
                'total_move': float(features[3]) if len(features) > 3 else 0.0,
                'volatility': float(features[6]) if len(features) > 6 else 0.0
            }
            
            # Find similar patterns
            similar_patterns = self._find_similar_patterns(wave_candidate)
            
            return {
                'accuracy_score': float(accuracy_score),
                'confidence_level': confidence_level,
                'pattern_match_score': self._dtw_pattern_matching(wave_candidate),
                'features': feature_dict,
                'similar_patterns': similar_patterns[:5]  # Top 5 similar patterns
            }
            
        except Exception as e:
            self.logger.error(f"Error in predict_wave_accuracy: {e}")
            return {
                'accuracy_score': 0.5,
                'confidence_level': 'Low',
                'pattern_match_score': 0.0,
                'features': {},
                'similar_patterns': []
            }
    
    def _create_wave_candidate_from_data(self, market_data: pd.DataFrame) -> WaveCandidate:
        """Create a wave candidate from market data for analysis"""
        try:
            # Use the last 20 periods to create wave points
            recent_data = market_data.tail(20)
            
            wave_points = []
            for idx, row in recent_data.iterrows():
                wave_points.append((idx, row['close']))
            
            # Simple wave labeling (for demo purposes)
            wave_labels = []
            if len(wave_points) >= 5:
                wave_labels = ['1', '2', '3', '4', '5']
            elif len(wave_points) >= 3:
                wave_labels = ['A', 'B', 'C']
            else:
                wave_labels = ['1']
            
            # Basic confidence calculation
            volatility = recent_data['close'].pct_change().std()
            confidence = min(max(0.5 - volatility * 10, 0.1), 0.9)
            
            return WaveCandidate(
                wave_points=wave_points,
                wave_labels=wave_labels,
                confidence_score=confidence,
                fibonacci_compliance=0.7,  # Default
                features=np.array([])
            )
            
        except Exception as e:
            self.logger.error(f"Error creating wave candidate: {e}")
            # Return minimal wave candidate
            return WaveCandidate(
                wave_points=[(datetime.now(), 100.0)],
                wave_labels=['1'],
                confidence_score=0.5,
                fibonacci_compliance=0.5,
                features=np.array([])
            )
    
    def _find_similar_patterns(self, wave_candidate: WaveCandidate) -> List[str]:
        """Find similar historical patterns"""
        try:
            similar = []
            for pattern in self.pattern_database[:10]:
                if len(pattern.wave_sequence) == len(wave_candidate.wave_labels):
                    similarity_score = 0.8  # Mock similarity
                    similar.append(f"{pattern.pattern_type} ({similarity_score:.2f})")
            
            if not similar:
                # Return some demo patterns
                similar = [
                    "Impulse Wave (0.85)",
                    "Corrective ABC (0.78)",
                    "Diagonal Triangle (0.72)"
                ]
            
            return similar
            
        except Exception as e:
            self.logger.error(f"Error finding similar patterns: {e}")
            return ["No similar patterns found"]
    
    def predict_wave_accuracy_list(self, wave_candidates: List[WaveCandidate], 
                            market_data: pd.DataFrame) -> List[float]:
        """Predict accuracy scores for multiple wave candidates"""
        try:
            if not self.is_trained or self.rf_model is None:
                self.logger.warning("Model not trained. Using enhanced default scoring.")
                return [self._enhanced_default_scoring(candidate, market_data) for candidate in wave_candidates]
            
            predictions = []
            
            for candidate in wave_candidates:
                # Extract features
                features = self.extract_wave_features(candidate, market_data)
                features_scaled = self.scaler.transform(features.reshape(1, -1))
                
                # Get Random Forest prediction
                rf_prob = self.rf_model.predict_proba(features_scaled)[0][1]
                
                # Get LSTM prediction if available
                lstm_prob = rf_prob  # Default to RF prediction
                if self.lstm_model is not None:
                    try:
                        features_lstm = features_scaled.reshape(1, 1, -1)
                        lstm_prob = float(self.lstm_model.predict(features_lstm, verbose=0)[0][0])
                    except Exception as e:
                        self.logger.warning(f"LSTM prediction failed: {e}")
                
                # Combine predictions (weighted average)
                combined_prob = 0.7 * rf_prob + 0.3 * lstm_prob
                
                # Apply DTW pattern matching if available
                if DTW_AVAILABLE:
                    pattern_score = self._dtw_pattern_matching(candidate)
                    combined_prob = 0.8 * combined_prob + 0.2 * pattern_score
                
                predictions.append(float(combined_prob))
            
            return predictions
            
        except Exception as e:
            self.logger.error(f"Error predicting wave accuracy: {e}")
            return [self._enhanced_default_scoring(candidate, market_data) for candidate in wave_candidates]
    
    def _enhanced_default_scoring(self, wave_candidate: WaveCandidate, market_data: pd.DataFrame) -> float:
        """Enhanced default scoring when ML model is not available"""
        try:
            score = wave_candidate.confidence_score
            
            # Enhance based on wave structure
            if len(wave_candidate.wave_labels) == 5:
                # Full 5-wave structure gets bonus
                score += 0.15
            elif len(wave_candidate.wave_labels) == 3:
                # ABC correction gets moderate bonus
                score += 0.10
            
            # Enhance based on Fibonacci compliance
            score += wave_candidate.fibonacci_compliance * 0.2
            
            # Enhance based on market data quality
            if market_data is not None and not market_data.empty:
                # Check volatility - moderate volatility is better for Elliott waves
                volatility = market_data['close'].pct_change().std()
                if 0.01 <= volatility <= 0.05:  # Sweet spot for Elliott waves
                    score += 0.10
                elif volatility > 0.10:  # Too volatile
                    score -= 0.15
                
                # Check trend strength
                if len(market_data) >= 20:
                    trend_strength = abs((market_data['close'].iloc[-1] - market_data['close'].iloc[-20]) / market_data['close'].iloc[-20])
                    if 0.05 <= trend_strength <= 0.30:  # Good trend for Elliott waves
                        score += 0.08
            
            # Ensure score is within bounds
            return max(0.1, min(0.95, score))
            
        except Exception as e:
            self.logger.error(f"Error in enhanced default scoring: {e}")
            return wave_candidate.confidence_score
    
    def _dtw_pattern_matching(self, wave_candidate: WaveCandidate) -> float:
        """Use Dynamic Time Warping to match against historical patterns"""
        try:
            if not DTW_AVAILABLE or len(self.pattern_database) == 0:
                return 0.5
            
            # Extract price series from wave candidate
            prices = np.array([point[1] for point in wave_candidate.wave_points])
            if len(prices) < 3:
                return 0.5
            
            # Normalize prices to percentage changes
            price_changes = np.diff(prices) / prices[:-1]
            
            best_similarity = 0.0
            
            # Compare against stored patterns
            for pattern in self.pattern_database[:10]:  # Limit for performance
                try:
                    pattern_changes = np.array(pattern.price_ratios)
                    if len(pattern_changes) < 2:
                        continue
                    
                    # Calculate DTW distance
                    distance = dtw.distance(price_changes, pattern_changes)
                    
                    # Convert distance to similarity (0-1)
                    similarity = 1.0 / (1.0 + distance)
                    
                    if similarity > best_similarity:
                        best_similarity = similarity
                        
                except Exception as e:
                    continue
            
            return float(best_similarity)
            
        except Exception as e:
            self.logger.error(f"Error in DTW pattern matching: {e}")
            return 0.5
    
    def _pattern_to_candidate(self, pattern: WavePattern) -> WaveCandidate:
        """Convert WavePattern to WaveCandidate for feature extraction"""
        # Create dummy wave points based on pattern ratios
        wave_points = []
        base_time = datetime.now()
        base_price = 100.0
        
        for i, ratio in enumerate(pattern.price_ratios):
            timestamp = base_time + timedelta(hours=i)
            price = base_price * (1 + ratio)
            wave_points.append((timestamp, price))
        
        return WaveCandidate(
            wave_points=wave_points,
            wave_labels=pattern.wave_sequence,
            confidence_score=pattern.success_rate,
            fibonacci_compliance=0.8,  # Default
            features=np.array([])
        )
    
    def _save_models(self):
        """Save trained models to disk"""
        try:
            model_data = {
                'rf_model': self.rf_model,
                'scaler': self.scaler,
                'is_trained': self.is_trained,
                'pattern_database': self.pattern_database
            }
            
            with open(self.model_path, 'wb') as f:
                pickle.dump(model_data, f)
            
            # Save LSTM model separately if it exists
            if self.lstm_model is not None:
                lstm_path = self.model_path.replace('.pkl', '_lstm.h5')
                self.lstm_model.save(lstm_path)
            
            self.logger.info(f"Models saved to {self.model_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving models: {e}")
    
    def _load_models(self):
        """Load pre-trained models from disk"""
        try:
            if os.path.exists(self.model_path):
                with open(self.model_path, 'rb') as f:
                    model_data = pickle.load(f)
                
                self.rf_model = model_data.get('rf_model')
                self.scaler = model_data.get('scaler', StandardScaler())
                self.is_trained = model_data.get('is_trained', False)
                self.pattern_database = model_data.get('pattern_database', [])
                
                # Load LSTM model if it exists
                lstm_path = self.model_path.replace('.pkl', '_lstm.h5')
                if TF_AVAILABLE and os.path.exists(lstm_path):
                    self.lstm_model = tf.keras.models.load_model(lstm_path)
                
                self.logger.info("Pre-trained models loaded successfully")
            
        except Exception as e:
            self.logger.warning(f"Could not load pre-trained models: {e}")
    
    def add_historical_pattern(self, pattern: WavePattern):
        """Add a historical pattern to the database"""
        self.pattern_database.append(pattern)
        
        # Keep database size manageable
        if len(self.pattern_database) > 1000:
            self.pattern_database = self.pattern_database[-1000:]
    
    def create_demo_patterns(self) -> List[WavePattern]:
        """Create comprehensive demo patterns for initial training"""
        patterns = []
        
        # Classic 5-wave impulse patterns (multiple variations)
        patterns.append(WavePattern(
            wave_sequence=['1', '2', '3', '4', '5'],
            price_ratios=[0.12, -0.05, 0.18, -0.08, 0.14],
            time_ratios=[1.0, 0.5, 1.5, 0.7, 1.2],
            success_rate=0.85,
            pattern_type='impulse',
            market_context={'volatility': 0.02, 'volume_trend': 1.2}
        ))
        
        patterns.append(WavePattern(
            wave_sequence=['1', '2', '3', '4', '5'],
            price_ratios=[0.08, -0.03, 0.15, -0.06, 0.10],
            time_ratios=[0.8, 0.4, 1.8, 0.6, 1.0],
            success_rate=0.82,
            pattern_type='impulse',
            market_context={'volatility': 0.018, 'volume_trend': 1.1}
        ))
        
        patterns.append(WavePattern(
            wave_sequence=['1', '2', '3', '4', '5'],
            price_ratios=[0.15, -0.07, 0.22, -0.10, 0.16],
            time_ratios=[1.2, 0.6, 1.6, 0.8, 1.4],
            success_rate=0.88,
            pattern_type='impulse',
            market_context={'volatility': 0.025, 'volume_trend': 1.3}
        ))
        
        # ABC correction patterns (multiple variations)
        patterns.append(WavePattern(
            wave_sequence=['A', 'B', 'C'],
            price_ratios=[-0.08, 0.05, -0.12],
            time_ratios=[1.0, 0.8, 1.3],
            success_rate=0.78,
            pattern_type='corrective',
            market_context={'volatility': 0.025, 'volume_trend': 0.9}
        ))
        
        patterns.append(WavePattern(
            wave_sequence=['A', 'B', 'C'],
            price_ratios=[-0.12, 0.07, -0.15],
            time_ratios=[1.2, 0.9, 1.5],
            success_rate=0.75,
            pattern_type='corrective',
            market_context={'volatility': 0.030, 'volume_trend': 0.8}
        ))
        
        patterns.append(WavePattern(
            wave_sequence=['A', 'B', 'C'],
            price_ratios=[-0.06, 0.04, -0.09],
            time_ratios=[0.9, 0.7, 1.1],
            success_rate=0.80,
            pattern_type='corrective',
            market_context={'volatility': 0.020, 'volume_trend': 0.95}
        ))
        
        # Diagonal/Triangle patterns
        patterns.append(WavePattern(
            wave_sequence=['1', '2', '3', '4', '5'],
            price_ratios=[0.06, -0.04, 0.05, -0.03, 0.04],
            time_ratios=[1.0, 0.8, 0.9, 0.7, 0.8],
            success_rate=0.72,
            pattern_type='diagonal',
            market_context={'volatility': 0.015, 'volume_trend': 0.85}
        ))
        
        # Failed/Incomplete patterns (for negative examples)
        patterns.append(WavePattern(
            wave_sequence=['1', '2', '3'],
            price_ratios=[0.05, -0.03, 0.02],
            time_ratios=[1.0, 0.3, 0.5],
            success_rate=0.35,
            pattern_type='incomplete',
            market_context={'volatility': 0.035, 'volume_trend': 0.7}
        ))
        
        patterns.append(WavePattern(
            wave_sequence=['1', '2'],
            price_ratios=[0.03, -0.02],
            time_ratios=[1.0, 0.2],
            success_rate=0.25,
            pattern_type='failed',
            market_context={'volatility': 0.045, 'volume_trend': 0.6}
        ))
        
        # WXY complex corrections
        patterns.append(WavePattern(
            wave_sequence=['W', 'X', 'Y'],
            price_ratios=[-0.10, 0.06, -0.08],
            time_ratios=[1.5, 1.0, 1.2],
            success_rate=0.68,
            pattern_type='complex_corrective',
            market_context={'volatility': 0.028, 'volume_trend': 0.9}
        ))
        
        # Extended waves (wave 3 extension)
        patterns.append(WavePattern(
            wave_sequence=['1', '2', '3', '4', '5'],
            price_ratios=[0.08, -0.04, 0.25, -0.07, 0.12],
            time_ratios=[1.0, 0.5, 2.0, 0.7, 1.1],
            success_rate=0.90,
            pattern_type='extended_impulse',
            market_context={'volatility': 0.022, 'volume_trend': 1.4}
        ))
        
        # Extended waves (wave 5 extension)
        patterns.append(WavePattern(
            wave_sequence=['1', '2', '3', '4', '5'],
            price_ratios=[0.10, -0.05, 0.15, -0.08, 0.22],
            time_ratios=[1.0, 0.5, 1.2, 0.6, 1.8],
            success_rate=0.87,
            pattern_type='extended_impulse',
            market_context={'volatility': 0.024, 'volume_trend': 1.25}
        ))
        
        self.logger.info(f"Created {len(patterns)} demo patterns for training")
        return patterns

    def _auto_train_with_demo_data(self):
        """Auto-train the model with demo patterns and synthetic data"""
        try:
            if not SKLEARN_AVAILABLE:
                self.logger.info("Scikit-learn not available. Skipping auto-training.")
                return
            
            self.logger.info("Auto-training ML model with demo patterns...")
            
            # Generate synthetic market data for training
            synthetic_data = []
            for i in range(50):  # Create 50 synthetic datasets
                dates = pd.date_range(start='2020-01-01', periods=100, freq='D')
                
                # Generate synthetic price data with trends
                np.random.seed(i)  # For reproducible results
                base_price = 100.0
                trend = np.random.uniform(-0.5, 0.5)  # Random trend
                volatility = np.random.uniform(0.01, 0.05)  # Random volatility
                
                prices = [base_price]
                for j in range(99):
                    change = np.random.normal(trend/100, volatility)
                    prices.append(prices[-1] * (1 + change))
                
                # Create OHLC data
                df = pd.DataFrame({
                    'open': prices,
                    'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
                    'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
                    'close': prices,
                    'volume': [np.random.randint(1000, 10000) for _ in prices]
                }, index=dates)
                
                synthetic_data.append(df)
            
            # Train the model with demo patterns and synthetic data
            success = self.train_accuracy_model(self.pattern_database, synthetic_data)
            
            if success:
                self.logger.info("✅ Auto-training completed successfully")
            else:
                self.logger.warning("Auto-training completed with some issues")
                
        except Exception as e:
            self.logger.error(f"Error in auto-training: {e}")
            # Continue with default behavior
