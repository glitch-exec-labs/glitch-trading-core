"""
XAU Regime Detector - HMM-based market state detection
Only for XAUUSD/Gold - BTC unchanged
"""
import numpy as np
import pandas as pd
from hmmlearn import hmm
import logging

logger = logging.getLogger(__name__)

class XAURegimeDetector:
    """HMM-based regime detection for XAUUSD only"""
    
    def __init__(self, n_states=3):
        self.n_states = n_states
        self.model = None
        self.fitted = False
        # State meanings: 0=ranging, 1=trending_up, 2=trending_down
        self.state_map = {0: 'ranging', 1: 'trending_up', 2: 'trending_down'}
        
    def fit(self, rates):
        """Fit HMM model on historical rates"""
        try:
            # Extract features: returns, volatility, range
            df = pd.DataFrame(rates, columns=['time', 'open', 'high', 'low', 'close', 'volume'])
            df['returns'] = df['close'].pct_change()
            df['volatility'] = df['returns'].rolling(20).std()
            df['range'] = (df['high'] - df['low']) / df['close']
            
            # Clean data
            features = df[['returns', 'volatility', 'range']].dropna().values
            if len(features) < 50:
                return False
                
            # Fit HMM - use 'diag' covariance for numerical stability (BUG-FIX: was 'full')
            self.model = hmm.GaussianHMM(n_components=self.n_states, covariance_type="diag", n_iter=100)
            self.model.fit(features)
            self.fitted = True
            logger.info("[REGIME] XAU HMM model fitted successfully")
            return True
            
        except Exception as e:
            logger.error(f"[REGIME] Failed to fit HMM: {e}")
            return False
    
    def predict(self, rates):
        """Predict current regime"""
        if not self.fitted or self.model is None:
            return {'regime': 'unknown', 'confidence': 0, 'state': -1}
            
        try:
            # Extract features
            df = pd.DataFrame(rates, columns=['time', 'open', 'high', 'low', 'close', 'volume'])
            df['returns'] = df['close'].pct_change()
            df['volatility'] = df['returns'].rolling(20).std()
            df['range'] = (df['high'] - df['low']) / df['close']
            
            features = df[['returns', 'volatility', 'range']].dropna().values[-20:]  # Last 20 bars
            if len(features) < 5:
                return {'regime': 'unknown', 'confidence': 0, 'state': -1}
            
            # Predict states
            states = self.model.predict(features)
            state_probs = self.model.predict_proba(features)
            
            # Most common recent state
            current_state = states[-1]
            confidence = np.max(state_probs[-1])
            
            regime = self.state_map.get(current_state, 'unknown')
            
            return {
                'regime': regime,
                'confidence': float(confidence),
                'state': int(current_state)
            }
            
        except Exception as e:
            logger.error(f"[REGIME] Prediction error: {e}")
            return {'regime': 'unknown', 'confidence': 0, 'state': -1}


class XAUSessionFilter:
    """Session filter for XAU - only trade during optimal hours"""
    
    # Optimal XAU trading sessions (GMT/UTC)
    SESSIONS = {
        'london': {'start': 8, 'end': 12},      # 08:00-12:00 GMT
        'ny': {'start': 13, 'end': 18},         # 13:00-18:00 GMT
        'overlap': {'start': 13, 'end': 16}     # 13:00-16:00 GMT (best)
    }
    
    @staticmethod
    def is_trading_hour(current_hour_utc):
        """Check if current hour is in optimal trading session"""
        # Allow London session
        if 8 <= current_hour_utc < 12:
            return True, 'london'
        # Allow NY session  
        if 13 <= current_hour_utc < 18:
            return True, 'ny'
        return False, 'off_hours'
    
    @staticmethod
    def get_session_info(current_hour_utc):
        """Get detailed session info"""
        if 13 <= current_hour_utc < 16:
            return {'trade': True, 'session': 'overlap', 'quality': 'excellent'}
        elif 8 <= current_hour_utc < 12:
            return {'trade': True, 'session': 'london', 'quality': 'good'}
        elif 13 <= current_hour_utc < 18:
            return {'trade': True, 'session': 'ny', 'quality': 'good'}
        else:
            return {'trade': False, 'session': 'off_hours', 'quality': 'poor'}


# Global instances
xau_regime_detector = XAURegimeDetector()
xau_session_filter = XAUSessionFilter()


def check_xau_regime(symbol, rates):
    """Check regime for XAU only - returns regime info or None for BTC"""
    if 'XAU' not in symbol.upper() and 'GOLD' not in symbol.upper():
        return None  # Not XAU, skip regime check
    
    # Fit model if not fitted
    if not xau_regime_detector.fitted:
        xau_regime_detector.fit(rates)
    
    return xau_regime_detector.predict(rates)


def check_xau_session(symbol):
    """Check session for XAU only - returns session info or None for BTC"""
    if 'XAU' not in symbol.upper() and 'GOLD' not in symbol.upper():
        return None  # Not XAU, skip session check
    
    from datetime import datetime
    current_hour = datetime.now().hour
    return xau_session_filter.get_session_info(current_hour)
