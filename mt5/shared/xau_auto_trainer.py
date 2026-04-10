"""
XAU Auto-Retraining Module
Retrains ML model when market conditions change significantly
Only for XAU - BTC unchanged
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
import pickle
import os

logger = logging.getLogger(__name__)

class XAUAutoTrainer:
    """Auto-retraining for XAU ML model"""
    
    def __init__(self, model_path='models/xau_xgb_model.pkl', retrain_interval_hours=24):
        self.model_path = model_path
        self.retrain_interval = timedelta(hours=retrain_interval_hours)
        self.last_retrain = None
        self.performance_history = []
        self.volatility_threshold = 0.02  # 2% daily vol change triggers retrain
        
    def should_retrain(self, recent_trades, current_rates):
        """Check if retraining is needed"""
        # Check 1: Time-based
        if self.last_retrain is None:
            return True, "First run - initial training needed"
            
        time_since_last = datetime.now() - self.last_retrain
        if time_since_last > self.retrain_interval:
            return True, f"Time-based: {time_since_last.total_seconds()/3600:.1f}h since last retrain"
        
        # Check 2: Performance degradation
        if len(recent_trades) >= 10:
            win_rate = sum(1 for t in recent_trades if t.get('profit', 0) > 0) / len(recent_trades)
            if win_rate < 0.40:  # Below 40% win rate
                return True, f"Performance-based: Win rate {win_rate:.0%} below 40%"
        
        # Check 3: Volatility regime change
        if current_rates is not None and len(current_rates) > 50:
            recent_vol = self._calculate_volatility(current_rates[-50:])
            historical_vol = self._calculate_volatility(current_rates[:-50])
            
            if historical_vol > 0 and abs(recent_vol - historical_vol) / historical_vol > 0.5:
                return True, f"Volatility regime change: {historical_vol:.2%} -> {recent_vol:.2%}"
        
        return False, "No retraining needed"
    
    def _calculate_volatility(self, rates):
        """Calculate daily volatility from rates"""
        try:
            df = pd.DataFrame(rates, columns=['time', 'open', 'high', 'low', 'close', 'volume'])
            returns = df['close'].pct_change().dropna()
            return returns.std() * np.sqrt(24)  # Annualized hourly volatility
        except:
            return 0
    
    def retrain(self, rates, symbol='XAUUSD'):
        """Retrain the XAU model"""
        try:
            from ml_engine import create_ml_engine
            
            logger.info(f"[AUTO-TRAIN] Starting retraining for {symbol}...")
            
            # Create fresh ML engine
            ml_engine = create_ml_engine()
            
            # Fit regime detector first
            if hasattr(ml_engine, 'xau_model') and hasattr(ml_engine.xau_model, 'fit'):
                ml_engine.xau_model.fit(rates)
                logger.info(f"[AUTO-TRAIN] {symbol} model retrained successfully")
                
                self.last_retrain = datetime.now()
                return True
            else:
                logger.warning(f"[AUTO-TRAIN] Model doesn't support retraining")
                return False
                
        except Exception as e:
            logger.error(f"[AUTO-TRAIN] Retraining failed: {e}")
            return False
    
    def log_performance(self, trade_result):
        """Log trade for performance tracking"""
        self.performance_history.append({
            'timestamp': datetime.now(),
            'profit': trade_result.get('profit', 0),
            'symbol': trade_result.get('symbol', 'unknown')
        })
        
        # Keep only last 50 trades
        if len(self.performance_history) > 50:
            self.performance_history = self.performance_history[-50:]


# Global auto-trainer instance
xau_auto_trainer = XAUAutoTrainer()


def check_and_retrain_xau(symbol, rates, recent_trades=None):
    """Check if XAU needs retraining and do it - XAU ONLY"""
    if 'XAU' not in symbol.upper() and 'GOLD' not in symbol.upper():
        return None  # Not XAU, skip
    
    should_train, reason = xau_auto_trainer.should_retrain(recent_trades or [], rates)
    
    if should_train:
        logger.info(f"[AUTO-TRAIN] Triggered: {reason}")
        success = xau_auto_trainer.retrain(rates, symbol)
        return {'retrained': success, 'reason': reason}
    
    return {'retrained': False, 'reason': reason}


def log_xau_trade(trade_result):
    """Log XAU trade for performance tracking - XAU ONLY"""
    if 'XAU' not in trade_result.get('symbol', '').upper():
        return  # Not XAU, skip
    
    xau_auto_trainer.log_performance(trade_result)
