"""
Risk Manager ULTRA - Enhanced with ML-based Risk Assessment
Portfolio-level risk management with NumPy optimizations
"""
import numpy as np
from numba import jit
import json
from datetime import datetime, timezone, date
import warnings
warnings.filterwarnings('ignore')

# Import base risk manager
import sys
sys.path.insert(0, '.')
from risk_manager import RiskManager

class RiskManagerUltra(RiskManager):
    """
    Enhanced risk manager with:
    - Portfolio correlation analysis
    - Value at Risk (VaR)
    - Kelly Criterion sizing
    - ML-based risk prediction
    - Trade log reconstruction
    """
    
    def __init__(self, 
                 daily_loss_limit: float = -100.0,
                 max_consecutive_losses: int = 6,
                 max_drawdown_percent: float = 10.0,
                 var_confidence: float = 0.95,
                 kelly_fraction: float = 0.25,
                 state_file: str = "risk_state_ultra.json"):
        
        super().__init__(daily_loss_limit, max_consecutive_losses, 
                        max_drawdown_percent, state_file)
        
        self.var_confidence = var_confidence
        self.kelly_fraction = kelly_fraction
        
        # Track position correlations
        self.position_history = []
        self.returns_history = []
        
    def calculate_var(self, returns, confidence=0.95):
        """
        Calculate Value at Risk using historical simulation
        VaR: Maximum expected loss at given confidence level
        """
        if len(returns) < 30:
            return None
        
        returns = np.array(returns)
        var_threshold = np.percentile(returns, (1 - confidence) * 100)
        
        return {
            'var_95': var_threshold,
            'var_99': np.percentile(returns, 1),
            'expected_shortfall': np.mean(returns[returns <= var_threshold]),
            'confidence': confidence
        }
    
    def calculate_correlation_matrix(self, symbol_returns):
        """
        Calculate correlation matrix between symbols
        symbol_returns: dict of {symbol: [returns]}
        """
        if len(symbol_returns) < 2:
            return None
        
        symbols = list(symbol_returns.keys())
        n = len(symbols)
        corr_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                if i == j:
                    corr_matrix[i, j] = 1.0
                else:
                    r1 = np.array(symbol_returns[symbols[i]])
                    r2 = np.array(symbol_returns[symbols[j]])
                    
                    # Match lengths
                    min_len = min(len(r1), len(r2))
                    if min_len > 1:
                        corr = np.corrcoef(r1[-min_len:], r2[-min_len:])[0, 1]
                        corr_matrix[i, j] = corr if not np.isnan(corr) else 0
        
        return {
            'symbols': symbols,
            'matrix': corr_matrix,
            'diversification_score': self._calculate_diversification(corr_matrix)
        }
    
    def _calculate_diversification(self, corr_matrix):
        """Calculate portfolio diversification score"""
        # Average off-diagonal correlation
        n = len(corr_matrix)
        if n <= 1:
            return 1.0
        
        off_diag = []
        for i in range(n):
            for j in range(i+1, n):
                off_diag.append(abs(corr_matrix[i, j]))
        
        avg_corr = np.mean(off_diag) if off_diag else 0
        # Score: 1 = perfectly diversified, 0 = perfectly correlated
        return max(0, 1 - avg_corr)
    
    def kelly_criterion(self, win_rate, avg_win, avg_loss):
        """
        Calculate Kelly Criterion position size
        f* = (p*b - q) / b
        where p = win rate, q = loss rate, b = avg_win/avg_loss
        """
        if avg_loss == 0 or win_rate <= 0 or win_rate >= 1:
            return 0
        
        b = avg_win / avg_loss
        q = 1 - win_rate
        
        kelly = (win_rate * b - q) / b
        
        # Apply fraction (half-Kelly is safer)
        return max(0, kelly * self.kelly_fraction)
    
    def calculate_position_size(self, symbol, account_balance, technical_score, 
                                win_rate=0.55, avg_win=100, avg_loss=50):
        """
        Calculate optimal position size using Kelly + risk constraints
        """
        # Base Kelly calculation
        kelly_size = self.kelly_criterion(win_rate, avg_win, avg_loss)
        
        # Adjust by technical score (0-1)
        adjusted_size = kelly_size * technical_score
        
        # Risk-based caps
        max_risk = 0.02  # Max 2% of account per trade
        max_position = min(adjusted_size, max_risk)
        
        # Adjust for consecutive losses
        if self.state['consecutive_losses'] > 0:
            reduction = min(self.state['consecutive_losses'] * 0.1, 0.5)
            max_position *= (1 - reduction)
        
        # Check daily limit
        remaining = self.daily_loss_limit - self.state['daily_pnl']
        if remaining < 0:
            return 0  # Can't trade
        
        return max_position
    
    def assess_trade_risk(self, symbol, direction, entry_price, stop_loss, 
                         take_profit, position_size, account_balance):
        """
        Comprehensive trade risk assessment
        Returns risk score and approval status
        """
        risks = []
        risk_score = 0  # 0 = lowest risk, 10 = highest risk
        
        # 1. Risk/Reward ratio
        risk_amount = abs(entry_price - stop_loss)
        reward_amount = abs(take_profit - entry_price)
        
        if risk_amount > 0:
            rr_ratio = reward_amount / risk_amount
            if rr_ratio < 1:
                risks.append(f"Risk/Reward = {rr_ratio:.2f} (should be > 1)")
                risk_score += 3
            elif rr_ratio < 1.5:
                risk_score += 1
        
        # 2. Position size check
        risk_percent = (risk_amount * position_size) / account_balance * 100
        if risk_percent > 2:
            risks.append(f"Risk percent = {risk_percent:.2f}% (max 2%)")
            risk_score += 2
        
        # 3. Stop distance check
        stop_pct = risk_amount / entry_price * 100
        if stop_pct > 3:
            risks.append(f"Stop distance = {stop_pct:.2f}% (wide stop)")
            risk_score += 1
        
        # 4. Account status
        can_trade, reason = self.can_trade()
        if not can_trade:
            risks.append(reason)
            risk_score += 10
        
        # Decision
        approved = risk_score < 5 and can_trade
        
        return {
            'approved': approved,
            'risk_score': risk_score,
            'risk_level': 'LOW' if risk_score < 3 else 'MEDIUM' if risk_score < 5 else 'HIGH',
            'risks': risks,
            'rr_ratio': rr_ratio if risk_amount > 0 else 0,
            'risk_percent': risk_percent,
            'position_size_recommended': self.calculate_position_size(
                symbol, account_balance, 0.8, 0.55, 100, 50
            )
        }
    
    def get_portfolio_risk(self, positions, account_equity):
        """
        Calculate portfolio-level risk metrics
        """
        if not positions:
            return {
                'total_exposure': 0,
                'gross_exposure': 0,
                'net_exposure': 0,
                'concentration_risk': 0,
                'margin_utilization': 0
            }
        
        # Calculate exposures
        long_exposure = sum(p['volume'] * p['price'] for p in positions if p['type'] == 'buy')
        short_exposure = sum(p['volume'] * p['price'] for p in positions if p['type'] == 'sell')
        
        gross_exposure = long_exposure + short_exposure
        net_exposure = long_exposure - short_exposure
        
        # Concentration risk (max position / total)
        position_values = [p['volume'] * p['price'] for p in positions]
        max_position = max(position_values) if position_values else 0
        concentration = max_position / gross_exposure if gross_exposure > 0 else 0
        
        # Margin utilization (simplified)
        margin_used = gross_exposure * 0.01  # Assume 1% margin
        margin_util = margin_used / account_equity if account_equity > 0 else 0
        
        return {
            'total_exposure': gross_exposure,
            'gross_exposure': gross_exposure,
            'net_exposure': net_exposure,
            'long_exposure': long_exposure,
            'short_exposure': short_exposure,
            'concentration_risk': concentration,
            'concentration_risk_pct': concentration * 100,
            'margin_utilization': margin_util,
            'margin_utilization_pct': margin_util * 100,
            'num_positions': len(positions)
        }
    
    def get_ultra_stats(self):
        """Get enhanced risk statistics"""
        base_stats = self.get_stats()
        
        # Calculate VaR if we have returns history
        var_stats = None
        if len(self.returns_history) >= 30:
            var_stats = self.calculate_var(self.returns_history)
        
        return {
            **base_stats,
            'var_95': var_stats['var_95'] if var_stats else None,
            'var_99': var_stats['var_99'] if var_stats else None,
            'expected_shortfall': var_stats['expected_shortfall'] if var_stats else None,
            'kelly_fraction': self.kelly_fraction,
            'returns_count': len(self.returns_history)
        }


# ============================================================
# ML-BASED RISK PREDICTION
# ============================================================

class MLRiskPredictor:
    """
    Use ML to predict probability of trade success
    """
    
    def __init__(self):
        self.trade_history = []
        self.feature_importance = {}
    
    def record_trade_outcome(self, features, success):
        """
        Record trade outcome for learning
        features: dict of trade features
        success: bool (profitable or not)
        """
        self.trade_history.append({
            'features': features,
            'success': success,
            'timestamp': datetime.now().isoformat()
        })
    
    def predict_success_probability(self, features):
        """
        Predict probability of trade success
        Simple heuristic-based for now, can upgrade to ML model
        """
        if len(self.trade_history) < 50:
            return {'probability': 0.5, 'confidence': 'low', 'reason': 'Insufficient history'}
        
        # Find similar trades
        similar = []
        for trade in self.trade_history[-200:]:  # Last 200 trades
            similarity = self._calculate_similarity(features, trade['features'])
            if similarity > 0.7:
                similar.append(trade)
        
        if len(similar) < 10:
            return {'probability': 0.5, 'confidence': 'low', 'reason': 'Few similar trades'}
        
        # Calculate success rate
        success_rate = sum(1 for t in similar if t['success']) / len(similar)
        
        return {
            'probability': success_rate,
            'confidence': 'high' if len(similar) > 30 else 'medium',
            'similar_trades': len(similar),
            'reason': f'Based on {len(similar)} similar trades'
        }
    
    def _calculate_similarity(self, feat1, feat2):
        """Calculate feature similarity (0-1)"""
        keys = ['rsi', 'adx', 'trend_strength', 'volume_ratio']
        similarities = []
        
        for key in keys:
            if key in feat1 and key in feat2:
                v1, v2 = feat1[key], feat2[key]
                # Normalize similarity
                diff = abs(v1 - v2)
                sim = max(0, 1 - diff / max(abs(v1), abs(v2), 0.001))
                similarities.append(sim)
        
        return np.mean(similarities) if similarities else 0


# ============================================================
# TEST
# ============================================================

def test_risk_ultra():
    """Test ultra risk manager"""
    print("=" * 60)
    print("RISK MANAGER ULTRA TEST")
    print("=" * 60)
    print()
    
    risk = RiskManagerUltra()
    
    # Initialize
    risk.initialize_equity(10000)
    
    # Test VaR calculation
    np.random.seed(42)
    returns = np.random.randn(100) * 0.01  # 1% daily vol
    var = risk.calculate_var(returns)
    print("Value at Risk:")
    print(f"  VaR 95%: {var['var_95']:.4f}")
    print(f"  VaR 99%: {var['var_99']:.4f}")
    print(f"  Expected Shortfall: {var['expected_shortfall']:.4f}")
    print()
    
    # Test correlation
    symbol_returns = {
        'BTCUSD': np.random.randn(50) * 0.02,
        'XAUUSD': np.random.randn(50) * 0.01,
        'EURUSD': np.random.randn(50) * 0.005
    }
    corr = risk.calculate_correlation_matrix(symbol_returns)
    print("Correlation Matrix:")
    print(f"  Symbols: {corr['symbols']}")
    print(f"  Diversification Score: {corr['diversification_score']:.2f}")
    print()
    
    # Test Kelly
    kelly = risk.kelly_criterion(0.55, 100, 50)
    print(f"Kelly Criterion: {kelly:.4f} ({kelly*100:.1f}%)")
    print()
    
    # Test trade risk assessment
    assessment = risk.assess_trade_risk(
        'BTCUSD', 'LONG', 65000, 64000, 67000, 0.02, 10000
    )
    print("Trade Risk Assessment:")
    print(f"  Approved: {assessment['approved']}")
    print(f"  Risk Score: {assessment['risk_score']}")
    print(f"  Risk Level: {assessment['risk_level']}")
    print(f"  R/R Ratio: {assessment['rr_ratio']:.2f}")
    print()
    
    # Test portfolio risk
    positions = [
        {'type': 'buy', 'volume': 0.1, 'price': 65000},
        {'type': 'sell', 'volume': 0.05, 'price': 2900}
    ]
    portfolio = risk.get_portfolio_risk(positions, 10000)
    print("Portfolio Risk:")
    print(f"  Gross Exposure: ${portfolio['gross_exposure']:,.2f}")
    print(f"  Net Exposure: ${portfolio['net_exposure']:,.2f}")
    print(f"  Concentration: {portfolio['concentration_risk_pct']:.1f}%")
    print(f"  Margin Util: {portfolio['margin_utilization_pct']:.1f}%")
    print()
    
    print("=" * 60)


if __name__ == "__main__":
    test_risk_ultra()
