"""
Strategy Selector Module
Chooses the best strategy based on market conditions
"""
from market_regime import detect_market_regime
from technical_indicators import get_rsi, get_macd, get_bollinger_bands, get_atr

def analyze_trade_opportunity(symbol, rates, config=None):
    """
    Main analysis function - returns trade recommendation
    Returns: {
        'action': 'BUY' | 'SELL' | 'HOLD',
        'confidence': 0-1,
        'strategy': 'trend_follow' | 'mean_reversion' | 'breakout',
        'setup': dict with entry details,
        'reason': str explanation
    }
    """
    
    # Get market regime
    regime = detect_market_regime(rates)
    
    # Get indicators
    rsi = get_rsi(rates)
    macd = get_macd(rates)
    bb = get_bollinger_bands(rates)
    atr = get_atr(rates)
    
    result = {
        'action': 'HOLD',
        'confidence': 0,
        'strategy': None,
        'setup': {},
        'reason': 'No clear setup',
        'regime': regime
    }
    
    # === TREND FOLLOWING STRATEGY ===
    if regime['regime'] == 'trending' and regime['direction'] != 'sideways':
        
        if regime['direction'] == 'bullish':
            # Uptrend - look for pullback to buy
            if rsi and rsi['rsi'] < 50 and rsi['rsi'] > 30:
                if macd and macd['is_bullish']:
                    result = {
                        'action': 'BUY',
                        'confidence': regime['confidence'] * 0.8,
                        'strategy': 'trend_follow',
                        'setup': {
                            'entry': 'market',
                            'sl_type': 'atr',
                            'tp_type': 'trailing'
                        },
                        'reason': f"Uptrend pullback - RSI {rsi['rsi']:.1f}, ADX {regime['adx']['adx']:.1f}",
                        'regime': regime
                    }
        
        elif regime['direction'] == 'bearish':
            # Downtrend - look for pullback to sell
            if rsi and rsi['rsi'] > 50 and rsi['rsi'] < 70:
                if macd and macd['is_bearish']:
                    result = {
                        'action': 'SELL',
                        'confidence': regime['confidence'] * 0.8,
                        'strategy': 'trend_follow',
                        'setup': {
                            'entry': 'market',
                            'sl_type': 'atr',
                            'tp_type': 'trailing'
                        },
                        'reason': f"Downtrend pullback - RSI {rsi['rsi']:.1f}, ADX {regime['adx']['adx']:.1f}",
                        'regime': regime
                    }
    
    # === MEAN REVERSION STRATEGY ===
    elif regime['regime'] == 'ranging':
        
        if bb and rsi:
            # Buy at lower band
            if bb['position'] == 'lower' and rsi['is_oversold']:
                result = {
                    'action': 'BUY',
                    'confidence': 0.7,
                    'strategy': 'mean_reversion',
                    'setup': {
                        'entry': 'market',
                        'target': bb['middle'],  # Take profit at middle band
                        'sl_type': 'percent',
                        'sl_value': 1.5  # 1.5% stop
                    },
                    'reason': f"Oversold bounce - RSI {rsi['rsi']:.1f}, at lower BB",
                    'regime': regime
                }
            
            # Sell at upper band
            elif bb['position'] == 'upper' and rsi['is_overbought']:
                result = {
                    'action': 'SELL',
                    'confidence': 0.7,
                    'strategy': 'mean_reversion',
                    'setup': {
                        'entry': 'market',
                        'target': bb['middle'],
                        'sl_type': 'percent',
                        'sl_value': 1.5
                    },
                    'reason': f"Overbought pullback - RSI {rsi['rsi']:.1f}, at upper BB",
                    'regime': regime
                }
    
    # === BREAKOUT STRATEGY ===
    elif bb and bb['is_squeeze']:
        # Bollinger squeeze = low volatility, potential breakout
        if macd and abs(macd['histogram']) > 0:
            direction = 'BUY' if macd['is_bullish'] else 'SELL'
            result = {
                'action': direction,
                'confidence': 0.6,
                'strategy': 'breakout',
                'setup': {
                    'entry': 'market',
                    'sl_type': 'atr',
                    'tp_type': 'trailing'
                },
                'reason': f"Bollinger squeeze breakout - bandwidth {bb['bandwidth']:.3f}",
                'regime': regime
            }
    
    return result


def calculate_position_size(analysis, account_balance, risk_config):
    """Calculate position size based on setup quality"""
    
    base_risk = risk_config.get('base_risk_percent', 2.0)
    confidence = analysis['confidence']
    
    # Scale risk by confidence
    risk_percent = base_risk * confidence
    
    # Different strategies have different risk profiles
    if analysis['strategy'] == 'trend_follow':
        risk_percent *= 1.0  # Standard risk
    elif analysis['strategy'] == 'mean_reversion':
        risk_percent *= 0.8  # Lower risk for counter-trend
    elif analysis['strategy'] == 'breakout':
        risk_percent *= 0.9  # Medium risk
    
    # Cap at max risk
    risk_percent = min(risk_percent, risk_config.get('max_risk_per_trade', 3.0))
    
    return {
        'risk_percent': risk_percent,
        'risk_amount': account_balance * (risk_percent / 100),
        'confidence': confidence,
        'reason': f"{analysis['strategy']} with {confidence:.0%} confidence"
    }
