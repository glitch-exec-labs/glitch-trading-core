"""
Market Regime Detection Module
Detects if market is trending or ranging
"""
import numpy as np

def get_adx(rates, period=14):
    """Calculate Average Directional Index (ADX)
    ADX > 25 = Trending
    ADX < 20 = Ranging
    """
    if len(rates) < period + 1:
        return None
    
    highs = [r[2] for r in rates]
    lows = [r[3] for r in rates]
    closes = [r[4] for r in rates]
    
    # Calculate +DM and -DM
    plus_dm = []
    minus_dm = []
    tr = []
    
    for i in range(1, len(rates)):
        up_move = highs[i] - highs[i-1]
        down_move = lows[i-1] - lows[i]
        
        if up_move > down_move and up_move > 0:
            plus_dm.append(up_move)
        else:
            plus_dm.append(0)
        
        if down_move > up_move and down_move > 0:
            minus_dm.append(down_move)
        else:
            minus_dm.append(0)
        
        # True Range
        tr1 = highs[i] - lows[i]
        tr2 = abs(highs[i] - closes[i-1])
        tr3 = abs(lows[i] - closes[i-1])
        tr.append(max(tr1, tr2, tr3))
    
    if len(tr) < period:
        return None
    
    # Smoothed averages
    atr = sum(tr[-period:]) / period
    plus_di = 100 * (sum(plus_dm[-period:]) / period) / atr if atr > 0 else 0
    minus_di = 100 * (sum(minus_dm[-period:]) / period) / atr if atr > 0 else 0
    
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di) if (plus_di + minus_di) > 0 else 0
    
    # Simplified ADX (smoothed DX)
    adx = dx  # In real implementation, would smooth over period
    
    return {
        'adx': adx,
        'plus_di': plus_di,
        'minus_di': minus_di,
        'is_trending': adx > 25,
        'is_ranging': adx < 20,
        'strength': 'strong' if adx > 40 else 'moderate' if adx > 25 else 'weak'
    }


def get_ema_alignment(rates, fast=20, slow=50):
    """Check EMA alignment for trend direction
    Returns: 'bullish', 'bearish', or 'neutral'
    """
    if len(rates) < slow:
        return 'unknown'
    
    closes = [r[4] for r in rates]
    
    # Calculate EMAs
    def ema(prices, period):
        multiplier = 2 / (period + 1)
        ema_values = [prices[0]]
        for price in prices[1:]:
            ema_values.append((price - ema_values[-1]) * multiplier + ema_values[-1])
        return ema_values[-1]
    
    ema_fast = ema(closes[-fast:], fast)
    ema_slow = ema(closes[-slow:], slow)
    current_price = closes[-1]
    
    if current_price > ema_fast > ema_slow:
        return 'bullish'
    elif current_price < ema_fast < ema_slow:
        return 'bearish'
    else:
        return 'neutral'


def detect_market_regime(rates):
    """Main function - returns full market analysis"""
    adx_data = get_adx(rates)
    ema_trend = get_ema_alignment(rates)
    
    if not adx_data:
        return {'regime': 'unknown', 'confidence': 0}
    
    regime = {
        'adx': adx_data,
        'ema_trend': ema_trend,
        'regime': 'trending' if adx_data['is_trending'] else 'ranging' if adx_data['is_ranging'] else 'transition',
        'direction': ema_trend if adx_data['is_trending'] else 'sideways',
        'confidence': min(adx_data['adx'] / 50, 1.0)  # 0-1 scale
    }
    
    return regime
