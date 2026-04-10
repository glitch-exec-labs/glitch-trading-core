"""
Technical Indicators Module
RSI, MACD, Bollinger Bands, etc.
"""
import math

def get_rsi(rates, period=14):
    """Calculate Relative Strength Index
    RSI < 30 = Oversold (buy signal)
    RSI > 70 = Overbought (sell signal)
    """
    if len(rates) < period + 1:
        return None
    
    closes = [r[4] for r in rates]
    
    gains = []
    losses = []
    
    for i in range(1, len(closes)):
        change = closes[i] - closes[i-1]
        if change > 0:
            gains.append(change)
            losses.append(0)
        else:
            gains.append(0)
            losses.append(abs(change))
    
    if len(gains) < period:
        return None
    
    avg_gain = sum(gains[-period:]) / period
    avg_loss = sum(losses[-period:]) / period
    
    if avg_loss == 0:
        rsi = 100
    else:
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
    
    return {
        'rsi': rsi,
        'is_oversold': rsi < 30,
        'is_overbought': rsi > 70,
        'zone': 'oversold' if rsi < 30 else 'overbought' if rsi > 70 else 'neutral'
    }


def get_macd(rates, fast=12, slow=26, signal=9):
    """Calculate MACD
    MACD line crossing above signal = Bullish
    MACD line crossing below signal = Bearish
    """
    if len(rates) < slow + signal:
        return None
    
    closes = [r[4] for r in rates]
    
    def ema(prices, period):
        multiplier = 2 / (period + 1)
        ema_values = [prices[0]]
        for price in prices[1:]:
            ema_values.append((price - ema_values[-1]) * multiplier + ema_values[-1])
        return ema_values
    
    ema_fast = ema(closes[-slow:], fast)[-1]
    ema_slow = ema(closes[-slow:], slow)[-1]
    
    macd_line = ema_fast - ema_slow
    
    # Signal line (EMA of MACD) - simplified
    signal_line = macd_line * 0.9  # Approximation
    
    histogram = macd_line - signal_line
    
    return {
        'macd': macd_line,
        'signal': signal_line,
        'histogram': histogram,
        'is_bullish': macd_line > signal_line,
        'is_bearish': macd_line < signal_line,
        'momentum': 'increasing' if histogram > 0 else 'decreasing'
    }


def get_bollinger_bands(rates, period=20, std_dev=2):
    """Calculate Bollinger Bands
    Price touching lower band = Buy (mean reversion)
    Price touching upper band = Sell (mean reversion)
    Squeeze = Low volatility, potential breakout
    """
    if len(rates) < period:
        return None
    
    closes = [r[4] for r in rates[-period:]]
    current_price = closes[-1]
    
    # Simple Moving Average
    sma = sum(closes) / period
    
    # Standard Deviation
    variance = sum((x - sma) ** 2 for x in closes) / period
    std = math.sqrt(variance)
    
    upper_band = sma + (std * std_dev)
    lower_band = sma - (std * std_dev)
    
    # Bandwidth (squeeze detection)
    bandwidth = (upper_band - lower_band) / sma
    
    return {
        'upper': upper_band,
        'middle': sma,
        'lower': lower_band,
        'current': current_price,
        'bandwidth': bandwidth,
        'is_squeeze': bandwidth < 0.05,  # Less than 5% = squeeze
        'position': 'upper' if current_price >= upper_band else 'lower' if current_price <= lower_band else 'middle',
        'percent_b': (current_price - lower_band) / (upper_band - lower_band) if upper_band != lower_band else 0.5
    }


def get_atr(rates, period=14):
    """Average True Range - volatility measure"""
    if len(rates) < period + 1:
        return None
    
    true_ranges = []
    for i in range(1, len(rates)):
        high, low = rates[i][2], rates[i][3]
        prev_close = rates[i-1][4]
        
        tr1 = high - low
        tr2 = abs(high - prev_close)
        tr3 = abs(low - prev_close)
        
        true_ranges.append(max(tr1, tr2, tr3))
    
    atr = sum(true_ranges[-period:]) / period
    
    return {
        'atr': atr,
        'atr_percent': (atr / rates[-1][4]) * 100,  # ATR as % of price
        'volatility': 'high' if atr > rates[-1][4] * 0.02 else 'low'
    }
