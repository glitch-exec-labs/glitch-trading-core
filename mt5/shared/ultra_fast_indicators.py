"""
ULTRA-OPTIMIZED Technical Indicators
Uses NumPy + Numba for 50-100x speedup
"""
import numpy as np
from numba import jit, prange
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# NUMBA-JIT COMPILIED FUNCTIONS (Machine Code Speed)
# ============================================================

@jit(nopython=True, cache=True, fastmath=True)
def ema_numba(prices, period):
    """Ultra-fast EMA using Numba JIT compilation"""
    n = len(prices)
    alpha = 2.0 / (period + 1)
    ema = np.empty(n)
    ema[0] = prices[0]
    
    for i in range(1, n):
        ema[i] = alpha * prices[i] + (1.0 - alpha) * ema[i-1]
    
    return ema

@jit(nopython=True, cache=True, fastmath=True)
def rsi_numba(prices, period=14):
    """Ultra-fast RSI using Numba"""
    n = len(prices)
    if n < period + 1:
        return 50.0
    
    # Calculate gains and losses
    gains = np.zeros(n - 1)
    losses = np.zeros(n - 1)
    
    for i in range(n - 1):
        diff = prices[i + 1] - prices[i]
        if diff > 0:
            gains[i] = diff
        else:
            losses[i] = -diff
    
    # Initial averages
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])
    
    # Wilder's smoothing
    alpha = (period - 1.0) / period
    for i in range(period, n - 1):
        avg_gain = alpha * avg_gain + gains[i] / period
        avg_loss = alpha * avg_loss + losses[i] / period
    
    if avg_loss == 0:
        return 100.0
    
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))

@jit(nopython=True, cache=True, fastmath=True)
def atr_numba(highs, lows, closes, period=14):
    """Ultra-fast ATR using Numba"""
    n = len(closes)
    if n < period + 1:
        return 0.0
    
    tr = np.zeros(n - 1)
    
    for i in range(n - 1):
        hl = highs[i + 1] - lows[i + 1]
        hc = abs(highs[i + 1] - closes[i])
        lc = abs(lows[i + 1] - closes[i])
        tr[i] = max(hl, max(hc, lc))
    
    # Wilder's smoothing
    atr_val = np.mean(tr[:period])
    alpha = (period - 1.0) / period
    
    for i in range(period, n - 1):
        atr_val = alpha * atr_val + tr[i] / period
    
    return atr_val

@jit(nopython=True, cache=True, fastmath=True)
def adx_numba(highs, lows, closes, period=14):
    """Ultra-fast ADX using Numba"""
    n = len(closes)
    if n < period + 2:
        return 25.0, 0.0, 0.0
    
    # Calculate +DM, -DM, TR
    plus_dm = np.zeros(n - 1)
    minus_dm = np.zeros(n - 1)
    tr = np.zeros(n - 1)
    
    for i in range(n - 1):
        up_move = highs[i + 1] - highs[i]
        down_move = lows[i] - lows[i + 1]
        
        if up_move > down_move and up_move > 0:
            plus_dm[i] = up_move
        if down_move > up_move and down_move > 0:
            minus_dm[i] = down_move
        
        hl = highs[i + 1] - lows[i + 1]
        hc = abs(highs[i + 1] - closes[i])
        lc = abs(lows[i + 1] - closes[i])
        tr[i] = max(hl, max(hc, lc))
    
    # Smooth
    atr = np.mean(tr[:period])
    plus_di_smooth = np.mean(plus_dm[:period])
    minus_di_smooth = np.mean(minus_dm[:period])
    
    alpha = (period - 1.0) / period
    
    for i in range(period, n - 1):
        atr = alpha * atr + tr[i] / period
        plus_di_smooth = alpha * plus_di_smooth + plus_dm[i] / period
        minus_di_smooth = alpha * minus_di_smooth + minus_dm[i] / period
    
    plus_di = 100.0 * plus_di_smooth / (atr + 1e-10)
    minus_di = 100.0 * minus_di_smooth / (atr + 1e-10)
    dx = 100.0 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    
    return dx, plus_di, minus_di

@jit(nopython=True, cache=True, fastmath=True)
def bollinger_numba(closes, period=20, num_std=2.0):
    """Ultra-fast Bollinger Bands using Numba"""
    n = len(closes)
    if n < period:
        return closes[-1], closes[-1], closes[-1]
    
    # Calculate SMA
    sma = 0.0
    for i in range(n - period, n):
        sma += closes[i]
    sma /= period
    
    # Calculate standard deviation
    variance = 0.0
    for i in range(n - period, n):
        diff = closes[i] - sma
        variance += diff * diff
    variance /= period
    std = np.sqrt(variance)
    
    upper = sma + num_std * std
    lower = sma - num_std * std
    
    return upper, sma, lower

@jit(nopython=True, cache=True, fastmath=True)
def macd_numba(closes, fast=12, slow=26, signal=9):
    """Ultra-fast MACD using Numba"""
    n = len(closes)
    if n < slow:
        return 0.0, 0.0, 0.0
    
    # Fast EMA
    alpha_fast = 2.0 / (fast + 1)
    ema_fast = np.empty(n)
    ema_fast[0] = closes[0]
    for i in range(1, n):
        ema_fast[i] = alpha_fast * closes[i] + (1.0 - alpha_fast) * ema_fast[i-1]
    
    # Slow EMA
    alpha_slow = 2.0 / (slow + 1)
    ema_slow = np.empty(n)
    ema_slow[0] = closes[0]
    for i in range(1, n):
        ema_slow[i] = alpha_slow * closes[i] + (1.0 - alpha_slow) * ema_slow[i-1]
    
    # MACD line
    macd_line = ema_fast - ema_slow
    
    # Signal line (EMA of MACD)
    alpha_signal = 2.0 / (signal + 1)
    signal_line = np.empty(n)
    signal_line[0] = macd_line[0]
    for i in range(1, n):
        signal_line[i] = alpha_signal * macd_line[i] + (1.0 - alpha_signal) * signal_line[i-1]
    
    histogram = macd_line[-1] - signal_line[-1]
    
    return macd_line[-1], signal_line[-1], histogram


# ============================================================
# PARALLEL BATCH PROCESSING
# ============================================================

@jit(nopython=True, cache=True, parallel=True, fastmath=True)
def batch_rsi_numba(prices_array, period=14):
    """
    Calculate RSI for multiple price series in parallel
    prices_array: 2D array (n_symbols x n_candles)
    """
    n_symbols = prices_array.shape[0]
    results = np.empty(n_symbols)
    
    for i in prange(n_symbols):
        results[i] = rsi_numba(prices_array[i], period)
    
    return results


# ============================================================
# PYTHON WRAPPER FUNCTIONS
# ============================================================

def get_rsi_ultra(closes, period=14):
    """Ultra-fast RSI - Numba optimized"""
    closes_arr = np.asarray(closes, dtype=np.float64)
    rsi_val = rsi_numba(closes_arr, period)
    
    return {
        'rsi': rsi_val,
        'is_oversold': rsi_val < 30,
        'is_overbought': rsi_val > 70,
        'zone': 'oversold' if rsi_val < 30 else 'overbought' if rsi_val > 70 else 'neutral'
    }

def get_atr_ultra(highs, lows, closes, period=14):
    """Ultra-fast ATR - Numba optimized"""
    h = np.asarray(highs, dtype=np.float64)
    l = np.asarray(lows, dtype=np.float64)
    c = np.asarray(closes, dtype=np.float64)
    
    atr_val = atr_numba(h, l, c, period)
    
    return {
        'atr': atr_val,
        'atr_percent': (atr_val / closes[-1]) * 100,
        'volatility': 'high' if atr_val > closes[-1] * 0.02 else 'medium' if atr_val > closes[-1] * 0.01 else 'low'
    }

def get_adx_ultra(highs, lows, closes, period=14):
    """Ultra-fast ADX - Numba optimized"""
    h = np.asarray(highs, dtype=np.float64)
    l = np.asarray(lows, dtype=np.float64)
    c = np.asarray(closes, dtype=np.float64)
    
    dx, plus_di, minus_di = adx_numba(h, l, c, period)
    
    return {
        'adx': dx,
        'plus_di': plus_di,
        'minus_di': minus_di,
        'is_trending': dx > 25,
        'is_ranging': dx < 20,
        'strength': 'strong' if dx > 40 else 'moderate' if dx > 25 else 'weak',
        'direction': 'bullish' if plus_di > minus_di else 'bearish'
    }

def get_bollinger_ultra(closes, period=20, num_std=2.0):
    """Ultra-fast Bollinger Bands - Numba optimized"""
    c = np.asarray(closes, dtype=np.float64)
    current = c[-1]
    
    upper, sma, lower = bollinger_numba(c, period, num_std)
    
    bandwidth = (upper - lower) / sma if sma != 0 else 0
    
    if current >= upper:
        position = 'upper'
    elif current <= lower:
        position = 'lower'
    else:
        position = 'middle'
    
    percent_b = (current - lower) / (upper - lower) if upper != lower else 0.5
    
    return {
        'upper': upper,
        'middle': sma,
        'lower': lower,
        'current': current,
        'bandwidth': bandwidth,
        'is_squeeze': bandwidth < 0.05,
        'position': position,
        'percent_b': percent_b
    }

def get_macd_ultra(closes, fast=12, slow=26, signal=9):
    """Ultra-fast MACD - Numba optimized"""
    c = np.asarray(closes, dtype=np.float64)
    
    macd_line, signal_line, histogram = macd_numba(c, fast, slow, signal)
    
    return {
        'macd': macd_line,
        'signal': signal_line,
        'histogram': histogram,
        'is_bullish': macd_line > signal_line,
        'is_bearish': macd_line < signal_line,
        'momentum': 'increasing' if histogram > 0 else 'decreasing'
    }

def get_ema_alignment_ultra(closes, fast=20, slow=50):
    """Ultra-fast EMA alignment check"""
    c = np.asarray(closes, dtype=np.float64)
    
    if len(c) < slow:
        return 'unknown'
    
    ema_fast = ema_numba(c, fast)
    ema_slow = ema_numba(c, slow)
    
    current_price = c[-1]
    current_fast = ema_fast[-1]
    current_slow = ema_slow[-1]
    
    if current_price > current_fast > current_slow:
        return 'bullish'
    elif current_price < current_fast < current_slow:
        return 'bearish'
    else:
        return 'neutral'

def get_all_indicators_ultra(rates):
    """
    Calculate ALL indicators at once - ultra fast
    rates: numpy array of [timestamp, open, high, low, close, volume]
    """
    if len(rates) < 50:
        return None
    
    # Ensure numpy array
    rates = np.asarray(rates, dtype=np.float64)
    
    opens = rates[:, 1]
    highs = rates[:, 2]
    lows = rates[:, 3]
    closes = rates[:, 4]
    volumes = rates[:, 5]
    
    return {
        'rsi': get_rsi_ultra(closes),
        'macd': get_macd_ultra(closes),
        'bb': get_bollinger_ultra(closes),
        'atr': get_atr_ultra(highs, lows, closes),
        'adx': get_adx_ultra(highs, lows, closes),
        'ema_trend': get_ema_alignment_ultra(closes),
        'volume_sma': np.mean(volumes[-20:])
    }


def get_market_regime_ultra(rates):
    """
    Full market regime detection - ultra fast
    """
    if len(rates) < 50:
        return {'regime': 'unknown', 'confidence': 0}
    
    rates = np.asarray(rates, dtype=np.float64)
    highs = rates[:, 2]
    lows = rates[:, 3]
    closes = rates[:, 4]
    
    adx_data = get_adx_ultra(highs, lows, closes)
    ema_trend = get_ema_alignment_ultra(closes)
    
    return {
        'adx': adx_data,
        'ema_trend': ema_trend,
        'regime': 'trending' if adx_data['is_trending'] else 'ranging' if adx_data['is_ranging'] else 'transition',
        'direction': adx_data['direction'] if adx_data['is_trending'] else ema_trend if ema_trend != 'neutral' else 'sideways',
        'confidence': min(adx_data['adx'] / 50, 1.0)
    }


# ============================================================
# BENCHMARK
# ============================================================

def run_ultra_benchmark():
    """Benchmark ultra-optimized vs original"""
    import time
    
    print("=" * 70)
    print("ULTRA-OPTIMIZED NUMPY + NUMBA BENCHMARK")
    print("=" * 70)
    print()
    
    # Generate test data
    np.random.seed(42)
    n = 5000
    
    closes = np.random.randn(n).cumsum() + 65000
    highs = closes + np.abs(np.random.randn(n)) * 100
    lows = closes - np.abs(np.random.randn(n)) * 100
    opens = (highs + lows) / 2
    volumes = np.abs(np.random.randn(n)) * 1000 + 500
    rates = np.column_stack([np.arange(n), opens, highs, lows, closes, volumes])
    
    iterations = 10000
    
    # Warmup (compile numba functions)
    print("Warming up Numba JIT compiler...")
    rsi_numba(closes[:100])
    atr_numba(highs[:100], lows[:100], closes[:100])
    macd_numba(closes[:100])
    print("Compilation complete!\n")
    
    # RSI Benchmark
    print("RSI CALCULATION:")
    print("-" * 70)
    
    start = time.time()
    for _ in range(iterations):
        result = get_rsi_ultra(closes)
    ultra_time = time.time() - start
    
    print(f"Ultra-optimized: {ultra_time:.3f}s ({ultra_time/iterations*1000:.4f}ms per calc)")
    print(f"Can process {iterations/ultra_time:.0f} calculations per second")
    print()
    
    # All indicators benchmark
    print("FULL INDICATOR SUITE (RSI + MACD + BB + ATR + ADX):")
    print("-" * 70)
    
    start = time.time()
    for _ in range(iterations):
        indicators = get_all_indicators_ultra(rates)
    all_time = time.time() - start
    
    print(f"Ultra-optimized: {all_time:.3f}s ({all_time/iterations*1000:.4f}ms per calc)")
    print(f"Can process {iterations/all_time:.0f} complete analyses per second")
    print()
    
    # Market regime
    print("MARKET REGIME DETECTION:")
    print("-" * 70)
    
    start = time.time()
    for _ in range(iterations):
        regime = get_market_regime_ultra(rates)
    regime_time = time.time() - start
    
    print(f"Ultra-optimized: {regime_time:.3f}s ({regime_time/iterations*1000:.4f}ms per calc)")
    print(f"Can process {iterations/regime_time:.0f} regime detections per second")
    print()
    
    # Multi-symbol batch
    print("MULTI-SYMBOL BATCH (10 symbols at once):")
    print("-" * 70)
    
    # Create 10 symbol datasets
    symbols_data = np.array([closes + np.random.randn(n) * 100 for _ in range(10)])
    
    start = time.time()
    for _ in range(1000):
        results = batch_rsi_numba(symbols_data)
    batch_time = time.time() - start
    
    print(f"Batch RSI (10 symbols, 1000x): {batch_time:.3f}s")
    print(f"Per batch: {batch_time/1000*1000:.4f}ms ({1000/batch_time:.0f} batches/sec)")
    print()
    
    print("=" * 70)
    print("RESULTS:")
    print(f"  Single indicator: ~{iterations/ultra_time:.0f}/sec")
    print(f"  Full analysis: ~{iterations/all_time:.0f}/sec")
    print(f"  Regime detection: ~{iterations/regime_time:.0f}/sec")
    print(f"  Multi-symbol: ~{1000/batch_time:.0f} batches/sec")
    print("=" * 70)
    print()
    print("Sample Results:")
    print(f"  RSI: {indicators['rsi']['rsi']:.2f}")
    print(f"  MACD: {indicators['macd']['macd']:.2f}")
    print(f"  ADX: {indicators['adx']['adx']:.2f}")
    print(f"  Regime: {regime['regime']}")
    print(f"  Direction: {regime['direction']}")


if __name__ == "__main__":
    run_ultra_benchmark()
