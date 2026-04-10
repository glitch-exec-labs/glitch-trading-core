"""
Backtest Engine - Vectorized Backtesting
Fast strategy testing with NumPy
"""
import numpy as np
from datetime import datetime
import json

class BacktestEngine:
    """
    Ultra-fast vectorized backtesting engine
    Tests strategies on historical data
    """
    
    def __init__(self, initial_balance=10000, commission=0.0001):
        self.initial_balance = initial_balance
        self.commission = commission  # 0.01% default
        self.results = None
    
    def run_backtest(self, rates, strategy_func, strategy_config=None):
        """
        Run vectorized backtest
        
        rates: OHLCV array [timestamp, open, high, low, close, volume]
        strategy_func: Function that returns signals
        strategy_config: Strategy parameters
        
        Returns: Backtest results dict
        """
        rates = np.asarray(rates, dtype=np.float64)
        n = len(rates)
        
        if n < 100:
            return {'error': 'Insufficient data'}
        
        closes = rates[:, 4]
        
        # Get signals from strategy
        signals = strategy_func(rates, strategy_config)
        
        # Vectorized backtest calculation
        positions = np.zeros(n)
        entry_prices = np.zeros(n)
        pnl = np.zeros(n)
        equity = np.zeros(n)
        equity[0] = self.initial_balance
        
        current_position = 0  # 0=none, 1=long, -1=short
        entry_price = 0
        
        trades = []
        
        for i in range(1, n):
            signal = signals[i]
            
            # Entry logic
            if current_position == 0 and signal != 0:
                current_position = signal
                entry_price = closes[i]
                entry_prices[i] = entry_price
                positions[i] = current_position
            
            # Exit logic
            elif current_position != 0:
                # Check for exit signal or stop loss
                exit_signal = signal != current_position or signal == 0
                
                if exit_signal or i == n - 1:  # Force exit on last bar
                    # Calculate P&L
                    if current_position == 1:  # Long
                        trade_pnl = (closes[i] - entry_price) / entry_price
                    else:  # Short
                        trade_pnl = (entry_price - closes[i]) / entry_price
                    
                    # Apply commission (entry + exit)
                    trade_pnl -= self.commission * 2
                    
                    # Update equity
                    pnl[i] = trade_pnl
                    equity[i] = equity[i-1] * (1 + trade_pnl)
                    
                    # Record trade
                    trades.append({
                        'entry_idx': np.where(entry_prices == entry_price)[0][0],
                        'exit_idx': i,
                        'direction': 'LONG' if current_position == 1 else 'SHORT',
                        'entry_price': entry_price,
                        'exit_price': closes[i],
                        'pnl_pct': trade_pnl * 100,
                        'equity': equity[i]
                    })
                    
                    # Reset position
                    current_position = 0
                    entry_price = 0
                else:
                    # Hold position
                    positions[i] = current_position
                    equity[i] = equity[i-1]
            else:
                equity[i] = equity[i-1]
        
        # Calculate metrics
        self.results = self._calculate_metrics(trades, equity)
        self.results['trades'] = trades
        self.results['equity_curve'] = equity.tolist()
        
        return self.results
    
    def _calculate_metrics(self, trades, equity):
        """Calculate performance metrics"""
        if not trades:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'total_return': 0,
                'max_drawdown': 0,
                'sharpe_ratio': 0
            }
        
        pnl_list = [t['pnl_pct'] for t in trades]
        wins = [p for p in pnl_list if p > 0]
        losses = [p for p in pnl_list if p <= 0]
        
        total_trades = len(trades)
        win_rate = len(wins) / total_trades * 100 if total_trades > 0 else 0
        
        gross_profit = sum(wins) if wins else 0
        gross_loss = abs(sum(losses)) if losses else 1
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        total_return = (equity[-1] - self.initial_balance) / self.initial_balance * 100
        
        # Max drawdown
        running_max = np.maximum.accumulate(equity)
        drawdown = (equity - running_max) / running_max * 100
        max_drawdown = abs(np.min(drawdown))
        
        # Sharpe ratio (simplified)
        returns = np.diff(equity) / equity[:-1]
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
        
        return {
            'total_trades': total_trades,
            'winning_trades': len(wins),
            'losing_trades': len(losses),
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'total_return_pct': total_return,
            'max_drawdown_pct': max_drawdown,
            'sharpe_ratio': sharpe,
            'final_equity': equity[-1],
            'avg_trade_pct': np.mean(pnl_list),
            'best_trade_pct': max(pnl_list) if pnl_list else 0,
            'worst_trade_pct': min(pnl_list) if pnl_list else 0
        }
    
    def optimize_parameters(self, rates, strategy_func, param_grid):
        """
        Grid search for best parameters
        
        param_grid: dict of lists, e.g., {'rsi_period': [10, 14, 20], 'threshold': [30, 40, 50]}
        """
        from itertools import product
        
        # Generate all combinations
        keys = list(param_grid.keys())
        values = list(param_grid.values())
        combinations = list(product(*values))
        
        best_result = None
        best_params = None
        best_score = -np.inf
        
        print(f"Testing {len(combinations)} parameter combinations...")
        
        for combo in combinations:
            params = dict(zip(keys, combo))
            result = self.run_backtest(rates, strategy_func, params)
            
            # Score based on risk-adjusted return
            score = result['total_return_pct'] - result['max_drawdown_pct'] * 0.5
            
            if score > best_score:
                best_score = score
                best_result = result
                best_params = params
        
        return {
            'best_params': best_params,
            'best_result': best_result,
            'all_combinations_tested': len(combinations)
        }
    
    def walk_forward_analysis(self, rates, strategy_func, train_size=0.7, n_splits=3):
        """
        Walk-forward optimization
        Train on first 70%, test on next 30%, repeat
        """
        rates = np.asarray(rates)
        n = len(rates)
        split_size = n // n_splits
        
        results = []
        
        for i in range(n_splits):
            start_idx = i * split_size
            train_end = start_idx + int(split_size * train_size)
            test_end = min((i + 1) * split_size, n)
            
            if train_end >= n or test_end >= n:
                break
            
            train_data = rates[start_idx:train_end]
            test_data = rates[train_end:test_end]
            
            # Train (placeholder - would optimize params here)
            # For now just run on test data
            result = self.run_backtest(test_data, strategy_func)
            result['period'] = i + 1
            results.append(result)
        
        # Aggregate results
        avg_return = np.mean([r['total_return_pct'] for r in results])
        avg_drawdown = np.mean([r['max_drawdown_pct'] for r in results])
        avg_winrate = np.mean([r['win_rate'] for r in results])
        
        return {
            'period_results': results,
            'avg_return_pct': avg_return,
            'avg_drawdown_pct': avg_drawdown,
            'avg_win_rate': avg_winrate,
            'is_robust': min(r['total_return_pct'] for r in results) > 0
        }


# ============================================================
# EXAMPLE STRATEGIES
# ============================================================

def rsi_strategy(rates, config=None):
    """
    Simple RSI strategy for backtesting
    Buy when RSI < oversold, sell when RSI > overbought
    """
    config = config or {'period': 14, 'oversold': 30, 'overbought': 70}
    
    closes = rates[:, 4]
    n = len(closes)
    
    # Calculate RSI
    period = config['period']
    deltas = np.diff(closes)
    
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    
    # Wilder's smoothing
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])
    
    rsi_values = np.zeros(n)
    rsi_values[:period] = 50
    
    alpha = (period - 1) / period
    for i in range(period, n-1):
        avg_gain = alpha * avg_gain + gains[i] / period
        avg_loss = alpha * avg_loss + losses[i] / period
        rs = avg_gain / (avg_loss + 1e-10)
        rsi_values[i+1] = 100 - (100 / (1 + rs))
    
    # Generate signals
    signals = np.zeros(n)
    signals[rsi_values < config['oversold']] = 1   # Buy
    signals[rsi_values > config['overbought']] = -1  # Sell
    
    return signals


def trend_following_strategy(rates, config=None):
    """
    Trend following using EMA crossover
    """
    config = config or {'fast': 20, 'slow': 50}
    
    closes = rates[:, 4]
    n = len(closes)
    
    # Calculate EMAs
    def ema(data, period):
        alpha = 2 / (period + 1)
        result = np.zeros_like(data)
        result[0] = data[0]
        for i in range(1, len(data)):
            result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
        return result
    
    ema_fast = ema(closes, config['fast'])
    ema_slow = ema(closes, config['slow'])
    
    # Generate signals
    signals = np.zeros(n)
    signals[ema_fast > ema_slow] = 1  # Long when fast > slow
    signals[ema_fast < ema_slow] = -1  # Short when fast < slow
    
    return signals


def smc_strategy(rates, config=None):
    """
    Smart Money Concepts strategy
    """
    from smart_money_ultra import analyze_smc_setup_ultra
    
    n = len(rates)
    signals = np.zeros(n)
    
    for i in range(50, n):
        window = rates[i-50:i]
        setup = analyze_smc_setup_ultra(window)
        
        if setup['has_setup']:
            if setup['direction'] == 'LONG':
                signals[i] = 1
            elif setup['direction'] == 'SHORT':
                signals[i] = -1
    
    return signals


# ============================================================
# BENCHMARK
# ============================================================

def benchmark_backtest():
    """Benchmark backtest speed"""
    import time
    
    print("=" * 60)
    print("BACKTEST ENGINE BENCHMARK")
    print("=" * 60)
    print()
    
    # Generate test data
    np.random.seed(42)
    n = 10000
    
    closes = np.random.randn(n).cumsum() + 65000
    highs = closes + np.abs(np.random.randn(n)) * 100
    lows = closes - np.abs(np.random.randn(n)) * 100
    opens = (highs + lows) / 2
    volumes = np.abs(np.random.randn(n)) * 1000 + 500
    rates = np.column_stack([np.arange(n), opens, highs, lows, closes, volumes])
    
    engine = BacktestEngine(initial_balance=10000)
    
    # Test RSI strategy
    print(f"Testing RSI strategy on {n} candles...")
    start = time.time()
    result = engine.run_backtest(rates, rsi_strategy)
    elapsed = time.time() - start
    
    print(f"Completed in {elapsed:.3f}s ({n/elapsed:.0f} candles/sec)")
    print()
    print("Results:")
    print(f"  Total Trades: {result['total_trades']}")
    print(f"  Win Rate: {result['win_rate']:.1f}%")
    print(f"  Profit Factor: {result['profit_factor']:.2f}")
    print(f"  Total Return: {result['total_return_pct']:.2f}%")
    print(f"  Max Drawdown: {result['max_drawdown_pct']:.2f}%")
    print(f"  Sharpe Ratio: {result['sharpe_ratio']:.2f}")
    print()
    
    # Test trend following
    print(f"Testing Trend Following strategy on {n} candles...")
    start = time.time()
    result = engine.run_backtest(rates, trend_following_strategy)
    elapsed = time.time() - start
    
    print(f"Completed in {elapsed:.3f}s ({n/elapsed:.0f} candles/sec)")
    print(f"  Total Return: {result['total_return_pct']:.2f}%")
    print()
    
    print("=" * 60)


if __name__ == "__main__":
    benchmark_backtest()
