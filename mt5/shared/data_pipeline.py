"""
Data Pipeline - Data handling and storage
MT5 integration, caching, and feature preparation
"""
import numpy as np
import json
import sqlite3
from datetime import datetime
import os
import pickle

class DataPipeline:
    """
    Handles all data operations:
    - MT5 data export
    - Caching
    - Feature engineering
    - Storage (SQLite/CSV)
    """
    
    def __init__(self, db_path='data/trading_data.db', cache_dir='data/cache'):
        self.db_path = db_path
        self.cache_dir = cache_dir
        
        # Create directories
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        os.makedirs(cache_dir, exist_ok=True)
        
        # Initialize database
        self._init_db()
    
    def _init_db(self):
        """Initialize SQLite database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # OHLCV data table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS ohlcv (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                timestamp INTEGER NOT NULL,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume REAL,
                UNIQUE(symbol, timeframe, timestamp)
            )
        ''')
        
        # Trades table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp INTEGER NOT NULL,
                symbol TEXT NOT NULL,
                direction TEXT NOT NULL,
                entry_price REAL,
                exit_price REAL,
                volume REAL,
                pnl REAL,
                strategy TEXT,
                metadata TEXT
            )
        ''')
        
        # Performance metrics table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL,
                symbol TEXT,
                strategy TEXT,
                metric_name TEXT NOT NULL,
                metric_value REAL,
                UNIQUE(date, symbol, strategy, metric_name)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    # ============================================================
    # MT5 DATA OPERATIONS
    # ============================================================
    
    def fetch_from_mt5(self, symbol, timeframe='M5', count=1000):
        """
        Fetch data from MT5
        
        timeframe: 'M1', 'M5', 'M15', 'M30', 'H1', 'H4', 'D1'
        """
        try:
            import MetaTrader5 as mt5
            
            if not mt5.initialize():
                print("MT5 not initialized")
                return None
            
            # Map timeframe strings to MT5 constants
            tf_map = {
                'M1': mt5.TIMEFRAME_M1,
                'M5': mt5.TIMEFRAME_M5,
                'M15': mt5.TIMEFRAME_M15,
                'M30': mt5.TIMEFRAME_M30,
                'H1': mt5.TIMEFRAME_H1,
                'H4': mt5.TIMEFRAME_H4,
                'D1': mt5.TIMEFRAME_D1,
            }
            
            mt5_tf = tf_map.get(timeframe, mt5.TIMEFRAME_M5)
            
            rates = mt5.copy_rates_from_pos(symbol, mt5_tf, 0, count)
            mt5.shutdown()
            
            if rates is None or len(rates) == 0:
                print(f"No data returned for {symbol}")
                return None
            
            # Convert to numpy array
            data = np.array([
                [r['time'], r['open'], r['high'], r['low'], r['close'], r['tick_volume']]
                for r in rates
            ])
            
            return data
            
        except ImportError:
            print("MetaTrader5 module not available")
            return None
        except Exception as e:
            print(f"Error fetching from MT5: {e}")
            return None
    
    def save_to_db(self, symbol, timeframe, data):
        """Save OHLCV data to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for row in data:
            try:
                cursor.execute('''
                    INSERT OR REPLACE INTO ohlcv 
                    (symbol, timeframe, timestamp, open, high, low, close, volume)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (symbol, timeframe, int(row[0]), row[1], row[2], row[3], row[4], row[5]))
            except Exception as e:
                print(f"Error saving row: {e}")
        
        conn.commit()
        conn.close()
        print(f"Saved {len(data)} records for {symbol} {timeframe}")
    
    def load_from_db(self, symbol, timeframe, start_time=None, end_time=None):
        """Load OHLCV data from database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        query = '''
            SELECT timestamp, open, high, low, close, volume
            FROM ohlcv
            WHERE symbol = ? AND timeframe = ?
        '''
        params = [symbol, timeframe]
        
        if start_time:
            query += ' AND timestamp >= ?'
            params.append(start_time)
        if end_time:
            query += ' AND timestamp <= ?'
            params.append(end_time)
        
        query += ' ORDER BY timestamp'
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()
        
        if not rows:
            return None
        
        return np.array(rows)
    
    # ============================================================
    # CACHING
    # ============================================================
    
    def cache_data(self, key, data):
        """Cache data to disk"""
        cache_path = os.path.join(self.cache_dir, f"{key}.pkl")
        with open(cache_path, 'wb') as f:
            pickle.dump(data, f)
    
    def load_cached(self, key, max_age_hours=24):
        """Load cached data if not expired"""
        cache_path = os.path.join(self.cache_dir, f"{key}.pkl")
        
        if not os.path.exists(cache_path):
            return None
        
        # Check age
        age = datetime.now().timestamp() - os.path.getmtime(cache_path)
        if age > max_age_hours * 3600:
            return None
        
        with open(cache_path, 'rb') as f:
            return pickle.load(f)
    
    def get_data(self, symbol, timeframe='M5', count=1000, use_cache=True):
        """
        Get data with caching
        Tries: Cache -> DB -> MT5
        """
        cache_key = f"{symbol}_{timeframe}_{count}"
        
        # Try cache
        if use_cache:
            cached = self.load_cached(cache_key)
            if cached is not None:
                return cached
        
        # Try database
        data = self.load_from_db(symbol, timeframe)
        if data is not None and len(data) >= count:
            data = data[-count:]
            if use_cache:
                self.cache_data(cache_key, data)
            return data
        
        # Fetch from MT5
        data = self.fetch_from_mt5(symbol, timeframe, count)
        if data is not None:
            self.save_to_db(symbol, timeframe, data)
            if use_cache:
                self.cache_data(cache_key, data)
            return data
        
        return None
    
    # ============================================================
    # FEATURE ENGINEERING
    # ============================================================
    
    def engineer_features(self, rates, include_indicators=True):
        """
        Create ML features from OHLCV data
        Returns: Feature matrix
        """
        rates = np.asarray(rates, dtype=np.float64)
        
        if len(rates) < 50:
            return None
        
        opens = rates[:, 1]
        highs = rates[:, 2]
        lows = rates[:, 3]
        closes = rates[:, 4]
        volumes = rates[:, 5]
        
        features = []
        
        for i in range(50, len(rates)):
            feat = []
            
            # Price window
            window_c = closes[i-50:i]
            window_h = highs[i-50:i]
            window_l = lows[i-50:i]
            window_v = volumes[i-50:i]
            
            # Returns
            returns = np.diff(window_c) / window_c[:-1]
            feat.extend(returns[-5:])  # Last 5 returns
            
            # Moving averages
            feat.append(np.mean(window_c[-10:]))  # SMA10
            feat.append(np.mean(window_c[-20:]))  # SMA20
            feat.append(np.mean(window_c[-50:]))  # SMA50
            
            # Price vs MAs
            feat.append((closes[i] - np.mean(window_c[-10:])) / closes[i])
            feat.append((closes[i] - np.mean(window_c[-20:])) / closes[i])
            feat.append((closes[i] - np.mean(window_c[-50:])) / closes[i])
            
            # Volatility
            feat.append(np.std(returns[-20:]))
            
            # Range
            feat.append((window_h[-1] - window_l[-1]) / closes[i])
            feat.append(np.mean(window_h[-20:] - window_l[-20:]) / closes[i])
            
            # Volume
            feat.append(volumes[i] / np.mean(window_v[-20:]) if np.mean(window_v[-20:]) > 0 else 1)
            
            # Price position within candle
            feat.append((closes[i] - lows[i]) / (highs[i] - lows[i]) if highs[i] != lows[i] else 0.5)
            
            # Momentum
            feat.append((closes[i] - closes[i-10]) / closes[i-10])
            feat.append((closes[i] - closes[i-20]) / closes[i-20])
            
            features.append(feat)
        
        return np.array(features)
    
    # ============================================================
    # TRADE RECORDING
    # ============================================================
    
    def record_trade(self, symbol, direction, entry_price, exit_price, 
                     volume, pnl, strategy='unknown', metadata=None):
        """Record a trade to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO trades 
            (timestamp, symbol, direction, entry_price, exit_price, volume, pnl, strategy, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            int(datetime.now().timestamp()),
            symbol,
            direction,
            entry_price,
            exit_price,
            volume,
            pnl,
            strategy,
            json.dumps(metadata) if metadata else None
        ))
        
        conn.commit()
        conn.close()
    
    def get_trade_history(self, symbol=None, strategy=None, limit=100):
        """Get trade history"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        query = 'SELECT * FROM trades WHERE 1=1'
        params = []
        
        if symbol:
            query += ' AND symbol = ?'
            params.append(symbol)
        if strategy:
            query += ' AND strategy = ?'
            params.append(strategy)
        
        query += ' ORDER BY timestamp DESC LIMIT ?'
        params.append(limit)
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()
        
        return rows
    
    # ============================================================
    # EXPORT/IMPORT
    # ============================================================
    
    def export_to_csv(self, symbol, timeframe, filepath):
        """Export data to CSV"""
        data = self.load_from_db(symbol, timeframe)
        if data is None:
            return False
        
        import pandas as pd
        df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
        df.to_csv(filepath, index=False)
        return True
    
    def import_from_csv(self, filepath, symbol, timeframe):
        """Import data from CSV"""
        import pandas as pd
        df = pd.read_csv(filepath)
        
        if 'datetime' in df.columns:
            df['timestamp'] = pd.to_datetime(df['datetime']).astype(int) // 10**9
        
        data = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].values
        self.save_to_db(symbol, timeframe, data)
        return True


# ============================================================
# DATA VALIDATION
# ============================================================

def validate_data(rates):
    """Validate OHLCV data for issues"""
    issues = []
    
    if rates is None or len(rates) == 0:
        return ['No data provided']
    
    # Check for NaN
    if np.any(np.isnan(rates)):
        issues.append('Contains NaN values')
    
    # Check for negative prices
    if np.any(rates[:, 1:5] <= 0):
        issues.append('Contains zero or negative prices')
    
    # Check OHLC logic
    for i in range(len(rates)):
        o, h, l, c = rates[i, 1:5]
        if h < l:
            issues.append(f'Candle {i}: High < Low')
        if h < max(o, c):
            issues.append(f'Candle {i}: High < max(Open, Close)')
        if l > min(o, c):
            issues.append(f'Candle {i}: Low > min(Open, Close)')
    
    # Check for gaps
    timestamps = rates[:, 0]
    diffs = np.diff(timestamps)
    if len(diffs) > 0:
        median_diff = np.median(diffs)
        large_gaps = diffs[diffs > median_diff * 3]
        if len(large_gaps) > 0:
            issues.append(f'Found {len(large_gaps)} large time gaps')
    
    return issues if issues else ['Data is valid']


# ============================================================
# TEST
# ============================================================

def test_pipeline():
    """Test data pipeline"""
    print("=" * 60)
    print("DATA PIPELINE TEST")
    print("=" * 60)
    print()
    
    pipeline = DataPipeline()
    
    # Generate test data
    np.random.seed(42)
    n = 1000
    closes = np.random.randn(n).cumsum() + 65000
    highs = closes + np.abs(np.random.randn(n)) * 100
    lows = closes - np.abs(np.random.randn(n)) * 100
    opens = (highs + lows) / 2
    volumes = np.abs(np.random.randn(n)) * 1000 + 500
    test_data = np.column_stack([np.arange(n) + 1609459200, opens, highs, lows, closes, volumes])
    
    # Save to DB
    print("Saving test data to database...")
    pipeline.save_to_db('BTCUSD', 'M5', test_data)
    
    # Load from DB
    print("Loading from database...")
    loaded = pipeline.load_from_db('BTCUSD', 'M5')
    print(f"Loaded {len(loaded)} records")
    
    # Feature engineering
    print("\nEngineering features...")
    features = pipeline.engineer_features(loaded)
    print(f"Generated {features.shape[1]} features for {features.shape[0]} samples")
    
    # Validation
    print("\nValidating data...")
    issues = validate_data(loaded)
    for issue in issues:
        print(f"  - {issue}")
    
    # Record trade
    print("\nRecording test trade...")
    pipeline.record_trade('BTCUSD', 'LONG', 65000, 66000, 0.1, 100, 'test_strategy')
    
    # Get trade history
    trades = pipeline.get_trade_history()
    print(f"Trade history: {len(trades)} records")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    test_pipeline()
