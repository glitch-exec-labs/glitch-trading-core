"""
BTC & XAU ML Engine - XGBoost Price Direction Predictor
Multi-asset ML engine supporting BTCUSD and XAUUSD
"""
import numpy as np
import pandas as pd
import pickle
import json
import warnings
from pathlib import Path
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

# Try to import XGBoost
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    logger.warning("XGBoost not installed. Install with: pip install xgboost")


def _set_model_device(model, device: str) -> bool:
    """Best-effort device switch for loaded XGBoost sklearn/booster objects."""
    if model is None:
        return False
    try:
        if hasattr(model, 'set_params'):
            model.set_params(device=device)
        booster = model.get_booster() if hasattr(model, 'get_booster') else model
        if hasattr(booster, 'set_param'):
            booster.set_param({'device': device})
        return True
    except Exception:
        return False


SUPER_XAU_MODEL_NAME = 'xau_xgb_super.json'
SUPER_XAU_FEATURES_NAME = 'xau_features.json'
SUPER_BTC_MODEL_NAME = 'btc_xgb_super.json'
SUPER_BTC_FEATURES_NAME = 'btc_features.json'
SUPER_XAU_DEFAULT_FEATURES = [
    'timeframe_minutes',
    'open',
    'high',
    'low',
    'close',
    'ema_3',
    'ema_8',
    'ema_20',
    'ema_50',
    'ema_cross_3_8',
    'ema_cross_8_20',
    'price_dist_ema_3_atr',
    'price_dist_ema_8_atr',
    'price_dist_ema_20_atr',
    'price_dist_ema_50_atr',
    'rsi_14',
    'rsi_roc_3',
    'adx_14',
    'plus_di_14',
    'minus_di_14',
    'macd',
    'macd_signal',
    'macd_hist',
    'atr_14',
    'atr_roc',
    'bb_width',
    'bb_pos',
    'body_ratio',
    'upper_wick_ratio',
    'lower_wick_ratio',
    'volume_ratio_20',
    'volume_roc_3',
    'h1_ema20',
    'h1_rsi14',
    'h1_adx14',
    'h1_trend_dir',
    'consecutive_bullish',
    'consecutive_bearish',
    'dist_session_high_atr',
    'dist_session_low_atr',
    'higher_highs_10',
    'lower_lows_10',
]


def _resolve_existing_path(*candidates: Path) -> Optional[Path]:
    for candidate in candidates:
        if candidate and candidate.exists():
            return candidate
    return None


def _ema_series(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def _rsi_series(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100.0 - (100.0 / (1.0 + rs))
    return out.fillna(50.0)


def _atr_series(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def _adx_components(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> Tuple[pd.Series, pd.Series, pd.Series]:
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr_vals = tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    plus_di = 100 * pd.Series(plus_dm, index=high.index).ewm(alpha=1 / period, adjust=False, min_periods=period).mean() / atr_vals
    minus_di = 100 * pd.Series(minus_dm, index=high.index).ewm(alpha=1 / period, adjust=False, min_periods=period).mean() / atr_vals
    dx = ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)) * 100
    adx = dx.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    return adx, plus_di, minus_di


def _macd_components(series: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
    ema12 = _ema_series(series, 12)
    ema26 = _ema_series(series, 26)
    macd_line = ema12 - ema26
    signal_line = _ema_series(macd_line, 9)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def _bollinger_components(series: pd.Series, period: int = 20, std_mult: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
    mid = series.rolling(period, min_periods=period).mean()
    std = series.rolling(period, min_periods=period).std()
    upper = mid + std_mult * std
    lower = mid - std_mult * std
    return upper, mid, lower


def _consecutive_counts(flags: pd.Series) -> pd.Series:
    out = []
    streak = 0
    for flag in flags.fillna(False).astype(bool):
        if flag:
            streak += 1
        else:
            streak = 0
        out.append(streak)
    return pd.Series(out, index=flags.index, dtype=float)


def _rates_to_frame(rates: np.ndarray) -> pd.DataFrame:
    arr = np.asarray(rates)
    if getattr(arr.dtype, 'names', None):
        volume_key = 'volume'
        if volume_key not in arr.dtype.names:
            volume_key = 'tick_volume' if 'tick_volume' in arr.dtype.names else 'real_volume'
        data = {
            'time': arr['time'],
            'open': arr['open'],
            'high': arr['high'],
            'low': arr['low'],
            'close': arr['close'],
            'volume': arr[volume_key] if volume_key in arr.dtype.names else np.zeros(len(arr)),
        }
        df = pd.DataFrame(data)
    else:
        if arr.ndim != 2 or arr.shape[1] < 5:
            raise ValueError("rates must be a 2D array with at least OHLC columns")
        columns = ['time', 'open', 'high', 'low', 'close']
        if arr.shape[1] >= 6:
            columns.append('volume')
            values = arr[:, :6]
        else:
            values = np.column_stack([arr[:, :5], np.zeros(len(arr), dtype=float)])
            columns.append('volume')
        df = pd.DataFrame(values, columns=columns)
    df['time'] = pd.to_numeric(df['time'], errors='coerce')
    for column in ['open', 'high', 'low', 'close', 'volume']:
        df[column] = pd.to_numeric(df[column], errors='coerce')
    df['timestamp_utc'] = pd.to_datetime(df['time'], unit='s', utc=True, errors='coerce')
    return df.dropna(subset=['timestamp_utc', 'open', 'high', 'low', 'close']).copy()


def _infer_timeframe_minutes(df: pd.DataFrame) -> int:
    diffs = df['time'].diff().dropna()
    diffs = diffs[diffs > 0]
    if diffs.empty:
        return 5
    median_seconds = float(diffs.median())
    approx_minutes = max(1, int(round(median_seconds / 60.0)))
    valid = np.array([1, 5, 15, 30, 60, 240, 1440], dtype=int)
    return int(valid[np.argmin(np.abs(valid - approx_minutes))])


def _add_h1_features(df: pd.DataFrame, timeframe_minutes: int) -> pd.DataFrame:
    df = df.sort_values('timestamp_utc').copy()
    if timeframe_minutes < 60:
        base = df.set_index('timestamp_utc')
        agg = base.resample('1h').agg(
            {
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum',
            }
        ).dropna(subset=['open', 'high', 'low', 'close'])
        if agg.empty:
            df['h1_ema20'] = np.nan
            df['h1_rsi14'] = np.nan
            df['h1_adx14'] = np.nan
            df['h1_trend_dir'] = np.nan
            return df
        agg['h1_ema20'] = _ema_series(agg['close'], 20)
        agg['h1_rsi14'] = _rsi_series(agg['close'], 14)
        agg['h1_adx14'], _, _ = _adx_components(agg['high'], agg['low'], agg['close'], 14)
        agg['h1_trend_dir'] = np.sign(agg['close'] - agg['h1_ema20']).replace(0, 1)
        return pd.merge_asof(
            df.sort_values('timestamp_utc'),
            agg[['h1_ema20', 'h1_rsi14', 'h1_adx14', 'h1_trend_dir']].sort_index().reset_index(),
            on='timestamp_utc',
            direction='backward',
        )

    df['h1_ema20'] = _ema_series(df['close'], 20)
    df['h1_rsi14'] = _rsi_series(df['close'], 14)
    df['h1_adx14'], _, _ = _adx_components(df['high'], df['low'], df['close'], 14)
    df['h1_trend_dir'] = np.sign(df['close'] - df['h1_ema20']).replace(0, 1)
    return df


def _build_super_xau_features(rates: np.ndarray, feature_columns: Optional[list] = None) -> pd.DataFrame:
    df = _rates_to_frame(rates)
    timeframe_minutes = _infer_timeframe_minutes(df)
    df = df.sort_values('timestamp_utc').copy()
    df['timeframe_minutes'] = timeframe_minutes

    close = df['close']
    high = df['high']
    low = df['low']
    open_ = df['open']
    volume = df['volume']

    df['ema_3'] = _ema_series(close, 3)
    df['ema_8'] = _ema_series(close, 8)
    df['ema_20'] = _ema_series(close, 20)
    df['ema_50'] = _ema_series(close, 50)
    df['ema_cross_3_8'] = (
        ((df['ema_3'] > df['ema_8']) & (df['ema_3'].shift(1) <= df['ema_8'].shift(1))).astype(float)
        - ((df['ema_3'] < df['ema_8']) & (df['ema_3'].shift(1) >= df['ema_8'].shift(1))).astype(float)
    )
    df['ema_cross_8_20'] = (
        ((df['ema_8'] > df['ema_20']) & (df['ema_8'].shift(1) <= df['ema_20'].shift(1))).astype(float)
        - ((df['ema_8'] < df['ema_20']) & (df['ema_8'].shift(1) >= df['ema_20'].shift(1))).astype(float)
    )

    df['rsi_14'] = _rsi_series(close, 14)
    df['rsi_roc_3'] = df['rsi_14'] - df['rsi_14'].shift(3)
    df['atr_14'] = _atr_series(high, low, close, 14)
    df['atr_roc'] = df['atr_14'].pct_change(3)
    df['adx_14'], df['plus_di_14'], df['minus_di_14'] = _adx_components(high, low, close, 14)
    df['macd'], df['macd_signal'], df['macd_hist'] = _macd_components(close)
    df['bb_upper'], df['bb_mid'], df['bb_lower'] = _bollinger_components(close, 20, 2.0)
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_mid'].replace(0, np.nan)
    df['bb_pos'] = (close - df['bb_lower']) / (df['bb_upper'] - df['bb_lower']).replace(0, np.nan)

    candle_range = (high - low).replace(0, np.nan)
    body = (close - open_).abs()
    df['body_ratio'] = body / candle_range
    df['upper_wick_ratio'] = (high - np.maximum(open_, close)) / candle_range
    df['lower_wick_ratio'] = (np.minimum(open_, close) - low) / candle_range

    for period in (3, 8, 20, 50):
        df[f'price_dist_ema_{period}_atr'] = (close - df[f'ema_{period}']) / df['atr_14'].replace(0, np.nan)

    if volume.notna().any():
        df['volume_avg_20'] = volume.rolling(20, min_periods=5).mean()
        df['volume_ratio_20'] = volume / df['volume_avg_20'].replace(0, np.nan)
        df['volume_roc_3'] = volume.pct_change(3)
    else:
        df['volume_ratio_20'] = np.nan
        df['volume_roc_3'] = np.nan

    df['consecutive_bullish'] = _consecutive_counts(close > open_)
    df['consecutive_bearish'] = _consecutive_counts(close < open_)

    session_key = df['timestamp_utc'].dt.floor('D')
    df['session_high'] = high.groupby(session_key).cummax()
    df['session_low'] = low.groupby(session_key).cummin()
    df['dist_session_high_atr'] = (df['session_high'] - close) / df['atr_14'].replace(0, np.nan)
    df['dist_session_low_atr'] = (close - df['session_low']) / df['atr_14'].replace(0, np.nan)
    df['higher_highs_10'] = (high.diff() > 0).rolling(10, min_periods=1).sum()
    df['lower_lows_10'] = (low.diff() < 0).rolling(10, min_periods=1).sum()

    df = _add_h1_features(df, timeframe_minutes)

    wanted = feature_columns or SUPER_XAU_DEFAULT_FEATURES
    for col in wanted:
        if col not in df.columns:
            df[col] = 0.0
    df[wanted] = df[wanted].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return df[wanted]


class FeatureEngineer:
    """Engineer features from OHLCV data for ML model"""
    
    @staticmethod
    def calculate_features(rates: np.ndarray) -> pd.DataFrame:
        """
        Calculate comprehensive features from rate data
        rates: np.array of shape (N, 6) [time, open, high, low, close, volume]
        Returns: DataFrame with features
        """
        df = pd.DataFrame(rates, columns=['time', 'open', 'high', 'low', 'close', 'volume'])
        
        # Basic returns
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # Price features
        df['body'] = df['close'] - df['open']
        df['body_pct'] = df['body'] / df['open']
        df['upper_shadow'] = df['high'] - df[['open', 'close']].max(axis=1)
        df['lower_shadow'] = df[['open', 'close']].min(axis=1) - df['low']
        df['range'] = df['high'] - df['low']
        df['range_pct'] = df['range'] / df['open']
        
        # Trend features - multiple timeframes
        for window in [5, 10, 20, 50]:
            df[f'sma_{window}'] = df['close'].rolling(window=window).mean()
            df[f'ema_{window}'] = df['close'].ewm(span=window, adjust=False).mean()
            df[f'close_vs_sma_{window}'] = (df['close'] - df[f'sma_{window}']) / df[f'sma_{window}']
        
        # Volatility features
        for window in [5, 10, 20]:
            df[f'volatility_{window}'] = df['returns'].rolling(window=window).std()
            df[f'atr_{window}'] = df['range'].rolling(window=window).mean()
        
        # Volume features
        df['volume_sma_10'] = df['volume'].rolling(window=10).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma_10']
        df['volume_trend'] = df['volume'].rolling(window=5).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0)
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi_14'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema_12 = df['close'].ewm(span=12, adjust=False).mean()
        ema_26 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        bb_std = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Momentum
        for window in [3, 5, 10]:
            df[f'momentum_{window}'] = df['close'].diff(window)
            df[f'momentum_{window}_pct'] = df[f'momentum_{window}'] / df['close'].shift(window)
        
        # Lagged returns (for autocorrelation)
        for lag in [1, 2, 3, 5]:
            df[f'returns_lag_{lag}'] = df['returns'].shift(lag)
        
        # Time features (if timestamp available)
        if 'time' in df.columns:
            df['hour'] = pd.to_datetime(df['time'], unit='s').dt.hour
            df['day_of_week'] = pd.to_datetime(df['time'], unit='s').dt.dayofweek
        
        return df
    
    @staticmethod
    def create_target(df: pd.DataFrame, threshold: float = 0.001) -> pd.Series:
        """
        Create target variable: next bar direction
        threshold: minimum move to be considered up/down (0.1% default)
        Returns: 0=down, 1=flat, 2=up
        """
        future_return = df['close'].shift(-1) / df['close'] - 1
        
        target = pd.Series(1, index=df.index)  # default flat
        target[future_return > threshold] = 2  # up
        target[future_return < -threshold] = 0  # down
        
        return target


class BTCMLModel:
    """XGBoost model for BTC M5 direction prediction"""
    
    def __init__(self, model_path: str = None, feature_config: dict = None):
        self.model = None
        self.feature_scaler = None
        self.feature_columns = None
        self.is_trained = False
        self.model_path = model_path or 'models/btc_xgb_model.pkl'
        self.feature_config = feature_config or {}
        self.uses_super_model = False
        self.super_model_path = None
        self.super_features_path = None
        
        # Try to load existing model
        self._load_model()
    
    def _load_model(self):
        if XGB_AVAILABLE:
            super_model_path, features_path = self._candidate_super_paths()
            if super_model_path and features_path:
                try:
                    self.model = xgb.Booster()
                    self.model.load_model(str(super_model_path))
                    self.feature_columns = json.loads(features_path.read_text(encoding='utf-8'))
                    self.feature_scaler = None
                    self.is_trained = True
                    self.uses_super_model = True
                    self.super_model_path = super_model_path
                    self.super_features_path = features_path
                    _set_model_device(self.model, 'cuda')
                    logger.info(f"Loaded BTC super model from {super_model_path}")
                    return
                except Exception as exc:
                    logger.warning(f"Could not load BTC super model: {exc}")

        self.uses_super_model = False
        try:
            legacy_path = _resolve_existing_path(*self._candidate_legacy_paths())
            if legacy_path is not None:
                self.model_path = str(legacy_path)
                with open(legacy_path, 'rb') as f:
                    data = pickle.load(f)
                    self.model = data['model']
                    self.feature_columns = data['feature_columns']
                    self.feature_scaler = data.get('scaler')
                    self.is_trained = True
                logger.info(f"Loaded BTC ML model from {self.model_path}")
        except Exception as e:
            logger.warning(f"Could not load model: {e}")
    
    def save_model(self):
        """Save model to disk"""
        try:
            Path(self.model_path).parent.mkdir(parents=True, exist_ok=True)
            data = {
                'model': self.model,
                'feature_columns': self.feature_columns,
                'scaler': self.feature_scaler
            }
            with open(self.model_path, 'wb') as f:
                pickle.dump(data, f)
            logger.info(f"Saved BTC ML model to {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
    
    def _candidate_super_paths(self) -> Tuple[Optional[Path], Optional[Path]]:
        module_dir = Path(__file__).resolve().parent
        cwd = Path.cwd()
        model_path = _resolve_existing_path(
            module_dir / 'models' / SUPER_BTC_MODEL_NAME,
            cwd / 'pro_modules' / 'models' / SUPER_BTC_MODEL_NAME,
            cwd / 'models' / SUPER_BTC_MODEL_NAME,
        )
        features_path = _resolve_existing_path(
            module_dir / 'models' / SUPER_BTC_FEATURES_NAME,
            cwd / 'pro_modules' / 'models' / SUPER_BTC_FEATURES_NAME,
            cwd / 'models' / SUPER_BTC_FEATURES_NAME,
        )
        return model_path, features_path

    def _candidate_legacy_paths(self) -> Tuple[Path, ...]:
        raw = Path(self.model_path)
        module_dir = Path(__file__).resolve().parent
        cwd = Path.cwd()
        return (
            raw,
            module_dir / raw,
            cwd / raw,
            module_dir / 'models' / raw.name,
            cwd / 'models' / raw.name,
        )

    def prepare_features(self, rates: np.ndarray) -> np.ndarray:
        if self.uses_super_model:
            features = _build_super_xau_features(rates, self.feature_columns)
            return features[self.feature_columns].to_numpy(dtype=np.float32)
        return self._prepare_legacy_features(rates)

    def _prepare_legacy_features(self, rates: np.ndarray) -> np.ndarray:
        """
        Prepare features from raw rates
        rates: np.array of shape (N, 6) [time, open, high, low, close, volume]
        Returns: Feature array ready for prediction
        """
        df = FeatureEngineer.calculate_features(rates)
        
        # Select feature columns (exclude raw price/volume and target)
        exclude = ['time', 'open', 'high', 'low', 'close', 'volume']
        feature_cols = [c for c in df.columns if c not in exclude]
        
        # Fill NaN values
        df[feature_cols] = df[feature_cols].fillna(0)
        
        # Store feature columns for consistency
        if self.feature_columns is None:
            self.feature_columns = feature_cols
        
        # Ensure same columns
        for col in self.feature_columns:
            if col not in df.columns:
                df[col] = 0
        
        return df[self.feature_columns].values
    
    def train(self, rates: np.ndarray, threshold: float = 0.001, test_size: float = 0.2):
        """
        Train the model on historical data
        rates: np.array of shape (N, 6)
        threshold: classification threshold for up/down
        """
        if not XGB_AVAILABLE:
            logger.error("XGBoost not available. Cannot train.")
            return False
        
        logger.info("Preparing features...")
        df = FeatureEngineer.calculate_features(rates)
        df['target'] = FeatureEngineer.create_target(df, threshold)
        
        # Prepare features
        X = self.prepare_features(rates)
        y = df['target'].values
        
        # Remove rows with NaN targets
        valid_idx = ~np.isnan(y)
        X = X[valid_idx]
        y = y[valid_idx].astype(int)
        
        if len(X) < 500:
            logger.error(f"Insufficient data: {len(X)} samples (need 500+)")
            return False
        
        # Train/test split (time series - use last portion for test)
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        logger.info(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
        logger.info(f"Class distribution - Down: {(y==0).sum()}, Flat: {(y==1).sum()}, Up: {(y==2).sum()}")
        
        # Train XGBoost
        logger.info("Training XGBoost model...")
        self.model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='multi:softprob',
            num_class=3,
            eval_metric='mlogloss',
            use_label_encoder=False,
            n_jobs=4  # Use 4 cores on c5a.xlarge
        )
        
        # XGBoost 2.x+ uses callbacks instead of early_stopping_rounds
        try:
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_test, y_test)],
                verbose=False
            )
        except TypeError:
            # Fallback for older XGBoost versions
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_test, y_test)],
                early_stopping_rounds=20,
                verbose=False
            )
        
        self.is_trained = True
        
        # Evaluate
        train_pred = self.model.predict(X_train)
        test_pred = self.model.predict(X_test)
        
        train_acc = (train_pred == y_train).mean()
        test_acc = (test_pred == y_test).mean()
        
        logger.info(f"Training accuracy: {train_acc:.2%}")
        logger.info(f"Test accuracy: {test_acc:.2%}")
        
        # Feature importance
        importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        logger.info("Top 10 features:")
        for _, row in importance.head(10).iterrows():
            logger.info(f"  {row['feature']}: {row['importance']:.4f}")
        
        # Save model
        self.save_model()
        
        return True
    
    def predict(self, rates: np.ndarray) -> Dict:
        """
        Predict direction of next bar
        rates: np.array of shape (N, 6)
        Returns: dict with action, confidence, prediction
        """
        if not self.is_trained or self.model is None:
            return {
                'action': 'HOLD',
                'confidence': 0.0,
                'prediction': 'neutral',
                'probabilities': [0.33, 0.34, 0.33]
        }
        
        try:
            if self.uses_super_model:
                X = self.prepare_features(rates)
                if len(X) == 0:
                    raise ValueError("No features generated")

                dmatrix = xgb.DMatrix(X[-1:], feature_names=self.feature_columns)
                try:
                    probs = np.asarray(self.model.predict(dmatrix)[0], dtype=float)
                except Exception as pred_err:
                    err_text = str(pred_err).lower()
                    if 'gpu' in err_text or 'cuda' in err_text:
                        logger.warning("BTC super model GPU inference failed; retrying on CPU")
                        if _set_model_device(self.model, 'cpu'):
                            probs = np.asarray(self.model.predict(dmatrix)[0], dtype=float)
                        else:
                            raise pred_err
                    else:
                        raise pred_err

                pred_class = int(np.argmax(probs))
                confidence = float(probs[pred_class])
                class_map = {0: 'down', 1: 'flat', 2: 'up'}
                action_map = {0: 'SELL', 1: 'HOLD', 2: 'BUY'}
                return {
                    'action': action_map[pred_class],
                    'confidence': confidence,
                    'prediction': class_map[pred_class],
                    'probabilities': probs.tolist(),
                }

            # Prepare features
            X = self.prepare_features(rates)
            
            if len(X) == 0:
                raise ValueError("No features generated")
            
            # Get last row for prediction
            X_last = X[-1:]
            
            # Predict
            try:
                probs = self.model.predict_proba(X_last)[0]
            except Exception as pred_err:
                if 'gpu_id' in str(pred_err):
                    logger.warning("GPU predictor config is stale; retrying BTC model on CPU")
                    if _set_model_device(self.model, 'cpu'):
                        probs = self.model.predict_proba(X_last)[0]
                    else:
                        raise pred_err
                else:
                    raise pred_err
            pred_class = np.argmax(probs)
            confidence = probs[pred_class]
            
            # Map to output format
            class_map = {0: 'down', 1: 'flat', 2: 'up'}
            action_map = {0: 'SELL', 1: 'HOLD', 2: 'BUY'}
            
            return {
                'action': action_map[pred_class],
                'confidence': float(confidence),
                'prediction': class_map[pred_class],
                'probabilities': probs.tolist()
            }
            
        except Exception as e:
            logger.error(f"Prediction error in {self.__class__.__name__}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {
                'action': 'HOLD',
                'confidence': 0.0,
                'prediction': 'neutral',
                'probabilities': [0.33, 0.34, 0.33]
            }


class XAUMLModel(BTCMLModel):
    """XAUUSD (Gold) specific ML model with session features"""
    
    def __init__(self, model_path: str = 'models/xau_xgb_model.pkl'):
        self.is_xau = True
        self.uses_super_model = False
        self.super_model_path = None
        self.super_features_path = None
        super().__init__(model_path=model_path)
    
    def prepare_features(self, rates: np.ndarray) -> np.ndarray:
        """Prepare XAU features for either the new super model or the legacy pickle model."""
        if self.uses_super_model:
            features = _build_super_xau_features(rates, self.feature_columns)
            return features[self.feature_columns].to_numpy(dtype=np.float32)
        return self._prepare_legacy_features(rates)

    def _candidate_super_paths(self) -> Tuple[Optional[Path], Optional[Path]]:
        module_dir = Path(__file__).resolve().parent
        cwd = Path.cwd()
        model_path = _resolve_existing_path(
            module_dir / 'models' / SUPER_XAU_MODEL_NAME,
            cwd / 'pro_modules' / 'models' / SUPER_XAU_MODEL_NAME,
            cwd / 'models' / SUPER_XAU_MODEL_NAME,
        )
        features_path = _resolve_existing_path(
            module_dir / 'models' / SUPER_XAU_FEATURES_NAME,
            cwd / 'pro_modules' / 'models' / SUPER_XAU_FEATURES_NAME,
            cwd / 'models' / SUPER_XAU_FEATURES_NAME,
        )
        return model_path, features_path

    def _candidate_legacy_paths(self) -> Tuple[Path, ...]:
        raw = Path(self.model_path)
        module_dir = Path(__file__).resolve().parent
        cwd = Path.cwd()
        return (
            raw,
            module_dir / raw,
            cwd / raw,
            module_dir / 'models' / raw.name,
            cwd / 'models' / raw.name,
        )

    def _load_model(self):
        if XGB_AVAILABLE:
            super_model_path, features_path = self._candidate_super_paths()
            if super_model_path and features_path:
                try:
                    self.model = xgb.Booster()
                    self.model.load_model(str(super_model_path))
                    self.feature_columns = json.loads(features_path.read_text(encoding='utf-8'))
                    self.feature_scaler = None
                    self.is_trained = True
                    self.uses_super_model = True
                    self.super_model_path = super_model_path
                    self.super_features_path = features_path
                    _set_model_device(self.model, 'cuda')
                    logger.info(f"Loaded XAU super model from {super_model_path}")
                    return
                except Exception as exc:
                    logger.warning(f"Could not load XAU super model: {exc}")

        self.uses_super_model = False
        legacy_path = _resolve_existing_path(*self._candidate_legacy_paths())
        if legacy_path is not None:
            self.model_path = str(legacy_path)
        super()._load_model()

    def _prepare_legacy_features(self, rates: np.ndarray) -> np.ndarray:
        df = pd.DataFrame(rates, columns=['time', 'open', 'high', 'low', 'close', 'volume'])
        df['datetime'] = pd.to_datetime(df['time'], unit='s')
        df['returns'] = df['close'].pct_change()
        df['range'] = df['high'] - df['low']
        df['range_pct'] = df['range'] / df['open']
        df['body'] = df['close'] - df['open']
        df['body_pct'] = df['body'] / df['open']

        for window in [5, 10, 20, 50]:
            df[f'sma_{window}'] = df['close'].rolling(window=window).mean()
            df[f'close_vs_sma_{window}'] = (df['close'] - df[f'sma_{window}']) / df[f'sma_{window}']

        for window in [5, 10, 20]:
            df[f'volatility_{window}'] = df['returns'].rolling(window=window).std()
            df[f'atr_{window}'] = df['range'].rolling(window=window).mean()

        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi_14'] = 100 - (100 / (1 + rs))

        df['hour'] = df['datetime'].dt.hour
        df['asian_session'] = ((df['hour'] >= 0) & (df['hour'] < 8)).astype(int)
        df['london_session'] = ((df['hour'] >= 8) & (df['hour'] < 16)).astype(int)
        df['ny_session'] = ((df['hour'] >= 13) & (df['hour'] < 21)).astype(int)
        df['london_ny_overlap'] = ((df['hour'] >= 13) & (df['hour'] < 16)).astype(int)

        feature_cols = [c for c in df.columns if c not in ['time', 'open', 'high', 'low', 'close', 'volume', 'datetime']]
        df[feature_cols] = df[feature_cols].fillna(0)

        if self.feature_columns is None:
            self.feature_columns = feature_cols
        else:
            for col in self.feature_columns:
                if col not in df.columns:
                    df[col] = 0

        return df[self.feature_columns].values

    def predict(self, rates: np.ndarray) -> Dict:
        if not self.uses_super_model:
            return super().predict(rates)

        if not self.is_trained or self.model is None:
            return {'action': 'HOLD', 'confidence': 0.0, 'prediction': 'neutral', 'probabilities': [0.33, 0.34, 0.33]}

        try:
            X = self.prepare_features(rates)
            if len(X) == 0:
                raise ValueError("No features generated")

            dmatrix = xgb.DMatrix(X[-1:], feature_names=self.feature_columns)
            try:
                probs = np.asarray(self.model.predict(dmatrix)[0], dtype=float)
            except Exception as pred_err:
                err_text = str(pred_err).lower()
                if 'gpu' in err_text or 'cuda' in err_text:
                    logger.warning("XAU super model GPU inference failed; retrying on CPU")
                    if _set_model_device(self.model, 'cpu'):
                        probs = np.asarray(self.model.predict(dmatrix)[0], dtype=float)
                    else:
                        raise pred_err
                else:
                    raise pred_err

            pred_class = int(np.argmax(probs))
            confidence = float(probs[pred_class])
            class_map = {0: 'down', 1: 'flat', 2: 'up'}
            action_map = {0: 'SELL', 1: 'HOLD', 2: 'BUY'}
            return {
                'action': action_map[pred_class],
                'confidence': confidence,
                'prediction': class_map[pred_class],
                'probabilities': probs.tolist(),
            }
        except Exception as exc:
            logger.error(f"Prediction error in XAU super model: {exc}")
            return {'action': 'HOLD', 'confidence': 0.0, 'prediction': 'neutral', 'probabilities': [0.33, 0.34, 0.33]}


class GenericMLModel:
    """Generic ML model for any symbol - uses same structure as BTCMLModel"""
    
    def __init__(self, symbol: str, model_path: str = None):
        self.symbol = symbol
        self.model = None
        self.feature_scaler = None
        self.feature_columns = None
        self.is_trained = False
        self.model_path = model_path or f'models/{symbol.lower()}_xgb_model.pkl'
        
        # Try to load existing model
        self._load_model()
    
    def _load_model(self):
        """Load model from disk if exists - handles multiple formats"""
        try:
            model_file = Path(self.model_path)
            if model_file.exists():
                with open(model_file, 'rb') as f:
                    data = pickle.load(f)
                    
                    # Handle different save formats
                    if isinstance(data, dict):
                        # New format: {'model': ..., 'feature_cols': ...}
                        self.model = data.get('model') or data.get('xgb_model')
                        self.feature_columns = data.get('feature_columns') or data.get('feature_cols')
                        self.feature_scaler = data.get('scaler')
                    else:
                        # Direct model save (just the XGBClassifier) - old format
                        self.model = data
                        # Use default feature columns based on training
                        self.feature_columns = None  # Will be set on first prediction
                    
                    if self.model is not None:
                        self.is_trained = True
                        logger.info(f"Loaded {self.symbol.upper()} ML model from {self.model_path}")
        except Exception as e:
            logger.warning(f"Could not load {self.symbol} model: {e}")
    
    def _get_expected_n_features(self):
        """Get expected number of features from model"""
        try:
            # XGBoost stores n_features_in_ after training
            return getattr(self.model, 'n_features_in_', None)
        except:
            return None
    
    def _prepare_simple_features(self, rates: np.ndarray, n_features: int) -> np.ndarray:
        """Prepare simple features for models trained with basic indicators"""
        # Extract OHLC
        closes = rates[:, 4]
        highs = rates[:, 2]
        lows = rates[:, 3]
        
        # Calculate basic indicators
        def ema(data, period):
            alpha = 2 / (period + 1)
            ema = np.zeros(len(data))
            ema[0] = data[0]
            for i in range(1, len(data)):
                ema[i] = alpha * data[i] + (1 - alpha) * ema[i-1]
            return ema
        
        def rsi(data, period=14):
            delta = np.diff(data, prepend=data[0])
            gain = np.where(delta > 0, delta, 0)
            loss = np.where(delta < 0, -delta, 0)
            avg_gain = np.convolve(gain, np.ones(period)/period, mode='same')
            avg_loss = np.convolve(loss, np.ones(period)/period, mode='same')
            rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
            return 100 - (100 / (1 + rs))
        
        def atr(high, low, close, period=14):
            tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
            tr[0] = high[0] - low[0]
            return np.convolve(tr, np.ones(period)/period, mode='same')
        
        # Build feature matrix based on expected count
        if n_features == 5:
            # EURUSD model: EMA5, EMA13, EMA21, RSI, ATR
            features = np.column_stack([
                ema(closes, 5),
                ema(closes, 13),
                ema(closes, 21),
                rsi(closes, 14),
                atr(highs, lows, closes, 14)
            ])
        else:
            # Use full feature set then pad/trim
            df = FeatureEngineer.calculate_features(rates)
            feature_cols = [c for c in df.columns if c not in ['time', 'open', 'high', 'low', 'close', 'volume', 'datetime']]
            features = df[feature_cols].fillna(0).values
        
        # Pad or trim to match expected count
        if features.shape[1] < n_features:
            # Pad with zeros
            padding = np.zeros((features.shape[0], n_features - features.shape[1]))
            features = np.hstack([features, padding])
        elif features.shape[1] > n_features:
            # Trim to expected size
            features = features[:, :n_features]
        
        return features
    
    def prepare_features(self, rates: np.ndarray) -> np.ndarray:
        """Prepare features from rate data - handles different model expectations"""
        expected_n = self._get_expected_n_features()
        
        if expected_n is not None and expected_n <= 10:
            # Simple model (like EURUSD with 5 features)
            return self._prepare_simple_features(rates, expected_n)
        else:
            # Full feature model (like WTI/Brent with 54 features)
            df = FeatureEngineer.calculate_features(rates)
            feature_cols = [c for c in df.columns if c not in ['time', 'open', 'high', 'low', 'close', 'volume', 'datetime']]
            df[feature_cols] = df[feature_cols].fillna(0)
            
            if self.feature_columns is None:
                self.feature_columns = feature_cols
            
            features = df[self.feature_columns].values if self.feature_columns else df[feature_cols].values
            
            # Pad or trim to match expected feature count
            if expected_n is not None:
                if features.shape[1] < expected_n:
                    padding = np.zeros((features.shape[0], expected_n - features.shape[1]))
                    features = np.hstack([features, padding])
                elif features.shape[1] > expected_n:
                    features = features[:, :expected_n]
            
            return features
    
    def predict(self, rates: np.ndarray) -> Dict:
        """Predict direction"""
        if not self.is_trained or self.model is None:
            return {'action': 'HOLD', 'confidence': 0.0, 'prediction': 'neutral', 'probabilities': [0.33, 0.34, 0.33]}
        
        try:
            X = self.prepare_features(rates)
            if len(X) == 0:
                raise ValueError("No features generated")
            
            X_last = X[-1:]
            try:
                probs = self.model.predict_proba(X_last)[0]
            except Exception as pred_err:
                if 'gpu_id' in str(pred_err):
                    logger.warning(f"GPU predictor config is stale for {self.symbol}; retrying on CPU")
                    if _set_model_device(self.model, 'cpu'):
                        probs = self.model.predict_proba(X_last)[0]
                    else:
                        raise pred_err
                else:
                    raise pred_err
            pred_class = np.argmax(probs)
            confidence = probs[pred_class]
            
            class_map = {0: 'down', 1: 'flat', 2: 'up'}
            action_map = {0: 'SELL', 1: 'HOLD', 2: 'BUY'}
            
            return {
                'action': action_map[pred_class],
                'confidence': float(confidence),
                'prediction': class_map[pred_class],
                'probabilities': probs.tolist()
            }
        except Exception as e:
            logger.error(f"Prediction error in {self.symbol}: {e}")
            return {'action': 'HOLD', 'confidence': 0.0, 'prediction': 'neutral', 'probabilities': [0.33, 0.34, 0.33]}


class MLEngine:
    """Multi-asset ML Engine - supports BTCUSD, XAUUSD, EURUSD, WTI, Brent"""
    
    def __init__(self):
        self.btc_model = BTCMLModel(model_path='models/btc_xgb_model.pkl')
        self.xau_model = XAUMLModel(model_path='models/xau_xgb_model.pkl')
        
        # NEW: Additional symbol models
        self.eurusd_model = GenericMLModel('eurusd', 'models/eurusd_xgb_model.pkl')
        self.wti_model = GenericMLModel('wti', 'models/wti_xgb_model.pkl')
        self.brent_model = GenericMLModel('brent', 'models/brent_xgb_model.pkl')
        
        self.is_ready = any([
            self.btc_model.is_trained,
            self.xau_model.is_trained,
            self.eurusd_model.is_trained,
            self.wti_model.is_trained,
            self.brent_model.is_trained
        ])
        
        # Log status
        for name, model in [
            ('BTC', self.btc_model),
            ('XAU', self.xau_model),
            ('EURUSD', self.eurusd_model),
            ('WTI', self.wti_model),
            ('Brent', self.brent_model)
        ]:
            if model.is_trained:
                logger.info(f"{name} ML model ready")
            else:
                logger.warning(f"{name} ML model not found")
    
    def _detect_symbol(self, rates: np.ndarray) -> str:
        """Detect symbol from price levels"""
        if len(rates) == 0:
            return 'UNKNOWN'
        
        latest_close = rates[-1, 4]  # Close price
        
        # XAU typically $1500-2500, BTC typically $10000-100000
        # EURUSD typically 0.8-1.3, Oil typically $60-120
        if 1000 < latest_close < 5000:
            return 'XAUUSD'
        elif latest_close > 10000:
            return 'BTCUSD'
        elif 0.5 < latest_close < 2.0:
            return 'EURUSD'
        elif 40 < latest_close < 150:
            return 'OIL'  # WTI or Brent
        else:
            return 'UNKNOWN'
    
    def predict(self, rates: np.ndarray, symbol: str = None, technical_analysis=None, smc_analysis=None) -> Dict:
        """
        Main prediction interface for BEAST
        Auto-detects symbol if not provided
        """
        # Detect symbol from price if not provided
        if symbol is None:
            symbol = self._detect_symbol(rates)
        
        symbol_upper = symbol.upper()
        
        # Route to appropriate model based on symbol
        if 'XAU' in symbol_upper or 'GOLD' in symbol_upper:
            if self.xau_model.is_trained:
                return self.xau_model.predict(rates)
            else:
                logger.debug("XAU model not available, using fallback")
                return {'action': 'HOLD', 'confidence': 0, 'prediction': 'neutral'}
        
        elif 'EUR' in symbol_upper:
            if self.eurusd_model.is_trained:
                return self.eurusd_model.predict(rates)
            else:
                logger.debug("EURUSD model not available, using fallback")
                return {'action': 'HOLD', 'confidence': 0, 'prediction': 'neutral'}
        
        elif 'WTI' in symbol_upper or 'USOUSD' in symbol_upper or 'OIL' in symbol_upper:
            if self.wti_model.is_trained:
                return self.wti_model.predict(rates)
            else:
                logger.debug("WTI model not available, using fallback")
                return {'action': 'HOLD', 'confidence': 0, 'prediction': 'neutral'}
        
        elif 'BRENT' in symbol_upper:
            if self.brent_model.is_trained:
                return self.brent_model.predict(rates)
            else:
                logger.debug("Brent model not available, using fallback")
                return {'action': 'HOLD', 'confidence': 0, 'prediction': 'neutral'}
        
        else:  # Default to BTC for BTC and any other symbol
            if self.btc_model.is_trained:
                return self.btc_model.predict(rates)
            else:
                logger.debug("BTC model not available, using fallback")
                return {'action': 'HOLD', 'confidence': 0, 'prediction': 'neutral'}
    
    def get_price_prediction(self, rates: np.ndarray, symbol: str = None):
        """Get price direction prediction"""
        return self.predict(rates, symbol=symbol)
    
    def get_setup_quality(self, rates: np.ndarray, symbol: str = None):
        """Get setup quality"""
        result = self.predict(rates, symbol=symbol)
        conf = result['confidence']
        if conf > 0.7:
            quality = 'high'
        elif conf > 0.5:
            quality = 'medium'
        else:
            quality = 'low'
        
        return {
            'quality': quality,
            'confidence': conf,
            'probabilities': result['probabilities']
        }


class MLStub:
    """Fallback when XGBoost not available"""
    
    def __init__(self):
        self.is_ready = False
        logger.warning("ML Engine running in stub mode (XGBoost not installed)")
    
    def train(self, rates, **kwargs):
        return False
    
    def predict(self, rates, technical_analysis=None, smc_analysis=None):
        return {'action': 'HOLD', 'confidence': 0, 'prediction': 'neutral'}
    
    def get_price_prediction(self, rates):
        return None
    
    def get_setup_quality(self, rates):
        return None


def create_ml_engine():
    """Factory function - BEAST calls this to get ML engine"""
    if XGB_AVAILABLE:
        return MLEngine()
    else:
        return MLStub()


if __name__ == "__main__":
    # Test the engine
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("BTC ML Engine Test")
    print("=" * 60)
    
    engine = create_ml_engine()
    
    # Test with dummy data
    np.random.seed(42)
    n = 200
    rates = np.column_stack([
        np.arange(n),
        np.random.randn(n).cumsum() * 100 + 40000,
        np.random.randn(n).cumsum() * 100 + 40100,
        np.random.randn(n).cumsum() * 100 + 39900,
        np.random.randn(n).cumsum() * 100 + 40000,
        np.abs(np.random.randn(n)) * 1000
    ])
    
    print("\nTesting prediction (untrained)...")
    result = engine.predict(rates)
    print(f"Result: {result}")
    
    if XGB_AVAILABLE:
        print("\nTraining on dummy data...")
        engine.train(rates, threshold=0.001)
        
        print("\nTesting prediction (trained)...")
        result = engine.predict(rates)
        print(f"Result: {result}")
    else:
        print("\nXGBoost not installed. Install with: pip install xgboost")
