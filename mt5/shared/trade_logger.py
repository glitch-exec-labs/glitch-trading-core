"""
Trade Decision Logger - Captures WHY trades were taken
For Phase 2 analysis and strategy improvement
"""
import json
import os
from datetime import datetime
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

class TradeDecisionLogger:
    """Logs detailed trade decisions for analysis"""
    
    def __init__(self, bot_name: str, log_dir: str = None):
        self.bot_name = bot_name
        self.log_dir = log_dir or os.path.expanduser('~/.openclaw/workspace/trade_logs')
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Daily log file
        self.current_date = datetime.now().strftime('%Y-%m-%d')
        self.log_file = os.path.join(self.log_dir, f'{bot_name}_trades_{self.current_date}.jsonl')
        
        # Statistics file
        self.stats_file = os.path.join(self.log_dir, f'{bot_name}_stats.json')
        
        logger.info(f"[TRADE_LOG] Initialized for {bot_name}")
        logger.info(f"[TRADE_LOG] Log file: {self.log_file}")
    
    def log_entry(self, symbol: str, direction: str, price: float, size: float,
                  decision_factors: Dict, market_context: Dict) -> str:
        """Log entry decision with full context"""
        
        trade_id = f"{symbol}_{datetime.now().strftime('%H%M%S')}_{direction}"
        
        entry_data = {
            'trade_id': trade_id,
            'timestamp': datetime.now().isoformat(),
            'bot': self.bot_name,
            'symbol': symbol,
            'direction': direction,
            'entry_price': price,
            'size': size,
            'entry_usd_value': price * size,
            'decision_factors': decision_factors,
            'market_context': market_context,
            'status': 'OPEN'
        }
        
        self._append_log(entry_data)
        logger.info(f"[TRADE_LOG] Entry logged: {trade_id} @ ${price:.2f}")
        
        return trade_id
    
    def log_exit(self, trade_id: str, exit_price: float, exit_reason: str,
                 pnl: float, pnl_pct: float, holding_time_minutes: float):
        """Log exit with outcome"""
        
        exit_data = {
            'trade_id': trade_id,
            'exit_timestamp': datetime.now().isoformat(),
            'exit_price': exit_price,
            'exit_reason': exit_reason,
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'holding_time_minutes': holding_time_minutes,
            'status': 'CLOSED'
        }
        
        self._update_trade(trade_id, exit_data)
        logger.info(f"[TRADE_LOG] Exit logged: {trade_id} | P&L: ${pnl:.2f} ({pnl_pct:.2f}%) | Reason: {exit_reason}")
    
    def log_decision_rejected(self, symbol: str, reason: str, decision_factors: Dict):
        """Log why a trade was NOT taken"""
        
        rejected_data = {
            'timestamp': datetime.now().isoformat(),
            'bot': self.bot_name,
            'symbol': symbol,
            'decision': 'REJECTED',
            'rejection_reason': reason,
            'decision_factors': decision_factors
        }
        
        self._append_log(rejected_data)
        logger.debug(f"[TRADE_LOG] Rejected: {symbol} - {reason}")
    
    def log_market_snapshot(self, symbol: str, snapshot: Dict):
        """Log market conditions for analysis"""
        
        snapshot_data = {
            'timestamp': datetime.now().isoformat(),
            'type': 'MARKET_SNAPSHOT',
            'symbol': symbol,
            'data': snapshot
        }
        
        self._append_log(snapshot_data)
    
    def _append_log(self, data: Dict):
        """Append to daily log file"""
        try:
            # Check if date changed
            current_date = datetime.now().strftime('%Y-%m-%d')
            if current_date != self.current_date:
                self.current_date = current_date
                self.log_file = os.path.join(self.log_dir, f'{self.bot_name}_trades_{self.current_date}.jsonl')
            
            with open(self.log_file, 'a') as f:
                f.write(json.dumps(data) + '\n')
        except Exception as e:
            logger.error(f"[TRADE_LOG] Failed to write: {e}")
    
    def _update_trade(self, trade_id: str, exit_data: Dict):
        """Update existing trade with exit data"""
        try:
            # Read all trades
            trades = []
            if os.path.exists(self.log_file):
                with open(self.log_file, 'r') as f:
                    for line in f:
                        if line.strip():
                            trades.append(json.loads(line))
            
            # Find and update trade
            for trade in trades:
                if trade.get('trade_id') == trade_id:
                    trade.update(exit_data)
                    break
            
            # Write back
            with open(self.log_file, 'w') as f:
                for trade in trades:
                    f.write(json.dumps(trade) + '\n')
                    
        except Exception as e:
            logger.error(f"[TRADE_LOG] Failed to update trade: {e}")
    
    def get_stats(self) -> Dict:
        """Get trading statistics"""
        try:
            trades = self._load_all_trades()
            
            closed_trades = [t for t in trades if t.get('status') == 'CLOSED']
            winning_trades = [t for t in closed_trades if t.get('pnl', 0) > 0]
            losing_trades = [t for t in closed_trades if t.get('pnl', 0) <= 0]
            
            total_pnl = sum(t.get('pnl', 0) for t in closed_trades)
            
            return {
                'bot': self.bot_name,
                'total_trades': len(closed_trades),
                'winning_trades': len(winning_trades),
                'losing_trades': len(losing_trades),
                'win_rate': len(winning_trades) / len(closed_trades) * 100 if closed_trades else 0,
                'total_pnl': total_pnl,
                'avg_win': sum(t.get('pnl', 0) for t in winning_trades) / len(winning_trades) if winning_trades else 0,
                'avg_loss': sum(t.get('pnl', 0) for t in losing_trades) / len(losing_trades) if losing_trades else 0,
                'profit_factor': abs(sum(t.get('pnl', 0) for t in winning_trades) / sum(t.get('pnl', 0) for t in losing_trades)) if losing_trades and sum(t.get('pnl', 0) for t in losing_trades) != 0 else float('inf')
            }
        except Exception as e:
            logger.error(f"[TRADE_LOG] Failed to get stats: {e}")
            return {}
    
    def _load_all_trades(self) -> List[Dict]:
        """Load all trades from log files"""
        trades = []
        try:
            # Load current day's trades
            if os.path.exists(self.log_file):
                with open(self.log_file, 'r') as f:
                    for line in f:
                        if line.strip():
                            trades.append(json.loads(line))
        except Exception as e:
            logger.error(f"[TRADE_LOG] Failed to load trades: {e}")
        
        return trades
    
    def generate_report(self) -> str:
        """Generate text report of trading activity"""
        stats = self.get_stats()
        trades = self._load_all_trades()
        
        report = []
        report.append("=" * 60)
        report.append(f"TRADE REPORT: {self.bot_name}")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("=" * 60)
        report.append("")
        
        if stats:
            report.append("STATISTICS:")
            report.append(f"  Total Trades: {stats['total_trades']}")
            report.append(f"  Win Rate: {stats['win_rate']:.1f}%")
            report.append(f"  Total P&L: ${stats['total_pnl']:.2f}")
            report.append(f"  Avg Win: ${stats['avg_win']:.2f}")
            report.append(f"  Avg Loss: ${stats['avg_loss']:.2f}")
            report.append(f"  Profit Factor: {stats['profit_factor']:.2f}")
            report.append("")
        
        # Recent trades
        recent_trades = [t for t in trades if t.get('status') == 'CLOSED'][-10:]
        if recent_trades:
            report.append("RECENT TRADES:")
            for t in recent_trades:
                symbol = t.get('symbol', 'N/A')
                direction = t.get('direction', 'N/A')
                entry = t.get('entry_price', 0)
                exit_p = t.get('exit_price', 0)
                pnl = t.get('pnl', 0)
                reason = t.get('exit_reason', 'N/A')
                report.append(f"  {symbol} {direction}: ${entry:.2f} -> ${exit_p:.2f} | P&L: ${pnl:.2f} | {reason}")
        
        return '\n'.join(report)

# Global instance (will be initialized in each bot)
trade_logger = None

def init_logger(bot_name: str) -> TradeDecisionLogger:
    """Initialize global trade logger"""
    global trade_logger
    trade_logger = TradeDecisionLogger(bot_name)
    return trade_logger
