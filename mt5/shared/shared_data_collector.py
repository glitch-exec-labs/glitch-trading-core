"""
Shared ML Data Collector for COBRA, ANACONDA, and VIPER.
Each bot writes to its own separate directory.
Thread-safe. Never affects trading — all writes wrapped in try/except.
"""
import os
import csv
import threading
from datetime import datetime, timezone

class SharedDataCollector:
    def __init__(self, bot_name, account_number):
        """
        bot_name: 'cobra', 'anaconda', or 'viper'
        account_number: MT5 account number
        """
        self.bot_name = bot_name
        self.account = account_number
        self.data_dir = os.path.join(os.path.dirname(__file__), 'ml_data', bot_name)
        os.makedirs(self.data_dir, exist_ok=True)
        self._lock = threading.Lock()
        self._current_date = None
        self._file_handle = None
        self._writer = None
        self._headers_written = False

    def log_signal(self, row: dict):
        """Append one row to today's CSV. Thread-safe. Never raises."""
        try:
            with self._lock:
                today = datetime.now(timezone.utc).strftime('%Y-%m-%d')
                # Rotate file at midnight UTC or on first call
                if today != self._current_date:
                    if self._file_handle:
                        self._file_handle.close()
                    self._current_date = today
                    filepath = os.path.join(self.data_dir, f'{self.bot_name}_signals_{today}.csv')
                    file_exists = os.path.exists(filepath)
                    if file_exists:
                        # Read existing header so appended rows stay schema-aligned
                        with open(filepath, 'r', newline='', encoding='utf-8') as rf:
                            existing_headers = next(csv.reader(rf), None)
                        fieldnames = existing_headers if existing_headers else list(row.keys())
                    else:
                        fieldnames = list(row.keys())
                    self._file_handle = open(filepath, 'a', newline='', encoding='utf-8')
                    self._writer = csv.DictWriter(
                        self._file_handle, fieldnames=fieldnames, extrasaction='ignore'
                    )
                    if not file_exists:
                        self._writer.writeheader()
                self._writer.writerow(row)
                self._file_handle.flush()
        except Exception:
            pass  # NEVER affect trading

    def close(self):
        try:
            if self._file_handle:
                self._file_handle.close()
        except Exception:
            pass
