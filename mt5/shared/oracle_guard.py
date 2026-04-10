import requests as http_requests


DEFAULT_ORACLE_URL = "http://127.0.0.1:8070"


def _resolve_guard_config(config):
    guard_cfg = config.get("oracle_guard", {}) if isinstance(config, dict) else {}
    return {
        "enabled": bool(guard_cfg.get("enabled", True)),
        "url": str(guard_cfg.get("url", DEFAULT_ORACLE_URL)).rstrip("/"),
        "timeout": float(guard_cfg.get("timeout", 2.0)),
        "fail_open": bool(guard_cfg.get("fail_open", False)),
        "api_key": str(guard_cfg.get("api_key", "")),
    }


def request_oracle_approval(config, bot_name, profile_name, symbol, direction, volume,
                            entry_price=None, sl=None, tp=None):
    guard_cfg = _resolve_guard_config(config)
    if not guard_cfg["enabled"]:
        return True, "oracle_guard_disabled", None

    payload = {
        "bot": bot_name,
        "profile": profile_name,
        "symbol": symbol,
        "direction": str(direction).upper(),
        "volume": float(volume),
        "entry_price": float(entry_price) if entry_price is not None else None,
        "sl": float(sl) if sl is not None else None,
        "tp": float(tp) if tp is not None else None,
    }

    headers = {}
    if guard_cfg["api_key"]:
        headers["X-API-Key"] = guard_cfg["api_key"]

    try:
        response = http_requests.post(
            f"{guard_cfg['url']}/can_open",
            json=payload,
            headers=headers,
            timeout=guard_cfg["timeout"],
        )
        data = response.json()
    except Exception as exc:
        if guard_cfg["fail_open"]:
            return True, f"oracle_unreachable_fail_open: {exc}", None
        return False, f"oracle_unreachable: {exc}", None

    if response.status_code != 200:
        reason = data.get("reason") if isinstance(data, dict) else None
        if guard_cfg["fail_open"]:
            return True, f"oracle_http_{response.status_code}_fail_open", data
        return False, reason or f"oracle_http_{response.status_code}", data

    allowed = bool(data.get("allowed", False))
    reason = data.get("reason", "oracle_no_reason")
    return allowed, reason, data
