"""Hard risk limits for the Panic Buyer, enforced in the execution layer (account level, all sleeves together).

Every limit is off when its value is 0. Values come from the "risk" block of panic_config.json:

  daily_dd_pct        X  flatten everything and lock new orders until the next UTC day when the bot's money
                         (cash + coins at market) is X% under its highest value of the current UTC day
  monthly_dd_pct      Y  flatten everything and lock for monthly_lock_hours when the money is Y% under its
                         highest hourly value of the last 30 days
  monthly_lock_hours     lock length after a 30-day breach (default 168)
  stop_loss_pct       S  sell a position when an hourly candle closes S% or more under its entry
                         (checked by the bot each hour, not resting on Binance: gaps can fill worse)
  max_trade_risk_pct  Z  an order may risk at most Z% of the bot's money: order <= Z% x money / S
                         (with no stop, the whole order is at risk: order <= Z% x money)
  max_spread_pct         skip a buy when the bid/ask spread is wider than this
  max_latency_ms      N  no new buys this hour if any market-data request took longer than N ms
  atr_spike_mult         skip a coin whose last hourly range is more than k x its median hourly range of the
                         last 30 days
  loss_streak         N  after N losing trades in a row (any sleeve) ...
  loss_pause_hours    T  ... no new buys for T hours

State is kept in guard_state.json so a restart never forgets a lock.
"""
import json
import os
from datetime import datetime, timedelta, timezone

HOUR_MS = 3600 * 1000
DAY_MS = 24 * HOUR_MS
KEYS = ("daily_dd_pct", "monthly_dd_pct", "monthly_lock_hours", "stop_loss_pct", "max_trade_risk_pct",
        "max_spread_pct", "max_latency_ms", "atr_spike_mult", "loss_streak", "loss_pause_hours")


def next_utc_midnight(ms):
    d = datetime.fromtimestamp(ms / 1000, timezone.utc).date() + timedelta(days=1)
    return int(datetime(d.year, d.month, d.day, tzinfo=timezone.utc).timestamp() * 1000)


def utc_day(ms):
    return datetime.fromtimestamp(ms / 1000, timezone.utc).strftime("%Y-%m-%d")


class RiskGuard:
    def __init__(self, risk, root, clock, log=lambda msg, alert=False: print(msg)):
        unknown = set(risk) - set(KEYS)
        if unknown:
            raise ValueError(f"unknown risk settings: {sorted(unknown)}")
        self.r = {k: float(risk.get(k, 0) or 0) for k in KEYS}
        if not self.r["monthly_lock_hours"]:
            self.r["monthly_lock_hours"] = 168.0
        self.path = os.path.join(root, "guard_state.json")
        self.clock = clock
        self.log = log
        self.s = self._load()

    # ------------------------------------------------------------- state
    def _load(self):
        if os.path.exists(self.path):
            with open(self.path) as f:
                return json.load(f)
        return {"day": None, "day_high": None, "hist": [], "locked_until": 0, "lock_reason": "",
                "streak": 0, "paused_until": 0}

    def save(self):
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.s, f, indent=2)
        os.replace(tmp, self.path)

    # ------------------------------------------------------------- kill-switch
    def update_equity(self, eq, t):
        """Record the bot's money at time t. Returns a reason string when a kill-switch fires (caller flattens)."""
        s = self.s
        if s["day"] != utc_day(t):
            s["day"], s["day_high"] = utc_day(t), eq
        s["day_high"] = max(s["day_high"], eq)
        s["hist"] = [h for h in s["hist"] if h[0] > t - 30 * DAY_MS] + [[t, eq]]
        reason = None
        if t >= s["locked_until"]:
            x, y = self.r["daily_dd_pct"], self.r["monthly_dd_pct"]
            high = max(h[1] for h in s["hist"])
            if x and eq <= s["day_high"] * (1 - x / 100):
                reason = f"daily drop {(eq / s['day_high'] - 1) * 100:.1f}% (limit -{x:g}%)"
                s["locked_until"] = next_utc_midnight(t)
            elif y and eq <= high * (1 - y / 100):
                reason = f"30-day drop {(eq / high - 1) * 100:.1f}% (limit -{y:g}%)"
                s["locked_until"] = t + int(self.r["monthly_lock_hours"] * HOUR_MS)
            if reason:
                s["lock_reason"] = reason
                s["hist"] = [[t, eq]]        # measure the next drops from here, not from the old highs
                s["day_high"] = eq
        self.save()
        return reason

    # ------------------------------------------------------------- entry gates
    def entries_allowed(self, t):
        s = self.s
        if t < s["locked_until"]:
            return False, f"kill-switch: {s['lock_reason']}"
        if t < s["paused_until"]:
            return False, f"{int(self.r['loss_streak'])} losses in a row"
        return True, ""

    def on_trade_closed(self, pnl, t):
        n, hrs = int(self.r["loss_streak"]), self.r["loss_pause_hours"]
        self.s["streak"] = self.s["streak"] + 1 if pnl < 0 else 0
        if n and hrs and self.s["streak"] >= n:
            self.s["paused_until"] = t + int(hrs * HOUR_MS)
            self.s["streak"] = 0
            self.log(f"RISK: {n} losing trades in a row, no new buys for {hrs:g}h", alert=True)
        self.save()

    def max_order(self, money):
        """Largest order (USDT) allowed by the per-trade risk cap, or None when the cap is off."""
        z, stop = self.r["max_trade_risk_pct"], self.r["stop_loss_pct"]
        if not z:
            return None
        return money * z / 100 / (stop / 100 if stop else 1.0)

    def latency_ok(self, ms):
        return not self.r["max_latency_ms"] or ms <= self.r["max_latency_ms"]

    def spread_ok(self, bid, ask):
        if not self.r["max_spread_pct"]:
            return True
        if bid <= 0 or ask <= 0:
            return False
        return (ask - bid) / ((ask + bid) / 2) * 100 <= self.r["max_spread_pct"]

    def atr_ok(self, candles):
        """candles: closed hourly candles, oldest first; the last one is the hour just closed."""
        k = self.r["atr_spike_mult"]
        if not k or len(candles) < 50:
            return True
        rng = sorted((c["high"] - c["low"]) / c["close"] for c in candles[-721:-1] if c["close"] > 0)
        med = rng[len(rng) // 2]
        last = candles[-1]
        return med <= 0 or (last["high"] - last["low"]) / last["close"] <= k * med

    def status(self, t):
        s = self.s
        lines = []
        if t < s["locked_until"]:
            lines.append(f"LOCKED by kill-switch until {datetime.fromtimestamp(s['locked_until'] / 1000, timezone.utc):%Y-%m-%d %H:%M} UTC: {s['lock_reason']}")
        if t < s["paused_until"]:
            lines.append(f"PAUSED after losing streak until {datetime.fromtimestamp(s['paused_until'] / 1000, timezone.utc):%Y-%m-%d %H:%M} UTC")
        on = ", ".join(f"{k}={v:g}" for k, v in self.r.items() if v and k != "monthly_lock_hours")
        lines.append(f"Risk limits: {on or 'none'}")
        return "\n".join(lines)
