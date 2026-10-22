"""Minimal Binance SPOT client for the pump-rider. Python standard library only.

Safety: this client has NO withdraw, transfer, convert, margin or futures functions. It can only read
market data and balances, and place, check or cancel SPOT orders.

Keys come from the environment (BINANCE_API_KEY, BINANCE_API_SECRET). They are never written to disk
by this code and never printed.
"""

import hashlib
import hmac
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from decimal import ROUND_DOWN, ROUND_UP, Decimal

BASE = "https://api.binance.com"


class BinanceError(Exception):
    def __init__(self, message, code=None):
        super().__init__(message)
        self.code = code


class BinanceSpot:
    def __init__(self, key=None, secret=None, base=BASE, timeout=15):
        self.key = key if key is not None else os.environ.get("BINANCE_API_KEY", "")
        self.secret = (secret if secret is not None else os.environ.get("BINANCE_API_SECRET", "")).encode()
        self.base = base
        self.timeout = timeout
        self.offset_ms = 0

    # ----------------------------------------------------------------- transport
    def _send(self, method, path, params=None, signed=False):
        params = dict(params or {})
        headers = {"User-Agent": "halal-rider/1.0"}
        if signed:
            if not self.key or not self.secret:
                raise BinanceError("BINANCE_API_KEY / BINANCE_API_SECRET are not set")
            params["timestamp"] = int(time.time() * 1000) + self.offset_ms
            params["recvWindow"] = 5000
            query = urllib.parse.urlencode(params)
            sig = hmac.new(self.secret, query.encode(), hashlib.sha256).hexdigest()
            query += "&signature=" + sig
            headers["X-MBX-APIKEY"] = self.key
        else:
            query = urllib.parse.urlencode(params)
        url = self.base + path
        data = None
        if method == "GET" or method == "DELETE":
            url += ("?" + query) if query else ""
        else:
            data = query.encode()
            headers["Content-Type"] = "application/x-www-form-urlencoded"
        attempts = 3 if method == "GET" else 1  # never repeat an order request
        last = None
        for n in range(attempts):
            req = urllib.request.Request(url, data=data, headers=headers, method=method)
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    return json.loads(r.read().decode() or "{}")
            except urllib.error.HTTPError as err:
                body = err.read().decode(errors="replace")[:300]
                code = None
                try:
                    code = json.loads(body).get("code")
                except ValueError:
                    pass
                last = BinanceError(f"{method} {path} HTTP {err.code}: {body}", code)
                if err.code < 500 and err.code != 429:
                    break
            except (urllib.error.URLError, TimeoutError, OSError) as err:
                last = BinanceError(f"{method} {path} failed: {err}")
            if n + 1 < attempts:
                time.sleep(1.5 * (n + 1))
        raise last

    # ------------------------------------------------------------- market data
    def sync_time(self):
        server = self._send("GET", "/api/v3/time")["serverTime"]
        self.offset_ms = int(server - time.time() * 1000)
        return self.offset_ms

    def exchange_info(self, symbols):
        data = self._send("GET", "/api/v3/exchangeInfo", {"symbols": json.dumps(sorted(symbols), separators=(",", ":"))})
        out = {}
        for s in data["symbols"]:
            f = {x["filterType"]: x for x in s["filters"]}
            notional = f.get("NOTIONAL", {}).get("minNotional") or f.get("MIN_NOTIONAL", {}).get("minNotional") or "5"
            out[s["symbol"]] = {
                "status": s["status"], "base": s["baseAsset"], "quote": s["quoteAsset"],
                "tick": f["PRICE_FILTER"]["tickSize"], "step": f["LOT_SIZE"]["stepSize"],
                "min_qty": float(f["LOT_SIZE"]["minQty"]), "min_notional": float(notional),
                "order_types": s.get("orderTypes", []),
            }
        return out

    def klines(self, symbol, interval="5m", limit=100):
        rows = self._send("GET", "/api/v3/klines", {"symbol": symbol, "interval": interval, "limit": limit})
        return [{"open_ms": int(k[0]), "open": float(k[1]), "high": float(k[2]), "low": float(k[3]),
                 "close": float(k[4]), "volume": float(k[5]), "close_ms": int(k[6])} for k in rows]

    def prices(self, symbols):
        rows = self._send("GET", "/api/v3/ticker/price", {"symbols": json.dumps(sorted(symbols), separators=(",", ":"))})
        return {r["symbol"]: float(r["price"]) for r in rows}

    # ---------------------------------------------------------------- account
    def balances(self):
        acc = self._send("GET", "/api/v3/account", {"omitZeroBalances": "true"}, signed=True)
        return {b["asset"]: {"free": float(b["free"]), "locked": float(b["locked"])} for b in acc["balances"]}

    def book_ticker(self, symbol):
        r = self._send("GET", "/api/v3/ticker/bookTicker", {"symbol": symbol})
        return float(r["bidPrice"]), float(r["askPrice"])

    def get_order_by_cid(self, symbol, client_order_id):
        return self._send("GET", "/api/v3/order", {"symbol": symbol, "origClientOrderId": client_order_id}, signed=True)

    def get_order(self, symbol, order_id):
        return self._send("GET", "/api/v3/order", {"symbol": symbol, "orderId": order_id}, signed=True)

    def cancel_order(self, symbol, order_id):
        return self._send("DELETE", "/api/v3/order", {"symbol": symbol, "orderId": order_id}, signed=True)

    def new_order(self, **params):
        """Spot order. params as Binance expects (symbol, side, type, quantity, price, stopPrice...)."""
        params.setdefault("newOrderRespType", "FULL")
        return self._send("POST", "/api/v3/order", params, signed=True)


# ------------------------------------------------------------------- rounding
def round_step(value, step, up=False):
    q = Decimal(str(step)).normalize()
    v = Decimal(repr(float(value)))
    out = (v / q).to_integral_value(rounding=ROUND_UP if up else ROUND_DOWN) * q
    return format(out.normalize(), "f") if out != 0 else "0"
