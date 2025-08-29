"""
app.py - Single-file Flask + AdvancedTradingSystem web app

Features:
- AdvancedTradingSystem (your trading code preserved and enhanced)
- Flask web frontend (single file, templates embedded)
- PayPal server-side order creation and client-side capture (sandbox by default)
- $2.64 per selected symbol
- Modern responsive dashboard display for signals
- Sponsors, SEO meta tags, monetization placeholders

USAGE:
python app.py
Open http://127.0.0.1:5000
"""

from flask import Flask, request, jsonify, render_template_string
import json, logging, http.client, urllib.parse, time, random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import requests

# Optional: XGBoost (if installed); app runs without it.
try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None

# ---------------------------
# CONFIG (move to env vars for production)
# ---------------------------
ALPHA_VANTAGE_API_KEY = "N3028IPGPX7SVW1F"

PAYPAL_CLIENT_ID = "AXEBIxFQXxIPCaamboADh1yhvTuny5OvvXRVEmtYPy8R8fM5sNZqJyfDCvc6v4ASDvhq2n-FiKroEQBX"
PAYPAL_SECRET = "EPngCVR_Cos0x-3LMQ92OVj6LQiubjMWkj2KeKMLXDGNbuqDbeS1CYMdtMzonDxIMhHcJvKCuJ4noF-C"
PAYPAL_API = "https://api-m.sandbox.paypal.com"  # switch to live for production

PRICE_PER_SYMBOL = 2.64  # USD

SPONSORS = [
    {"name": "NVIDIA", "url": "https://www.nvidia.com", "logo": "https://www.nvidia.com/content/dam/en-zz/Solutions/geforce/news/rtx-30-series/rtx-30-series-share/rtx-30-share.jpg"},
    {"name": "Intel", "url": "https://www.intel.com", "logo": "https://www.intel.com/content/dam/www/public/us/en/images/logos/intel-logo.svg"},
    {"name": "Microsoft Azure", "url": "https://azure.microsoft.com", "logo": "https://azure.microsoft.com/s/cms/images/og-image-azure.png"},
    {"name": "Amazon AWS", "url": "https://aws.amazon.com", "logo": "https://a0.awsstatic.com/libra-css/images/logos/aws_logo_smile_1200x630.png"},
    {"name": "Google Cloud", "url": "https://cloud.google.com", "logo": "https://cloud.google.com/images/social-icon-google-cloud-1200-630.png"},
    {"name": "IBM", "url": "https://www.ibm.com", "logo": "https://www.ibm.com/images/logos/ibm-logo.png"},
    {"name": "Meta for Developers", "url": "https://developers.facebook.com", "logo": "https://static.xx.fbcdn.net/rsrc.php/yo/r/iRmz9lCMJ2P.svg"},
    {"name": "Coinbase", "url": "https://www.coinbase.com", "logo": "https://www.coinbase.com/assets/branding/coinbase-logo.png"},
]

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ---------------------------
# AdvancedTradingSystem (preserved and adapted)
# ---------------------------
class AdvancedTradingSystem:
    def __init__(self, api_key):
        self.api_key = api_key
        self.model = XGBClassifier() if XGBClassifier else None
        self.max_retries = 5
        self.backoff_factor = 2

    def fetch_data_alpha_vantage(self, symbol):
        # Note: Alpha Vantage has symbol limitations for indices/commodities;
        # for symbols that AlphaVantage doesn't support this will return None.
        for attempt in range(self.max_retries):
            try:
                conn = http.client.HTTPSConnection("www.alphavantage.co")
                params = urllib.parse.urlencode({
                    'function': 'TIME_SERIES_DAILY_ADJUSTED',
                    'symbol': symbol,
                    'apikey': self.api_key,
                    'outputsize': 'compact'
                })
                conn.request("GET", f"/query?{params}")
                response = conn.getresponse()
                if response.status == 200:
                    data = json.loads(response.read())
                    conn.close()
                    time_series = data.get('Time Series (Daily)', {})
                    if not time_series:
                        logging.warning(f"No daily series for {symbol} from AlphaVantage.")
                        return None
                    prices = []
                    for date, values in time_series.items():
                        try:
                            prices.append((
                                date,
                                float(values.get('4. close', 0)),
                                float(values.get('5. adjusted close', 0)),
                                float(values.get('2. high', 0)),
                                float(values.get('3. low', 0)),
                                float(values.get('6. volume', 0))
                            ))
                        except Exception:
                            continue
                    prices.sort(reverse=True)
                    return prices
                elif response.status == 429:
                    wait_time = self.backoff_factor ** attempt
                    logging.warning(f"Too many requests, waiting {wait_time}s (attempt {attempt+1})")
                    time.sleep(wait_time)
                else:
                    logging.error(f"Error fetching data: {response.status} {response.reason}")
                    return None
            except Exception as e:
                logging.error(f"Error fetching data for {symbol}: {e}")
                return None

    def calculate_ema(self, prices, period):
        if len(prices) < period:
            return None
        multiplier = 2 / (period + 1)
        sma = sum(price[1] for price in prices[:period]) / period
        ema = sma
        for price in prices[period:]:
            ema = (price[1] - ema) * multiplier + ema
        return ema

    def calculate_rsi(self, prices, period=14):
        if len(prices) < period:
            return None
        gains, losses = [], []
        for i in range(1, len(prices)):
            change = prices[i][1] - prices[i-1][1]
            gains.append(max(change, 0))
            losses.append(abs(min(change, 0)))
        avg_gain, avg_loss = np.mean(gains[-period:]), np.mean(losses[-period:])
        if avg_loss == 0:
            return 100
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def calculate_macd(self, prices, short_period=12, long_period=26, signal_period=9):
        if len(prices) < long_period:
            return None, None
        short_ema = self.calculate_ema(prices, short_period)
        long_ema = self.calculate_ema(prices, long_period)
        macd = short_ema - long_ema
        # Simplified: reuse macd value across prices for signal_line calc
        signal_line = self.calculate_ema([(i[0], macd) for i in prices], signal_period)
        return macd, signal_line

    def detect_candlestick_patterns(self, prices):
        patterns = []
        if len(prices) < 2:
            return patterns
        latest = prices[0]
        prev = prices[1]
        body = abs(latest[1] - prev[1]) or 1e-9
        candle_range = (latest[3] - latest[4]) or 1e-9
        if body < 0.3 * candle_range:
            patterns.append("Doji")
        if (latest[1] - latest[4]) > 2 * body and latest[1] > prev[1]:
            patterns.append("Hammer")
        if (latest[3] - latest[1]) > 2 * body and latest[1] < prev[1]:
            patterns.append("Shooting Star")
        return patterns

    def calculate_trend(self, prices):
        if len(prices) < 6:
            return "Sideways"
        highs = [p[3] for p in prices[:6]]
        lows = [p[4] for p in prices[:6]]
        if all(x < y for x, y in zip(lows, lows[1:])):
            return "Uptrend"
        elif all(x > y for x, y in zip(highs, highs[1:])):
            return "Downtrend"
        return "Sideways"

    def calculate_support_resistance(self, prices):
        sample = prices[:60] if len(prices) >= 60 else prices
        lows = [p[4] for p in sample if p[4] is not None]
        highs = [p[3] for p in sample if p[3] is not None]
        if not lows or not highs:
            return 0.0, 0.0
        support = float(np.percentile(lows, 10))
        resistance = float(np.percentile(highs, 90))
        return support, resistance

    def check_volume_spike(self, prices):
        if len(prices) < 10:
            return False
        volumes = [p[5] for p in prices[:20] if p[5] is not None]
        if len(volumes) < 2:
            return False
        latest_vol = volumes[0]
        avg_vol = float(np.mean(volumes[1:])) if len(volumes) > 1 else latest_vol
        return latest_vol > 1.5 * avg_vol

    def generate_signals(self, symbol):
        prices = self.fetch_data_alpha_vantage(symbol)
        if not prices:
            return {
                "symbol": symbol,
                "signals": ["No Data"],
                "Entry_Point": None,
                "Stop_Loss": None,
                "Take_Profit": None,
                "Entry_Time": None,
                "Duration": None
            }

        short_ema = self.calculate_ema(prices, 12)
        long_ema = self.calculate_ema(prices, 26)
        rsi = self.calculate_rsi(prices)
        macd, signal_line = self.calculate_macd(prices)
        patterns = self.detect_candlestick_patterns(prices)
        trend = self.calculate_trend(prices)
        support, resistance = self.calculate_support_resistance(prices)
        volume_spike = self.check_volume_spike(prices)

        latest_price = prices[0][1]
        signals = []

        if short_ema and long_ema:
            if short_ema > long_ema:
                signals.append("Bullish Signal")
            elif short_ema < long_ema:
                signals.append("Bearish Signal")
        if rsi:
            if rsi < 30:
                signals.append("RSI Oversold - Buy Signal")
            elif rsi > 70:
                signals.append("RSI Overbought - Sell Signal")
        if macd is not None and signal_line is not None:
            if macd > signal_line:
                signals.append("MACD Bullish Crossover")
            elif macd < signal_line:
                signals.append("MACD Bearish Crossover")

        signals.extend(patterns)
        signals.append(f"Trend: {trend}")
        signals.append(f"Support: {support:.4f}, Resistance: {resistance:.4f}")
        if volume_spike:
            signals.append("Volume Spike Detected")

        bullish = any("Bullish" in s or "Buy" in s for s in signals)
        entry_point = latest_price
        stop_loss = entry_point * (0.98 if bullish else 1.02)
        take_profit = entry_point * (1.02 if bullish else 0.98)
        duration = "1h"

        entry_time = (datetime.utcnow() + timedelta(seconds=random.randint(0, 300))).strftime('%Y-%m-%d %H:%M:%S UTC')

        return {
            "symbol": symbol,
            "signals": signals,
            "Entry_Point": float(entry_point),
            "Stop_Loss": float(stop_loss),
            "Take_Profit": float(take_profit),
            "Entry_Time": entry_time,
            "Duration": duration
        }

    def process_signals(self, symbols):
        all_signals = []
        for symbol in symbols:
            logging.info(f"Generating signals for {symbol} ...")
            signal_info = self.generate_signals(symbol)
            all_signals.append(signal_info)
            logging.info(f"{symbol} => Entry: {signal_info['Entry_Point']}, SL: {signal_info['Stop_Loss']}, TP: {signal_info['Take_Profit']}")
        with open('trading_signals.json', 'w') as f:
            json.dump(all_signals, f, indent=2)
        return all_signals

# ---------------------------
# Flask app & HTML template
# ---------------------------
app = Flask(__name__, static_url_path='/static')

# A comprehensive symbols list covering many markets (not exhaustive but large)
ALL_SYMBOLS = [
    # Forex majors & minors
    "EURUSD","GBPUSD","USDJPY","USDCHF","USDCAD","AUDUSD","NZDUSD","EURGBP","EURJPY","GBPJPY",
    # Exotic/common crosses
    "EURAUD","EURNZD","AUDJPY","GBPCHF","CADJPY",

    # Indices (common tickers)
    "SPY","^GSPC","DAX","FTSE","NQ=F","^N225","GDAXI","UKX","^HSI","ASX200",

    # Stocks (popular US & global)
    "AAPL","MSFT","AMZN","GOOGL","TSLA","META","NVDA","NFLX","INTC","CSCO","BABA","JPM","BAC","V","MA","WMT","DIS","KO","PFE","MRNA",

    # Metals
    "XAUUSD","XAGUSD","XPTUSD","XPDUSD",

    # Energy
    "USOIL","UKOIL","NGAS","BRN",

    # Crypto (common USD pairs)
    "BTCUSD","ETHUSD","LTCUSD","BCHUSD","XRPUSD","ADAUSD",

    # Commodities / Softs
    "CORN","WHEAT","SOYB","COFFEE","SUGAR",

    # ETFs (popular)
    "QQQ","IWM","GLD","SLV","USO","UNG"
]

# HTML template (rendered via render_template_string)
INDEX_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>SIGY11 — Advanced Trading Signals</title>
  <meta name="description" content="Buy per-symbol premium trading signals (EMA, RSI, MACD, candlesticks, volume, trend, S/R and more).">
  <meta property="og:title" content="SIGY11 — Advanced Trading Signals">
  <meta property="og:description" content="Premium pay-per-symbol trading signals.">
  <meta property="og:type" content="website">
  <meta name="robots" content="index, follow">
  <link rel="canonical" href="{{ url }}">

  <script src="https://www.paypal.com/sdk/js?client-id={{ paypal_client_id }}&currency=USD"></script>
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">

  <style>
    body { background: linear-gradient(180deg,#f6f8fb,#ffffff); font-family: Inter, system-ui, -apple-system, 'Segoe UI', Roboto, Arial; }
    .card-rounded { border-radius: 12px; box-shadow: 0 6px 20px rgba(18,38,63,.06); }
    .sponsor-logo { height:36px; object-fit:contain; }
    .muted { color:#6b7280; }
    .header { padding: 1.5rem 0; }
    .footer { padding: 1.5rem 0; color:#6b7280; font-size:.9rem; }
    .symbol-card { cursor:pointer; transition: transform .08s ease-in-out; }
    .symbol-card:hover { transform: translateY(-3px); }
  </style>
</head>
<body>
<div class="container py-4">
  <div class="header text-center">
    <h1>SIGY11 — Advanced Trading Signals</h1>
    <p class="muted">Select symbols from Forex, Indices, Stocks, Metals, Energy, Crypto & more. ${{ price }} per symbol.</p>
  </div>

  <div class="row g-4">
    <div class="col-lg-8">
      <div class="card card-rounded p-3 mb-3">
        <h5>Select symbols (search or pick):</h5>
        <div class="mb-2">
          <input id="searchBox" class="form-control" placeholder="Search symbols (e.g., EURUSD, AAPL, BTCUSD)">
        </div>
        <div id="symbolsGrid" class="row gy-2" style="max-height:360px; overflow:auto;">
          {% for s in all_symbols %}
          <div class="col-6 col-sm-4 col-md-3">
            <div class="card p-2 symbol-card" data-symbol="{{ s }}">
              <div class="form-check">
                <input class="form-check-input me-2 symbol-checkbox" type="checkbox" value="{{ s }}" id="sym_{{ loop.index }}">
                <label class="form-check-label" for="sym_{{ loop.index }}"><strong>{{ s }}</strong></label>
              </div>
            </div>
          </div>
          {% endfor %}
        </div>
        <p class="muted mt-2">Price per symbol: <strong>${{ price }}</strong></p>

        <div id="paypal-button-container" class="mt-3"></div>
        <small class="muted">After payment you'll see signals displayed below.</small>
      </div>

      <div id="signalsArea"></div>
    </div>

    <div class="col-lg-4">
      <div class="card card-rounded p-3 mb-3">
        <h6>Sponsored by</h6>
        <div class="d-flex flex-wrap gap-2 align-items-center">
          {% for sp in sponsors %}
            <a href="{{ sp.url }}" target="_blank" title="{{ sp.name }}">
              <img src="{{ sp.logo }}" class="sponsor-logo me-2" alt="{{ sp.name }}">
            </a>
          {% endfor %}
        </div>
      </div>

      <div class="card card-rounded p-3 mb-3">
        <h6>Monetization</h6>
        <ul class="muted small">
          <li>Affiliate links (brokers, exchanges, cloud providers)</li>
          <li>Ad slot placeholder</li>
          <li>Newsletter / referrals</li>
        </ul>
        <div class="ad-placeholder my-2 p-3 text-center">Ad / Affiliate Slot</div>

        <div class="mt-3">
          <label class="form-label">Join newsletter</label>
          <div class="input-group">
            <input type="email" id="newsletterEmail" class="form-control" placeholder="email@example.com">
            <button class="btn btn-primary" type="button" onclick="subscribeNewsletter()">Subscribe</button>
          </div>
        </div>
      </div>

      <div class="card card-rounded p-3">
        <h6>Affiliate Tools</h6>
        <ul class="small">
          <li><a href="https://www.tradingview.com" target="_blank">TradingView</a></li>
          <li><a href="https://www.binance.com" target="_blank">Binance</a></li>
          <li><a href="https://www.coinbase.com" target="_blank">Coinbase</a></li>
        </ul>
      </div>
    </div>
  </div>

  <div class="footer text-center mt-4">
    <small>© {{ year }} SIGY11 · Built by Mateu. Signals are informational only.</small>
  </div>
</div>

<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/js/bootstrap.bundle.min.js"></script>

<script>
  const pricePerSymbol = {{ price }};
  const paypalClientId = "{{ paypal_client_id }}";

  // keep selected
  let selectedSymbols = [];

  // wire symbol clicks
  document.addEventListener('click', function(e){
    const card = e.target.closest('.symbol-card');
    if(card){
      const checkbox = card.querySelector('.symbol-checkbox');
      checkbox.checked = !checkbox.checked;
      updateSelected();
    }
  });

  document.querySelectorAll('.symbol-checkbox').forEach(cb => {
    cb.addEventListener('change', updateSelected);
  });

  document.getElementById('searchBox').addEventListener('input', function(e){
    const q = e.target.value.trim().toUpperCase();
    document.querySelectorAll('#symbolsGrid [data-symbol]').forEach(node=>{
      const sym = node.getAttribute('data-symbol');
      node.style.display = sym.includes(q) ? '' : 'none';
    });
  });

  function updateSelected(){
    selectedSymbols = Array.from(document.querySelectorAll('.symbol-checkbox:checked')).map(x => x.value);
  }

  paypal.Buttons({
    createOrder: function(data, actions) {
      updateSelected();
      if(selectedSymbols.length === 0){
        alert("Select at least one symbol.");
        return;
      }
      const total = (selectedSymbols.length * pricePerSymbol).toFixed(2);
      return fetch("/create_order", {
        method: "POST",
        headers: {"Content-Type":"application/json"},
        body: JSON.stringify({ total: total, symbols: selectedSymbols })
      }).then(res=>res.json()).then(order=>{
        if(order && order.id) return order.id;
        throw new Error("Failed to create order");
      });
    },
    onApprove: function(data, actions){
      return fetch("/capture_order", {
        method: "POST",
        headers: {"Content-Type":"application/json"},
        body: JSON.stringify({ orderID: data.orderID })
      }).then(res=>res.json()).then(capture=>{
        // capture may return different structure in sandbox; check status
        const status = capture.status || (capture.statuses && capture.statuses[0]);
        if(capture.status && capture.status !== 'COMPLETED' && capture.status !== 'COMPLETED'){
          // continue anyway but warn
          console.warn("PayPal capture response:", capture);
        }
        // fetch signals now
        return fetch("/get_signals", {
          method: "POST",
          headers: {"Content-Type":"application/json"},
          body: JSON.stringify({ symbols: selectedSymbols })
        })
        .then(r=>r.json())
        .then(renderSignals)
        .catch(err=>alert("Failed to fetch signals: "+err));
      });
    }
  }).render('#paypal-button-container');

  function renderSignals(data){
    const container = document.getElementById('signalsArea');
    if(!data || data.length===0){
      container.innerHTML = '<div class="card p-3 mt-3">No signals returned.</div>';
      return;
    }
    let html = '';
    data.forEach(sig=>{
      html += `<div class="card card-rounded p-3 mt-3">
        <div class="d-flex justify-content-between align-items-center">
          <h5>${sig.symbol}</h5>
          <small class="muted">${sig.Entry_Time || ''}</small>
        </div>
        <div class="mt-2">
          <div><strong>Entry:</strong> ${sig.Entry_Point !== null ? sig.Entry_Point.toFixed(6) : 'N/A'}</div>
          <div><strong>Stop Loss:</strong> ${sig.Stop_Loss !== null ? sig.Stop_Loss.toFixed(6) : 'N/A'}</div>
          <div><strong>Take Profit:</strong> ${sig.Take_Profit !== null ? sig.Take_Profit.toFixed(6) : 'N/A'}</div>
          <div><strong>Duration:</strong> ${sig.Duration || '1h'}</div>
          <div class="mt-2"><strong>Signals:</strong> ${Array.isArray(sig.signals) ? sig.signals.join(', ') : sig.signals}</div>
        </div>
      </div>`;
    });
    container.innerHTML = html;
    window.scrollTo({ top: container.offsetTop - 60, behavior: 'smooth' });
  }

  function subscribeNewsletter(){
    const email = document.getElementById('newsletterEmail').value;
    if(!email){ alert('Enter an email'); return; }
    alert('Thanks! We will send updates to ' + email);
  }
</script>
</body>
</html>
"""

# ---------------------------
# PayPal helpers
# ---------------------------
def paypal_get_access_token():
    auth = (PAYPAL_CLIENT_ID, PAYPAL_SECRET)
    headers = {"Accept":"application/json","Accept-Language":"en_US"}
    data = {"grant_type": "client_credentials"}
    try:
        r = requests.post(f"{PAYPAL_API}/v1/oauth2/token", headers=headers, data=data, auth=auth, timeout=15)
        if r.status_code == 200:
            return r.json().get("access_token")
        logging.error("PayPal token error: %s %s", r.status_code, r.text)
    except Exception as e:
        logging.error("PayPal auth exception: %s", e)
    return None

# ---------------------------
# Routes
# ---------------------------
@app.route("/")
def index():
    return render_template_string(INDEX_HTML,
                                  url=request.url,
                                  paypal_client_id=PAYPAL_CLIENT_ID,
                                  price=PRICE_PER_SYMBOL,
                                  all_symbols=ALL_SYMBOLS,
                                  sponsors=SPONSORS,
                                  year=datetime.utcnow().year)

@app.route("/create_order", methods=["POST"])
def create_order():
    payload = request.json or {}
    total = payload.get("total")
    symbols = payload.get("symbols", [])
    if not total or not symbols:
        return jsonify({"error":"missing total or symbols"}), 400

    token = paypal_get_access_token()
    if not token:
        return jsonify({"error":"paypal auth failed"}), 500

    headers = {"Content-Type":"application/json", "Authorization": f"Bearer {token}"}
    order_payload = {
        "intent": "CAPTURE",
        "purchase_units": [{"amount": {"currency_code":"USD","value": str(total)}}],
        "application_context": {"return_url": request.url_root, "cancel_url": request.url_root}
    }
    r = requests.post(f"{PAYPAL_API}/v2/checkout/orders", headers=headers, json=order_payload, timeout=15)
    if r.status_code in (200,201):
        return jsonify(r.json())
    logging.error("Create order failed: %s %s", r.status_code, r.text)
    return jsonify({"error":"create order failed", "details": r.text}), 500

@app.route("/capture_order", methods=["POST"])
def capture_order():
    payload = request.json or {}
    order_id = payload.get("orderID")
    if not order_id:
        return jsonify({"error":"missing orderID"}), 400
    token = paypal_get_access_token()
    if not token:
        return jsonify({"error":"paypal auth failed"}), 500
    headers = {"Content-Type":"application/json","Authorization": f"Bearer {token}"}
    r = requests.post(f"{PAYPAL_API}/v2/checkout/orders/{order_id}/capture", headers=headers, timeout=15)
    if r.status_code in (200,201):
        return jsonify(r.json())
    logging.error("Capture failed: %s %s", r.status_code, r.text)
    return jsonify({"error":"capture failed", "details": r.text}), 500

@app.route("/get_signals", methods=["POST"])
def get_signals():
    payload = request.json or {}
    symbols = payload.get("symbols", [])
    if not symbols:
        return jsonify([])

    trading_system = AdvancedTradingSystem(ALPHA_VANTAGE_API_KEY)
    results = trading_system.process_signals(symbols)

    # Normalize keys (replace spaces with underscores) to ensure safe JSON keys for frontend
    cleaned = []
    for r in results:
        newr = {}
        for k,v in r.items():
            newk = k.replace(' ', '_')
            # convert numpy types
            if isinstance(v, (np.generic, np.ndarray)):
                try:
                    v = v.item()
                except Exception:
                    v = float(v)
            newr[newk] = v
        cleaned.append(newr)
    return jsonify(cleaned)

# ---------------------------
# Run
# ---------------------------
if __name__ == "__main__":
    app.run(debug=True, port=5000)

