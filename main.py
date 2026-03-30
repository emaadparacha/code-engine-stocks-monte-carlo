"""
IBM Cloud Code Engine Demo – Monte Carlo Stock Simulator
=========================================================
Each parallel job worker picks one stock from the STOCKS env var,
runs a Monte Carlo simulation using 10 years of historical data,
and sends the results via SMS (Twilio).

Environment variables:
  STOCKS            – comma-separated tickers, e.g. "AAPL,MSFT,GOOGL,AMZN"
  INVEST_AMOUNT     – dollar amount to simulate (capped at $10,000)
  TWILIO_ACCOUNT_SID
  TWILIO_AUTH_TOKEN
  TWILIO_FROM_NUMBER
  TO_NUMBER
  JOB_INDEX         – set automatically by Code Engine (0-based)
"""

import os
import numpy as np
import yfinance as yf
from datetime import datetime, timezone, timedelta
from twilio.rest import Client


# ── Configuration ────────────────────────────────────────────────────
NUM_SIMULATIONS = 10_000   # number of Monte Carlo paths
FORECAST_DAYS = 252        # 1 trading year (~252 days)
HISTORY_YEARS = 10         # how far back to pull historical data
MAX_INVEST = 10_000        # hard cap on investment amount


def fetch_historical_returns(ticker: str) -> np.ndarray:
    """Download 10 years of adjusted-close prices and return daily log returns."""
    end = datetime.now()
    start = end - timedelta(days=HISTORY_YEARS * 365)

    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)

    if df.empty or len(df) < 252:
        raise ValueError(f"Not enough historical data for {ticker}")

    # Daily log returns
    closes = df["Close"].values.flatten()
    log_returns = np.diff(np.log(closes))
    return log_returns


def run_monte_carlo(log_returns: np.ndarray, invest_amount: float) -> dict:
    """
    Run Monte Carlo simulation using Geometric Brownian Motion (GBM).

    Each simulated day:
        price *= exp(drift + volatility * random_normal)

    where:
        drift     = mean(daily log returns) - 0.5 * variance   (risk-adjusted)
        volatility = std(daily log returns)
    """
    mu = np.mean(log_returns)
    sigma = np.std(log_returns)
    drift = mu - 0.5 * sigma ** 2

    # Simulate: each row is one path of FORECAST_DAYS daily returns
    random_shocks = np.random.normal(size=(NUM_SIMULATIONS, FORECAST_DAYS))
    daily_returns = drift + sigma * random_shocks          # (sims × days)
    cumulative = np.cumsum(daily_returns, axis=1)          # cumulative log return
    price_paths = invest_amount * np.exp(cumulative)       # simulated portfolio values

    final_values = price_paths[:, -1]

    return {
        "mean_final":    float(np.mean(final_values)),
        "median_final":  float(np.median(final_values)),
        "p5":            float(np.percentile(final_values, 5)),
        "p95":           float(np.percentile(final_values, 95)),
        "prob_profit":   float(np.mean(final_values > invest_amount) * 100),
        "best_case":     float(np.max(final_values)),
        "worst_case":    float(np.min(final_values)),
        "annual_vol":    float(sigma * np.sqrt(252) * 100),  # annualised volatility %
    }


def format_sms(ticker: str, amount: float, results: dict) -> str:
    """Build a concise SMS body with the simulation results."""
    gain = results["mean_final"] - amount
    gain_pct = (gain / amount) * 100

    return (
        f"📈 Monte Carlo results for {ticker}\n"
        f"Invested: ${amount:,.0f} | Sims: {NUM_SIMULATIONS:,}\n"
        f"────────────────────\n"
        f"Expected value: ${results['mean_final']:,.0f} ({gain_pct:+.1f}%)\n"
        f"Median value:   ${results['median_final']:,.0f}\n"
        f"5th–95th %%ile: ${results['p5']:,.0f} – ${results['p95']:,.0f}\n"
        f"Prob of profit: {results['prob_profit']:.1f}%\n"
        f"Best / Worst:   ${results['best_case']:,.0f} / ${results['worst_case']:,.0f}\n"
        f"Annual vol:     {results['annual_vol']:.1f}%\n"
        f"(1-year forecast from {HISTORY_YEARS}yr history)"
    )


def main():
    # ── Read env vars ────────────────────────────────────────────────
    job_index = int(os.getenv("JOB_INDEX", "0"))

    stocks_raw = os.environ.get("STOCKS", "AAPL,MSFT,GOOGL")
    tickers = [s.strip().upper() for s in stocks_raw.split(",") if s.strip()]

    if len(tickers) > 7:
        tickers = tickers[:7]
        print("Warning: capped to 7 stocks max")

    if job_index >= len(tickers):
        print(f"Worker {job_index} has no stock assigned (only {len(tickers)} tickers). Exiting.")
        return

    ticker = tickers[job_index]

    invest_amount = min(float(os.environ.get("INVEST_AMOUNT", "1000")), MAX_INVEST)

    # ── Twilio setup ─────────────────────────────────────────────────
    account_sid  = "AC9c4c7cc858805c2d3661e38214d8e505"
    auth_token   = os.environ["TWILIO_TOKEN"]
    from_number  = "+18674571410"
    to_number    = os.environ["PHONE_NUMBER"]
    twilio_client = Client(account_sid, auth_token)

    # ── Run simulation ───────────────────────────────────────────────
    print(f"Worker {job_index}: running Monte Carlo for {ticker} "
          f"(${invest_amount:,.0f}, {NUM_SIMULATIONS:,} sims)...")

    log_returns = fetch_historical_returns(ticker)
    results = run_monte_carlo(log_returns, invest_amount)

    # ── Send SMS ─────────────────────────────────────────────────────
    body = format_sms(ticker, invest_amount, results)
    message = twilio_client.messages.create(
        body=body,
        from_=from_number,
        to=to_number,
    )

    print(f"Worker {job_index} ({ticker}): SMS sent, SID={message.sid}")
    print(body)


if __name__ == "__main__":
    main()
