"""
IBM Cloud Code Engine Demo – Enhanced Monte Carlo Stock Simulator
=================================================================
Advanced stock analysis with Monte Carlo simulation, fundamental analysis,
risk metrics, and portfolio theory for more robust investment insights.

Environment variables:
  STOCKS            – comma-separated tickers, e.g. "AAPL,MSFT,GOOGL,AMZN"
  INVEST_AMOUNT     – dollar amount to simulate (capped at $10,000)
  TWILIO_ACCOUNT_SID
  TWILIO_TOKEN
  TWILIO_FROM_NUMBER
  PHONE_NUMBER
  JOB_INDEX         – set automatically by Code Engine (0-based)
"""

import os
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from twilio.rest import Client
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


# ── Configuration ────────────────────────────────────────────────────
NUM_SIMULATIONS = 20_000   # increased for better statistical significance
FORECAST_DAYS = 252        # 1 trading year (~252 days)
HISTORY_YEARS = 10         # how far back to pull historical data
MAX_INVEST = 10_000        # hard cap on investment amount
TRANSACTION_COST_PCT = 0.001  # 0.1% transaction cost
TAX_RATE = 0.15            # 15% capital gains tax (short-term)
RISK_FREE_RATE = 0.045     # 4.5% annual risk-free rate (approximate)


def fetch_stock_data(ticker: str) -> tuple:
    """
    Download historical data and extract both price returns and fundamental info.
    Returns: (log_returns, stock_info_dict)
    """
    end = datetime.now()
    start = end - timedelta(days=HISTORY_YEARS * 365)
    
    stock = yf.Ticker(ticker)
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    
    if df.empty or len(df) < 252:
        raise ValueError(f"Not enough historical data for {ticker}")
    
    # Daily log returns
    closes = df["Close"].values.flatten()
    log_returns = np.diff(np.log(closes))
    
    # Get fundamental data
    info = stock.info
    fundamentals = {
        "pe_ratio": info.get("trailingPE", None),
        "forward_pe": info.get("forwardPE", None),
        "peg_ratio": info.get("pegRatio", None),
        "debt_to_equity": info.get("debtToEquity", None),
        "current_ratio": info.get("currentRatio", None),
        "roe": info.get("returnOnEquity", None),
        "profit_margin": info.get("profitMargins", None),
        "market_cap": info.get("marketCap", None),
        "beta": info.get("beta", None),
        "52w_high": info.get("fiftyTwoWeekHigh", None),
        "52w_low": info.get("fiftyTwoWeekLow", None),
        "current_price": info.get("currentPrice", closes[-1]),
        "sector": info.get("sector", "Unknown"),
        "industry": info.get("industry", "Unknown"),
    }
    
    return log_returns, fundamentals


def detect_regime_changes(log_returns: np.ndarray, window: int = 60) -> dict:
    """
    Detect market regime changes using rolling volatility.
    Returns regime statistics for risk assessment.
    """
    rolling_vol = np.array([
        np.std(log_returns[max(0, i-window):i+1]) 
        for i in range(len(log_returns))
    ])
    
    # Identify high/low volatility regimes
    vol_median = np.median(rolling_vol)
    high_vol_periods = rolling_vol > vol_median * 1.5
    
    return {
        "current_volatility": float(rolling_vol[-1]),
        "median_volatility": float(vol_median),
        "high_vol_ratio": float(np.mean(high_vol_periods)),
        "regime": "High Volatility" if rolling_vol[-1] > vol_median * 1.5 else "Normal"
    }


def analyze_fundamentals(fundamentals: dict) -> dict:
    """
    Score the stock based on fundamental metrics.
    Returns a score (0-100) and analysis.
    """
    score = 50  # Start neutral
    warnings = []
    strengths = []
    
    # P/E Ratio analysis
    pe = fundamentals.get("pe_ratio")
    if pe:
        if pe < 15:
            score += 10
            strengths.append(f"Low P/E ({pe:.1f})")
        elif pe > 30:
            score -= 10
            warnings.append(f"High P/E ({pe:.1f})")
    
    # PEG Ratio (P/E to Growth)
    peg = fundamentals.get("peg_ratio")
    if peg:
        if peg < 1:
            score += 10
            strengths.append(f"Good PEG ({peg:.2f})")
        elif peg > 2:
            score -= 10
            warnings.append(f"High PEG ({peg:.2f})")
    
    # Debt to Equity
    dte = fundamentals.get("debt_to_equity")
    if dte is not None:
        if dte < 50:
            score += 5
            strengths.append("Low debt")
        elif dte > 200:
            score -= 10
            warnings.append(f"High debt ({dte:.0f}%)")
    
    # Return on Equity
    roe = fundamentals.get("roe")
    if roe:
        if roe > 0.15:
            score += 10
            strengths.append(f"Strong ROE ({roe*100:.1f}%)")
        elif roe < 0.05:
            score -= 10
            warnings.append(f"Weak ROE ({roe*100:.1f}%)")
    
    # Profit Margin
    margin = fundamentals.get("profit_margin")
    if margin:
        if margin > 0.20:
            score += 5
            strengths.append(f"High margins ({margin*100:.1f}%)")
        elif margin < 0.05:
            score -= 5
            warnings.append(f"Low margins ({margin*100:.1f}%)")
    
    # Beta (volatility vs market)
    beta = fundamentals.get("beta")
    if beta:
        if beta > 1.5:
            score -= 5
            warnings.append(f"High volatility (β={beta:.2f})")
        elif beta < 0.8:
            score += 5
            strengths.append(f"Low volatility (β={beta:.2f})")
    
    # Ensure score stays in 0-100 range
    score = max(0, min(100, score))
    
    return {
        "score": score,
        "warnings": warnings,
        "strengths": strengths,
        "grade": "A" if score >= 80 else "B" if score >= 65 else "C" if score >= 50 else "D" if score >= 35 else "F"
    }


def run_enhanced_monte_carlo(log_returns: np.ndarray, invest_amount: float, 
                            fundamentals: dict) -> dict:
    """
    Enhanced Monte Carlo with fat-tailed distributions and transaction costs.
    Uses Student's t-distribution to better model extreme events.
    """
    mu = np.mean(log_returns)
    sigma = np.std(log_returns)
    
    # Fit Student's t-distribution for fat tails
    df_param, loc, scale = stats.t.fit(log_returns)
    
    # Use t-distribution if it fits better (captures crashes better)
    use_t_dist = df_param < 10  # Low df means fatter tails
    
    if use_t_dist:
        # Generate returns using t-distribution (better for crashes)
        random_shocks = stats.t.rvs(df_param, size=(NUM_SIMULATIONS, FORECAST_DAYS))
        daily_returns = loc + scale * random_shocks
    else:
        # Standard GBM with normal distribution
        drift = mu - 0.5 * sigma ** 2
        random_shocks = np.random.normal(size=(NUM_SIMULATIONS, FORECAST_DAYS))
        daily_returns = drift + sigma * random_shocks
    
    # Calculate portfolio values
    cumulative = np.cumsum(daily_returns, axis=1)
    gross_values = invest_amount * np.exp(cumulative)
    
    # Apply transaction costs (entry and exit)
    transaction_cost = invest_amount * TRANSACTION_COST_PCT * 2
    final_values = gross_values[:, -1] - transaction_cost
    
    # Calculate gains and apply taxes
    gains = final_values - invest_amount
    taxed_gains = np.where(gains > 0, gains * (1 - TAX_RATE), gains)
    net_final_values = invest_amount + taxed_gains
    
    # Risk metrics
    returns = (net_final_values - invest_amount) / invest_amount
    
    # Sharpe Ratio (risk-adjusted return)
    mean_return = np.mean(returns)
    std_return = np.std(returns)
    sharpe_ratio = (mean_return - RISK_FREE_RATE/252*FORECAST_DAYS) / std_return if std_return > 0 else 0
    
    # Value at Risk (VaR) - 5% worst case
    var_95 = np.percentile(net_final_values, 5)
    
    # Conditional VaR (CVaR/Expected Shortfall) - average of worst 5%
    cvar_95 = np.mean(net_final_values[net_final_values <= var_95])
    
    # Maximum Drawdown simulation
    max_drawdowns = []
    for path in gross_values:
        running_max = np.maximum.accumulate(path)
        drawdown = (path - running_max) / running_max
        max_drawdowns.append(np.min(drawdown))
    avg_max_drawdown = np.mean(max_drawdowns)
    
    # Probability metrics
    prob_loss = float(np.mean(net_final_values < invest_amount) * 100)
    prob_significant_loss = float(np.mean(net_final_values < invest_amount * 0.9) * 100)
    
    # Best and worst cases
    best_case = float(np.percentile(net_final_values, 95))
    worst_case = float(np.percentile(net_final_values, 5))
    
    return {
        "mean_final": float(np.mean(net_final_values)),
        "median_final": float(np.median(net_final_values)),
        "prob_profit": float(100 - prob_loss),
        "prob_loss": prob_loss,
        "prob_significant_loss": prob_significant_loss,
        "best_case": best_case,
        "worst_case": worst_case,
        "var_95": float(var_95),
        "cvar_95": float(cvar_95),
        "sharpe_ratio": float(sharpe_ratio),
        "max_drawdown": float(avg_max_drawdown * 100),
        "used_fat_tails": use_t_dist,
        "transaction_cost": transaction_cost,
        "expected_tax": float(np.mean(np.maximum(0, taxed_gains - gains))),
    }


def calculate_risk_score(results: dict, regime: dict, fundamental_score: int) -> dict:
    """
    Calculate overall risk score combining technical and fundamental analysis.
    """
    risk_score = 50  # Start neutral
    
    # Adjust based on probability of profit
    if results["prob_profit"] > 70:
        risk_score -= 15
    elif results["prob_profit"] < 40:
        risk_score += 20
    
    # Adjust based on Sharpe ratio
    if results["sharpe_ratio"] > 1.0:
        risk_score -= 10
    elif results["sharpe_ratio"] < 0:
        risk_score += 15
    
    # Adjust based on max drawdown
    if abs(results["max_drawdown"]) > 30:
        risk_score += 15
    elif abs(results["max_drawdown"]) < 15:
        risk_score -= 5
    
    # Adjust based on market regime
    if regime["regime"] == "High Volatility":
        risk_score += 10
    
    # Adjust based on fundamentals
    risk_score += (50 - fundamental_score) * 0.3
    
    # Ensure 0-100 range
    risk_score = max(0, min(100, risk_score))
    
    risk_level = (
        "Very Low" if risk_score < 20 else
        "Low" if risk_score < 40 else
        "Moderate" if risk_score < 60 else
        "High" if risk_score < 80 else
        "Very High"
    )
    
    return {
        "score": risk_score,
        "level": risk_level
    }


def generate_recommendation(ticker: str, amount: float, results: dict, 
                          fundamentals: dict, fundamental_analysis: dict,
                          regime: dict, risk_assessment: dict) -> str:
    """
    Generate comprehensive investment recommendation.
    """
    mean_val = results["mean_final"]
    gain_pct = ((mean_val - amount) / amount) * 100
    
    worst_val = results["worst_case"]
    worst_pct = ((worst_val - amount) / amount) * 100
    
    best_val = results["best_case"]
    best_pct = ((best_val - amount) / amount) * 100
    
    prob = results["prob_profit"]
    
    # Recommendation logic
    fund_score = fundamental_analysis["score"]
    risk_score = risk_assessment["score"]
    
    if prob >= 70 and fund_score >= 65 and risk_score < 50:
        recommendation = "STRONG BUY"
        reason = f"High probability of profit ({prob:.0f}%), solid fundamentals (grade {fundamental_analysis['grade']}), and manageable risk."
    elif prob >= 60 and fund_score >= 50 and risk_score < 60:
        recommendation = "BUY"
        reason = f"Good odds ({prob:.0f}%) with acceptable fundamentals and moderate risk."
    elif prob >= 50 and fund_score >= 45:
        recommendation = "HOLD/CAUTIOUS BUY"
        reason = f"Slightly favorable odds ({prob:.0f}%) but requires careful position sizing."
    elif prob >= 40:
        recommendation = "HOLD/AVOID"
        reason = f"Marginal probability ({prob:.0f}%). Better opportunities likely exist."
    else:
        recommendation = "AVOID"
        reason = f"Poor odds ({prob:.0f}%) and/or weak fundamentals. High risk of loss."
    
    # Build comprehensive message
    msg = f"{'='*40}\n"
    msg += f"{ticker} - ENHANCED ANALYSIS\n"
    msg += f"{'='*40}\n\n"
    
    msg += f"💰 FINANCIAL PROJECTION\n"
    msg += f"Investment: ${amount:,.0f}\n"
    msg += f"Expected value: ${mean_val:,.0f} ({gain_pct:+.1f}%)\n"
    msg += f"Median outcome: ${results['median_final']:,.0f}\n"
    msg += f"Best case (95%): ${best_val:,.0f} ({best_pct:+.1f}%)\n"
    msg += f"Worst case (5%): ${worst_val:,.0f} ({worst_pct:+.1f}%)\n\n"
    
    msg += f"📊 PROBABILITY ANALYSIS\n"
    msg += f"Profit probability: {prob:.1f}%\n"
    msg += f"Loss >10% probability: {results['prob_significant_loss']:.1f}%\n"
    msg += f"Simulations: {NUM_SIMULATIONS:,}\n"
    msg += f"Model: {'Fat-tailed (crash-aware)' if results['used_fat_tails'] else 'Normal distribution'}\n\n"
    
    msg += f"⚠️ RISK METRICS\n"
    msg += f"Risk level: {risk_assessment['level']} ({risk_assessment['score']:.0f}/100)\n"
    msg += f"Sharpe ratio: {results['sharpe_ratio']:.2f}\n"
    msg += f"Max drawdown: {results['max_drawdown']:.1f}%\n"
    msg += f"VaR (95%): ${results['var_95']:,.0f}\n"
    msg += f"Market regime: {regime['regime']}\n\n"
    
    msg += f"🏢 FUNDAMENTALS (Grade: {fundamental_analysis['grade']})\n"
    msg += f"Score: {fund_score}/100\n"
    if fundamentals.get("pe_ratio"):
        msg += f"P/E: {fundamentals['pe_ratio']:.1f}\n"
    if fundamentals.get("beta"):
        msg += f"Beta: {fundamentals['beta']:.2f}\n"
    if fundamental_analysis["strengths"]:
        msg += f"✓ {', '.join(fundamental_analysis['strengths'][:2])}\n"
    if fundamental_analysis["warnings"]:
        msg += f"⚠ {', '.join(fundamental_analysis['warnings'][:2])}\n"
    msg += f"\n"
    
    msg += f"💵 COSTS & TAXES\n"
    msg += f"Transaction costs: ${results['transaction_cost']:.2f}\n"
    msg += f"Expected taxes: ${results['expected_tax']:.2f}\n\n"
    
    msg += f"🎯 RECOMMENDATION: {recommendation}\n"
    msg += f"{reason}\n\n"
    
    msg += f"{'='*40}\n"
    msg += f"⚠️ DISCLAIMER: This is educational analysis\n"
    msg += f"based on historical data and statistical\n"
    msg += f"modeling. Past performance ≠ future results.\n"
    msg += f"Consult a financial advisor before investing.\n"
    msg += f"Not financial advice.\n"
    
    return msg


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
    
    # ── Twilio setup (now using env vars) ───────────────────────────
    account_sid = os.environ["TWILIO_ACCOUNT_SID"]
    auth_token = os.environ["TWILIO_TOKEN"]
    from_number = os.environ["TWILIO_FROM_NUMBER"]
    to_number = os.environ["PHONE_NUMBER"]
    twilio_client = Client(account_sid, auth_token)
    
    # ── Run enhanced analysis ────────────────────────────────────────
    print(f"Worker {job_index}: Enhanced analysis for {ticker}...")
    
    try:
        # Fetch data
        log_returns, fundamentals = fetch_stock_data(ticker)
        
        # Analyze market regime
        regime = detect_regime_changes(log_returns)
        
        # Fundamental analysis
        fundamental_analysis = analyze_fundamentals(fundamentals)
        
        # Run Monte Carlo simulation
        results = run_enhanced_monte_carlo(log_returns, invest_amount, fundamentals)
        
        # Risk assessment
        risk_assessment = calculate_risk_score(results, regime, fundamental_analysis["score"])
        
        # Generate recommendation
        message_body = generate_recommendation(
            ticker, invest_amount, results, fundamentals,
            fundamental_analysis, regime, risk_assessment
        )
        
        # ── Send SMS ─────────────────────────────────────────────────
        message = twilio_client.messages.create(
            body=message_body,
            from_=from_number,
            to=to_number,
        )
        
        print(f"Worker {job_index} ({ticker}): SMS sent, SID={message.sid}")
        print(message_body)
        
    except Exception as e:
        error_msg = f"Error analyzing {ticker}: {str(e)}"
        print(error_msg)
        # Send error notification
        twilio_client.messages.create(
            body=f"❌ {ticker} analysis failed: {str(e)}",
            from_=from_number,
            to=to_number,
        )


if __name__ == "__main__":
    main()

# Made with Bob
