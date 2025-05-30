import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List

def generate_realistic_market_data(start, end, num_stocks=30):
        dates = pd.date_range(start=start, end=end, freq='ME')
        np.random.seed(42)
        # market
        m_r, d = [], {"high": False, "p":0.95}
        for _ in dates:
            if np.random.random()>d['p']: d['high']=not d['high']
            vol = 0.035*(2 if d['high'] else 1)
            r = np.random.normal(0.007, vol)
            m_r.append(np.clip(r, -0.15,0.15))
        m_p = 100*np.cumprod(1+np.array(m_r))
        # stocks
        S = np.zeros((len(dates), num_stocks))
        for i in range(num_stocks):
            beta = np.clip(np.random.normal(1,0.4),0.2,2.5)
            alpha = np.random.normal(0,0.002)
            iv = np.random.uniform(0.02,0.06)
            sr = []
            for r in m_r:
                s = alpha + beta*r + np.random.normal(0,iv)
                sr.append(np.clip(s, -0.25,0.25))
            S[:,i] = 100*np.cumprod(1+np.array(sr))
        return pd.DataFrame(S, index=dates, columns=[f'STK{i:03d}' for i in range(num_stocks)]), pd.DataFrame(m_p, index=dates, columns=['MKT'])

def analyze_market_data(stocks: pd.DataFrame, market: pd.DataFrame) -> Dict:
    """Analyze the generated market data to understand patterns"""
    
    # Calculate returns
    stock_returns = stocks.pct_change().fillna(0)
    market_returns = market.pct_change().fillna(0)
    
    print("Market Data Analysis")
    print("=" * 50)
    
    # Basic statistics
    print(f"Date range: {market.index[0]} to {market.index[-1]}")
    print(f"Number of periods: {len(market)}")
    print(f"Number of stocks: {len(stocks.columns)}")
    
    # Market return statistics
    market_ret = market_returns.iloc[:, 0]
    print(f"\nMarket Returns:")
    print(f"Mean monthly return: {market_ret.mean():.4f} ({market_ret.mean()*12:.2%} annually)")
    print(f"Volatility (monthly): {market_ret.std():.4f} ({market_ret.std()*np.sqrt(12):.2%} annually)")
    print(f"Min return: {market_ret.min():.4f}")
    print(f"Max return: {market_ret.max():.4f}")
    
    # Check for regime changes (high volatility periods)
    rolling_vol = market_ret.rolling(6).std()
    high_vol_periods = rolling_vol > rolling_vol.quantile(0.8)
    print(f"\nHigh volatility periods: {high_vol_periods.sum()} out of {len(rolling_vol)}")
    
    # Analyze last 12 months specifically
    last_12_returns = market_ret.tail(12)
    print(f"\nLast 12 months analysis:")
    print(f"Mean return: {last_12_returns.mean():.4f}")
    print(f"Volatility: {last_12_returns.std():.4f}")
    print(f"Cumulative return: {(1 + last_12_returns).prod() - 1:.2%}")
    
    # Check for systematic patterns at the end
    end_period_returns = market_ret.tail(24)  # Last 2 years
    print(f"\nLast 24 months trend:")
    print(f"Mean return: {end_period_returns.mean():.4f}")
    print(f"Positive months: {(end_period_returns > 0).sum()}/24")
    
    return {
        'market_returns': market_ret,
        'stock_returns': stock_returns,
        'high_vol_periods': high_vol_periods,
        'rolling_vol': rolling_vol
    }

def plot_market_analysis(stocks: pd.DataFrame, market: pd.DataFrame, results: Dict = None):
    """Plot comprehensive market analysis"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Market price evolution
    axes[0, 0].plot(market.index, market.iloc[:, 0])
    axes[0, 0].set_title('Market Price Evolution')
    axes[0, 0].set_ylabel('Price')
    axes[0, 0].grid(True)
    
    # 2. Market returns
    market_returns = market.pct_change().fillna(0).iloc[:, 0]
    axes[0, 1].plot(market.index[1:], market_returns[1:])
    axes[0, 1].set_title('Market Monthly Returns')
    axes[0, 1].set_ylabel('Return')
    axes[0, 1].grid(True)
    axes[0, 1].axhline(y=0, color='r', linestyle='--', alpha=0.5)
    
    # 3. Rolling volatility
    rolling_vol = market_returns.rolling(6).std()
    axes[1, 0].plot(market.index, rolling_vol)
    axes[1, 0].set_title('Rolling 6-Month Volatility')
    axes[1, 0].set_ylabel('Volatility')
    axes[1, 0].grid(True)
    
    # 4. Portfolio performance (if results provided)
    if results:
        axes[1, 1].plot(pd.to_datetime(results['dates']), results['values'])
        axes[1, 1].set_title('Portfolio Performance')
        axes[1, 1].set_ylabel('Portfolio Value')
        axes[1, 1].grid(True)
        
        # Highlight the dropdown period
        if len(results['values']) > 12:
            last_12_values = results['values'][-12:]
            last_12_dates = pd.to_datetime(results['dates'][-12:])
            axes[1, 1].plot(last_12_dates, last_12_values, 'r-', linewidth=2, label='Last 12 months')
            axes[1, 1].legend()
    else:
        # Plot some stock prices
        sample_stocks = stocks.iloc[:, :5]  # First 5 stocks
        for col in sample_stocks.columns:
            axes[1, 1].plot(stocks.index, sample_stocks[col], alpha=0.7, label=col)
        axes[1, 1].set_title('Sample Stock Prices')
        axes[1, 1].set_ylabel('Price')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.show()

# Usage in your main code:
if __name__ == "__main__":
    print("Testing original data generation:")

    
    
    stocks_orig, market_orig = generate_realistic_market_data('2010-01-01', '2020-12-31', 50)
    analysis_orig = analyze_market_data(stocks_orig, market_orig)
    
    
    # Plot comparison
    plot_market_analysis(stocks_orig, market_orig)