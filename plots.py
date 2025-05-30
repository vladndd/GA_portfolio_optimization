import matplotlib.pyplot as plt
from typing import Dict, Optional
import pandas as pd
import numpy as np


def plot_evolution_summary(backtest_results: Dict, benchmark_data: Optional[pd.DataFrame] = None):
    """
    Evolution of all key metrics
    """
    if not backtest_results['dates']:
        print("No backtest results available for plotting")
        return
        
    # Convert to pandas for easier plotting
    df = pd.DataFrame({
        'Date': backtest_results['dates'],
        'Portfolio_Value': backtest_results['values'],
        'Returns': backtest_results['returns'],
        'Selected_Stocks': backtest_results['selected_stocks'],
        'Fitness': backtest_results['fitness_history']
    })
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    
    # Calculate cumulative returns
    initial_value = backtest_results['values'][0] / (1 + backtest_results['returns'][0])
    df['Cumulative_Return'] = (df['Portfolio_Value'] / initial_value - 1) * 100
    
    # Create subplot figure
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Portfolio GA Evolution Analysis', fontsize=16, fontweight='bold')
    
    # 1. Portfolio Value Evolution
    axes[0, 0].plot(df.index, df['Portfolio_Value'], 'b-', linewidth=2, label='Portfolio Value')
    if benchmark_data is not None:
        # Align benchmark data with portfolio dates
        benchmark_aligned = benchmark_data.reindex(df.index, method='ffill')
        initial_benchmark = benchmark_aligned.iloc[0, 0]
        benchmark_value = initial_value * (benchmark_aligned.iloc[:, 0] / initial_benchmark)
        axes[0, 0].plot(df.index, benchmark_value, 'r--', linewidth=2, label='Benchmark', alpha=0.7)
    
    axes[0, 0].set_title('Portfolio Value Over Time')
    axes[0, 0].set_ylabel('Portfolio Value ($)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # 2. Monthly Returns
    axes[0, 1].bar(df.index, df['Returns'] * 100, alpha=0.7, 
                    color=['green' if x > 0 else 'red' for x in df['Returns']])
    axes[0, 1].set_title('Monthly Returns')
    axes[0, 1].set_ylabel('Return (%)')
    axes[0, 1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # 3. Number of Stocks Selected
    axes[1, 0].plot(df.index, df['Selected_Stocks'], 'purple', marker='o', 
                    markersize=4, linewidth=1.5)
    axes[1, 0].set_title('Number of Stocks Selected Over Time')
    axes[1, 0].set_ylabel('Number of Stocks')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # 4. Fitness Evolution
    axes[1, 1].plot(df.index, df['Fitness'], 'orange', linewidth=2)
    axes[1, 1].set_title('Fitness Score Over Time')
    axes[1, 1].set_ylabel('Fitness Score')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('portfolio_evolution.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Additional statistics
    print_performance_stats(df, initial_value)

def print_performance_stats(df: pd.DataFrame, initial_value: float):
    """Comprehensive performance statistics"""
    total_return = (df['Portfolio_Value'].iloc[-1] / initial_value - 1) * 100
    annual_return = ((df['Portfolio_Value'].iloc[-1] / initial_value) ** (12/len(df)) - 1) * 100
    volatility = df['Returns'].std() * np.sqrt(12) * 100
    sharpe_ratio = (df['Returns'].mean() * 12) / (df['Returns'].std() * np.sqrt(12))
    max_drawdown = ((df['Portfolio_Value'] / df['Portfolio_Value'].expanding().max()) - 1).min() * 100
    win_rate = (df['Returns'] > 0).sum() / len(df) * 100
    avg_monthly_return = df['Returns'].mean() * 100
    
    print("\n" + "="*50)
    print("PORTFOLIO PERFORMANCE STATISTICS")
    print("="*50)
    print(f"Total Return:              {total_return:.2f}%")
    print(f"Annualized Return:         {annual_return:.2f}%")
    print(f"Volatility (Annual):       {volatility:.2f}%")
    print(f"Sharpe Ratio:              {sharpe_ratio:.3f}")
    print(f"Maximum Drawdown:          {max_drawdown:.2f}%")
    print(f"Win Rate:                  {win_rate:.1f}%")
    print(f"Average Monthly Return:    {avg_monthly_return:.2f}%")
    print(f"Average Stocks Selected:   {df['Selected_Stocks'].mean():.1f}")
    print(f"Max Stocks Selected:       {df['Selected_Stocks'].max()}")
    print(f"Min Stocks Selected:       {df['Selected_Stocks'].min()}")