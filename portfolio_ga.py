from typing import Dict
import numpy as np
import pandas as pd
from dataclasses import dataclass
import logging

from market_analysis import generate_realistic_market_data
from plots import plot_evolution_summary

logger = logging.getLogger(__name__)

@dataclass
class GAParameters:
    # Population size (default 50) 
    population_size: int = 50
    # Crossover rate 100% (two-point crossover) 
    crossover_rate: float = 1.0
    # Mutation rate 1% per gene 
    mutation_rate: float = 0.01
    # Number of generations (e.g., 300 for larger universes)
    num_generations: int = 300
    # Tournament size = 3 for selection pressure
    tournament_size: int = 3
    # Elitism: carry top 2 individuals each generation
    elitism_count: int = 2
    # Stop if no improvement for 20 generations
    convergence_generations: int = 20

@dataclass
class MarketParameters:
    # Annual risk-free rate (2%) 
    risk_free_rate: float = 0.02      
    # Transaction fee (0.015%) and tax (0.3%) 
    transaction_fee_rate: float = 0.00015
    transaction_tax_rate: float = 0.003
    # Window for rolling beta: 36 months, min 12 periods
    beta_window: int = 36
    min_beta_periods: int = 12

class PortfolioGA:
    def __init__(self, stock_data: pd.DataFrame, market_data: pd.DataFrame,
                 ga_params: GAParameters=None,
                 market_params: MarketParameters=None,
                 use_capm: bool=True,
                 lookback: int=12):
        # Data preparation: compute monthly returns and align indices
        self.stock_data   = stock_data.copy()
        self.market_data  = market_data.copy()
        self.ga_params    = ga_params    or GAParameters()
        self.market_params = market_params or MarketParameters()
        self.use_capm     = use_capm    # toggle CAPM mispricing
        self.lookback     = lookback    # months for historical averages
        self.num_stocks   = stock_data.shape[1]

        # Monthly percent changes, fill NA → 0
        self.stock_returns  = self.stock_data.pct_change().fillna(0)
        self.market_returns = self.market_data.pct_change().fillna(0)
        common = self.stock_returns.index.intersection(self.market_returns.index)
        self.stock_returns  = self.stock_returns.loc[common]
        self.market_returns = self.market_returns.loc[common]
        self.analysis_period = 1  # rebalance monthly -> in research paper specified that its most efficient
        print(f"Initialized with {self.num_stocks} stocks")

    def calculate_capm_expected_return(self, beta: float, market_return: float) -> float:
        """
        CAPM expected return per Security Market Line: E(R_i) = R_f + β_i (E(R_m)-R_f)
        """
        risk_free_monthly = self.market_params.risk_free_rate / 12
        return risk_free_monthly + beta * (market_return - risk_free_monthly)

    def calculate_rolling_beta(self, stock_series: pd.Series, market_series: pd.Series, end_idx: int) -> float:
        """
        3-year rolling beta: window=36 mo (min 12), β = Cov(R_i,R_m)/Var(R_m)
        """
        window = min(self.market_params.beta_window, end_idx + 1)
        window = max(window, self.market_params.min_beta_periods)
        start = max(0, end_idx - window + 1)
        stock_ret = stock_series.iloc[start:end_idx+1]
        market_ret = market_series.iloc[start:end_idx+1]
        if len(stock_ret) < self.market_params.min_beta_periods:
            return 1.0
        market_var = np.var(market_ret)
        if market_var < 1e-10:
            return 1.0
        beta = np.cov(stock_ret, market_ret)[0,1] / market_var
        return np.clip(beta, -3.0, 3.0)

    def calculate_capm_mispricing(self, stock_series: pd.Series, market_series: pd.Series, end_idx: int) -> float:
        """
        Mispricing = actual - CAPM-expected over lookback
        """
        if not self.use_capm:
            return 0.0
        beta = self.calculate_rolling_beta(stock_series, market_series, end_idx)
        start = max(0, end_idx - self.lookback + 1)
        market_hist_return = market_series.iloc[start:end_idx+1].mean()
        expected = self.calculate_capm_expected_return(beta, market_hist_return)
        actual = stock_series.iloc[start:end_idx+1].mean()
        return actual - expected

    def two_point_crossover(
        self,
        parent1: np.ndarray,
        parent2: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Two-point crossover at 100% rate 
        """
        if np.random.rand() > self.ga_params.crossover_rate:
            return parent1.copy(), parent2.copy()
        pts = np.random.choice(range(1, self.num_stocks), size=2, replace=False)
        p1, p2 = sorted(pts)
        o1, o2 = parent1.copy(), parent2.copy()
        o1[p1:p2], o2[p1:p2] = parent2[p1:p2], parent1[p1:p2]
        return o1, o2

    def calculate_fitness(
        self,
        chromosome: np.ndarray,
        end_idx: int
    ) -> float:
        """
        Fitness = (ROI - Risk-free-rate) / Risk + Portfolio CAPM
        """
        idxs = np.where(chromosome == 1)[0]
        if len(idxs) == 0:
            return -np.inf
        start = max(0, end_idx - self.lookback + 1)
        rets = self.stock_returns.iloc[start:end_idx+1, idxs]
        cost_pct = (self.market_params.transaction_fee_rate + self.market_params.transaction_tax_rate)
        net_rets = rets - cost_pct
        avg_return = net_rets.mean(axis=0).mean() * 100
        risk = max(net_rets.mean(axis=1).std(), 1e-6)
        if self.use_capm:
            misps = [
                self.calculate_capm_mispricing(
                    self.stock_returns.iloc[:, i],
                    self.market_returns.iloc[:, 0],
                    end_idx
                ) for i in idxs
            ]
            avg_mispricing = np.mean(misps)
        else:
            avg_mispricing = 0.0
        rf_pct = self.market_params.risk_free_rate * 100
        return (avg_return - rf_pct) / risk + avg_mispricing

    def initialize_population(self):
        """
        Binary encoding of stocks 
        """
        pop = np.zeros((self.ga_params.population_size, self.num_stocks), dtype=int)
        for i in range(self.ga_params.population_size):
            k = np.random.randint(1, self.num_stocks + 1)
            sel = np.random.choice(self.num_stocks, k, replace=False)
            pop[i, sel] = 1
        return pop

    def tournament_selection(self, pop, fits):
        """
        Tournament selection (size 3)
        """
        idxs = np.random.choice(len(pop), self.ga_params.tournament_size, replace=False)
        return pop[idxs[np.argmax(fits[idxs])]].copy()

    def mutate(self, chrom):
        """
        Bit-flip mutation 1% rate 
        """
        m = chrom.copy()
        for i in range(self.num_stocks):
            if np.random.rand() < self.ga_params.mutation_rate:
                m[i] = 1 - m[i]
        if m.sum() == 0:
            m[np.random.randint(self.num_stocks)] = 1
        return m

    def run_optimization(self, end_idx):
        """
        Main GA loop with elitism & convergence check
        """
        pop = self.initialize_population()
        best_fit, best_chr, no_imp, history = -np.inf, None, 0, []
        for gen in range(self.ga_params.num_generations):
            fits = np.array([self.calculate_fitness(ind, end_idx) for ind in pop])
            i_best = np.nanargmax(fits)
            if fits[i_best] > best_fit:
                best_fit, best_chr, no_imp = fits[i_best], pop[i_best].copy(), 0
            else:
                no_imp += 1
            history.append({'gen': gen, 'best': fits.max(), 'avg': np.nanmean(fits[np.isfinite(fits)]), 'worst': np.nanmin(fits[np.isfinite(fits)])})
            if no_imp >= self.ga_params.convergence_generations:
                print(f"[v2] Converged at gen {gen}")
                break
            new_pop = []
            elites = np.argsort(fits)[-self.ga_params.elitism_count:]
            for e in elites:
                new_pop.append(pop[e].copy())
            while len(new_pop) < self.ga_params.population_size:
                p1 = self.tournament_selection(pop, fits)
                p2 = self.tournament_selection(pop, fits)
                o1, o2 = self.two_point_crossover(p1, p2)
                new_pop.extend([self.mutate(o1), self.mutate(o2)])
            pop = np.array(new_pop[:self.ga_params.population_size])
        return best_chr, history

    def backtest_strategy(self, start_date: str, end_date: str, initial_investment: float) -> Dict:
        """
        Backtest monthly strategy over period using GA selections
        """
        print(f"Backtesting strategy from {start_date} to {end_date}")
        start_idx = self.stock_data.index.get_loc(start_date)
        end_idx   = self.stock_data.index.get_loc(end_date)
        portfolio_value, prev_weights = initial_investment, np.zeros(self.num_stocks)
        results = { 'dates': [], 'values': [], 'selected_stocks': [], 'returns': [], 'fitness_history': [], 'optimization_histories': [] }
        current_idx = max(start_idx, self.lookback)
        while current_idx + self.analysis_period <= end_idx:
            best_chr, opt_hist = self.run_optimization(current_idx)
            fitness = self.calculate_fitness(best_chr, current_idx)
            idxs = np.where(best_chr == 1)[0]
            next_idx = current_idx + self.analysis_period
            if len(idxs) > 0:
                rets = self.stock_returns.iloc[next_idx, idxs]
                port_ret = rets.mean()
                w_new = best_chr / best_chr.sum()
                turnover = np.abs(w_new - prev_weights).sum()
                cost_pct = turnover * (self.market_params.transaction_fee_rate + self.market_params.transaction_tax_rate)
                port_ret -= cost_pct
                prev_weights = w_new
            else:
                port_ret = 0.0
            portfolio_value *= (1 + port_ret)
            date = self.stock_data.index[next_idx]
            results['dates'].append(date)
            results['values'].append(portfolio_value)
            results['selected_stocks'].append(len(idxs))
            results['returns'].append(port_ret)
            results['fitness_history'].append(fitness)
            results['optimization_histories'].append(opt_hist)
            print(f"Month: {date.strftime('%Y-%m')}, Selected: {len(idxs)}, Return: {port_ret:.2%}, Fitness: {fitness:.4f}, Value: ${portfolio_value:,.2f}")
            current_idx += self.analysis_period
        return results

if __name__ == "__main__":
    print("starting algorithm...")
    stocks, market = generate_realistic_market_data('2010-01-01', '2020-12-31', 160)
    print("generated market data")
    ga = PortfolioGA(stocks, market, use_capm=True, lookback=12)
    res = ga.backtest_strategy('2015-01-31', '2020-12-31', 100_000_000)
    print(f"Final portfolio value: {res['values'][-1]:.2f}")
    plot_evolution_summary(res, market)
