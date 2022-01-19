import numpy as np
import pandas as pd
import pyfolio as pf

class CrossAssetMomentum():
    def __init__(
        self, 
        prices, 
        lookback_period, 
        holding_period, 
        n_selection, 
        cost=0.001, 
        signal_method='dm', # 기본값: 듀얼 모멘텀
        weightings='emv', 
        long_only=False, 
        show_analytics=True
        ):

        # price dataframe을 return으로 바꿔준다. = 일별수익률
        self.returns = self.get_returns(prices)
        # 그리고 holding_period를 감안해 shift를 걸어준다. = holding_period 만큼의 수익률
        self.holding_returns = self.get_holding_returns(prices, holding_period)

        if signal_method == 'am': # 절대 모멘텀. 시장/섹터 neutral 없이 + 수익률이면 long, -면 short
            self.signal = self.absolute_momentum(prices, lookback_period, long_only)
        elif signal_method == 'rm': # 상대 모멘텀. 그룹 내 순위로 winner/loser 분류. 
            self.signal = self.relative_momentum(prices, lookback_period, n_selection, long_only)
        elif signal_method == 'dm': # 절대 모멘텀과 상대 모멘텀을 차례대로 operation 걸어줌. 
            self.signal = self.dual_momentum(prices, lookback_period, n_selection, long_only)

        if weightings == 'ew': # 동일 가중 배분
            self.cs_risk_weight = self.equal_weight(self.signal)
        elif weightings == 'emv': # 개별 ii에서 최대 손실가능한 weight만 주는 배분
            self.cs_risk_weight = self.equal_marginal_volatility(self.returns, self.signal)

        # rebalance weight 준다는 것은 한 번에 다 사지 않고 holding period동안 천천히 조금씩 사는 것. 
        self.rebalance_weight = 1 / holding_period
        #  거래비용. 하지만 여기선 시그널에 그냥 bool 씌우고 * 0.001 같이, 주문 물량이나 유동성 등 상관 없이 단순화시켰다. 
        self.cost = self.transaction_cost(self.signal, cost)

        # portfolio returns without cash. 현금성 자산 제외하고 single asset에 대한 간단한 총 수익 (또는 수익률?)
        self.port_rets_wo_cash = self.backtest(
            self.holding_returns, 
            self.signal, 
            self.cost, 
            self.rebalance_weight, 
            self.cs_risk_weight
            )
        
        # 종적(time series) 방향으로 risk weight를 정해준다...? ########## 질문
        self.ts_risk_weight = self.volatility_targeting(self.port_rets_wo_cash)
        
        # 최종적인 portfolio return: 최대 인내 가능 risk weight를 줬을 때
        self.port_rets = self.port_rets_wo_cash * self.ts_risk_weight
        
        if show_analytics == True:
            self.performance_analytics(self.port_rets)                          
                
    def get_returns(self, prices):
        """Returns the historical daily returns
        
        Paramters
        ---------
        prices : dataframe
            Historical daily prices
            
        Returns
        -------
        returns : dataframe
            Historical daily returns
        """
        returns = prices.pct_change().fillna(0)
        return returns

    def get_holding_returns(self, prices, holding_period):
        """Returns the periodic returns for each holding period
        
        Paramters
        ---------
        returns : dataframe
            Historical daily returns
        holding_period : int
            Holding Period
            
        Returns
        -------
        holding_returns : dataframe
            Periodic returns for each holding period. Pulled by N (holding_period) days forward to keep inline with trading signals.
        """
        holding_returns = prices.pct_change(periods=holding_period).shift(-holding_period).fillna(0)
        return holding_returns

    def absolute_momentum(self, prices, lookback, long_only=False):
        """Returns Absolute Momentum Signals
        
        Parameters
        ----------
        prices : dataframe
            Historical daily prices
        lookback : int
            Lookback window for signal generation
        long_only : bool, optional
            Indicator for long-only momentum, False is default value
        
        Returns
        -------
        returns : dataframe
            Absolute momentum signals     
        """    
        returns = prices.pct_change(periods=lookback).fillna(0)
        long_signal = (returns > 0).applymap(self.bool_converter) # 그냥 단순히 0보다 크면 산다 1
        short_signal = -(returns < 0).applymap(self.bool_converter) # 그냥 단순히 0보다 작으면 -1
        if long_only == True:
            signal = long_signal
        else:
            signal = long_signal + short_signal
        return signal
    
    def relative_momentum(self, prices, lookback, n_selection, long_only=False):
        """Returns Relative Momentum Signals
        
        Parameters
        ----------
        prices : dataframe
            Historical daily prices
        lookback : int
            Lookback Window for Signal Generation
        n_selection : int
            Number of asset to be traded at one side
        long_only : bool, optional
            Indicator for long-only momentum, False is default value
        
        Returns
        -------
        returns : dataframe
            Relative momentum signals     
        """
        returns = prices.pct_change(periods=lookback).fillna(0)
        rank = returns.rank(axis=1, ascending=False) # di 에서 모든 종목 rank
        long_signal = (rank <= n_selection).applymap(self.bool_converter) # winners 골라서 산다 1
        short_signal = -(rank >= len(rank.columns) - n_selection + 1).applymap(self.bool_converter) # losers 골라서 판다 1
        if long_only == True:
            signal = long_signal
        else:
            signal = long_signal + short_signal
        return signal
    
    def dual_momentum(self, prices, lookback, n_selection, long_only=False):
        """Returns Dual Momentum Signals
        
        Parameters
        ----------
        prices : dataframe
            Historical daily prices
        lookback : int
            Lookback Window for Signal Generation
        n_selection : int
            Number of asset to be traded at one side
        long_only : bool, optional
            Indicator for long-only momentum, False is default value
        
        Returns
        -------
        returns : dataframe
            Dual momentum signals     
        """
        abs_signal = self.absolute_momentum(prices, lookback, long_only) # 먼저 abs 기준으로 시그널 내놓고
        rel_signal = self.relative_momentum(prices, lookback, n_selection, long_only) # rel 기준으로도 시그널을 내놓고
        signal = (abs_signal == rel_signal).applymap(self.bool_converter) * abs_signal # 두 조건 모두 만족하는 것만 골라서 남긴다. 
        return signal

    def equal_weight(self, signal):
        """Returns Equal Weights

        Parameters
        ----------
        signal : dataframe
            Momentum signal dataframe

        Returns
        -------
        weight : dataframe
            Equal weights for cross-asset momentum portfolio
        """
        total_signal = 1 / abs(signal).sum(axis=1)
        total_signal.replace([np.inf, -np.inf], 0, inplace=True)
        weight = pd.DataFrame(index=signal.index, columns=signal.columns).fillna(value=1)
        weight = weight.mul(total_signal, axis=0)
        return weight

    def equal_marginal_volatility(self, returns, signal):
        """Returns Equal Marginal Volatility (Inverse Volatility)
        
        Parameters
        ----------
        returns : dataframe
            Historical daily returns
        signal : dataframe
            Momentum signal dataframe

        Returns
        -------
        weight : dataframe
            Weights using equal marginal volatility

        """
        vol = (returns.rolling(252).std() * np.sqrt(252)).fillna(0)
        vol_signal = vol * abs(signal)
        inv_vol = 1 / vol_signal
        inv_vol.replace([np.inf, -np.inf], 0, inplace=True)
        weight = inv_vol.div(inv_vol.sum(axis=1), axis=0).fillna(0)
        return weight

    def volatility_targeting(self, returns, target_vol=0.01):
        """Returns Weights based on Vol Target
        
        Parameters
        ----------
        returns : dataframe
            Historical daily returns of backtested portfolio
        target_vol : float, optional
            Target volatility, Default target volatility is 1%

        Returns
        -------
        weights : dataframe
            Weights using equal marginal volatility

        """
        weight = target_vol / (returns.rolling(252).std() * np.sqrt(252)).fillna(0)
        weight.replace([np.inf, -np.inf], 0, inplace=True)
        weight = weight.shift(1).fillna(0)
        return weight

    def transaction_cost(self, signal, cost=0.001):
        """Returns Transaction Costs
        
        Parameters
        ----------
        signal : dataframe
            Momentum signal dataframe
        cost : float, optional
            Transaction cost (%) per each trade. The default is 0.001.

        Returns
        -------
        cost_df : dataframe
            Transaction cost dataframe

        """
        cost_df = (signal.diff() != 0).applymap(self.bool_converter) * cost
        cost_df.iloc[0] = 0
        return cost_df
    
    def backtest(self, returns, signal, cost, rebalance_weight, weighting):
        """Returns Portfolio Returns without Time-Series Risk Weights

        Parameters
        ----------
        returns : dataframe
            Historical daily returns
        signal : dataframe
            Momentum signal dataframe
        cost : dataframe
            Transaction cost dataframe
        rebalance_weight : float
            Rebalance weight
        weighting : dataframe
            Weighting dataframe

        Returns
        -------
        port_rets : dataframe
            Portfolio returns dataframe without applying time-series risk model

        """
        port_rets = ((signal * returns - cost) * rebalance_weight * weighting).sum(axis=1)
        return port_rets

    def performance_analytics(self, returns):
        """Returns Perforamnce Analytics using pyfolio package

        Parameters
        ----------
        returns : series
            backtestd portfolio returns

        Returns
        -------
        None

        """
        pf.create_returns_tear_sheet(returns)

    def bool_converter(self, bool_var):
        """Returns Integer Value from Boolean Value

        Parameters
        ----------
        bool_var : boolean
            Boolean variables representing trade signals

        Returns
        -------
        result : int
            Integer variables representing trade signals

        """
        if bool_var == True:
            result = 1
        elif bool_var == False:
            result = 0
        return result

def get_price_df(url):
    """Returns price dataframe from given URL

    Parameters
    ----------
    url : string
        URL which contains dataset

    Returns
    -------
    df : dataframe
        Imported price dataframe from URL
    """
    df = pd.read_csv(url).dropna()
    df.index = pd.to_datetime(df['Date'])
    df = df.drop(columns=['Date'])
    return df

if __name__ == "__main__":
    url = 'https://raw.githubusercontent.com/davidkim0523/Momentum/main/Data.csv'
    prices = get_price_df(url)
    lookback_period = 120
    holding_period = 20
    n_selection = 19
    momentum = CrossAssetMomentum(prices, lookback_period, holding_period, n_selection)
