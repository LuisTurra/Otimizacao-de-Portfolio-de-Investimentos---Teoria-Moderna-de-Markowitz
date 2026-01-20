import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import requests
from typing import Tuple, List, Dict
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

# ==============================
# CONFIGURAÇÕES DA PÁGINA
# ==============================
st.set_page_config(page_title="Otimização de Portfólio com ML", layout="wide")
st.title("🚀 Otimização de Portfólio com Machine Learning - Markowitz Enhanced")
st.markdown("""
Dashboard separado para versão avançada: compara média histórica vs. Random Forest vs. XGBoost para prever retornos futuros.
Ambos os modelos usam regularização forte para evitar overfitting extremo.
Inclui validação walk-forward, feature importance e comparação direta.
""")

# ==============================
# FUNÇÕES AUXILIARES
# ==============================

def get_risk_free_rate() -> float:
    """Busca Selic meta via API BCB ou fallback."""
    try:
        url = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.432/dados/ultimos/1?formato=json"
        data = requests.get(url, timeout=10).json()
        return float(data[0]['valor']) / 100
    except Exception:
        return 0.149


def download_single_ticker(ticker: str, period: str) -> pd.Series:
    """Baixa preços ajustados (Close) de um ticker."""
    try:
        hist = yf.Ticker(ticker).history(period=period)
        if hist.empty or 'Close' not in hist.columns:
            return None
        return hist['Close']
    except Exception:
        return None


def download_data_robust(tickers: List[str], benchmark: str, period: str) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """Download robusto ticker por ticker."""
    valid_prices = {}
    failed_tickers = []

    with st.spinner("Baixando dados..."):
        for ticker in tickers:
            series = download_single_ticker(ticker, period)
            if series is not None and len(series) > 252:
                valid_prices[ticker] = series
            else:
                failed_tickers.append(ticker)

        bench_series = download_single_ticker(benchmark, period)

    if len(valid_prices) < 2:
        st.error("Menos de 2 ativos válidos.")
        st.stop()

    prices_df = pd.DataFrame(valid_prices).dropna()
    return prices_df, bench_series, failed_tickers


def calculate_returns(prices_df: pd.DataFrame, bench_series: pd.Series) -> Tuple[pd.DataFrame, pd.Series, pd.Series, pd.DataFrame]:
    returns = prices_df.pct_change().dropna()
    
    if bench_series is not None:
        bench_returns = bench_series.pct_change().dropna()
        common_index = returns.index.intersection(bench_returns.index)
        returns = returns.loc[common_index]
        bench_returns = bench_returns.loc[common_index]
    else:
        bench_returns = None

    mean_returns = returns.mean() * 252
    cov_matrix = returns.cov() * 252
    return returns, bench_returns, mean_returns, cov_matrix


def add_technical_features(prices: pd.Series) -> pd.DataFrame:
    """Feature engineering em série de preços de um único ativo."""
    features = pd.DataFrame(index=prices.index)
    for lag in [1, 5, 10, 20]:
        features[f'return_lag_{lag}'] = prices.pct_change(lag)
    features['vol_20'] = prices.pct_change().rolling(20).std()
    delta = prices.diff()
    up = delta.clip(lower=0).rolling(14).mean()
    down = -delta.clip(upper=0).rolling(14).mean()
    down = down.replace(0, np.nan)
    rsi = 100 - (100 / (1 + up / down))
    features['rsi_14'] = rsi
    return features.dropna()


def prepare_ml_data(prices_df: pd.DataFrame, horizon: int = 21) -> Dict[str, pd.DataFrame]:
    data_dict = {}
    for ticker in prices_df.columns:
        asset_prices = prices_df[ticker]
        features = add_technical_features(asset_prices)
        df = pd.DataFrame(index=features.index)
        df['target'] = asset_prices.pct_change(horizon).shift(-horizon)
        df = pd.concat([df, features.add_suffix(f'_{ticker}')], axis=1)
        df = df.dropna()
        data_dict[ticker] = df
    return data_dict


@st.cache_resource
def train_models(data_dict: Dict[str, pd.DataFrame]) -> Tuple[Dict, Dict, Dict, Dict]:
    """Treina RandomForest e XGBoost por ativo com regularização."""
    rf_models = {}
    xgb_models = {}
    rf_metrics = {}
    xgb_metrics = {}
    
    for ticker, df in data_dict.items():
        X = df.drop('target', axis=1)
        y = df['target']
        
        tscv = TimeSeriesSplit(n_splits=5)
        
        # Random Forest com regularização
        rf = RandomForestRegressor(
            n_estimators=100, max_depth=6, min_samples_leaf=20,
            random_state=42, n_jobs=-1
        )
        rf_maes, rf_r2s = [], []
        for train_idx, test_idx in tscv.split(X):
            rf.fit(X.iloc[train_idx], y.iloc[train_idx])
            pred = rf.predict(X.iloc[test_idx])
            rf_maes.append(mean_absolute_error(y.iloc[test_idx], pred))
            rf_r2s.append(r2_score(y.iloc[test_idx], pred))
        rf.fit(X, y)
        rf_models[ticker] = rf
        rf_metrics[ticker] = {'MAE': np.mean(rf_maes), 'R2': np.mean(rf_r2s)}
        
        # XGBoost com regularização
        xgb = XGBRegressor(
            n_estimators=100, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=42
        )
        xgb_maes, xgb_r2s = [], []
        for train_idx, test_idx in tscv.split(X):
            xgb.fit(X.iloc[train_idx], y.iloc[train_idx])
            pred = xgb.predict(X.iloc[test_idx])
            xgb_maes.append(mean_absolute_error(y.iloc[test_idx], pred))
            xgb_r2s.append(r2_score(y.iloc[test_idx], pred))
        xgb.fit(X, y)
        xgb_models[ticker] = xgb
        xgb_metrics[ticker] = {'MAE': np.mean(xgb_maes), 'R2': np.mean(xgb_r2s)}
    
    return rf_models, xgb_models, rf_metrics, xgb_metrics


def get_ml_expected_returns(models: Dict[str, object], data_dict: Dict[str, pd.DataFrame], horizon: int) -> pd.Series:
    expected = {}
    for ticker, df in data_dict.items():
        X_latest = df.drop('target', axis=1).iloc[-1:]
        pred_horizon = models[ticker].predict(X_latest)[0]
        
        if pred_horizon > -1:
            annual_pred = (1 + pred_horizon) ** (252 / horizon) - 1
        else:
            annual_pred = pred_horizon * (252 / horizon)
        
        annual_pred = np.clip(annual_pred, -0.5, 1.0)
        expected[ticker] = annual_pred
    return pd.Series(expected)


def portfolio_performance(weights: np.ndarray, mean_returns: pd.Series, cov_matrix: pd.DataFrame, risk_free_rate: float):
    ret = np.dot(weights, mean_returns)
    risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    sharpe = (ret - risk_free_rate) / risk if risk > 0 else -np.inf
    return ret, risk, sharpe


def optimize_portfolio(mean_returns: pd.Series, cov_matrix: pd.DataFrame, risk_free_rate: float, allow_short: bool):
    num_assets = len(mean_returns)
    bounds = tuple((-1.0, 1.0) if allow_short else (0.0, 1.0) for _ in range(num_assets))
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
    initial = np.array([1.0 / num_assets] * num_assets)
    
    result = minimize(lambda w: -portfolio_performance(w, mean_returns, cov_matrix, risk_free_rate)[2],
                      initial, method='SLSQP', bounds=bounds, constraints=constraints)
    
    if not result.success:
        st.error("Otimização falhou.")
        st.stop()
    
    weights = result.x
    ret, risk, sharpe = portfolio_performance(weights, mean_returns, cov_matrix, risk_free_rate)
    return weights, ret, risk, sharpe


# ==============================
# SIDEBAR
# ==============================
st.sidebar.header("⚙️ Configurações")

available_tickers = ["PETR4.SA", "VALE3.SA", "ITUB4.SA", "BBDC4.SA", "ABEV3.SA",
                     "WEGE3.SA", "EQTL3.SA", "RENT3.SA", "LREN3.SA", "CSAN3.SA"]

with st.sidebar.expander("📈 Ativos", expanded=True):
    selected_tickers = st.multiselect("Ativos (mínimo 3)", available_tickers, default=available_tickers[:8])
    if len(selected_tickers) < 3:
        st.error("Selecione pelo menos 3 ativos.")
        st.stop()

with st.sidebar.expander("📊 Parâmetros"):
    benchmark = st.text_input("Benchmark", "^BVSP")
    period = st.selectbox("Período", ["5y", "3y"], index=0)
    horizon = st.selectbox("Horizonte previsão (dias)", [21, 63], index=0)
    allow_short = st.checkbox("Short selling", False)
    risk_free_rate = st.number_input("Taxa livre de risco", value=get_risk_free_rate(), format="%.4f")

# ==============================
# PROCESSAMENTO
# ==============================
prices_df, bench_series, failed = download_data_robust(selected_tickers, benchmark, period)
if failed:
    st.warning(f"Falharam: {', '.join(failed)}")

returns, bench_returns, hist_mean, cov_matrix = calculate_returns(prices_df, bench_series)

data_dict = prepare_ml_data(prices_df, horizon)
rf_models, xgb_models, rf_metrics, xgb_metrics = train_models(data_dict)

st.subheader("📊 Qualidade dos Modelos (walk-forward)")
col_rf, col_xgb = st.columns(2)
with col_rf:
    st.write("**Random Forest**")
    st.table(pd.DataFrame(rf_metrics).T.style.format({"MAE": "{:.4f}", "R2": "{:.4f}"}))
with col_xgb:
    st.write("**XGBoost**")
    st.table(pd.DataFrame(xgb_metrics).T.style.format({"MAE": "{:.4f}", "R2": "{:.4f}"}))

# Expected returns
rf_mean = get_ml_expected_returns(rf_models, data_dict, horizon)
xgb_mean = get_ml_expected_returns(xgb_models, data_dict, horizon)

# Otimização
st.subheader("🔄 Comparação Completa")

col_classic, col_rf, col_xgb = st.columns(3)

with col_classic:
    st.write("**Clássico (média histórica)**")
    w_hist, ret_hist, risk_hist, sharpe_hist = optimize_portfolio(hist_mean, cov_matrix, risk_free_rate, allow_short)
    st.metric("Retorno Esperado", f"{ret_hist:.2%}")
    st.metric("Risco", f"{risk_hist:.2%}")
    st.metric("Sharpe Ratio", f"{sharpe_hist:.2f}")
    st.bar_chart(pd.Series(w_hist, index=selected_tickers).round(4) * 100)

with col_rf:
    st.write("**Random Forest**")
    w_rf, ret_rf, risk_rf, sharpe_rf = optimize_portfolio(rf_mean, cov_matrix, risk_free_rate, allow_short)
    st.metric("Retorno Esperado", f"{ret_rf:.2%}")
    st.metric("Risco", f"{risk_rf:.2%}")
    st.metric("Sharpe Ratio", f"{sharpe_rf:.2f}")
    st.bar_chart(pd.Series(w_rf, index=selected_tickers).round(4) * 100)

with col_xgb:
    st.write("**XGBoost**")
    w_xgb, ret_xgb, risk_xgb, sharpe_xgb = optimize_portfolio(xgb_mean, cov_matrix, risk_free_rate, allow_short)
    st.metric("Retorno Esperado", f"{ret_xgb:.2%}")
    st.metric("Risco", f"{risk_xgb:.2%}")
    st.metric("Sharpe Ratio", f"{sharpe_xgb:.2f}")
    st.bar_chart(pd.Series(w_xgb, index=selected_tickers).round(4) * 100)

st.warning("⚠️ Previsões de ML em finanças são desafiadoras. Modelos regularizados evitam extremos, mas resultados variam por período.")
st.caption("XGBoost adicionado com regularização (max_depth=4, learning_rate=0.05). Ambos modelos leves para Streamlit Cloud.")