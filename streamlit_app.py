import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import plotly.graph_objects as go
import requests
from typing import Tuple, List

# ==============================
# CONFIGURAÇÕES DA PÁGINA
# ==============================
st.set_page_config(page_title="Otimização de Portfólio", layout="wide")
st.title("🚀 Otimização de Portfólio - Teoria Moderna de Markowitz")
st.markdown("""
Aplicação interativa para otimização de portfólio baseada na Teoria Moderna de Portfólios de Harry Markowitz.
- Calcula a **Fronteira Eficiente**
- Encontra o portfólio de **máximo Sharpe Ratio** (ajustado pela Selic)
- Inclui **Sortino Ratio** e comparação detalhada com o benchmark
- Backtesting histórico e análise de drawdown
""")

# ==============================
# FUNÇÕES AUXILIARES
# ==============================

def get_risk_free_rate() -> float:
    """
    Busca a taxa Selic meta atual via API oficial do Banco Central do Brasil (série SGS 432).
    Em caso de falha na conexão ou parsing, retorna um fallback conservador (valor aproximado de jan/2026).
    Retorna a taxa em formato decimal (ex: 0.149 para 14.9%).
    """
    try:
        url = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.432/dados/ultimos/1?formato=json"
        data = requests.get(url, timeout=10).json()
        rate = float(data[0]['valor']) / 100
        return rate
    except Exception:
        return 0.149  # fallback ≈ 14.9%


def download_single_ticker(ticker: str, period: str) -> pd.Series:
    """
    Baixa os preços ajustados (Close) de um único ticker usando yfinance.history().
    Nas versões recentes do yfinance, 'Close' já vem ajustado por dividendos e splits.
    Retorna uma pd.Series indexada por data ou None se houver falha ou dados insuficientes.
    """
    try:
        hist = yf.Ticker(ticker).history(period=period)
        if hist.empty or 'Close' not in hist.columns:
            return None
        return hist['Close']
    except Exception:
        return None


def download_data_robust(tickers: List[str], benchmark: str, period: str) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """
    Realiza download robusto, baixando cada ticker individualmente para evitar falhas em lote.
    Alinha todos os preços por datas comuns (dropna).
    Retorna:
    - DataFrame com preços dos ativos válidos
    - Série de preços do benchmark (ou None se falhar)
    - Lista de tickers que falharam no download
    """
    valid_prices = {}
    failed_tickers = []

    with st.spinner("Baixando dados de mercado..."):
        for ticker in tickers:
            series = download_single_ticker(ticker, period)
            if series is not None and len(series) > 100:
                valid_prices[ticker] = series
            else:
                failed_tickers.append(ticker)

        bench_series = download_single_ticker(benchmark, period)

    if len(valid_prices) < 2:
        st.error("Erro: menos de 2 ativos válidos foram baixados. Verifique os tickers.")
        st.stop()

    prices_df = pd.DataFrame(valid_prices).dropna()
    return prices_df, bench_series, failed_tickers


def calculate_returns(prices_df: pd.DataFrame, bench_series: pd.Series) -> Tuple[pd.DataFrame, pd.Series, pd.Series, pd.DataFrame]:
    """
    Calcula retornos diários dos ativos e do benchmark.
    Alinha os índices para datas comuns.
    Anualiza retorno médio e matriz de covariância (multiplicando por 252 dias úteis).
    Retorna retornos diários, retornos do benchmark, retorno médio anualizado e covariância anualizada.
    """
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


def portfolio_performance(weights: np.ndarray, mean_returns: pd.Series, cov_matrix: pd.DataFrame, risk_free_rate: float) -> Tuple[float, float, float]:
    """
    Calcula as métricas básicas de um portfólio dado os pesos:
    - Retorno esperado anualizado
    - Risco (volatilidade) anualizado
    - Sharpe Ratio (retorno excedente / risco)
    """
    portfolio_return = np.dot(weights, mean_returns)
    portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    sharpe = (portfolio_return - risk_free_rate) / portfolio_risk if portfolio_risk > 0 else -np.inf
    return portfolio_return, portfolio_risk, sharpe


def negative_sharpe(weights: np.ndarray, mean_returns: pd.Series, cov_matrix: pd.DataFrame, risk_free_rate: float) -> float:
    """
    Função objetivo para minimização no scipy.optimize.
    Retorna o negativo do Sharpe Ratio (para que a minimização maximize o Sharpe).
    """
    return -portfolio_performance(weights, mean_returns, cov_matrix, risk_free_rate)[2]


def optimize_max_sharpe(mean_returns: pd.Series, cov_matrix: pd.DataFrame, risk_free_rate: float, allow_short: bool) -> Tuple[np.ndarray, float, float, float]:
    """
    Executa a otimização para encontrar o portfólio de máximo Sharpe Ratio.
    Usa método SLSQP com restrição de soma de pesos = 1.
    Bounds configuráveis para permitir ou não short selling.
    Retorna pesos ótimos e as métricas correspondentes.
    """
    num_assets = len(mean_returns)
    bounds = tuple((-1.0, 1.0) if allow_short else (0.0, 1.0) for _ in range(num_assets))
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
    initial_weights = np.array([1.0 / num_assets] * num_assets)
    
    result = minimize(negative_sharpe, initial_weights,
                      args=(mean_returns, cov_matrix, risk_free_rate),
                      method='SLSQP', bounds=bounds, constraints=constraints)
    
    if not result.success:
        st.error("Falha na otimização do máximo Sharpe. Tente ajustar parâmetros.")
        st.stop()
    
    weights = result.x
    ret, risk, sharpe = portfolio_performance(weights, mean_returns, cov_matrix, risk_free_rate)
    return weights, ret, risk, sharpe


def calculate_sortino(daily_returns: pd.Series, annual_return: float, risk_free_rate: float) -> float:
    """
    Calcula o Sortino Ratio: penaliza apenas o risco downside (retornos negativos).
    Downside deviation = desvio padrão anualizado apenas dos retornos abaixo de zero.
    """
    downside = daily_returns[daily_returns < 0]
    downside_dev = np.sqrt(np.mean(downside**2)) * np.sqrt(252) if len(downside) > 0 else 0.001
    return (annual_return - risk_free_rate) / downside_dev if downside_dev > 0 else 0.0


def calculate_max_drawdown(cum_returns_normalized: pd.Series) -> float:
    """
    Calcula o máximo drawdown (maior perda percentual do pico ao vale).
    Recebe série cumulativa já normalizada (iniciando em 1.0).
    Retorna o valor em percentual negativo (ex: -35.2%).
    """
    rolling_max = cum_returns_normalized.cummax()
    drawdown = (cum_returns_normalized - rolling_max) / rolling_max
    return drawdown.min() * 100


def generate_efficient_frontier(mean_returns: pd.Series, cov_matrix: pd.DataFrame, risk_free_rate: float, allow_short: bool) -> Tuple[List[float], List[float]]:
    """
    Gera pontos da Fronteira Eficiente minimizando risco para vários níveis alvo de retorno.
    Usa otimização sequencial para 50 pontos.
    """
    num_assets = len(mean_returns)
    bounds = tuple((-1.0, 1.0) if allow_short else (0.0, 1.0) for _ in range(num_assets))
    constraints_base = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
    initial_weights = np.array([1.0 / num_assets] * num_assets)
    
    target_returns = np.linspace(mean_returns.min(), mean_returns.max() * 1.1, 50)
    frontier_risks = []
    frontier_rets = []
    
    for tr in target_returns:
        constraints = constraints_base + [{'type': 'eq', 'fun': lambda w: np.dot(w, mean_returns) - tr}]
        result = minimize(lambda w: portfolio_performance(w, mean_returns, cov_matrix, risk_free_rate)[1],
                          initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
        if result.success:
            ret, risk, _ = portfolio_performance(result.x, mean_returns, cov_matrix, risk_free_rate)
            frontier_rets.append(ret)
            frontier_risks.append(risk)
    
    return frontier_risks, frontier_rets


def create_frontier_plot(frontier_risks: List[float], frontier_rets: List[float], ms_risk: float, ms_ret: float, ms_sharpe: float) -> go.Figure:
    """Cria gráfico interativo da Fronteira Eficiente com destaque do portfólio ótimo."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=frontier_risks, y=frontier_rets, mode='lines', name='Fronteira Eficiente',
                             line=dict(color='royalblue', width=3)))
    fig.add_trace(go.Scatter(x=[ms_risk], y=[ms_ret], mode='markers',
                             marker=dict(color='red', size=14, symbol='star'),
                             name=f'Máximo Sharpe ({ms_sharpe:.2f})'))
    fig.update_layout(title="Fronteira Eficiente", xaxis_title="Risco Anualizado", yaxis_title="Retorno Esperado Anual",
                      template="plotly_dark", hovermode="closest")
    return fig


def create_backtest_plot(port_cum: pd.Series, bench_cum: pd.Series) -> go.Figure:
    """Gráfico de retorno cumulativo normalizado para 100% no início."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=port_cum.index, y=port_cum.values, name="Portfólio Otimizado", line=dict(width=3)))
    if bench_cum is not None:
        fig.add_trace(go.Scatter(x=bench_cum.index, y=bench_cum.values, name="Benchmark (Ibovespa)", line=dict(dash='dash')))
    fig.update_layout(title="Backtesting - Retorno Cumulativo (%)", xaxis_title="Data", yaxis_title="Retorno Cumulativo (%)",
                      template="plotly_dark")
    return fig


def create_drawdown_plot(port_cum: pd.Series, bench_cum: pd.Series, port_max_dd: float, bench_max_dd: float) -> go.Figure:
    """
    Gráfico de drawdown melhorado:
    - Linhas sólidas (sem preenchimento para evitar sobreposição confusa)
    - Cores contrastantes: vermelho para portfólio, cinza escuro para benchmark
    - Linhas horizontais marcando o máximo drawdown de cada um
    - Anotações com valores exatos do máximo drawdown
    - Eixo Y invertido não é necessário (valores negativos = perda)
    """
    # Calcula drawdown em %
    port_dd = ((port_cum / port_cum.cummax()) - 1) * 100
    bench_dd = ((bench_cum / bench_cum.cummax()) - 1) * 100 if bench_cum is not None else None
    
    fig = go.Figure()
    
    # Portfólio otimizado
    fig.add_trace(go.Scatter(
        x=port_dd.index, y=port_dd.values,
        name=f"Portfólio Otimizado (Máx: {port_max_dd:.1f}%)",
        line=dict(color='crimson', width=3),
        hovertemplate="Data: %{x}<br>Drawdown: %{y:.1f}%"
    ))
    
    # Linha horizontal do máximo drawdown do portfólio
    fig.add_hline(y=port_max_dd, line_dash="dot", line_color="crimson",
                  annotation_text=f"Máx Drawdown Portfólio: {port_max_dd:.1f}%", 
                  annotation_position="bottom right")
    
    # Benchmark
    if bench_dd is not None:
        fig.add_trace(go.Scatter(
            x=bench_dd.index, y=bench_dd.values,
            name=f"Benchmark (Máx: {bench_max_dd:.1f}%)",
            line=dict(color='gray', width=2, dash='dash'),
            hovertemplate="Data: %{x}<br>Drawdown: %{y:.1f}%"
        ))
        fig.add_hline(y=bench_max_dd, line_dash="dot", line_color="gray",
                      annotation_text=f"Máx Drawdown Benchmark: {bench_max_dd:.1f}%", 
                      annotation_position="top right")
    
    fig.update_layout(
        title="Drawdown ao Longo do Tempo (%)<br><sub>Quanto mais próximo de 0%, melhor (menor perda acumulada do pico)</sub>",
        xaxis_title="Data",
        yaxis_title="Drawdown (%)",
        template="plotly_dark",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig


# ==============================
# SIDEBAR - CONFIGURAÇÕES
# ==============================
st.sidebar.header("⚙️ Configurações do Portfólio")

available_tickers = [
    "PETR4.SA", "VALE3.SA", "ITUB4.SA", "BBDC4.SA", "ABEV3.SA",
    "WEGE3.SA", "EQTL3.SA", "RENT3.SA", "LREN3.SA", "CSAN3.SA",
    "SUZB3.SA", "GGBR4.SA", "TAEE11.SA", "BBAS3.SA", "MGLU3.SA"
]

with st.sidebar.expander("📈 Seleção de Ativos", expanded=True):
    st.write("Escolha os ativos do portfólio:")
    selected_tickers = st.multiselect(
        "Ativos principais (mínimo 2)",
        options=available_tickers,
        default=available_tickers[:10]
    )
    
    custom_tickers = st.text_input("Tickers adicionais (separados por vírgula)")
    if custom_tickers:
        extra = [t.strip().upper() for t in custom_tickers.split(",") if t.strip()]
        selected_tickers += extra

    if len(selected_tickers) < 2:
        st.error("Selecione pelo menos 2 ativos.")
        st.stop()

with st.sidebar.expander("📊 Benchmark e Período"):
    benchmark = st.text_input("Ticker do Benchmark", value="^BVSP")
    period = st.selectbox("Período histórico", ["5y", "3y", "10y", "max"], index=0)

with st.sidebar.expander("⚖️ Parâmetros Avançados"):
    allow_short = st.checkbox("Permitir short selling (pesos negativos)", value=False)
    risk_free_rate = st.number_input(
        "Taxa livre de risco anual (Selic automática)",
        min_value=0.0, value=get_risk_free_rate(), step=0.001, format="%.4f"
    )

# ==============================
# PROCESSAMENTO
# ==============================
prices_df, bench_series, failed_tickers = download_data_robust(selected_tickers, benchmark, period)

if failed_tickers:
    st.sidebar.warning(f"Tickers falhados: {', '.join(failed_tickers)}")
st.sidebar.info(f"Ativos utilizados: {len(prices_df.columns)}")

returns, bench_returns, mean_returns, cov_matrix = calculate_returns(prices_df, bench_series)

weights, ms_ret, ms_risk, ms_sharpe = optimize_max_sharpe(mean_returns, cov_matrix, risk_free_rate, allow_short)
port_daily_returns = returns @ weights
sortino = calculate_sortino(port_daily_returns, ms_ret, risk_free_rate)
frontier_risks, frontier_rets = generate_efficient_frontier(mean_returns, cov_matrix, risk_free_rate, allow_short)

# Backtesting e métricas
port_cum = (1 + port_daily_returns).cumprod() * 100
port_cum = port_cum / port_cum.iloc[0] * 100
total_return_port = port_cum.iloc[-1] - 100

port_cum_norm = (1 + port_daily_returns).cumprod()  # normalizado para 1.0
port_max_dd = calculate_max_drawdown(port_cum_norm)

bench_cum = None if bench_returns is None else (1 + bench_returns).cumprod() * 100
if bench_cum is not None:
    bench_cum = bench_cum / bench_cum.iloc[0] * 100
    total_return_bench = bench_cum.iloc[-1] - 100
    
    bench_cum_norm = (1 + bench_returns).cumprod()
    bench_max_dd = calculate_max_drawdown(bench_cum_norm)
else:
    total_return_bench = bench_max_dd = None

# Métricas do benchmark
if bench_returns is not None:
    bench_ret = bench_returns.mean() * 252
    bench_risk = bench_returns.std() * np.sqrt(252)
    bench_sharpe = (bench_ret - risk_free_rate) / bench_risk if bench_risk > 0 else 0
    bench_sortino = calculate_sortino(bench_returns, bench_ret, risk_free_rate)
else:
    bench_ret = bench_risk = bench_sharpe = bench_sortino = None

# ==============================
# EXIBIÇÃO
# ==============================
col1, col2 = st.columns(2)
with col1:
    st.plotly_chart(create_frontier_plot(frontier_risks, frontier_rets, ms_risk, ms_ret, ms_sharpe), use_container_width=True)
with col2:
    st.plotly_chart(create_backtest_plot(port_cum, bench_cum), use_container_width=True)

col3, col4 = st.columns(2)
with col3:
    st.plotly_chart(create_drawdown_plot(port_cum, bench_cum, port_max_dd, bench_max_dd), use_container_width=True)
with col4:
    st.empty()  # espaço reservado se quiser adicionar outro gráfico no futuro

st.header("📊 Comparação Detalhada: Portfólio Otimizado vs. Benchmark")

comparison_data = {
    "Métrica": ["Retorno Anualizado", "Risco Anualizado", "Sharpe Ratio", "Sortino Ratio", "Retorno Total Acumulado (%)", "Máximo Drawdown (%)"],
    "Portfólio Otimizado": [
        f"{ms_ret:.2%}", f"{ms_risk:.2%}", f"{ms_sharpe:.2f}", f"{sortino:.2f}",
        f"{total_return_port:+.2f}%", f"{port_max_dd:.2f}%"
    ],
    "Benchmark (Ibovespa)": [
        f"{bench_ret:.2%}" if bench_ret else "N/A",
        f"{bench_risk:.2%}" if bench_risk else "N/A",
        f"{bench_sharpe:.2f}" if bench_sharpe else "N/A",
        f"{bench_sortino:.2f}" if bench_sortino else "N/A",
        f"{total_return_bench:+.2f}%" if total_return_bench else "N/A",
        f"{bench_max_dd:.2f}%" if bench_max_dd else "N/A"
    ]
}

df_comparison = pd.DataFrame(comparison_data)
st.table(df_comparison)

st.subheader("Alocação Ótima dos Pesos")
weights_df = pd.Series(weights, index=prices_df.columns) * 100
weights_df = weights_df.round(2)
weights_df = weights_df[weights_df.abs() > 0.5].sort_values(ascending=False)
st.bar_chart(weights_df)

st.caption("Projeto desenvolvido para demonstração de habilidades em Finanças Quantitativas e Data Science.")