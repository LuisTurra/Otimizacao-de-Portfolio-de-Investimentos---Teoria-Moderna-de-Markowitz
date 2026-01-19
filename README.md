# Otimização de Portfólio de Investimentos - Teoria Moderna de Markowitz (Enhanced)

Projeto de Data Science / Quant Finance para portfólio profissional.  
Demonstra habilidades end-to-end: coleta de dados reais (yfinance + API BCB), otimização matemática (scipy), visualização interativa (Plotly) e deployment (Streamlit).

## 🎯 Problema de Negócio
Como alocar capital entre ações brasileiras para maximizar retorno ajustado ao risco, considerando a taxa Selic real como risco livre, possibilidade de short selling e desempenho histórico vs. Ibovespa?

## 🚀 Funcionalidades
- Fronteira Eficiente com portfólio de Máximo Sharpe Ratio (ajustado por Selic real)
- Opção de short selling (pesos negativos)
- Taxa livre de risco buscada automaticamente via API do BCB (Selic meta)
- Cálculo de Sortino Ratio (risco downside)
- Backtesting: retorno cumulativo do portfólio otimizado vs. Ibovespa
- Dashboard interativo com configurações customizáveis (tickers, período, etc.)

## 🛠 Tech Stack
- Python 3
- Streamlit (dashboard)
- yfinance (dados de mercado)
- pandas, numpy, scipy (cálculos e otimização)
- plotly (gráficos interativos)
- requests (API Selic)

## 📊 Resultados Típicos (exemplo com dados até jan/2026)
- Sharpe Ratio ~1.0–1.5 (melhor que Ibovespa ~0.6–0.9 no período)
- Retorno anualizado otimizado >15% com volatilidade controlada
- Backtesting mostra outperformance em períodos de alta diversificação

## 🚀 Como Rodar Localmente
```bash
pip install streamlit yfinance pandas numpy scipy plotly requests
streamlit run app.py