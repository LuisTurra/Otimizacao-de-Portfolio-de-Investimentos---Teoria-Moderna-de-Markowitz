# Otimização de Portfólio de Investimentos - Teoria Moderna de Markowitz (com versão Machine Learning)

Projeto completo de Data Science / Finanças Quantitativas para portfólio profissional.  
Implementa otimização de portfólio baseada na Teoria Moderna de Portfólios de Harry Markowitz, com dashboard interativo em Streamlit e versão avançada que integra Machine Learning (Random Forest + XGBoost) para prever retornos esperados.

**Deploy ao vivo**: [Acesse aqui] https://luisturra-otimizacao-de-portfolio-de-inves-streamlit-app-4bj7xl.streamlit.app/

## 🎯 Visão Geral
O projeto tem dois módulos integrados em um único app multi-page:

1. **Otimização Clássica (Markowitz)**  
   - Fronteira Eficiente  
   - Portfólio de máximo Sharpe Ratio (ajustado pela Selic real)  
   - Sortino Ratio (risco downside)  
   - Backtesting vs. Ibovespa  
   - Análise de drawdown  
   - Comparação detalhada de métricas (retorno, risco, Sharpe, drawdown)

2. **Otimização com Machine Learning**  
   - Previsão de retornos futuros por ativo usando Random Forest e XGBoost  
   - Features técnicas (lags, volatilidade rolling, RSI)  
   - Validação walk-forward para métricas out-of-sample  
   - Feature importance média  
   - Comparação direta: Clássico vs. Random Forest vs. XGBoost (retorno esperado, risco, Sharpe)

## 🚀 Funcionalidades Principais
- Seleção interativa de ativos brasileiros (pré-lista com ações líquidas + tickers customizados)
- Configuração de período histórico, benchmark (^BVSP), short selling e taxa livre de risco
- Gráficos interativos com Plotly (fronteira eficiente, backtesting, drawdown)
- Tabela comparativa detalhada com destaque de melhorias
- Regularização forte nos modelos ML para evitar overfitting extremo
- Resultados realistas e explicáveis (evita previsões absurdas com clip e hiperparâmetros conservadores)

## 🛠 Tech Stack
- **Python** 3.10+
- **Streamlit** (dashboard interativo multi-page)
- **yfinance** (dados de mercado)
- **pandas, numpy, scipy** (cálculos e otimização)
- **plotly** (visualizações)
- **scikit-learn** (Random Forest)
- **xgboost** (XGBoost)
- **requests** (API Selic BCB)
