# 🇹🇳 BVMT Intelligent Trading Assistant

**Assistant Intelligent de Trading pour la Bourse des Valeurs Mobilières de Tunisie**

A full-stack AI-powered platform for market analysis, forecasting, anomaly detection, portfolio management, and multi-agent orchestration on the Tunisian stock exchange (BVMT).

---

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Features](#features)
- [Application Flow](#application-flow)
- [Installation](#installation)
- [Running the Application](#running-the-application)
- [Project Structure](#project-structure)
- [Modules](#modules)
- [API Reference](#api-reference)
- [Agent System](#agent-system)
- [Documentation](#documentation)
- [Testing](#testing)
- [Demo Scenario](#demo-scenario)

---

## Overview

This project implements a complete intelligent trading assistant for the BVMT, featuring:

- **Forecasting** — EMA extrapolation + weighted regression with optional XGBoost ensemble for price and volume prediction
- **Sentiment Analysis** — GPT-4o powered multilingual (French/Arabic) market sentiment via OpenAI API
- **Anomaly Detection** — Statistical (Z-score, thresholds) + ML-based (Isolation Forest) market surveillance
- **Portfolio Management** — Decision engine with explainability, risk profiles, portfolio simulation with Sharpe ratio
- **Multi-Agent System** — 5-agent pipeline (Market Analyst, Forecast, Sentiment, Anomaly, Recommendation) with safety guardrails
- **Real-Time Scraping** — Background thread scraping ilboursa/bvmt for live prices every 60 seconds
- **Reinforcement Learning** — RL-based portfolio optimization with personalized learning from user feedback
- **GPT-4o Chat** — Context-aware AI assistant powered by OpenAI for natural language market Q&A

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    Web Application (Flask)                    │
│                                                              │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌───────────────┐   │
│  │ Marché   │ │ Trading  │ │Portfolio │ │ Surveillance  │   │
│  │ Overview │ │  Detail  │ │ Manager  │ │   & Alerts    │   │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘ └──────┬────────┘   │
│       │            │            │               │            │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐                     │
│  │  Agents  │ │   Chat   │ │  Login   │                     │
│  │Dashboard │ │Assistant │ │Onboarding│                     │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘                     │
│       │            │            │                            │
│  ┌────┴────────────┴────────────┴────────────────────────┐   │
│  │                  REST API Layer                       │   │
│  └────┬─────────────┬────────────┬──────────────┬────────┘   │
│       │             │            │              │            │
│  ┌────┴────┐  ┌─────┴────┐ ┌────┴─────┐  ┌─────┴─────┐     │
│  │Forecast │  │Sentiment │ │ Anomaly  │  │ Portfolio  │     │
│  │ Engine  │  │ Analyzer │ │ Detector │  │  + RL   │     │
│  └────┬────┘  └─────┬────┘ └────┬─────┘  └─────┬─────┘     │
│       └─────────────┴───────────┴───────────────┘           │
│                          │                                   │
│  ┌───────────────────────┴───────────────────────────────┐   │
│  │    Multi-Agent System (5 Agents + Safety + Logger)    │   │
│  └───────────────────────────────────────────────────────┘   │
│                          │                                   │
│  ┌───────────────────────┴───────────────────────────────┐   │
│  │  Data Layer (860K+ records · CSV + SQLite + Realtime) │   │
│  └───────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────┘
```

---

## Features

### Core Analytics
- 📈 **Price Forecasting** — EMA + weighted linear regression with damped trends, confidence intervals, and optional XGBoost ensemble
- 📊 **Volume Forecasting** — Liquidity probability estimation with high/low confidence bands
- 📰 **Sentiment Analysis** — GPT-4o via OpenAI API for French/Arabic market news analysis with real price context
- 🔍 **Anomaly Detection** — Volume Z-score (>3σ), price threshold (>5%), suspicious patterns, Isolation Forest multivariate scoring
- 💼 **Portfolio Simulator** — Buy/sell/track with Sharpe ratio, max drawdown, VaR, and decision explainability
- 🤖 **Decision Engine** — 4-signal weighted aggregation (forecast + sentiment + technical + anomaly) with risk profiles

### Intelligence
- 🧠 **5-Agent Pipeline** — MarketAnalyst → ForecastEngine → SentimentAnalyzer → AnomalyScanner → RecommendationEngine
- 💬 **GPT-4o Chat** — Context-aware AI assistant with market data, powered by OpenAI API
- 🎯 **RL Portfolio Optimization** — Reinforcement learning that adapts to user feedback and preferences
- 🔄 **Real-Time Data** — Background scraping of ilboursa.com and bvmt.com.tn every 60 seconds

### Interface
- 🌙 **Dark theme dashboard** with responsive design (Bootstrap 5)
- 📊 **Interactive charts** (Chart.js) — TUNINDEX, candlestick, volume, RSI, MACD, Bollinger Bands
- 🔒 **User authentication** — Login, registration, guided onboarding with investment profile
- 🚨 **Alert system** — Real-time anomaly notifications with severity levels (CRITICAL / HIGH / MEDIUM)
- ⚡ **Ultra-fast forecasts** — Precomputed tail cache for sub-20ms API responses in Trading view

---

## Application Flow

```
Login/Register → Onboarding (Investment Profile) → Dashboard
                                                       │
                    ┌──────────────────────────────────┼──────────────────────┐
                    │                                  │                      │
               ┌────▼────┐   ┌────────────┐   ┌───────▼──────┐   ┌──────────▼──┐
               │ Marché  │   │  Trading   │   │ Portefeuille │   │Surveillance │
               │Overview │   │   Detail   │   │   Manager    │   │  & Alertes  │
               └────┬────┘   └─────┬──────┘   └──────┬───────┘   └──────┬──────┘
                    │              │                  │                  │
                    │         ┌────▼──────┐           │                  │
                    │         │ Prévision │           │                  │
                    │         │  (5 jours)│           │                  │
                    │         └───────────┘           │                  │
                    │                                 │                  │
               ┌────▼────────────────────────────────▼──────────────────▼──┐
               │                    Agents Dashboard                       │
               │  MarketAnalyst → Forecast → Sentiment → Anomaly → Reco   │
               └──────────────────────────┬────────────────────────────────┘
                                          │
                                   ┌──────▼──────┐
                                   │  Assistant  │
                                   │  GPT-4o AI  │
                                   └─────────────┘
```

### Page Descriptions

| Page | Description |
|------|-------------|
| **Marché** | TUNINDEX chart, top gainers/losers, market volume, sector breakdown |
| **Trading** | Stock price chart, 5-day forecast preview, technical indicators (RSI, MACD, Bollinger), buy/sell actions |
| **Portefeuille** | Portfolio creation, holdings tracking, performance metrics (Sharpe, drawdown), AI-powered suggestions |
| **Surveillance** | Anomaly alerts across all stocks, severity-based filtering, suspicious pattern detection |
| **Agents** | Multi-agent analysis dashboard — trigger full pipeline analysis on any stock |
| **Assistant** | GPT-4o conversational AI with market context for trading advice in French |

---

## Installation

### Prerequisites
- Python 3.9+
- pip
- OpenAI API key (for chat & sentiment features)

### Setup

```bash
# 1. Clone the repository
git clone https://github.com/bechir23/ihec-fintech.git
cd ihec-fintech

# 2. Create virtual environment (recommended)
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/Mac

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
copy .env.example .env
# Edit .env and set your OPENAI_API_KEY
```

### Dependencies
Key packages: Flask, pandas, numpy, scikit-learn, statsmodels, xgboost, requests, python-dotenv, flask-cors.

Full list in `requirements.txt`.

---

## Running the Application

```bash
# Start the Flask server
python webapp/app.py
```

Open your browser at **http://localhost:5000**

The server will:
1. Load all historical market data (~860K records)
2. Build precomputed indexes for fast lookups
3. Start real-time background scraping (60s interval)
4. Serve the web application on port 5000

---

## Project Structure

```
BVMT/
├── data/                              # Historical BVMT data (2016–2025)
│   ├── histo_cotation_2016.txt        # 2016 daily quotes
│   ├── histo_cotation_2017.txt        # 2017 daily quotes
│   ├── histo_cotation_2018.txt        # 2018 daily quotes
│   ├── histo_cotation_2019.txt        # 2019 daily quotes
│   ├── histo_cotation_2020.txt        # 2020 daily quotes
│   ├── histo_cotation_2021.txt        # 2021 daily quotes
│   ├── histo_cotation_2022.csv        # 2022 daily quotes
│   ├── histo_cotation_2023.csv        # 2023 daily quotes
│   ├── histo_cotation_2024.csv        # 2024 daily quotes
│   ├── histo_cotation_2025.csv        # 2025 daily quotes (partial)
│   ├── web_histo_cotation_2022.csv    # Web-sourced 2022 data
│   └── web_histo_cotation_2023.csv    # Web-sourced 2023 data
│
├── modules/                           # Core ML/analytics modules
│   ├── common/
│   │   └── data_loader.py             # Data loading, TUNINDEX computation, technicals
│   ├── forecasting/
│   │   └── forecaster.py              # EMA + regression + XGBoost forecasting
│   ├── sentiment/
│   │   └── analyzer.py                # GPT-4o sentiment analysis via OpenAI
│   ├── anomaly/
│   │   └── detector.py                # Statistical + ML anomaly detection
│   ├── portfolio/
│   │   ├── manager.py                 # Portfolio simulator & decision engine
│   │   └── db.py                      # SQLite portfolio persistence
│   ├── drift/
│   │   └── analyzer.py                # Prediction drift & backtest analysis
│   ├── user/
│   │   └── manager.py                 # User authentication & profile management
│   ├── scraper/
│   │   └── realtime.py                # Real-time price scraper (ilboursa/bvmt)
│   └── rl/
│       └── portfolio_rl.py            # Reinforcement learning portfolio optimizer
│
├── agents/
│   ├── agent_system.py                # ChatAgent, ExecutionAgent, SafetyGuard
│   └── crew.py                        # OrchestratorAgent, MCPToolRegistry
│
├── webapp/
│   ├── app.py                         # Flask app — routes, API, startup logic
│   ├── templates/                     # Jinja2 HTML templates (dark theme)
│   │   ├── base.html                  # Base layout with navigation
│   │   ├── login.html                 # Login & registration page
│   │   ├── market_overview.html       # Market dashboard
│   │   ├── stock_detail.html          # Individual stock analysis
│   │   ├── trade.html                 # Trading view with forecast preview
│   │   ├── portfolio.html             # Portfolio management
│   │   ├── surveillance.html          # Anomaly surveillance
│   │   ├── agents.html                # Multi-agent analysis dashboard
│   │   └── chat.html                  # AI chat assistant
│   └── static/
│       ├── css/                       # Custom stylesheets
│       └── js/                        # Client-side JavaScript
│
├── notebooks/
│   ├── 01_eda.ipynb                   # Exploratory Data Analysis
│   └── 02_forecasting.ipynb           # Forecasting pipeline demo
│
├── docs/                              # Technical documentation
│   ├── 00_project_understanding.md
│   ├── 10_agent_architecture.md
│   ├── 20_model_choices.md
│   ├── 30_statistical_analysis.md
│   └── 40_user_stories_and_tests.md
│
├── tests/                             # Unit & integration tests
├── models/                            # Saved model artifacts
├── logs/                              # Agent & application logs
├── requirements.txt                   # Python dependencies
├── .env.example                       # Environment template
└── README.md
```

---

## Modules

### Forecasting (`modules/forecasting/forecaster.py`)
- **BVMTForecaster** — EMA extrapolation + weighted linear regression (fast mode) + optional XGBoost/SARIMA ensemble
- Stationarity testing (ADF + KPSS) with Box-Cox variance stabilization
- Precomputed tail cache at startup for ultra-fast (<20ms) per-request forecasts
- Confidence intervals from historical volatility with √t scaling
- Walk-forward backtesting support

### Sentiment (`modules/sentiment/analyzer.py`)
- **SentimentAnalyzer** — GPT-4o via OpenAI API with real price context injection
- French + Arabic keyword fallback when API is unavailable
- Structured output: sentiment label, score, positive/negative percentages, summary
- Caching with configurable TTL (1 hour default)

### Anomaly (`modules/anomaly/detector.py`)
- **AnomalyDetector** — Volume Z-score (>3σ), price threshold (>5%), gap detection
- Suspicious patterns: volume-price divergence, consecutive extremes, end-of-day manipulation
- Isolation Forest for multivariate anomaly scoring
- Severity classification: CRITICAL / HIGH / MEDIUM

### Portfolio (`modules/portfolio/manager.py`)
- **PortfolioSimulator** — Cash management, buy/sell execution, performance tracking
- **DecisionEngine** — 4-signal weighted aggregation (forecast, sentiment, technical, anomaly)
- **RiskProfile** — Conservative / Moderate / Aggressive configurations
- Sharpe ratio with 7% Tunisia risk-free rate, max drawdown, VaR

### User Management (`modules/user/manager.py`)
- JSON-file based user profiles with investment questionnaire
- Session-based authentication with Flask sessions
- Guided onboarding flow collecting risk tolerance, investment horizon, sector preferences

### Real-Time Scraper (`modules/scraper/realtime.py`)
- Background thread scraping ilboursa.com and bvmt.com.tn every 60 seconds
- Latest prices, intraday ticks, OHLCV candle aggregation
- Search functionality for ticker lookup

### RL Portfolio (`modules/rl/portfolio_rl.py`)
- Reinforcement learning-based portfolio optimization
- Learns from user feedback to personalize recommendations
- Model persistence and summary reporting

---

## API Reference

### Authentication
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/auth/register` | POST | Register new user `{username, password, display_name}` |
| `/api/auth/login` | POST | Login `{username, password}` |
| `/api/auth/me` | GET | Get current user info |
| `/api/user/onboarding` | POST | Complete investment profile |

### Market Data
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/market/overview` | GET | Market summary with TUNINDEX |
| `/api/stocks` | GET | List all available stocks |
| `/api/stock/<code>` | GET | Stock details, price history, technicals |
| `/api/stock/<code>/analysis` | GET | Deep analysis: forecast + sentiment + anomalies |

### Forecasting
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/forecast/<code>?fast=1&horizon=5` | GET | Price forecast with confidence intervals |

### Anomaly Detection
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/anomalies` | GET | All detected anomalies |
| `/api/anomalies/<code>` | GET | Stock-specific anomalies |

### Portfolio
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/portfolio/create` | POST | Create portfolio `{cash}` |
| `/api/portfolio/buy` | POST | Buy stock `{code, qty, price}` |
| `/api/portfolio/sell` | POST | Sell stock `{code, qty, price}` |
| `/api/portfolio/status` | GET | Portfolio performance & holdings |
| `/api/portfolio/suggest` | POST | AI suggestions `{budget, profile}` |

### Agents
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/agents/status` | GET | Agent system health |
| `/api/agents/analyze/<code>` | POST | Full 5-agent pipeline analysis |

### Chat
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/chat` | POST | Send message to GPT-4o `{message}` |
| `/api/chat/clear` | POST | Clear conversation history |

### Real-Time
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/realtime/status` | GET | Scraper status |
| `/api/realtime/latest` | GET | Latest prices for all stocks |
| `/api/realtime/quote/<ticker>` | GET | Latest quote for a stock |
| `/api/realtime/candles/<ticker>` | GET | OHLCV candles `?timeframe=5m&limit=100` |

---

## Agent System

The multi-agent pipeline runs 5 specialized agents in sequence to produce a comprehensive stock analysis:

```
┌─────────────────┐   ┌─────────────────┐   ┌──────────────────┐
│  MarketAnalyst  │──▶│  ForecastEngine │──▶│SentimentAnalyzer │
│  Price & volume │   │  EMA+régression │   │ GPT-4o analysis  │
│  trend analysis │   │  5-day forecast │   │  news sentiment  │
└─────────────────┘   └─────────────────┘   └────────┬─────────┘
                                                      │
                      ┌─────────────────┐   ┌─────────▼─────────┐
                      │ Recommendation  │◀──│  AnomalyScanner   │
                      │    Engine       │   │  Z-score + IF     │
                      │ BUY/SELL/HOLD   │   │  pattern detect   │
                      └─────────────────┘   └───────────────────┘
```

Each agent is wrapped with **SafetyGuard** validation and **AgentLogger** audit trailing.

---

## Documentation

| Document | Description |
|----------|-------------|
| `docs/00_project_understanding.md` | Project comprehension & assumptions |
| `docs/10_agent_architecture.md` | Agent system design & safety guardrails |
| `docs/20_model_choices.md` | Model selection justification |
| `docs/30_statistical_analysis.md` | Statistical methodology |
| `docs/40_user_stories_and_tests.md` | User stories & test matrix |

---

## Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ --cov=modules --cov=agents --cov-report=html
```

---

## Demo Scenario — "5000 TND"

1. Open the app at `http://localhost:5000`
2. **Register** a new account and complete the investment profile onboarding
3. Browse the **Marché** page to assess overall BVMT conditions and TUNINDEX
4. Navigate to **Trading** — select a stock, review the 5-day forecast and technicals
5. Go to **Portefeuille** — enter **5000 TND** budget, select **Moderate** risk profile
6. Click **"Obtenir Suggestions"** — the system analyzes all stocks and recommends an allocation
7. Review each suggestion's reasoning (forecast, sentiment, anomaly signals)
8. Execute trades and monitor performance
9. Check **Surveillance** for anomaly alerts across the market
10. Open **Agents** — run a full 5-agent pipeline analysis on any stock
11. Use the **Assistant** to ask: *"Quels sont les meilleurs titres à acheter aujourd'hui ?"*

---

## Technologies

| Component | Technology |
|-----------|-----------|
| Backend | Python 3.12, Flask, python-dotenv |
| Frontend | Bootstrap 5 (Dark), Chart.js 4.4 |
| AI / Chat | GPT-4o via OpenAI API |
| Forecasting | EMA + Weighted Regression + XGBoost |
| Sentiment | GPT-4o + French/Arabic keyword fallback |
| Anomaly | Isolation Forest, Z-score, pattern detection |
| Portfolio | Decision engine + RL optimization |
| Database | SQLite (portfolio), JSON (users) |
| Real-Time | Web scraping (ilboursa, bvmt.com.tn) |
| Data | BVMT historical quotes 2016–2025 (860K+ records) |

---

## License

Academic project — BVMT Trading Assistant for IHEC FinTech / Code Lab 2.0.

---

*Built for the modernization of financial market analysis in Tunisia 🇹🇳*
