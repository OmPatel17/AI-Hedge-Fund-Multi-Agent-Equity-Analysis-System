# AI Hedge Fund — Multi-Agent Equity Analysis System

A multi-agent AI system that simulates a hedge fund. Each agent embodies a different legendary investor, independently analyzes equities using real financial data, and collaborates to produce trading decisions. Includes a full backtesting engine to evaluate performance across historical periods.

---

## How It Works

```
Tickers + Date Range
        │
        ▼
┌───────────────────────────────────────────┐
│           Analyst Agents (parallel)        │
│  Warren Buffett  │  Charlie Munger         │
│  Ben Graham      │  Peter Lynch            │
│  Bill Ackman     │  Cathie Wood            │
│  Michael Burry   │  Stanley Druckenmiller  │
│  Nassim Taleb    │  Phil Fisher            │
│  Mohnish Pabrai  │  Rakesh Jhunjhunwala    │
│  Aswath Damodaran│  Valuation / Growth /   │
│                  │  Technicals / Sentiment │
└───────────────────────────────────────────┘
        │
        ▼
  Risk Manager Agent
        │
        ▼
  Portfolio Manager Agent
        │
        ▼
  Trading Decision (BUY / SELL / HOLD + sizing)
```

Each analyst agent fetches its own data, reasons independently, and outputs a signal with confidence. The risk manager aggregates signals and applies position-sizing constraints. The portfolio manager issues final orders.

---

## Data Sources

| Source | What it provides |
|--------|-----------------|
| **SEC EDGAR XBRL** | Income statements, balance sheets, cash flows, Form 4 insider trades — directly from SEC filings, no API key required |
| **Yahoo Finance** | Daily price history, real-time market cap, financial ratios |
| **GDELT** | News sentiment across global media |
| **FRED** | Macro indicators (interest rates, CPI, etc.) |

All data is fetched live and cached in-memory per run. No paid financial data subscription required.

---

## Analyst Agents

| Agent | Investment Style |
|-------|----------------|
| Warren Buffett | Quality businesses at fair prices, FCF, moat, long-term compounding |
| Charlie Munger | Mental models, quality over price, concentrated conviction |
| Ben Graham | Deep value, margin of safety, net-nets |
| Peter Lynch | GARP, PEG ratio, local knowledge, growth at reasonable price |
| Bill Ackman | Activist value, brand strength, FCF yield |
| Cathie Wood | Disruptive innovation, exponential growth, TAM expansion |
| Michael Burry | Contrarian deep value, hidden catalysts, asymmetric bets |
| Stanley Druckenmiller | Macro-driven, momentum, risk/reward |
| Nassim Taleb | Tail-risk awareness, convexity, fragility analysis |
| Phil Fisher | Scuttlebutt, R&D quality, management assessment |
| Mohnish Pabrai | Dhandho investor, cloning, heads-I-win-tails-I-don't-lose |
| Rakesh Jhunjhunwala | India-style growth value, earnings momentum |
| Aswath Damodaran | Valuation-first, DCF, intrinsic value |
| Valuation Agent | DCF + comparables |
| Growth Agent | Revenue/earnings acceleration |
| Technicals Agent | Trend, momentum, mean-reversion signals |
| Sentiment Agent | Insider trades + news tone |
| Fundamentals Agent | Ratio screening |

---

## Supported LLM Providers

| Provider | Models |
|----------|--------|
| **OpenAI** | GPT-4.1, GPT-4o, o1, o3 |
| **Anthropic** | Claude Sonnet 4, Claude Opus 4 |
| **Google** | Gemini 2.5 Flash, Gemini 2.5 Pro |
| **Groq** | Llama 3, DeepSeek (fast inference) |
| **DeepSeek** | DeepSeek Chat, DeepSeek Reasoner |
| **xAI** | Grok |
| **Ollama** | Any local model (Gemma 3, Llama 3, Qwen 3, Mistral, etc.) |

Run entirely locally with Ollama — no cloud API calls, no cost per token.

---

## Quickstart

### Prerequisites
- Python 3.11+
- [Poetry](https://python-poetry.org/)
- [Ollama](https://ollama.com/) (optional, for local inference)

### Install

```bash
git clone https://github.com/OmPatel17/AI-Hedge-Fund-Multi-Agent-Equity-Analysis-System.git
cd AI-Hedge-Fund-Multi-Agent-Equity-Analysis-System
poetry install
```

### Configure

```bash
cp .env.example .env
# Edit .env and add the API key for whichever LLM provider you want to use
```

---

## Usage

### Live Analysis

```bash
# OpenAI
poetry run python src/main.py --ticker AAPL,MSFT,NVDA

# Anthropic
poetry run python src/main.py --ticker AAPL,MSFT,NVDA --anthropic

# Local (Ollama — no API key needed)
poetry run python src/main.py --ticker AAPL,MSFT,NVDA --ollama
```

### Backtesting

```bash
# Backtest over a custom date range
poetry run python src/backtester.py --ticker AAPL,MSFT,NVDA --start-date 2024-01-01 --end-date 2024-12-31

# With Ollama
poetry run python src/backtester.py --ticker AAPL,MSFT,NVDA --ollama
```

### Options

| Flag | Description |
|------|-------------|
| `--ticker` | Comma-separated tickers (required) |
| `--start-date` | Start of analysis window (YYYY-MM-DD) |
| `--end-date` | End of analysis window (YYYY-MM-DD) |
| `--ollama` | Use local Ollama model |
| `--show-reasoning` | Print each agent's full reasoning |
| `--initial-cash` | Starting portfolio cash (default: $100,000) |

---

## Project Structure

```
src/
├── agents/          # One file per analyst agent + risk/portfolio managers
├── backtesting/     # Backtesting engine and performance metrics
├── data/
│   ├── providers/   # EDGAR, Yahoo Finance, GDELT, FRED data providers
│   ├── models.py    # Pydantic data models
│   └── cache.py     # In-memory request cache
├── graph/           # LangGraph workflow definition
├── llm/             # LLM provider selection and model configs
├── runtime/         # Dependency checks
├── tools/           # Agent-facing API tools
└── utils/           # Display, visualization, progress
```

---

## Tech Stack

- **[LangGraph](https://github.com/langchain-ai/langgraph)** — multi-agent orchestration and state management
- **[LangChain](https://github.com/langchain-ai/langchain)** — LLM abstraction layer
- **[SEC EDGAR API](https://www.sec.gov/developer)** — free XBRL financial data
- **[yfinance](https://github.com/ranaroussi/yfinance)** — Yahoo Finance price and market data
- **[Ollama](https://ollama.com/)** — local LLM inference
- **pandas / numpy** — data processing
- **Poetry** — dependency management

---

## Disclaimer

This project is for **educational and research purposes only**. It does not constitute financial advice. Do not use it to make real investment decisions. Past backtest performance does not guarantee future results.

---

## License

MIT
