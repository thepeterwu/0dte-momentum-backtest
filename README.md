git clone <https://github.com/thepeterwu/0dte-momentum-backtest>
cd <your-repo-folder>
python -m venv .venv

Activate the environment:
Windows: .venv\Scripts\activate
Mac/Linux: source .venv/bin/activate

Install Dependencies
pip install -r requirements.txt

# Momentum 



# Vol surface

### Project Structure

```text
0dte-momentum-backtest/
│
├── data/
│   └── spx_options_surface.csv       # Raw 1-minute delayed snapshots from IBKR
│
├── vol_surface/                      # Core modeling package
│   ├── __init__.py
│   ├── black76.py                    # Forward extraction, Black-76 pricer, IV root solver
│   ├── smile.py                      # 1D smile spline, wing extrapolation, delta-inversion solver
│   ├── surface.py                    # 2D (Delta, Tau) surface container & interpolation
│   ├── arbitrage.py                  # Butterfly density g(x) & calendar spread auditors
│   └── visualization.py              # 2D smile slices & 3D Plotly surface visualizers
│
├── vol_capture_option_price.py       # existing ib_async data collection loop
└── vol_run_surface_pipeline.py       # Top-level orchestrator & batch processing runner
```
