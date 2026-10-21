import asyncio
from datetime import datetime
import logging
import math
import os
from zoneinfo import ZoneInfo
from ib_async import IB, Index, Option
import pandas as pd

NY_TZ = ZoneInfo("America/New_York")


# =====================================================================
# 1. SILENCE IBKR LOG NOISE (Error 10090, 200, 2104, etc.)
# =====================================================================
class IBLogFilter(logging.Filter):

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        # Suppress routine delayed-market advisories and wing-strike misses
        ignored_patterns = ["10090", "Error 200", "2104", "2106", "2158"]
        return not any(pattern in msg for pattern in ignored_patterns)


# Attach filter to all ib_async child loggers (wrapper is where TWS errors originate)
log_filter = IBLogFilter()
for logger_name in ["ib_async", "ib_async.wrapper", "ib_async.client"]:
    logging.getLogger(logger_name).addFilter(log_filter)


# =====================================================================
# 2. CONTRACT DISCOVERY (Configured for 6 Expiries)
# =====================================================================
async def get_spx_contracts(
        ib: IB,
        strike_pct_range: float = 0.05,
        strike_step: int = 25,
        n_expiries: int = 6,
) -> tuple[Index, list[Option]]:
    # Resolve SPX Underlying Index
    spx = Index(symbol="SPX", exchange="CBOE", currency="USD")
    await ib.qualifyContractsAsync(spx)

    # Get Spot Reference Price
    tickers = await ib.reqTickersAsync(spx)
    spx_price = tickers[0].marketPrice() if tickers else None

    if spx_price is None or math.isnan(spx_price) or spx_price <= 0:
        bars = await ib.reqHistoricalDataAsync(
            spx,
            endDateTime="",
            durationStr="2 D",
            barSizeSetting="1 day",
            whatToShow="TRADES",
            useRTH=True,
            formatDate=1,
        )
        spx_price = bars[-1].close if bars else 5800.0

    print(f"Reference SPX spot: {spx_price:.2f}")

    # Request Option Chains
    chains = await ib.reqSecDefOptParamsAsync(
        underlyingSymbol=spx.symbol,
        futFopExchange="",
        underlyingSecType=spx.secType,
        underlyingConId=spx.conId,
    )

    chain = next(
        (
            c
            for c in chains
            if c.tradingClass == "SPXW" and c.exchange == "SMART"
        ),
        None,
    )
    if chain is None:
        chain = next((c for c in chains if c.tradingClass == "SPXW"), chains[0])

    target_trading_class = chain.tradingClass
    target_multiplier = chain.multiplier or "100"

    today_str = datetime.now(NY_TZ).strftime("%Y%m%d")
    expirations = sorted([exp for exp in chain.expirations if exp >= today_str])[
                  :n_expiries
                  ]

    # Cleaned strike boundaries
    min_k = spx_price * (1.0 - strike_pct_range)
    max_k = spx_price * (1.0 + strike_pct_range)

    strikes = sorted([
        k
        for k in chain.strikes
        if min_k <= k <= max_k and (k % strike_step == 0)
    ])
    if len(strikes) < 5:
        strikes = sorted([k for k in chain.strikes if min_k <= k <= max_k])[::5]

    print(
        f"Chain '{target_trading_class}': {len(expirations)} expiries,"
        f" {len(strikes)} strikes per expiry."
    )

    raw_contracts = []
    for exp in expirations:
        for right in ["C", "P"]:
            for strike in strikes:
                raw_contracts.append(
                    Option(
                        symbol="SPX",
                        lastTradeDateOrContractMonth=exp,
                        strike=strike,
                        right=right,
                        exchange="SMART",
                        currency="USD",
                        multiplier=target_multiplier,
                        tradingClass=target_trading_class,
                    )
                )

    qualified_contracts = []
    batch_size = 50
    for i in range(0, len(raw_contracts), batch_size):
        batch = raw_contracts[i: i + batch_size]
        res = await ib.qualifyContractsAsync(*batch)
        valid = [c for c in res if c is not None and getattr(c, "conId", 0) > 0]
        qualified_contracts.extend(valid)
        await asyncio.sleep(0.05)

    print(f"Successfully qualified {len(qualified_contracts)} option contracts.")
    return spx, qualified_contracts


# =====================================================================
# 3. SNAPSHOT ENGINE
# =====================================================================
async def record_surface_snapshot(
        ib: IB,
        spx_contract: Index,
        option_contracts: list[Option],
        output_csv: str,
        batch_size: int = 75,
):
    timestamp = datetime.now(NY_TZ).strftime("%Y-%m-%d %H:%M:%S")

    def clean_val(v):
        return v if (v is not None and not math.isnan(v) and v > 0) else None

    # Spot price lookup with robust delayed fallbacks
    spx_tickers = await ib.reqTickersAsync(spx_contract)
    spx_t = spx_tickers[0] if spx_tickers else None
    spot_price = None
    if spx_t:
        spot_price = (
                clean_val(spx_t.marketPrice())
                or clean_val(getattr(spx_t, "delayedLast", None))
                or clean_val(spx_t.close)
                or clean_val(getattr(spx_t, "delayedClose", None))
        )

    records = []

    for i in range(0, len(option_contracts), batch_size):
        batch = option_contracts[i: i + batch_size]
        tickers = await ib.reqTickersAsync(*batch)

        for t in tickers:
            bid = clean_val(t.bid) or clean_val(getattr(t, "delayedBid", None))
            ask = clean_val(t.ask) or clean_val(getattr(t, "delayedAsk", None))
            last = (
                    clean_val(t.last)
                    or clean_val(getattr(t, "delayedLast", None))
                    or clean_val(t.close)
                    or clean_val(getattr(t, "delayedClose", None))
            )
            volume = (
                    clean_val(t.volume)
                    or clean_val(getattr(t, "delayedVolume", None))
                    or 0
            )

            records.append({
                "timestamp": timestamp,
                "underlying_symbol": "SPX",
                "spot_price": spot_price,
                "expiry": t.contract.lastTradeDateOrContractMonth,
                "strike": t.contract.strike,
                "right": t.contract.right,
                "bid": bid,
                "ask": ask,
                "last": last,
                "volume": volume,
            })

        await asyncio.sleep(0.08)  # Pacing guard

    if records:
        df = pd.DataFrame(records)
        header = not os.path.exists(output_csv)
        df.to_csv(output_csv, mode="a", index=False, header=header)
        print(f"[{timestamp} ET] Saved {len(df)} quotes to {output_csv}")


# =====================================================================
# 4. MAIN SCHEDULER
# =====================================================================
async def main():
    ib = IB()
    await ib.connectAsync("127.0.0.1", 7496, clientId=25)

    try:
        # 4 = Delayed-Frozen (gives last quotes off-hours and delayed live quotes during RTH)
        ib.reqMarketDataType(4)

        output_file = "spx_options_surface.csv"
        spx, option_list = await get_spx_contracts(
            ib, strike_pct_range=0.05, strike_step=25, n_expiries=6
        )

        # # ONE-OFF TEST SNAPSHOT
        # print("Recording 1 test snapshot to verify CSV generation...")
        # await record_surface_snapshot(
        #     ib, spx, option_list, output_csv=output_file, batch_size=75
        # )
        # print(f"Test snapshot complete. Verify '{output_file}'.")

        # RTH SESSION SCHEDULER
        print("Scheduler active. Waiting for / running through RTH capture window...")

        while True:
            now_ny = datetime.now(NY_TZ)

            start_window = now_ny.replace(hour=9, minute=45, second=0, microsecond=0)
            end_window = now_ny.replace(hour=16, minute=32, second=0, microsecond=0)

            if now_ny < start_window:
                wait_seconds = (start_window - now_ny).total_seconds()
                print(f"Pre-market: sleeping for {wait_seconds / 60:.1f} minutes...")
                await asyncio.sleep(min(wait_seconds, 300))
                continue

            if now_ny > end_window:
                print(
                    "RTH capture window closed (16:32 ET reached). Extraction complete."
                )
                break

            start_time = asyncio.get_running_loop().time()
            await record_surface_snapshot(
                ib, spx, option_list, output_csv=output_file, batch_size=75
            )

            elapsed = asyncio.get_running_loop().time() - start_time
            await asyncio.sleep(max(0.0, 60.0 - elapsed))

    finally:
        ib.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
