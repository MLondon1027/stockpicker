import altair as alt
import pandas as pd
import pandas_market_calendars as mcal
import streamlit as st
import yfinance as yf

CONSTITUENTS_URL = (
    "https://raw.githubusercontent.com/datasets/"
    "s-and-p-500-companies/main/data/constituents.csv"
)
RSI_PERIOD = 14
OVERBOUGHT_LEVEL = 70
MARKET_TIMEZONE = "America/New_York"


@st.cache_data(ttl=86_400)
def load_constituents():
    stocks = pd.read_csv(
        CONSTITUENTS_URL, usecols=["Symbol", "Security", "GICS Sector"]
    )
    stocks["Yahoo symbol"] = stocks["Symbol"].str.replace(".", "-", regex=False)
    return stocks.set_index("Yahoo symbol")


def completed_market_weeks(now, lookback):
    today = now.tz_convert(MARKET_TIMEZONE).tz_localize(None).normalize()
    start = today - pd.Timedelta(weeks=lookback + 156)
    start -= pd.Timedelta(days=start.dayofweek)
    sessions = mcal.get_calendar("NYSE").schedule(
        start_date=start.date(),
        end_date=(today + pd.Timedelta(days=7)).date(),
    )
    sessions["Week ending"] = (
        sessions.index.to_period("W-FRI").end_time.normalize()
    )
    last_sessions = sessions.groupby("Week ending").tail(1)
    completed = last_sessions.loc[last_sessions["market_close"] <= now]
    return completed, sessions.loc[: completed.index[-1], "Week ending"]


@st.cache_data(ttl=300, show_spinner=False)
def load_daily_data(symbols, start, end):
    batches = []
    fields = ["Adj Close", "Close", "Volume"]
    for offset in range(0, len(symbols), 50):
        try:
            prices = yf.download(
                list(symbols[offset : offset + 50]),
                start=start,
                end=end,
                interval="1d",
                auto_adjust=False,
                group_by="column",
                multi_level_index=True,
                ignore_tz=True,
                threads=5,
                progress=False,
                timeout=20,
            )
            if prices is not None and not prices.empty:
                batches.append(prices[fields])
        except Exception:
            # Missing tickers are reported in the dashboard's coverage section.
            continue

    if not batches:
        empty = pd.DataFrame(columns=symbols)
        return empty, empty.copy(), empty.copy()

    daily = pd.concat(batches, axis=1).sort_index()
    daily.index = pd.DatetimeIndex(daily.index).normalize()
    return tuple(daily[field].reindex(columns=symbols) for field in fields)


@st.cache_data(ttl=3600, show_spinner=False)
def load_market_cap(symbol):
    try:
        value = yf.Ticker(symbol).fast_info.market_cap
        if pd.notna(value) and value > 0:
            return float(value)
    except Exception:
        pass
    return None


def wilder_average(values, period):
    averages = pd.Series(float("nan"), index=values.index, dtype=float)
    valid = values.notna()
    runs = (~valid).cumsum()
    for _, run in values[valid].groupby(runs[valid]):
        if len(run) < period:
            continue
        # Wilder's first average is the simple mean of the first 14 changes.
        seeded = run.iloc[period - 1 :].copy()
        seeded.iloc[0] = run.iloc[:period].mean()
        averages.loc[seeded.index] = seeded.ewm(
            alpha=1 / period, adjust=False
        ).mean()
    return averages


def wilder_rsi(closes):
    changes = closes.diff()
    gain = wilder_average(changes.clip(lower=0), RSI_PERIOD)
    loss = wilder_average(-changes.clip(upper=0), RSI_PERIOD)
    total = gain + loss
    return (100 * gain / total).mask(total.eq(0), 50.0)


def three_week_volume_change(daily_volume, sessions, week_endings):
    recent_weeks = pd.DatetimeIndex(week_endings.tail(4))
    recent_sessions = sessions[sessions.isin(recent_weeks)]
    grouped = daily_volume.reindex(recent_sessions.index).groupby(recent_sessions)
    weekly_volume = grouped.sum(min_count=1)
    complete = grouped.count().eq(grouped.size(), axis=0)
    weekly_volume = weekly_volume.where(complete).reindex(recent_weeks)
    baseline = weekly_volume.iloc[0]
    return weekly_volume.iloc[-1].div(baseline.where(baseline.gt(0))).sub(1).mul(100)


def price_performance(closing_price, symbol, signal_date):
    """Use stock and SPY closing prices on identical start/end trading dates."""
    prices = closing_price[symbol].loc[signal_date:]
    prices = prices.where(prices.gt(0)).dropna()
    result = {
        "Latest price (USD)": float("nan"),
        "Price as of": pd.NaT,
        "Gain / loss (%)": float("nan"),
        "SPY gain / loss (%)": float("nan"),
        "Gain / loss vs SPY (pp)": float("nan"),
    }
    if prices.empty:
        return result

    latest_date = prices.index[-1]
    latest_price = prices.iloc[-1]
    result["Latest price (USD)"] = float(latest_price)
    result["Price as of"] = latest_date
    signal_price = closing_price[symbol].get(signal_date, float("nan"))
    if pd.isna(signal_price) or signal_price <= 0:
        return result

    stock_return = (latest_price / signal_price - 1) * 100
    result["Gain / loss (%)"] = float(stock_return)
    spy_start = closing_price["SPY"].get(signal_date, float("nan"))
    spy_end = closing_price["SPY"].get(latest_date, float("nan"))
    if pd.notna(spy_start) and spy_start > 0 and pd.notna(spy_end) and spy_end > 0:
        spy_return = (spy_end / spy_start - 1) * 100
        result["SPY gain / loss (%)"] = float(spy_return)
        result["Gain / loss vs SPY (pp)"] = float(stock_return - spy_return)
    return result


def show_price_chart(ticker, prices, signals, first_date):
    history = (
        prices.loc[first_date:].dropna().rename("Price")
        .rename_axis("Date").reset_index()
    )
    markers = signals.loc[
        signals["Ticker"].eq(ticker),
        ["Signal date", "Signal close (USD)", "Signal RSI"],
    ].dropna().rename(columns={"Signal date": "Date", "Signal close (USD)": "Price"})

    price_line = alt.Chart(history).mark_line().encode(
        x=alt.X("Date:T", title="Trading session"),
        y=alt.Y("Price:Q", title="Price (USD)", scale=alt.Scale(zero=False)),
        tooltip=[
            alt.Tooltip("Date:T", format="%b %d, %Y"),
            alt.Tooltip("Price:Q", format="$,.2f"),
        ],
    )
    points = alt.Chart(markers).mark_point(
        color="#d1495b", filled=True, size=100
    ).encode(
        x="Date:T", y="Price:Q",
        tooltip=[
            alt.Tooltip("Date:T", title="Buy signal date", format="%b %d, %Y"),
            alt.Tooltip("Price:Q", title="Signal close", format="$,.2f"),
            alt.Tooltip("Signal RSI:Q", format=".2f"),
        ],
    )
    # Show each signal's price directly beside its marker.
    labels = alt.Chart(markers).mark_text(
        align="left", dx=8, dy=-12, color="#d1495b"
    ).encode(x="Date:T", y="Price:Q", text=alt.Text("Price:Q", format="$,.2f"))
    st.subheader(f"{ticker} price and buy signals")
    st.altair_chart(price_line + points + labels, use_container_width=True)


def main():
    st.set_page_config(page_title="S&P 500 Weekly RSI Scanner", layout="wide")
    st.title("S&P 500 Weekly RSI Scanner")
    threshold = st.sidebar.number_input(
        "RSI crossover level", min_value=0.0, max_value=100.0, value=36.0, step=0.5
    )
    lookback = int(st.sidebar.number_input(
        "Completed weeks to scan", min_value=1, value=2, step=1
    ))
    min_cap_billions = st.sidebar.number_input(
        "Minimum market cap ($ billions)", min_value=0.0, value=70.0, step=5.0
    )
    if st.sidebar.button("Refresh data"):
        st.cache_data.clear()
        st.rerun()

    now = pd.Timestamp.now(tz="UTC")
    today = now.tz_convert(MARKET_TIMEZONE).tz_localize(None).normalize()
    try:
        stocks = load_constituents()
        weeks, sessions = completed_market_weeks(now, lookback)
    except Exception as exc:
        st.error(f"Could not load the S&P 500 list or market calendar. {exc}")
        st.stop()

    # Include today's in-progress daily bar for the latest price, while weekly
    # RSI continues to use only completed weekly candles.
    symbols = tuple(stocks.index) + ("SPY",)
    with st.spinner("Downloading prices and calculating weekly RSI..."):
        adjusted, closing_price, volume = load_daily_data(
            symbols,
            sessions.index[0].date().isoformat(),
            (today + pd.Timedelta(days=1)).date().isoformat(),
        )
        if adjusted.empty or not adjusted.notna().any().any():
            st.error("Yahoo Finance returned no prices. Try refreshing shortly.")
            st.stop()
        weekly = adjusted.reindex(index=weeks.index, columns=stocks.index)
        weekly.index = pd.DatetimeIndex(weeks["Week ending"])
        rsi = weekly.apply(wilder_rsi)

    previous = rsi.shift(1)
    eligible = (rsi.notna() & previous.notna()).tail(lookback)
    crossovers = ((previous <= threshold) & (rsi > threshold)).tail(lookback)
    scanned = int(eligible.any(axis=0).sum())
    st.caption(
        f"Latest completed week ending {rsi.index[-1]:%b %d, %Y} · "
        f"{scanned} of {len(stocks)} tickers had at least one scorable week"
    )
    incomplete = stocks.loc[~eligible.all(axis=0), "Symbol"]
    if not incomplete.empty:
        with st.expander(f"{len(incomplete)} tickers with incomplete RSI coverage"):
            st.write(", ".join(incomplete))
    if scanned == 0:
        st.warning("Insufficient weekly closing prices to evaluate RSI.")
        st.stop()

    candidates = crossovers.columns[crossovers.any(axis=0)]
    if candidates.empty:
        st.info(f"No crossovers above {threshold:g} in the last {lookback} completed weeks.")
        st.stop()

    volume_change = three_week_volume_change(volume, sessions, weeks["Week ending"])
    session_for_week = pd.Series(
        weeks.index, index=pd.DatetimeIndex(weeks["Week ending"])
    )
    rows = []
    missing_caps = []
    with st.spinner("Checking market caps and calculating returns..."):
        for symbol in candidates:
            cap = load_market_cap(symbol)
            if cap is None:
                missing_caps.append(stocks.at[symbol, "Symbol"])
                continue
            if cap < min_cap_billions * 1_000_000_000:
                continue
            for week in crossovers.index[crossovers[symbol]]:
                signal_date = session_for_week.at[week]
                rows.append({
                    "Ticker": stocks.at[symbol, "Symbol"],
                    "Company": stocks.at[symbol, "Security"],
                    "Sector": stocks.at[symbol, "GICS Sector"],
                    "Market Cap ($B)": cap / 1_000_000_000,
                    "Week ending": week,
                    "Signal date": signal_date,
                    "Signal close (USD)": closing_price[symbol].get(signal_date, float("nan")),
                    **price_performance(closing_price, symbol, signal_date),
                    "Previous RSI": previous.at[week, symbol],
                    "Signal RSI": rsi.at[week, symbol],
                    "Latest RSI": rsi[symbol].iloc[-1],
                    "3-Week Volume Change (%)": volume_change[symbol],
                })

    if missing_caps:
        st.warning("Market cap unavailable. Excluded from results: " + ", ".join(missing_caps))
    if not rows:
        st.info(f"No signals meet the ${min_cap_billions:g} billion minimum market cap.")
        st.stop()

    signals = pd.DataFrame(rows).sort_values(
        ["Signal date", "Ticker"], ascending=[False, True]
    )
    config = {
        column: st.column_config.NumberColumn(format="%+.2f%%")
        for column in ["Gain / loss (%)", "SPY gain / loss (%)", "3-Week Volume Change (%)"]
    }
    config.update({
        column: st.column_config.DateColumn(format="YYYY-MM-DD")
        for column in ["Week ending", "Signal date", "Price as of"]
    })
    config.update({
        column: st.column_config.NumberColumn(format="$%.2f")
        for column in ["Signal close (USD)", "Latest price (USD)"]
    })
    config.update({
        column: st.column_config.NumberColumn(format="%.2f")
        for column in ["Previous RSI", "Signal RSI", "Latest RSI"]
    })
    config["Market Cap ($B)"] = st.column_config.NumberColumn(
        format="$%.2fB", help="Latest available market capitalization in billions of US dollars."
    )
    config["Gain / loss vs SPY (pp)"] = st.column_config.NumberColumn(
        format="%+.2f", help="Stock return minus SPY return over the same dates, in percentage points."
    )
    st.subheader(f"Buy signals ({len(signals)})")
    st.dataframe(signals, hide_index=True, column_config=config)
    st.caption(
        "Latest price includes today's available regular-session daily bar and may be delayed. "
        "Price as of identifies each row's trading date. On non-trading days, the last available "
        "close is used. Prices refresh from cache every five minutes."
    )
    st.caption(
        "Returns use closing prices and exclude dividends. SPY uses the same start and end "
        "dates as each stock. A stock gain of 10% and SPY gain of 7% produces +3 percentage "
        "points versus SPY. Missing benchmark prices leave comparison cells blank."
    )
    st.caption(
        "Volume change compares the latest completed week's total shares traded with three "
        "weeks earlier. Market cap and volume change use the latest data for every signal row."
    )
    st.download_button(
        "Download signals as CSV",
        data=signals.to_csv(index=False, date_format="%Y-%m-%d"),
        file_name="weekly_rsi_buy_signals.csv", mime="text/csv",
    )
    ticker = st.selectbox("View signal charts", signals["Ticker"].drop_duplicates())
    symbol = stocks.reset_index().set_index("Symbol").at[ticker, "Yahoo symbol"]
    display_weeks = max(26, lookback + 1)
    chart = rsi[symbol].tail(display_weeks).rename("RSI (14)").to_frame()
    chart.index = pd.DatetimeIndex(session_for_week.reindex(chart.index), name="Trading session")
    chart["Crossover level"] = threshold
    chart["Overbought level"] = OVERBOUGHT_LEVEL
    st.subheader(f"{ticker} weekly RSI")
    st.line_chart(chart)
    show_price_chart(ticker, closing_price[symbol], signals, chart.index[0])


if __name__ == "__main__":
    main()
