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
        CONSTITUENTS_URL,
        usecols=["Symbol", "Security", "GICS Sector"],
    )
    stocks["Yahoo symbol"] = stocks["Symbol"].str.replace(
        ".", "-", regex=False
    )
    return stocks.set_index("Yahoo symbol")


def completed_market_weeks(now, lookback):
    today = now.tz_convert(MARKET_TIMEZONE).tz_localize(None).normalize()

    # Include three years of warm-up before the scan window.
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
    sessions = sessions.loc[: completed.index[-1], "Week ending"]
    return completed, sessions


@st.cache_data(ttl=900, show_spinner=False)
def load_daily_data(symbols, start, end):
    batches = []
    batch_size = 50
    fields = ["Adj Close", "Close", "Volume"]

    for offset in range(0, len(symbols), batch_size):
        batch = symbols[offset : offset + batch_size]
        try:
            prices = yf.download(
                list(batch),
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
            # Coverage reporting identifies tickers missing from failed batches.
            continue

    if not batches:
        empty = pd.DataFrame(columns=symbols)
        return empty, empty.copy(), empty.copy()

    daily = pd.concat(batches, axis=1).sort_index()
    daily.index = pd.DatetimeIndex(daily.index).normalize()

    adjusted_close = daily["Adj Close"].reindex(columns=symbols)
    closing_price = daily["Close"].reindex(columns=symbols)
    volume = daily["Volume"].reindex(columns=symbols)
    return adjusted_close, closing_price, volume


@st.cache_data(ttl=3600, show_spinner=False)
def load_market_cap(symbol):
    try:
        value = yf.Ticker(symbol).fast_info.market_cap
        if pd.notna(value) and value > 0:
            return int(value)
    except Exception:
        return None

    return None


def wilder_average(values, period):
    averages = pd.Series(float("nan"), index=values.index, dtype=float)
    valid = values.notna()
    runs = (~valid).cumsum()

    # Seed each uninterrupted run with a simple average.
    for _, run in values[valid].groupby(runs[valid]):
        if len(run) < period:
            continue

        seeded = run.iloc[period - 1 :].copy()
        seeded.iloc[0] = run.iloc[:period].mean()
        averages.loc[seeded.index] = seeded.ewm(
            alpha=1 / period, adjust=False
        ).mean()

    return averages


def wilder_rsi(closes):
    changes = closes.diff()
    average_gain = wilder_average(changes.clip(lower=0), RSI_PERIOD)
    average_loss = wilder_average(-changes.clip(upper=0), RSI_PERIOD)
    total = average_gain + average_loss

    # A flat price series has a neutral RSI.
    return (100 * average_gain / total).mask(total.eq(0), 50.0)


def three_week_volume_change(daily_volume, sessions, week_endings):
    recent_weeks = pd.DatetimeIndex(week_endings.tail(4))
    recent_sessions = sessions[sessions.isin(recent_weeks)]
    volume_by_session = daily_volume.reindex(recent_sessions.index)
    grouped = volume_by_session.groupby(recent_sessions)

    weekly_volume = grouped.sum(min_count=1)
    complete = grouped.count().eq(grouped.size(), axis=0)
    weekly_volume = weekly_volume.where(complete).reindex(recent_weeks)

    baseline = weekly_volume.iloc[0]
    latest = weekly_volume.iloc[-1]
    return latest.div(baseline.where(baseline.gt(0))).sub(1).mul(100)


def show_price_chart(ticker, weekly_prices, signals, display_weeks):
    price_history = (
        weekly_prices.tail(display_weeks)
        .rename("Close")
        .rename_axis("Date")
        .reset_index()
    )
    markers = (
        signals.loc[
            signals["Ticker"].eq(ticker),
            ["Signal date", "Signal close (USD)", "Signal RSI"],
        ]
        .dropna()
        .rename(columns={"Signal date": "Date", "Signal close (USD)": "Close"})
    )

    price_line = (
        alt.Chart(price_history)
        .mark_line()
        .encode(
            x=alt.X("Date:T", title="Trading session"),
            y=alt.Y(
                "Close:Q",
                title="Closing price (USD)",
                scale=alt.Scale(zero=False),
            ),
            tooltip=[
                alt.Tooltip("Date:T", title="Trading session", format="%b %d, %Y"),
                alt.Tooltip("Close:Q", title="Close", format="$,.2f"),
            ],
        )
    )
    signal_points = (
        alt.Chart(markers)
        .mark_point(color="#d1495b", filled=True, size=100)
        .encode(
            x="Date:T",
            y="Close:Q",
            tooltip=[
                alt.Tooltip("Date:T", title="RSI crossover date", format="%b %d, %Y"),
                alt.Tooltip("Close:Q", title="Signal close", format="$,.2f"),
                alt.Tooltip("Signal RSI:Q", format=".2f"),
            ],
        )
    )

    st.subheader(f"{ticker} weekly closing price")
    st.altair_chart(price_line + signal_points, use_container_width=True)


def main():
    st.set_page_config(page_title="S&P 500 Weekly RSI Scanner", layout="wide")
    st.title("S&P 500 Weekly RSI Scanner")

    threshold = st.sidebar.number_input(
        "RSI crossover level",
        min_value=0.0,
        max_value=100.0,
        value=36.0,
        step=0.5,
    )
    lookback = int(
        st.sidebar.number_input(
            "Completed weeks to scan", min_value=1, value=2, step=1
        )
    )

    try:
        stocks = load_constituents()
        weeks, sessions = completed_market_weeks(
            pd.Timestamp.now(tz="UTC"), lookback
        )
    except Exception as exc:
        st.error(f"Could not load the S&P 500 list or market calendar. {exc}")
        st.stop()

    symbols = tuple(stocks.index)
    start = sessions.index[0].date().isoformat()
    end = (sessions.index[-1] + pd.Timedelta(days=1)).date().isoformat()

    with st.spinner("Downloading prices and calculating weekly RSI..."):
        adjusted_close, closing_price, daily_volume = load_daily_data(
            symbols, start, end
        )
        if adjusted_close.empty or not adjusted_close.notna().any().any():
            st.error("Yahoo Finance returned no closing prices. Try again shortly.")
            st.stop()

        weekly_adjusted_close = adjusted_close.reindex(weeks.index)
        weekly_adjusted_close.index = pd.DatetimeIndex(weeks["Week ending"])
        weekly_prices = closing_price.reindex(weeks.index)
        rsi = weekly_adjusted_close.apply(wilder_rsi)

    previous_rsi = rsi.shift(1)
    eligible = (rsi.notna() & previous_rsi.notna()).tail(lookback)
    crossovers = ((previous_rsi <= threshold) & (rsi > threshold)).tail(lookback)

    latest_week = rsi.index[-1].date()
    scanned = int(eligible.any(axis=0).sum())
    incomplete = stocks.loc[~eligible.all(axis=0), "Symbol"]

    st.caption(
        f"Latest completed week ending {latest_week:%b %d, %Y} · "
        f"{scanned} of {len(symbols)} tickers had at least one scorable week"
    )
    if not incomplete.empty:
        with st.expander(f"{len(incomplete)} tickers with incomplete RSI coverage"):
            st.write(", ".join(incomplete))

    if scanned == 0:
        st.warning("There were insufficient weekly closing prices to evaluate RSI.")
        st.stop()

    matching_symbols = crossovers.columns[crossovers.any(axis=0)]
    if matching_symbols.empty:
        st.info(
            f"No RSI crossovers above {threshold:g} "
            f"in the last {lookback} completed weeks."
        )
        st.stop()

    volume_change = three_week_volume_change(
        daily_volume, sessions, weeks["Week ending"]
    )
    session_for_week = pd.Series(
        weeks.index, index=pd.DatetimeIndex(weeks["Week ending"])
    )

    rows = []
    with st.spinner("Loading market caps for matching tickers..."):
        for symbol in matching_symbols:
            stock = stocks.loc[symbol]
            market_cap = load_market_cap(symbol)

            for week in crossovers.index[crossovers[symbol]]:
                signal_date = session_for_week.at[week]
                rows.append(
                    {
                        "Ticker": stock["Symbol"],
                        "Company": stock["Security"],
                        "Sector": stock["GICS Sector"],
                        "Week ending": week.date(),
                        "Signal date": signal_date,
                        "Signal close (USD)": round(
                            weekly_prices.at[signal_date, symbol], 2
                        ),
                        "Previous RSI": round(previous_rsi.at[week, symbol], 2),
                        "Signal RSI": round(rsi.at[week, symbol], 2),
                        "Latest RSI": round(rsi[symbol].iloc[-1], 2),
                        "Market Cap (USD)": market_cap,
                        "3-Week Volume Change (%)": round(volume_change[symbol], 2),
                    }
                )

    signals = pd.DataFrame(rows).sort_values(
        ["Week ending", "Ticker"], ascending=[False, True]
    )
    signals["Market Cap (USD)"] = signals["Market Cap (USD)"].astype("Int64")

    st.subheader(f"Buy signals ({len(signals)})")
    st.dataframe(
        signals,
        hide_index=True,
        column_config={
            "Signal date": st.column_config.DateColumn(format="YYYY-MM-DD"),
            "Signal close (USD)": st.column_config.NumberColumn(
                format="$%.2f",
                help="Closing price on the final trading session of the crossover week.",
            ),
            "Market Cap (USD)": st.column_config.NumberColumn(
                format="dollar",
                help="Latest available market cap in US dollars.",
            ),
            "3-Week Volume Change (%)": st.column_config.NumberColumn(
                format="%+.2f%%",
                help="Latest completed week's total volume versus three weeks earlier.",
            ),
        },
    )
    st.download_button(
        "Download signals as CSV",
        data=signals.to_csv(index=False, date_format="%Y-%m-%d"),
        file_name="weekly_rsi_buy_signals.csv",
        mime="text/csv",
    )

    ticker = st.selectbox("View signal charts", signals["Ticker"].drop_duplicates())
    symbol = stocks.reset_index().set_index("Symbol").at[ticker, "Yahoo symbol"]
    display_weeks = max(26, lookback + 1)

    st.subheader(f"{ticker} weekly RSI")
    chart = rsi[symbol].tail(display_weeks).rename("RSI (14)").to_frame()
    chart.index = pd.DatetimeIndex(
        session_for_week.reindex(chart.index), name="Trading session"
    )
    chart["Crossover level"] = threshold
    chart["Overbought level"] = OVERBOUGHT_LEVEL
    st.line_chart(chart)

    show_price_chart(ticker, weekly_prices[symbol], signals, display_weeks)


if __name__ == "__main__":
    main()