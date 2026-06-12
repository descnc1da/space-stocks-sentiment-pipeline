import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta, timezone
from database import (
    get_connection,
    query_rolling_sentiment,
    query_top_headlines,
    query_intraday_volume_zscore
)


st.set_page_config(
    page_title="Space stocks sentiment dashboard",
    layout="wide"
)

st.title("Space stocks sentiment dashboard (RKLB, ASTS, RDW, LUNR, PL)")

conn = get_connection()
last_update = pd.read_sql_query("""
    SELECT MAX(published_at) as last_updated FROM headlines
""", conn)
conn.close()

last_updated = last_update['last_updated'].iloc[0]

if last_updated:
    last_dt_utc = pd.to_datetime(last_updated)
    last_dt_local = last_dt_utc + timedelta(hours=3)
    st.caption(
        f"Data last updated: {last_dt_local.strftime('%Y-%m-%d %H:%M')} EEST | "
        f"{last_dt_utc.strftime('%H:%M')} UTC"
    )
else:
    st.caption("Data last updated: unknown")

TICKERS = ["RKLB.US", "ASTS.US", "RDW.US", "LUNR.US", "PL.US"]

colors = {
    "RKLB.US": "#00C2FF",
    "ASTS.US": "#FF6B6B",
    "RDW.US": "#FFD93D",
    "LUNR.US": "#6BCB77",
    "PL.US": "#C77DFF"
}

events = {
    "2026-05-29": "Blue Origin explosion",
    "2026-05-27": "SpaceX IPO S-1 filing",
}

highlight_events = {
    "2026-06-12": "SpaceX IPO",
}


def signal_color(zscore, sentiment):
    if zscore >= 2.0 and sentiment > 0.15:
        return "#00C853"
    elif zscore >= 2.0 and sentiment < -0.15:
        return "#D50000"
    elif zscore >= 2.0:
        return "#FFD600"
    elif sentiment > 0.15:
        return "#69F0AE"
    elif sentiment < -0.15:
        return "#FF5252"
    else:
        return "#AAAAAA"


def last_trading_day() -> str:
    date = datetime.now(timezone.utc) - timedelta(days=1)
    while date.weekday() >= 5:
        date -= timedelta(days=1)
    return date.strftime("%Y-%m-%d")


def prev_trading_day(date_str: str) -> str:
    date = datetime.strptime(date_str, "%Y-%m-%d") - timedelta(days=1)
    while date.weekday() >= 5:
        date -= timedelta(days=1)
    return date.strftime("%Y-%m-%d")


def next_trading_day(date_str: str) -> str:
    date = datetime.strptime(date_str, "%Y-%m-%d") + timedelta(days=1)
    while date.weekday() >= 5:
        date += timedelta(days=1)
    return date.strftime("%Y-%m-%d")

target_date = last_trading_day()

# --- SECTION 1: GUIDELINES ---
st.subheader("How to read this dashboard")
st.markdown("""
This is a learning tool for forming hypotheses about how news sentiment and trading volume interact in space tech stocks.
- **Rolling sentiment lines**: shows whether it's a sector-wide move or individual stocks reacting to its own catalyst. 
- **Volume & sentiment**: correlation of stock sentiment with volume z-score to predict daily moves.
- **Headlines**: explains what drove sentiment and by how much.
- **Entry rule**: the requirement is improving sentiment trajectory (last session > previous session) AND intraday volume z > 2.0.
- **Daily log**: I record my prediction before the market opens, score it the next day. """)

# --- SECTION 2: SCORECARDS ---
st.subheader("Last session signals")

conn = get_connection()
scorecard_rows = []
for ticker in TICKERS:
    z_df = query_intraday_volume_zscore(ticker, date=target_date)
    peak_z = z_df['volume_zscore'].max() if not z_df.empty else 0

    sent_df = pd.read_sql_query("""
        SELECT AVG(s.compound) as avg_sentiment
        FROM headlines h
        JOIN sentiment s ON h.id = s.headline_id
        WHERE h.ticker = ? AND DATE(h.published_at) = DATE(?)
    """, conn, params=(ticker, target_date))
    sentiment = sent_df['avg_sentiment'].iloc[0] if not sent_df.empty else None

    entry_rule = peak_z >= 2.0 and (sentiment or 0) > 0.15
    color = signal_color(peak_z, sentiment or 0)

    scorecard_rows.append({
        "Ticker": ticker.replace(".US", ""),
        "Sentiment": f"{sentiment:+.2f}" if sentiment is not None else "N/A",
        "Volume Z": f"{peak_z:.2f}",
        "Entry rule met": "✅" if entry_rule else "❌",
        "_color": color
    })
conn.close()

cols = st.columns(5)
for col, row in zip(cols, scorecard_rows):
    with col:
        st.markdown(f"""
        <div style="border-left: 4px solid {row['_color']}; padding: 8px 12px; border-radius: 4px; background: rgba(128,128,128,0.05)">
            <div style="font-size: 1.1rem; font-weight: 700">{row['Ticker']}</div>
            <div style="font-size: 0.85rem">Sentiment: <b>{row['Sentiment']}</b></div>
            <div style="font-size: 0.85rem">Vol Z: <b>{row['Volume Z']}</b></div>
            <div style="font-size: 0.85rem">{row['Entry rule met']} Entry rule</div>
        </div>
        """, unsafe_allow_html=True)

# --- SECTION 3: ROLLING SENTIMENT ---
st.subheader("Rolling sentiment")

rolling = query_rolling_sentiment()

if rolling.empty:
    st.warning("No sentiment data available.")
else:
    rolling['date'] = pd.to_datetime(rolling['date'])
    
    min_date = rolling['date'].min().date()
    max_date = rolling['date'].max().date()
    
    date_range = st.date_input(
        "Date range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )
    
    if len(date_range) == 2:
        start, end = date_range
        rolling = rolling[
            (rolling['date'].dt.date >= start) &
            (rolling['date'].dt.date <= end)
        ]

    fig = go.Figure()

    for ticker in rolling["ticker"].unique():
        t = rolling[rolling["ticker"] == ticker]
        fig.add_trace(go.Scatter(
            x=t["date"],
            y=t["rolling_avg_sentiment"],
            name=ticker,
            mode="lines+markers",
            line=dict(color=colors.get(ticker, "gray"), width=2),
            marker=dict(size=4)
        ))

    fig.add_hline(
        y=0,
        line_dash="dash",
        line_color="gray"
    )

    fig.update_layout(
        margin=dict(t=80),
        height=400,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        yaxis_title="Sentiment",
        xaxis_title="",
        hovermode="x unified",
        legend=dict(orientation="h", y=-0.2),
        yaxis=dict(gridcolor="rgba(128,128,128,0.2)"),
        xaxis=dict(gridcolor="rgba(128,128,128,0.2)",
            tickformat="%b %d"
)
    )

    for date, label in events.items():
        fig.add_vline(
            x=date,
            line_dash="dot",
            line_color="rgba(128,128,128,0.8)",
            line_width=1,
        )
        fig.add_annotation(
            x=date,
            y=1,
            yref="paper",
            text=label,
            showarrow=False,
            textangle=-45,
            font=dict(size=9),
            yshift=60
        )
    
    for date, label in highlight_events.items():
        fig.add_vline(
            x=date,
            line_width=2,
            line_color="rgba(255, 51, 51, 1)",
            line_dash="dot",
        )
        fig.add_annotation(
            x=date,
            y=1,
            yref="paper",
            text=label,
            showarrow=False,
            textangle=-45,
            font=dict(size=10, color="rgba(255, 51, 51, 1)"),
            yshift=60,
            xshift=10,
        )

    st.plotly_chart(fig, use_container_width=True)

# --- SECTION 4: VOLUME Z-SCORE + SENTIMENT PER TICKER ---
st.subheader("Volume & sentiment")

selected_ticker = st.selectbox("Select ticker", TICKERS)

time_range = st.radio(
    "Time range",
    ["1D", "5D", "1M"],
    horizontal=True
)

# --- SECTION 5: ENTRY RULE EVALUATOR ---
with st.expander("📋 Entry rule check", expanded=False):
    st.markdown("""
    **Rule:** Sentiment trajectory improving (last session > previous session) **AND** volume z-score > 2.0
    """)
    conn = get_connection()
    yesterday = prev_trading_day(target_date)

    rule_sent = pd.read_sql_query("""
        SELECT DATE(h.published_at) as date, AVG(s.compound) as avg_sentiment
        FROM headlines h
        JOIN sentiment s ON h.id = s.headline_id
        WHERE h.ticker = ?
        AND DATE(h.published_at) IN (?, ?)
        GROUP BY DATE(h.published_at)
        ORDER BY date DESC
    """, conn, params=(selected_ticker, target_date, yesterday))
    conn.close()

    if len(rule_sent) == 2 and rule_sent.iloc[0]['date'] == target_date and rule_sent.iloc[1]['date'] == yesterday:
        today_s = rule_sent.iloc[0]['avg_sentiment'] or 0
        yesterday_s = rule_sent.iloc[1]['avg_sentiment'] or 0
        improving = today_s > yesterday_s

        z_df = query_intraday_volume_zscore(selected_ticker, date=target_date)
        peak_z = z_df['volume_zscore'].max() if not z_df.empty else 0
        volume_confirms = peak_z >= 2.0

        col1, col2, col3 = st.columns(3)
        col1.metric(f"Sentiment {target_date}", f"{today_s:+.2f}", f"{today_s - yesterday_s:+.2f} vs prev session")
        col2.metric("Peak volume z", f"{peak_z:.2f}", "✅ confirms" if volume_confirms else "❌ no confirm")
        col3.metric("Entry rule", "✅ MET" if (improving and volume_confirms) else "❌ NOT MET")
    else:
        st.info("Not enough sentiment history to evaluate rule.")

if time_range == "1D":
    intraday_df = query_intraday_volume_zscore(
        selected_ticker,
        date=target_date,
    )

    if intraday_df.empty:
        st.warning("No intraday data for this date.")
    else:
        conn = get_connection()
        day_sentiment = pd.read_sql_query("""
            SELECT AVG(s.compound) as avg_sentiment
            FROM headlines h
            JOIN sentiment s ON h.id = s.headline_id
            WHERE h.ticker = ?
            AND DATE(h.published_at) = DATE(?)
        """, conn, params=(selected_ticker, target_date))
        conn.close()

        sentiment_val = day_sentiment['avg_sentiment'].iloc[0] if not day_sentiment.empty else 0
        sentiment_val = sentiment_val if sentiment_val is not None else 0

        latest_z = intraday_df['volume_zscore'].max()
        if latest_z >= 3.0 and sentiment_val > 0.3:
            st.warning(f"⚠️ POSSIBLE EXHAUSTION — Peak volume z: {latest_z:.2f}, Sentiment: {sentiment_val:+.2f}")
        elif latest_z >= 2.0 and sentiment_val > 0.15:
            st.success(f"🟢 BREAKOUT CONFIRMED — Peak volume z: {latest_z:.2f}, Sentiment: {sentiment_val:+.2f}")
        elif latest_z >= 2.0 and sentiment_val < -0.15:
            st.error(f"🔴 CONFIRMED SELLING — Peak volume z: {latest_z:.2f}, Sentiment: {sentiment_val:+.2f}")
        elif intraday_df['volume_zscore'].mean() <= -2.0:
            st.info(f"⚪ LOW CONVICTION — Avg volume z: {intraday_df['volume_zscore'].mean():.2f}, Sentiment: {sentiment_val:+.2f}")
        else:
            st.info(f"Peak volume z: {latest_z:.2f} | Sentiment: {sentiment_val:+.2f} — No clear signal.")

        bar_color = "#6BCB77" if sentiment_val > 0 else "#FF6B6B" if sentiment_val < 0 else "#AAAAAA"

        fig_intraday = go.Figure()
        fig_intraday.add_trace(go.Bar(
            x=intraday_df["datetime"],
            y=intraday_df["volume_zscore"],
            marker_color=bar_color,
            name="Volume Z-Score"
        ))

        fig_intraday.add_hline(y=2.0, line_dash="dot", line_color="orange",
            annotation_text="z=2", annotation_position="top right",
            annotation_font_size=9)
        fig_intraday.add_hline(y=-2.0, line_dash="dot", line_color="orange",
            annotation_text="z=-2", annotation_position="bottom right",
            annotation_font_size=9)
        fig_intraday.add_hline(y=0, line_dash="dash", line_color="gray")

        fig_intraday.update_layout(
            height=400,
            title=f"{selected_ticker} — Intraday Volume Z-Score ({target_date})",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            yaxis_title="Volume Z-Score",
            xaxis_title="Time",
            yaxis=dict(gridcolor="rgba(128,128,128,0.2)"),
            xaxis=dict(
                gridcolor="rgba(128,128,128,0.2)",
                tickformat="%H:%M"
)
        )
        st.plotly_chart(fig_intraday, use_container_width=True)

else:
    days_map = {"5D": 5, "1M": 30}
    days_back = days_map[time_range]

    conn = get_connection()
    volume_df = pd.read_sql_query("""
        WITH baseline AS (
            SELECT
                ticker,
                AVG(volume) as avg_volume,
                AVG(volume * volume) - AVG(volume) * AVG(volume) as variance
            FROM prices
            WHERE ticker = ?
            GROUP BY ticker
        )
        SELECT
            p.date,
            p.volume,
            p.close,
            ROUND((p.volume - b.avg_volume) / SQRT(b.variance), 2) as volume_zscore
        FROM prices p
        JOIN baseline b ON p.ticker = b.ticker
        WHERE p.ticker = ?
        AND p.date >= DATE('now', ?)
        ORDER BY p.date
    """, conn, params=(selected_ticker, selected_ticker, f'-{days_back} days'))

    sentiment_df = pd.read_sql_query("""
        SELECT
            DATE(h.published_at) as date,
            AVG(s.compound) as avg_sentiment
        FROM headlines h
        JOIN sentiment s ON h.id = s.headline_id
        WHERE h.ticker = ?
        AND DATE(h.published_at) >= DATE('now', ?)
        GROUP BY DATE(h.published_at)
        ORDER BY date
    """, conn, params=(selected_ticker, f'-{days_back} days'))
    conn.close()

    if volume_df.empty:
        st.warning("No volume data available.")
    else:
        if not sentiment_df.empty:
            latest_z = volume_df['volume_zscore'].iloc[-1]
            latest_date = volume_df['date'].iloc[-1]
            sent_on_date = sentiment_df[sentiment_df['date'] == latest_date]

            if sent_on_date.empty:
                st.info(f"Volume z: {latest_z:.2f} | No sentiment data for {latest_date} — no signal.")
            else:
                latest_sentiment = sent_on_date['avg_sentiment'].iloc[0]
                latest_sentiment = latest_sentiment if latest_sentiment is not None else 0

                if latest_z >= 3.0 and latest_sentiment > 0.3:
                    st.warning(f"⚠️ POSSIBLE EXHAUSTION — Volume z: {latest_z:.2f}, Sentiment: {latest_sentiment:+.2f}")
                elif latest_z >= 2.0 and latest_sentiment > 0.15:
                    st.success(f"🟢 BREAKOUT CONFIRMED — Volume z: {latest_z:.2f}, Sentiment: {latest_sentiment:+.2f}")
                elif latest_z >= 2.0 and latest_sentiment < -0.15:
                    st.error(f"🔴 CONFIRMED SELLING — Volume z: {latest_z:.2f}, Sentiment: {latest_sentiment:+.2f}")
                elif latest_z <= -2.0 and abs(latest_sentiment) > 0.15:
                    st.info(f"⚪ WEAK CONVICTION — Volume z: {latest_z:.2f}, Sentiment: {latest_sentiment:+.2f}")
                else:
                    st.info(f"Volume z: {latest_z:.2f} | Sentiment: {latest_sentiment:+.2f} — No clear signal.")

        merged = volume_df.merge(sentiment_df, on='date', how='left')

        bar_colors = [
            signal_color(z, s)
            for z, s in zip(merged['volume_zscore'], merged['avg_sentiment'].fillna(0))
        ]

        fig4 = go.Figure()

        fig4.add_trace(go.Bar(
            x=merged["date"],
            y=merged["volume_zscore"],
            marker_color=bar_colors,
            name="Volume Z-Score",
            yaxis="y1"
        ))

        fig4.add_trace(go.Scatter(
            x=sentiment_df["date"],
            y=sentiment_df["avg_sentiment"],
            mode="lines+markers",
            line=dict(color=colors.get(selected_ticker, "#00C2FF"), width=2),
            name="Sentiment",
            marker=dict(size=6),
            yaxis="y2"
        ))

        fig4.update_layout(
            height=450,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            hovermode="x unified",
            legend=dict(orientation="h", y=-0.2),
            yaxis=dict(
                title="Volume Z-Score",
                gridcolor="rgba(128,128,128,0.2)",
                zeroline=True,
                zerolinecolor="gray",
                zerolinewidth=1,
                autorange=True
            ),
            yaxis2=dict(
                title="Sentiment",
                overlaying="y",
                side="right",
                range=[-1, 1],
                gridcolor="rgba(0,0,0,0)",
                zeroline=False
            )
        )

        fig4.add_hline(y=2.0, line_dash="dot", line_color="rgba(255,180,0,0.6)",
            annotation_text="z=2", annotation_position="top left",
            annotation_font_size=9)
        fig4.add_hline(y=-2.0, line_dash="dot", line_color="rgba(255,180,0,0.6)",
            annotation_text="z=-2", annotation_position="bottom left",
            annotation_font_size=9)

        st.plotly_chart(fig4, use_container_width=True)
        st.caption("🟢 Breakout confirmed | 🔴 Confirmed selling | 🟡 Volume spike unclear | Light green/red = sentiment without volume | ⬜ No signal")

# --- SECTION 6: HEADLINES ---
st.subheader(f"What drove sentiment for {selected_ticker}?")
headlines_df = query_top_headlines(date=target_date)
t = headlines_df[headlines_df["ticker"] == selected_ticker]

if t.empty:
    st.write("No headlines for this ticker on the selected date.")
else:
    conn = get_connection()
    next_day = next_trading_day(target_date)
    price_df = pd.read_sql_query("""
        SELECT date, close FROM prices
        WHERE ticker = ? AND date IN (?, ?)
        ORDER BY date
    """, conn, params=(selected_ticker, target_date, next_day))
    conn.close()

    if len(price_df) == 2:
        price_today = price_df.iloc[0]['close']
        price_next = price_df.iloc[1]['close']
        pct_change = (price_next - price_today) / price_today * 100
        direction = "🟢" if pct_change > 0.5 else "🔴" if pct_change < -0.5 else "⚪"
        st.caption(f"Next day price move: {direction} {pct_change:+.2f}%")
    else:
        st.caption(f"Next session close: not yet available — check after market close")

    for _, row in t.head(5).iterrows():
        icon = "🟢" if row["label"] == "positive" else "🔴" if row["label"] == "negative" else "⚪"
        is_specific = any(
            kw.lower() in row['title'].lower()
            for kw in ["ast spacemobile", "asts", "rocket lab", "rklb",
                    "redwire", "rdw", "lunr", "planet labs"]
        )
        specificity = "🏢" if is_specific else "🌐"

        pub_dt_utc = pd.to_datetime(row['published_at'], utc=True)
        pub_dt_local = pub_dt_utc + timedelta(hours=3)
        meta = f"<span style='color:gray; font-size:0.78rem'>{pub_dt_local.strftime('%b %d %H:%M')} EEST · {pub_dt_utc.strftime('%H:%M')} UTC</span>"
        
        st.markdown(
            f"{icon} {specificity} [{row['title']}]({row['url']}) `{row['compound']:+.2f}`<br>{meta}",
            unsafe_allow_html=True
        )

# --- SECTION 7: DAILY LOG ---
st.subheader("📓 Daily log")
st.caption("Record your read before you see what happens. Come back tomorrow to check.")

with st.form("log_form"):
    log_ticker = st.selectbox("Ticker", TICKERS, key="log_ticker")
    log_read = st.selectbox("My read today", [
        "Sector noise — expect flat",
        "Company-specific bullish — expect up",
        "Company-specific bearish — expect down",
        "Volume spike, no conviction — watch only",
        "Entry rule met — high confidence long setup"
    ])
    log_notes = st.text_input("Notes (optional)")
    submitted = st.form_submit_button("Save read")

if submitted:
    conn = get_connection()
    conn.execute("""
        INSERT INTO daily_log (date, ticker, read, notes)
        VALUES (?, ?, ?, ?)
    """, (target_date, log_ticker, log_read, log_notes))
    conn.commit()
    conn.close()
    st.success("Saved.")

# Show past logs
conn = get_connection()
log_df = pd.read_sql_query("""
    SELECT id, date, ticker, read, was_right, notes
    FROM daily_log
    ORDER BY date DESC
    LIMIT 20
""", conn)
conn.close()

if not log_df.empty:
    st.markdown("**Past logs**")
    for _, row in log_df.iterrows():
        correct = "✅" if row['was_right'] == 1 else "❌" if row['was_right'] == 0 else "⬜ unscored"
        col1, col2, col3 = st.columns([8, 0.3, 0.3])
        with col1:
            st.markdown(
                f"`{row['date']}` **{row['ticker']}** — {row['read']} {correct}"
                + (f" — _{row['notes']}_" if row['notes'] else "")
            )
        with col2:
            if st.button("✏️", key=f"log_edit_{row['id']}"):
                st.session_state[f"log_editing_{row['id']}"] = True
        with col3:
            if st.button("🗑️", key=f"log_delete_{row['id']}"):
                conn = get_connection()
                conn.execute("DELETE FROM daily_log WHERE id = ?", (row['id'],))
                conn.commit()
                conn.close()
                st.rerun()

        if st.session_state.get(f"log_editing_{row['id']}", False):
            with st.form(key=f"edit_form_{row['id']}"):
                new_read = st.selectbox("Read", [
                    "Sector noise — expect flat",
                    "Company-specific bullish — expect up",
                    "Company-specific bearish — expect down",
                    "Volume spike, no conviction — watch only",
                    "Entry rule met — high confidence long setup"
                ], index=[
                    "Sector noise — expect flat",
                    "Company-specific bullish — expect up",
                    "Company-specific bearish — expect down",
                    "Volume spike, no conviction — watch only",
                    "Entry rule met — high confidence long setup"
                ].index(row['read']) if row['read'] in [
                    "Sector noise — expect flat",
                    "Company-specific bullish — expect up",
                    "Company-specific bearish — expect down",
                    "Volume spike, no conviction — watch only",
                    "Entry rule met — high confidence long setup"
                ] else 0)
                new_notes = st.text_input("Notes", value=row['notes'] or "")
                save = st.form_submit_button("Save")
                cancel = st.form_submit_button("Cancel")
                if save:
                    conn = get_connection()
                    conn.execute(
                        "UPDATE daily_log SET read = ?, notes = ? WHERE id = ?",
                        (new_read, new_notes, row['id'])
                    )
                    conn.commit()
                    conn.close()
                    st.session_state[f"log_editing_{row['id']}"] = False
                    st.rerun()
                if cancel:
                    st.session_state[f"log_editing_{row['id']}"] = False
                    st.rerun()

if not log_df.empty:
    st.markdown("**Score a past read**")
    options = {
        f"{row['date']} | {row['ticker']} | {row['read']}": row['id']
        for _, row in log_df.iterrows()
        if pd.isna(row['was_right'])
    }

    if not options:
        st.caption("All entries scored.")
    else:
        selected_log = st.selectbox("Select entry to score", list(options.keys()))
        score = st.radio("Was your read correct?", ["Yes", "No"], horizontal=True)
        if st.button("Save score"):
            conn = get_connection()
            conn.execute(
                "UPDATE daily_log SET was_right = ? WHERE id = ?",
                (1 if score == "Yes" else 0, options[selected_log])
            )
            conn.commit()
            conn.close()
            st.success("Score saved.")
            st.rerun()