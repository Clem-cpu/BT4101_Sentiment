import requests
from datetime import datetime, timedelta, timezone
import plotly.graph_objs as go
import plotly.offline as pyo
from flask import Flask, render_template
import time
import pandas as pd

app = Flask(__name__)


@app.route("/")
def bitcoin_price():
    # Load bitcoin price data from the Binance API
    api_url = "https://api.binance.com/api/v3/klines"
    symbol = "BTCUSDT"
    interval = "1d"
    start_date = "2022-01-01 00:00:00"
    start_timestamp = (
        int(time.mktime(time.strptime(start_date, "%Y-%m-%d %H:%M:%S"))) * 1000
    )
    dt = datetime.now(timezone.utc)
    utc_time = dt.replace(tzinfo=timezone.utc)
    utc_timestamp = utc_time.timestamp()
    end_timestamp = int(utc_timestamp * 1000)

    params = {
        "symbol": symbol,
        "interval": interval,
        "startTime": start_timestamp,
        "endTime": end_timestamp,
    }
    response = requests.get(api_url, params=params)
    data = response.json()

    # Parse the data into dates and prices
    dates = []
    prices = []
    for item in data:
        timestamp = int(item[0]) / 1000
        date = datetime.fromtimestamp(timestamp).date()
        price = float(item[4])
        dates.append(date)
        prices.append(price)

    proba_df = pd.read_csv("./test/application_files/output.csv")
    prob_dates = proba_df["date"].values
    probabilities = proba_df["probability"].values
    # Create the Plotly graph with two y-axes
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=prices, mode="lines", name="Bitcoin Price"))
    fig.add_trace(
        go.Scatter(
            x=prob_dates, y=probabilities, mode="lines", name="Probability", yaxis="y2"
        )
    )
    # Add the horizontal line at the middle of the y-axis
    fig.add_shape(
        type="line",
        y0=0.5,
        y1=0.5,
        x0=dates[0],
        x1=dates[-1],
        line=dict(color="red", width=2, dash="dot"),
        yref="y2",
    )

    fig.update_layout(
        title="Bitcoin Price and Probability",
        yaxis=dict(
            title="Price (USDT)",
            tickformat="$,.0f",
            tick0=5000,
            dtick=5000,
            gridcolor="lightgray",
            zeroline=False,
            showgrid=True,
        ),
        yaxis2=dict(
            title="Probability",
            overlaying="y",
            side="right",
            range=[0, 1],
            gridcolor="lightgray",
            tick0=0,
            dtick=0.25,
            zeroline=False,
            showgrid=True,
        ),
    )
    fig.update_layout(height=700, width=1000)

    # Convert the graph to HTML and render the template
    graph = pyo.plot(fig, output_type="div")
    return render_template("bitcoin_price.html", graph=graph)


if __name__ == "__main__":
    app.run(debug=True)
