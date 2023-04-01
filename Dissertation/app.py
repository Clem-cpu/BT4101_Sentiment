from flask import Flask, render_template
import requests
import json

app = Flask(__name__)


@app.route("/")
def index():
    # Get the daily price data for Bitcoin
    response = requests.get(
        "https://api.coindesk.com/v1/bpi/historical/close.json?start=2021-01-01&end=2023-02-02"
    )
    data = json.loads(response.text)

    # Prepare the data for Plotly
    x = [d for d in data["bpi"].keys()]
    y = [data["bpi"][d] for d in x]
    plot_data = [{"x": x, "y": y, "type": "line", "name": "Bitcoin Price"}]
    plot_layout = {
        "title": "Daily Bitcoin Price",
        "xaxis_title": "Date",
        "yaxis_title": "Price (USD)",
    }

    return render_template("index.html", plot_data=plot_data, plot_layout=plot_layout)


if __name__ == "__main__":
    app.run()
