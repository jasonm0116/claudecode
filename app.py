from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from scipy import stats

app = Flask(__name__)


def compute_stats(data):
    arr = np.array(data)
    n = len(arr)
    mean = float(np.mean(arr))
    median = float(np.median(arr))
    std = float(np.std(arr, ddof=1)) if n > 1 else 0.0
    variance = float(np.var(arr, ddof=1)) if n > 1 else 0.0
    skewness = float(stats.skew(arr)) if n > 2 else 0.0
    kurt = float(stats.kurtosis(arr)) if n > 3 else 0.0
    q1 = float(np.percentile(arr, 25))
    q3 = float(np.percentile(arr, 75))
    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    outliers = [float(x) for x in arr if x < lower or x > upper]

    if 3 <= n <= 5000:
        stat_sw, p_sw = stats.shapiro(arr)
        normality = {"statistic": round(float(stat_sw), 4), "p_value": round(float(p_sw), 4)}
    else:
        normality = None

    counts, bin_edges = np.histogram(arr, bins="auto")
    histogram = {
        "counts": counts.tolist(),
        "bin_edges": [round(float(e), 4) for e in bin_edges],
    }

    return {
        "n": n,
        "mean": round(mean, 4),
        "median": round(median, 4),
        "mode": round(float(stats.mode(arr, keepdims=True).mode[0]), 4),
        "std": round(std, 4),
        "variance": round(variance, 4),
        "min": round(float(arr.min()), 4),
        "max": round(float(arr.max()), 4),
        "range": round(float(arr.max() - arr.min()), 4),
        "q1": round(q1, 4),
        "q3": round(q3, 4),
        "iqr": round(iqr, 4),
        "skewness": round(skewness, 4),
        "kurtosis": round(kurt, 4),
        "outliers": outliers,
        "normality": normality,
        "histogram": histogram,
        "sorted": sorted([float(x) for x in arr]),
    }


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    payload = request.get_json()
    columns = payload.get("columns", {})
    results = {}
    for col, values in columns.items():
        try:
            clean = [float(v) for v in values if v != "" and v is not None]
            if len(clean) < 2:
                results[col] = {"error": "Need at least 2 numeric values"}
            else:
                results[col] = compute_stats(clean)
        except Exception as e:
            results[col] = {"error": str(e)}
    return jsonify(results)


@app.route("/correlate", methods=["POST"])
def correlate():
    payload = request.get_json()
    columns = payload.get("columns", {})
    df = pd.DataFrame({k: pd.to_numeric(v, errors="coerce") for k, v in columns.items()})
    df = df.dropna()
    if df.shape[1] < 2 or df.shape[0] < 3:
        return jsonify({"error": "Need at least 2 columns and 3 rows for correlation"})
    corr = df.corr().round(4)
    return jsonify({"matrix": corr.to_dict(), "columns": list(corr.columns)})


if __name__ == "__main__":
    app.run(debug=False)
