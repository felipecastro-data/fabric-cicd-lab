# Fabric notebook source

# METADATA ********************

# META {
# META   "kernel_info": {
# META     "name": "synapse_pyspark"
# META   },
# META   "dependencies": {
# META     "lakehouse": {
# META       "default_lakehouse": "ea4fef77-4858-4c38-b041-daf2f384c708",
# META       "default_lakehouse_name": "lh_divvy_bench",
# META       "default_lakehouse_workspace_id": "0cf64660-62b7-4220-a83a-4d4a1cf90511",
# META       "known_lakehouses": [
# META         {
# META           "id": "ea4fef77-4858-4c38-b041-daf2f384c708"
# META         }
# META       ]
# META     }
# META   }
# META }

# CELL ********************

import os, io, zipfile, requests

SRC_URL  = "https://divvy-tripdata.s3.amazonaws.com/202609-divvy-tripdata.zip"
RAW_DIR  = "/lakehouse/default/Files/raw/divvy"
CSV_NAME = "202609-divvy-tripdata.csv"
CSV_PATH = f"{RAW_DIR}/{CSV_NAME}"

os.makedirs(RAW_DIR, exist_ok=True)

if not os.path.exists(CSV_PATH):              
    resp = requests.get(SRC_URL, timeout=120)
    resp.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        zf.extract(CSV_NAME, RAW_DIR)         

print(CSV_PATH, f"{os.path.getsize(CSV_PATH)/1e6:.1f} MB")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

import pandas as pd
import numpy as np

COLS = ["ride_id", "rideable_type", "started_at", "ended_at",
        "start_lat", "start_lng", "end_lat", "end_lng", "member_casual"]

df_raw = pd.read_csv(CSV_PATH, usecols=COLS, parse_dates=["started_at", "ended_at"])
print("Raw shape:", df_raw.shape)
print("\nNulls per column:\n", df_raw.isna().sum())

GEO = ["start_lat", "start_lng", "end_lat", "end_lng"]
df = df_raw.dropna(subset=GEO).reset_index(drop=True)
df["duration_min"] = (df["ended_at"] - df["started_at"]).dt.total_seconds() / 60

print(f"\nBenchmark rows: {len(df):,}  (dropped {len(df_raw) - len(df):,} without end coords)")
print(f"Memory: {df.memory_usage(deep=True).sum()/1e6:.1f} MB")
df.head()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

import math, time

R_KM = 6371.0088          
results = []              

def timed(task, method, fn):
    t0 = time.perf_counter()
    out = fn()
    secs = time.perf_counter() - t0
    results.append({"task": task, "method": method, "rows": len(df), "seconds": secs})
    print(f"[{task}] {method}: {secs:.3f} s for {len(df):,} rows")
    return out

def haversine_km(lat1, lon1, lat2, lon2):
    """Scalar haversine - one trip at a time."""
    lat1, lon1, lat2, lon2 = map(math.radians, (lat1, lon1, lat2, lon2))
    a = (math.sin((lat2 - lat1) / 2) ** 2
         + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2)
    return 2 * R_KM * math.asin(math.sqrt(a))

def dist_iterrows(frame):
    out = []
    for _, row in frame.iterrows():           
        out.append(haversine_km(row["start_lat"], row["start_lng"],
                                row["end_lat"], row["end_lng"]))
    return pd.Series(out, index=frame.index)

dist_iter = timed("distance", "iterrows", lambda: dist_iterrows(df))
dist_iter.describe()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

dist_apply = timed(
    "distance", "apply_axis1",
    lambda: df.apply(lambda r: haversine_km(r["start_lat"], r["start_lng"],
                                            r["end_lat"], r["end_lng"]), axis=1)
)

print("Matches iterrows:", np.allclose(dist_apply, dist_iter))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

def haversine_km_vec(lat1, lon1, lat2, lon2):
    """Array haversine - all trips in one pass (C-level loops inside NumPy)."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
    return 2 * R_KM * np.arcsin(np.sqrt(a))

args = [df[c].to_numpy() for c in ["start_lat", "start_lng", "end_lat", "end_lng"]]
haversine_km_vec(*args) 

dist_vec = timed("distance", "vectorized_numpy",
                 lambda: pd.Series(haversine_km_vec(*args), index=df.index))

print("Matches iterrows:", np.allclose(dist_vec, dist_iter))
print(f"Speedup vs iterrows: {results[0]['seconds'] / results[-1]['seconds']:,.0f}x")
print(f"Speedup vs apply:    {results[1]['seconds'] / results[-1]['seconds']:,.0f}x")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Haversine distance on 769,918 real Divvy bike trips (Chicago, Sep 2026)
import time
cols = ["start_lat", "start_lng", "end_lat", "end_lng"]

# 1) Row-wise loop: Python runs the math once per ROW
t0 = time.perf_counter()
loop_km = [haversine_km(r["start_lat"], r["start_lng"], r["end_lat"], r["end_lng"])
           for _, r in df.iterrows()]
t_loop = time.perf_counter() - t0

# 2) Vectorized: NumPy runs the math once per COLUMN (compiled C)
t0 = time.perf_counter()
lat1, lon1, lat2, lon2 = np.radians(df[cols].to_numpy().T)
a = np.sin((lat2 - lat1) / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2)**2
vec_km = 2 * 6371.0088 * np.arcsin(np.sqrt(a))
t_vec = time.perf_counter() - t0

print(f"Row-wise loop : {t_loop:8.3f} s")
print(f"Vectorized    : {t_vec:8.3f} s")
print(f"Speedup       : {t_loop / t_vec:8.0f}x  | identical results: {np.allclose(loop_km, vec_km)}")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

import matplotlib.pyplot as plt

res = (pd.DataFrame(results)
         .query("task == 'distance'")
         .assign(speedup_vs_iterrows=lambda d: d["seconds"].iloc[0] / d["seconds"])
         .round({"seconds": 3, "speedup_vs_iterrows": 0}))
display(res)

fig, ax = plt.subplots(figsize=(8, 3))
bars = ax.barh(res["method"], res["seconds"], color=["#c0392b", "#e67e22", "#27ae60"])
ax.set_xscale("log")
ax.invert_yaxis()
ax.set_xlabel("Seconds (log scale) - lower is better")
ax.set_title(f"Haversine distance on {len(df):,} real Divvy trips (Sep 2026)")
for b, s, x in zip(bars, res["seconds"], res["speedup_vs_iterrows"]):
    ax.text(b.get_width() * 1.1, b.get_y() + b.get_height() / 2,
            f"{s:.3f} s  ({x:,.0f}x)", va="center")
ax.set_xlim(right=res["seconds"].max() * 8)
plt.tight_layout()
plt.savefig("/lakehouse/default/Files/raw/divvy/benchmark_vectorization.png", dpi=200)
plt.show()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }
