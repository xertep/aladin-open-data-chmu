import streamlit as st
import requests
import bz2
import xarray as xr
import matplotlib.pyplot as plt
import tempfile
import numpy as np

st.title("CHMI ALADIN 3h Precipitation 🌧️")

# ---- USER INPUT ----
date = st.text_input("Date (YYYYMMDD)", "20260412")
run = st.selectbox("Model run (UTC)", ["00", "06", "12", "18"])

url = f"https://opendata.chmi.cz/meteorology/weather/nwp_aladin/CZ_1km/{run}/ALADCZ1K4opendata_{date}{run}_SURFPREC_TOTAL.grb.bz2"

st.write("Data URL:", url)

# ---- LOAD DATA ----
@st.cache_data(show_spinner=True)
def load_data(url):
    r = requests.get(url)
    compressed = r.content

    # decompress bz2
    grib_bytes = bz2.decompress(compressed)

    # save to temp file (cfgrib needs file path)
    with tempfile.NamedTemporaryFile(suffix=".grb", delete=False) as f:
        f.write(grib_bytes)
        return f.name

if st.button("Load data"):
    st.session_state["path"] = load_data(url)

@st.cache_data
def open_grib(path):
    return xr.open_dataset(path, engine="cfgrib")

if "path" in st.session_state:
    path = st.session_state["path"]

    ds = open_grib(path)

    st.write("Dataset loaded")
    #st.write("Variables:", list(ds.data_vars))

    tp = ds[list(ds.data_vars)[0]]

    rain_3h = tp.diff(dim="step", n=3)
    rain_3h = rain_3h.isel(step=slice(3, None, 3))
    rain_3h = rain_3h.clip(min=0)

    step_idx = st.slider("3h interval", 0, len(rain_3h.step)-1, 0)

    data = rain_3h.isel(step=step_idx)

    import pandas as pd
    end_time = pd.to_datetime(data.valid_time.values)
    start_time = end_time - pd.Timedelta(hours=3)

    st.write(f"{start_time:%H:%M} – {end_time:%H:%M} UTC")

    data_small = data[::4, ::4]

    fig, ax = plt.subplots(figsize=(6, 5))
    data_small.plot(ax=ax)
    ax.set_title("3-hour precipitation")

    st.pyplot(fig)
