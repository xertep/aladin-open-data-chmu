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
    path = load_data(url)

    ds = xr.open_dataset(path, engine="cfgrib")

    st.write(ds)

    # ---- GET PRECIP ----
    tp = ds[list(ds.data_vars)[0]]  # safer than guessing name

    # ---- COMPUTE 3h PRECIP ----
    rain_3h = tp.diff(dim="step", n=3)
    rain_3h = rain_3h.pad(step=(3, 0))
    rain_3h = rain_3h.clip(min=0)

    # ---- SELECT STEP ----
    step_idx = st.slider("Forecast step", 0, len(rain_3h.step)-1, 10)

    data = rain_3h.isel(step=step_idx)

    # ---- PLOT ----
    fig, ax = plt.subplots(figsize=(6, 5))
    data.plot(ax=ax)
    ax.set_title(f"3h precipitation (step {step_idx})")

    st.pyplot(fig)
