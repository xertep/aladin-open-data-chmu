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

# st.write("Data URL:", url)

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

    import pandas as pd
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    st.write("Dataset loaded")

    tp = ds[list(ds.data_vars)[0]]

    rain_3h = tp.diff(dim="step", n=3)
    rain_3h = rain_3h.isel(step=slice(3, None, 3))
    rain_3h = rain_3h.clip(min=0)

    # LOOP THROUGH ALL TIME STEPS
    for i in range(len(rain_3h.step)):
        data = rain_3h.isel(step=i)

        # filter noise
        data = data.where(data >= 0.01)

        end_time = pd.to_datetime(data.valid_time.values)
        start_time = end_time - pd.Timedelta(hours=3)

        st.markdown(f"### {start_time:%H:%M} – {end_time:%H:%M} UTC")

        data_small = data[::2, ::2]

        # ---- MAP PLOT ----
        fig = plt.figure(figsize=(14, 6))
        ax = plt.axes(projection=ccrs.Mercator())

        # set extent (CZ)
        ax.set_extent([12, 19, 48.3, 51.2])


        data_small.plot(
            ax=ax,
            transform=ccrs.PlateCarree(),
            cmap="turbo",
            add_colorbar=False,
            add_labels=False
        )

        ax.add_feature(cfeature.BORDERS, edgecolor="magenta", linewidth=1)
        ax.add_feature(cfeature.COASTLINE, edgecolor="magenta", linewidth=1)

        ax.set_axis_off()

        st.pyplot(fig, use_container_width=False)
