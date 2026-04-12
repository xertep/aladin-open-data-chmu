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

    st.write("Dataset loaded")

    tp = ds[list(ds.data_vars)[0]]

    import pandas as pd
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    # All valid times
    all_times = pd.to_datetime(tp.valid_time.values)

    # Model run time
    run_time = pd.to_datetime(ds.time.values)

    # ---- FIND 3H WINDOWS ----
    target_indices = []

    for i, t in enumerate(all_times):
        diff_hours = (t - run_time).total_seconds() / 3600

        # keep only 3h steps (3, 6, 9, ...)
        if diff_hours > 0 and diff_hours % 3 == 0:
            target_indices.append(i)

    # ---- LOOP THROUGH WINDOWS ----
    for idx in target_indices:
        end_time = all_times[idx]
        start_time = end_time - pd.Timedelta(hours=3)

        start_idx = np.argmin(np.abs(all_times - start_time))

        data = tp.isel(step=idx) - tp.isel(step=start_idx)

        data = data.clip(min=0)
        data = data.where(data >= 0.1)

        st.markdown(f"### {start_time:%d %H:%M} – {end_time:%H:%M} UTC")
        data_small = data[::2, ::2]

        # ---- MAP ----
        fig = plt.figure(figsize=(10, 6))
        ax = plt.axes(projection=ccrs.Mercator())

        ax.set_extent([12, 19, 48.3, 51.2], crs=ccrs.PlateCarree())

        data_small.plot(
            ax=ax,
            transform=ccrs.PlateCarree(),
            cmap="Blues",
            vmin=0,
            vmax=10,
            add_colorbar=True,
            add_labels=False,
            cbar_kwargs={"label": "Precipitation (mm / 3h)"}
        )

        ax.add_feature(cfeature.BORDERS, edgecolor="magenta", linewidth=1)
        ax.add_feature(cfeature.COASTLINE, edgecolor="magenta", linewidth=1)

        ax.set_axis_off()

        st.pyplot(fig, use_container_width=False)
