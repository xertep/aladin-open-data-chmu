import streamlit as st
import requests
import bz2
import xarray as xr
import matplotlib.pyplot as plt
import tempfile
import numpy as np
import matplotlib.colors as mcolors

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

# 1. Define the boundaries (the numbers on the left of your image)
# These are the 'edges' of the color blocks
boundaries = [0.1, 0.3, 0.5, 1.0, 2.0, 4.0, 6.0, 10.0, 15.0, 20.0, 30.0, 40.0, 80.0, 100.0]

# 2. Define the colors corresponding to each boundary
# I have mapped these based on your image (Purple -> Blue -> Green -> Yellow -> Orange -> Red -> White)
colors = [
    "#4B0082", # Deep Purple (0.1 - 0.3)
    "#0000FF", # Blue (0.3 - 0.5)
    "#00008B", # Dark Blue (0.5 - 1.0)
    "#00FF00", # Green (1.0 - 2.0)
    "#32CD32", # Lime Green (2.0 - 4.0)
    "#ADFF2F", # Green Yellow (4.0 - 6.0)
    "#FFFF00", # Yellow (6.0 - 10.0)
    "#FFA500", # Orange (10.0 - 20.0)
    "#FF8C00", # Dark Orange (20.0 - 30.0)
    "#FF4500", # Orange Red (30.0 - 40.0)
    "#B22222", # Firebrick (40.0 - 80.0)
    "#8B0000", # Dark Red (80.0 - 100.0)
    "#FFFFFF", # White (100.0 - 150.0)
    "#960096"  # Dark Pink (> 150.0)
]

# 3. Create the Colormap and the Normalization object
custom_cmap = mcolors.ListedColormap(colors)
norm = mcolors.BoundaryNorm(boundaries, custom_cmap.N)
    

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
            cmap=custom_cmap,       # Use our new cmap
            norm=norm,              # Use the boundary norm instead of vmin/vmax
            add_colorbar=True,
            add_labels=False,
            cbar_kwargs={
                "label": "Precipitation (mm / 3h)",
                "boundaries": boundaries, # This makes the colorbar show the steps correctly
                "ticks": [0.1, 0.5, 1, 2, 4, 6, 10, 20, 40, 80, 100] # Specific labels for the bar
            }
        )

        ax.add_feature(cfeature.BORDERS, edgecolor="magenta", linewidth=1)
        ax.add_feature(cfeature.COASTLINE, edgecolor="magenta", linewidth=1)

        ax.set_axis_off()

        st.pyplot(fig, use_container_width=False)
