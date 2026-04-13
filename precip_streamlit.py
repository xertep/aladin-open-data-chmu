import streamlit as st
import requests
import bz2
import xarray as xr
import matplotlib.pyplot as plt
import tempfile
import numpy as np
import matplotlib.colors as mcolors
import matplotlib.ticker as ticker
import matplotlib.patheffects as pe
import pandas as pd
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from datetime import datetime

st.set_page_config(
    page_title="Aladin (open data ČHMÚ)",  # this changes the browser tab title
    page_icon="🌧️"                     # optional: emoji or path to an image
)

st.sidebar.title("Modelové vrstvy")

layers = {
    "precip": st.sidebar.checkbox("Srážky", True),
    "temp": st.sidebar.checkbox("Teplota", False),
    "tminmax": st.sidebar.checkbox("Tmin / Tmax", False),
    "wind": st.sidebar.checkbox("Vítr", False),
}

# ---- USER INPUT ----
date = st.text_input("Datum (YYYYMMDD)", datetime.today().strftime('%Y%m%d'))
run = st.selectbox("Běh modelu (UTC)", ["00", "06", "12", "18"])

url = f"https://opendata.chmi.cz/meteorology/weather/nwp_aladin/CZ_1km/{run}/ALADCZ1K4opendata_{date}{run}_SURFPREC_TOTAL.grb.bz2"

temp_url = f"https://opendata.chmi.cz/meteorology/weather/nwp_aladin/CZ_1km/{run}/ALADCZ1K4opendata_{date}{run}_CLSTEMPERATURE.grb.bz2"

tmax_url = f"https://opendata.chmi.cz/meteorology/weather/nwp_aladin/CZ_1km/{run}/ALADCZ1K4opendata_{date}{run}_CLSMAXI_TEMPERAT.grb.bz2"

tmin_url = f"https://opendata.chmi.cz/meteorology/weather/nwp_aladin/CZ_1km/{run}/ALADCZ1K4opendata_{date}{run}_CLSMINI_TEMPERAT.grb.bz2"

wind_speed_url = f"https://opendata.chmi.cz/meteorology/weather/nwp_aladin/CZ_1km/{run}/ALADCZ1K4opendata_{date}{run}_CLSWIND_SPEED.grb.bz2"
wind_dir_url   = f"https://opendata.chmi.cz/meteorology/weather/nwp_aladin/CZ_1km/{run}/ALADCZ1K4opendata_{date}{run}_CLSWIND_DIREC.grb.bz2"

gust_u_url = f"https://opendata.chmi.cz/meteorology/weather/nwp_aladin/CZ_1km/{run}/ALADCZ1K4opendata_{date}{run}_CLSRAFAL_MOD_XFU.grb.bz2"
gust_v_url = f"https://opendata.chmi.cz/meteorology/weather/nwp_aladin/CZ_1km/{run}/ALADCZ1K4opendata_{date}{run}_CLSRAFAL_MOD_XFU.grb.bz2"

@st.cache_data(show_spinner=True)
def load_data(url):
    r = requests.get(url)
    compressed = r.content

    grib_bytes = bz2.decompress(compressed)

    with tempfile.NamedTemporaryFile(suffix=".grb", delete=False) as f:
        f.write(grib_bytes)
        return f.name



# 1. Define the boundaries (the numbers on the left of your image)
# These are the 'edges' of the color blocks
boundaries = [0.1, 0.3, 0.5, 1.0, 2.0, 4.0, 6.0, 10.0, 15.0, 20.0, 30.0, 40.0, 60.0, 80.0, 100.0, 150.0]

# 2. Define the colors corresponding to each boundary
# I have mapped these based on your image (Purple -> Blue -> Green -> Yellow -> Orange -> Red -> White)
colors = [
    "#390071", # Deep Purple (0.1 - 0.3)
    "#3000aa", # Blue (0.3 - 0.5)
    "#0000ff", # Dark Blue (0.5 - 1.0)
    "#006dc2", # Green (1.0 - 2.0)
    "#00a200", # Lime Green (2.0 - 4.0)
    "#00be00", # Green Yellow (4.0 - 6.0)
    "#35db00", # Yellow (6.0 - 10.0)
    "#9edf00", # Orange (10.0 - 15.0)
    "#e3df00", # Dark Orange (15.0 - 20.0)
    "#ffb200", # Orange Red (20.0 - 30.0)
    "#ff8600", # Firebrick (30.0 - 40.0)
    "#ff5900", # Dark Red (40.0 - 60.0)
    "#ff0000", # White (60.0 - 80.0)
    "#a20000", # White (80.0 - 100.0)
    "#ffffff", # White (100.0 - 150.0)
    "#960096"  # Dark Pink (> 150.0)
]

# 3. Create the Colormap and the Normalization object
custom_cmap = mcolors.ListedColormap(colors)
norm = mcolors.BoundaryNorm(boundaries, custom_cmap.N)

# 1. Define a custom formatting function
def weather_formatter(x, pos):
    # If the value is less than 1, use 1 decimal place (e.g., 0.5)
    if x < 1:
        return f"{x:.1f}"
    # If the value is 1 or greater, return it as an integer (e.g., 10)
    else:
        return f"{int(x)}"
    
GUST_MIN = 11
GUST_MAX = 30


@st.cache_data
def open_grib(path):
    return xr.open_dataset(path, engine="cfgrib")


if st.sidebar.button("Načíst model"):

    if layers["precip"]:
        st.session_state["precip_path"] = load_data(url)

    if layers["temp"]:
        st.session_state["temp_path"] = load_data(temp_url)

    if layers["tminmax"]:
        st.session_state["tmax_path"] = load_data(tmax_url)
        st.session_state["tmin_path"] = load_data(tmin_url)

    if layers["wind"]:
        st.session_state["wind_speed_path"] = load_data(wind_speed_url)
        st.session_state["wind_dir_path"] = load_data(wind_dir_url)
        st.session_state["gust_u_path"] = load_data(gust_u_url)
        st.session_state["gust_v_path"] = load_data(gust_v_url)
        

if layers["precip"] and "precip_path" in st.session_state:
    path = st.session_state["precip_path"]

    ds = open_grib(path)

    st.write("Data načtena")

    tp = ds[list(ds.data_vars)[0]]


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
                "label": "Srážky (mm / 3h)",
                "boundaries": boundaries,
                "ticks": [0.1, 0.3, 0.5, 1, 2, 4, 6, 10, 15, 20, 30, 40, 60, 80, 100, 150],
                # 2. Apply the custom formatter to the colorbar axis
                "format": ticker.FuncFormatter(weather_formatter) 
            }
        )

        ax.add_feature(cfeature.BORDERS, edgecolor="magenta", linewidth=1)
        ax.add_feature(cfeature.COASTLINE, edgecolor="magenta", linewidth=1)

        ax.set_axis_off()

        st.pyplot(fig, use_container_width=False)

        ds.close()
        del ds, tp



# ---- LOAD TEMPERATURE ----

if layers["temp"] and "temp_path" in st.session_state:
    ds_temp = open_grib(st.session_state["temp_path"])

    st.write("Teplota načtena")

    temp = ds_temp[list(ds_temp.data_vars)[0]]

    all_times = pd.to_datetime(temp.valid_time.values)
    run_time = pd.to_datetime(ds_temp.time.values)

    # ---- SELECT 3H STEPS ----
    target_indices = []

    for i, t in enumerate(all_times):
        diff_hours = (t - run_time).total_seconds() / 3600

        if diff_hours % 3 == 0:
            target_indices.append(i)

    # ---- LOOP ----
    for idx in target_indices:
        valid_time = all_times[idx]

        # Kelvin -> Celsius
        data = temp.isel(step=idx) - 273.15

        st.markdown(f"### {valid_time:%d %H:%M} UTC")

        data_small = data[::2, ::2]
        label_data = data[::15, ::15]   # controls density (bigger = fewer labels)

        fig = plt.figure(figsize=(10, 6))
        ax = plt.axes(projection=ccrs.Mercator())

        ax.set_extent([12, 19, 48.3, 51.2], crs=ccrs.PlateCarree())

        im = data_small.plot(
            ax=ax,
            transform=ccrs.PlateCarree(),
            cmap="coolwarm",
            vmin=-20,
            vmax=35,
            add_colorbar=True,
            add_labels=False,
            cbar_kwargs={
                "label": "Teplota (°C)"
            }
        )

        lats = label_data.latitude.values
        lons = label_data.longitude.values
        values = label_data.values

        for i in range(len(lats)):
            for j in range(len(lons)):
                val = values[i, j]

                if np.isnan(val):
                    continue

                ax.text(
                    lons[j],
                    lats[i],
                    f"{int(val)}",   # no decimals
                    transform=ccrs.PlateCarree(),
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="black",
                    alpha=0.7
                )

        ax.add_feature(cfeature.BORDERS, edgecolor="black", linewidth=1)
        ax.add_feature(cfeature.COASTLINE, edgecolor="black", linewidth=1)

        ax.set_axis_off()

        st.pyplot(fig)

        ds_temp.close()

        del ds_temp, temp


# ---- LOAD TMIN / TMAX ----
if layers["tminmax"] and "tmax_path" in st.session_state:

    ds_tmax = open_grib(st.session_state["tmax_path"])
    ds_tmin = open_grib(st.session_state["tmin_path"])

    tmax = ds_tmax[list(ds_tmax.data_vars)[0]]
    tmin = ds_tmin[list(ds_tmin.data_vars)[0]]

    all_times = pd.to_datetime(tmax.valid_time.values)
    run_time = pd.to_datetime(ds_tmax.time.values)

    for idx, valid_time in enumerate(all_times):

        hour = valid_time.hour

        # ---- SELECT ONLY 06 and 18 UTC ----
        if hour == 6:
            data = tmin.isel(step=idx) - 273.15
            title = "Tmin"

        elif hour == 18:
            data = tmax.isel(step=idx) - 273.15
            title = "Tmax"

        else:
            continue

        st.markdown(f"### {title} – {valid_time:%d %H:%M} UTC")

        data_small = data[::2, ::2]
        label_data = data[::15, ::15]

        fig = plt.figure(figsize=(10, 6))
        ax = plt.axes(projection=ccrs.Mercator())

        ax.set_extent([12, 19, 48.3, 51.2], crs=ccrs.PlateCarree())

        data_small.plot(
            ax=ax,
            transform=ccrs.PlateCarree(),
            cmap="coolwarm",
            vmin=-20,
            vmax=35,
            add_colorbar=True,
            add_labels=False,
            cbar_kwargs={
                "label": f"{title} (°C)"
            }
        )

        # ---- NUMBERS ----
        lats = label_data.latitude.values
        lons = label_data.longitude.values
        values = label_data.values

        for i in range(len(lats)):
            for j in range(len(lons)):
                val = values[i, j]

                if np.isnan(val):
                    continue

                ax.text(
                    lons[j],
                    lats[i],
                    f"{int(val)}",   # no decimals
                    transform=ccrs.PlateCarree(),
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="black",
                    alpha=0.7
                )

        ax.add_feature(cfeature.BORDERS, edgecolor="black", linewidth=1)
        ax.add_feature(cfeature.COASTLINE, edgecolor="black", linewidth=1)

        ax.set_axis_off()

        st.pyplot(fig)

    ds_tmax.close()
    ds_tmin.close()

    del ds_tmax, tmax
    del ds_tmin, tmin


# ---- WIND ----
if layers["wind"] and "wind_speed_path" in st.session_state:

    ds_ws = open_grib(st.session_state["wind_speed_path"])
    ds_wd = open_grib(st.session_state["wind_dir_path"])
    ds_gust = open_grib(st.session_state["gust_u_path"])  # scalar gust field (assumed)

    ws = ds_ws[list(ds_ws.data_vars)[0]]
    wd = ds_wd[list(ds_wd.data_vars)[0]]
    gust = ds_gust[list(ds_gust.data_vars)[0]]

    all_times = pd.to_datetime(ws.valid_time.values)
    run_time = pd.to_datetime(ds_ws.time.values)

    for idx, t in enumerate(all_times):

        diff_hours = (t - run_time).total_seconds() / 3600

        # only 3-hour steps
        if diff_hours % 3 != 0:
            continue

        # -------------------------
        # DATA EXTRACTION
        # -------------------------
        speed = ws.isel(step=idx)
        direction = wd.isel(step=idx)

        gust_field = gust.isel(step=idx)

        # convert wind to u/v
        rad = np.deg2rad(direction)
        u = -speed * np.sin(rad)
        v = -speed * np.cos(rad)

        wind_mag = np.sqrt(u**2 + v**2)
        mask = wind_mag >= 1.5

        u = u.where(mask) # below 1.5 not showing
        v = v.where(mask)

        # thin grid (VERY important for barbs)
        skip = 10
        u_plot = u[::skip, ::skip]
        v_plot = v[::skip, ::skip]

        gust_plot = gust_field[::2, ::2]  # finer resolution for shading

        # -------------------------
        # PLOT
        # -------------------------
        st.markdown(f"### Wind – {t:%d %H:%M} UTC")

        fig = plt.figure(figsize=(10, 6))
        ax = plt.axes(projection=ccrs.Mercator())

        ax.set_extent([12, 19, 48.3, 51.2], crs=ccrs.PlateCarree())

        # ---- GUST OVERLAY ----
        gust_mask = gust_plot.where(gust_plot >= GUST_MIN)

        im = gust_mask.plot(
            ax=ax,
            transform=ccrs.PlateCarree(),
            cmap="inferno",
            vmin=GUST_MIN,
            vmax=GUST_MAX,
            alpha=0.6,
            add_colorbar=True,
            cbar_kwargs={
                "label": "Gust (m/s)",
                "ticks": [11, 15, 20, 25, 30]
            }
        )

        # ---- WIND ARROWS (QUIVER) ----
        ax.quiver(
            u_plot.longitude,
            u_plot.latitude,
            u_plot.values,
            v_plot.values,
            transform=ccrs.PlateCarree(),
            scale=400,        # adjust visibility
            width=0.0025,
            headwidth=3,
            headlength=4,
            headaxislength=3
        )

        # ---- MAP FEATURES ----
        ax.add_feature(cfeature.BORDERS, edgecolor="black", linewidth=1)
        ax.add_feature(cfeature.COASTLINE, edgecolor="black", linewidth=1)

        ax.set_axis_off()

        st.pyplot(fig)

    ds_ws.close()
    ds_wd.close()
    ds_gust.close()

