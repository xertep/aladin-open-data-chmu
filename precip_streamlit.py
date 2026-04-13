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

layer = st.sidebar.radio(
    "Vyber vrstvu",
    ["Srážky", "Teplota", "Tmin / Tmax", "Vítr", "Oblačnost"],
    index=0
)

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

cloud_total_url = f"https://opendata.chmi.cz/meteorology/weather/nwp_aladin/CZ_1km/{run}/ALADCZ1K4opendata_{date}{run}_SURFNEBUL_TOTALE.grb.bz2"
cloud_low_url   = f"https://opendata.chmi.cz/meteorology/weather/nwp_aladin/CZ_1km/{run}/ALADCZ1K4opendata_{date}{run}_SURFNEBUL_BASSE.grb.bz2"
cloud_mid_url   = f"https://opendata.chmi.cz/meteorology/weather/nwp_aladin/CZ_1km/{run}/ALADCZ1K4opendata_{date}{run}_SURFNEBUL_MOYENN.grb.bz2"
cloud_high_url  = f"https://opendata.chmi.cz/meteorology/weather/nwp_aladin/CZ_1km/{run}/ALADCZ1K4opendata_{date}{run}_SURFNEBUL_HAUTE.grb.bz2"

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


cloud_norm = mcolors.Normalize(vmin=0, vmax=100)

cmap_total = "Greys"

cmap_low = mcolors.LinearSegmentedColormap.from_list(
    "low_clouds", ["#00000000", "#ffff49"]
)

cmap_mid = mcolors.LinearSegmentedColormap.from_list(
    "mid_clouds", ["#00000000", "#b6ffb6"]
)

cmap_high = mcolors.LinearSegmentedColormap.from_list(
    "high_clouds", ["#00000000", "#b6b6ff"]
)


@st.cache_data
def open_grib(path):
    return xr.open_dataset(path, engine="cfgrib")


if st.sidebar.button("Načíst model"):

    if layer == "Srážky":
        st.session_state["precip_path"] = load_data(url)

    elif layer == "Teplota":
        st.session_state["temp_path"] = load_data(temp_url)

    elif layer == "Tmin / Tmax":
        st.session_state["tmax_path"] = load_data(tmax_url)
        st.session_state["tmin_path"] = load_data(tmin_url)

    elif layer == "Vítr":
        st.session_state["wind_speed_path"] = load_data(wind_speed_url)
        st.session_state["wind_dir_path"] = load_data(wind_dir_url)
        st.session_state["gust_u_path"] = load_data(gust_u_url)

    elif layer == "Oblačnost":
        st.session_state["cloud_total_path"] = load_data(cloud_total_url)
        st.session_state["cloud_low_path"]   = load_data(cloud_low_url)
        st.session_state["cloud_mid_path"]   = load_data(cloud_mid_url)
        st.session_state["cloud_high_path"]  = load_data(cloud_high_url)
        

if layer == "Srážky" and "precip_path" in st.session_state:
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

if layer == "Teplota" and "temp_path" in st.session_state:
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
if layer == "Tmin / Tmax" and "tmax_path" in st.session_state:

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
if layer == "Vítr" and "wind_speed_path" in st.session_state:

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
        skip = 20
        u_plot = u[::skip, ::skip]
        v_plot = v[::skip, ::skip]

        gust_plot = gust_field[::2, ::2]  # finer resolution for shading

        # -------------------------
        # PLOT
        # -------------------------
        st.markdown(f"### Vítr – {t:%d.%m.%y %H:%M} UTC")

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
                "label": "Vítr nárazy (m/s)",
                "ticks": [11, 15, 20, 25, 30]
            }
        )

        ax.set_title("")

        # ---- WIND ARROWS (QUIVER) ----
        ax.quiver(
            u_plot.longitude,
            u_plot.latitude,
            u_plot.values,
            v_plot.values,
            transform=ccrs.PlateCarree(),
            scale=150,        # adjust visibility
            width=0.0075,
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


# ---- CLOUDS ----
if layer == "Oblačnost" and "cloud_total_path" in st.session_state:

    ds_total = open_grib(st.session_state["cloud_total_path"])
    ds_low   = open_grib(st.session_state["cloud_low_path"])
    ds_mid   = open_grib(st.session_state["cloud_mid_path"])
    ds_high  = open_grib(st.session_state["cloud_high_path"])

    total = ds_total[list(ds_total.data_vars)[0]]
    low   = ds_low[list(ds_low.data_vars)[0]]
    mid   = ds_mid[list(ds_mid.data_vars)[0]]
    high  = ds_high[list(ds_high.data_vars)[0]]

    all_times = pd.to_datetime(total.valid_time.values)
    run_time = pd.to_datetime(ds_total.time.values)

    for idx, t in enumerate(all_times):

        diff_hours = (t - run_time).total_seconds() / 3600

        if diff_hours % 3 != 0:
            continue

        st.markdown(f"### Oblačnost – {t:%d.%m.%y %H:%M} UTC")

        fig = plt.figure(figsize=(10, 10))

        datasets = [
            (total, cmap_total, "Celková oblačnost"),
            (low, cmap_low, "Nízká"),
            (mid, cmap_mid, "Střední"),
            (high, cmap_high, "Vysoká"),
        ]

        for i, (ds_var, cmap, title) in enumerate(datasets, 1):

            ax = plt.subplot(2, 2, i, projection=ccrs.Mercator())
            ax.set_extent([12, 19, 48.3, 51.2], crs=ccrs.PlateCarree())

            data = ds_var.isel(step=idx) * 100
            data_small = data[::2, ::2]
            data = data.where(data > 1)

            data_small.plot(
                ax=ax,
                transform=ccrs.PlateCarree(),
                cmap=cmap,
                norm=cloud_norm,
                add_colorbar=True,
                cbar_kwargs={"label": "%"}
            )

            ax.set_title(title)

            ax.add_feature(cfeature.BORDERS, edgecolor="black", linewidth=1)
            ax.add_feature(cfeature.COASTLINE, edgecolor="black", linewidth=1)

            ax.set_axis_off()

        st.pyplot(fig)

    ds_total.close()
    ds_low.close()
    ds_mid.close()
    ds_high.close()

