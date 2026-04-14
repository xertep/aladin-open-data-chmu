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
import matplotlib.patches as mpatches
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
    ["Srážky", "Typ srážek", "Teplota", "Tmin / Tmax", "Vítr", "Oblačnost"],
    index=0
)

if layer == "Srážky":
    options = {
        "3 h": 3,
        "24 h": 24,
        "72 h": 72
    }

    label = st.sidebar.selectbox("Suma srážek", options.keys())
    window_hours = options[label]

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

ptype_url = f"https://opendata.chmi.cz/meteorology/weather/nwp_aladin/CZ_1km/{run}/ALADCZ1K4opendata_{date}{run}_PRECIP_TYPESEV.grb.bz2"


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
    "#ffffff" # White (100.0 - 150.0)
]

# 3. Create the Colormap and the Normalization object
custom_cmap = mcolors.ListedColormap(colors)
custom_cmap.set_over("#960096")  # >150 mm

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

cmap_total = "Greys_r"

cmap_low = mcolors.LinearSegmentedColormap.from_list(
    "low_clouds", ["black", "#ffff49"]
)

cmap_mid = mcolors.LinearSegmentedColormap.from_list(
    "mid_clouds", ["black", "#b6ffb6"]
)

cmap_high = mcolors.LinearSegmentedColormap.from_list(
    "high_clouds", ["black", "#b6b6ff"]
)




@st.cache_data
def open_grib(path):
    return xr.open_dataset(path, engine="cfgrib")


if st.sidebar.button("Načíst model"):

    if layer == "Srážky":
        st.session_state["precip_path"] = load_data(url)

    elif layer == "Typ srážek":
        st.session_state["ptype_path"] = load_data(ptype_url)

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

    # ---- HANDLE 72h (SPECIAL CASE) ----
    if window_hours == 72:

        start_idx = 0
        end_idx = len(all_times) - 1

        start_time = all_times[start_idx]
        end_time = all_times[end_idx]

        data = tp.isel(step=end_idx) - tp.isel(step=start_idx)

        data = data.clip(min=0)
        data = data.where(data >= 0.1)

        st.markdown(
            f"## Srážky {start_time:%d.%m. %H:%M} – {end_time:%d.%m.%y %H:%M} UTC (72h)"
        )

        data_small = data[::2, ::2]

        fig = plt.figure(figsize=(10, 6))
        ax = plt.axes(projection=ccrs.Mercator())

        ax.set_extent([12, 19, 48.3, 51.2], crs=ccrs.PlateCarree())

        data_small.plot(
            ax=ax,
            transform=ccrs.PlateCarree(),
            cmap=custom_cmap,
            norm=norm,
            add_colorbar=True,
            add_labels=False,
            cbar_kwargs={
                "label": "Srážky (mm / 72h)",
                "boundaries": boundaries,
                "ticks": boundaries,
                "format": ticker.FuncFormatter(weather_formatter),
                "extend": "max"
            }
        )

        ax.add_feature(cfeature.BORDERS, edgecolor="magenta", linewidth=1)
        ax.add_feature(cfeature.COASTLINE, edgecolor="magenta", linewidth=1)

        ax.set_axis_off()

        st.pyplot(fig)

    # ---- 3h and 24h (LOOP) ----
    else:

        target_indices = []

        for i, t in enumerate(all_times):
            diff_hours = (t - run_time).total_seconds() / 3600

            if diff_hours >= window_hours and diff_hours % 3 == 0:
                target_indices.append(i)

        for idx in target_indices:

            end_time = all_times[idx]
            start_time = end_time - pd.Timedelta(hours=window_hours)

            start_idx = np.argmin(np.abs(all_times - start_time))

            #if start_idx < 0:
             #   continue

            data = tp.isel(step=idx) - tp.isel(step=start_idx)

            data = data.clip(min=0)
            data = data.where(data >= 0.1)

            st.markdown(
                f"## Srážky {start_time:%d.%m. %H:%M} – {end_time:%d.%m.%y %H:%M} UTC ({window_hours}h)"
            )

            data_small = data[::2, ::2]

            fig = plt.figure(figsize=(10, 6))
            ax = plt.axes(projection=ccrs.Mercator())

            ax.set_extent([12, 19, 48.3, 51.2], crs=ccrs.PlateCarree())

            data_small.plot(
                ax=ax,
                transform=ccrs.PlateCarree(),
                cmap=custom_cmap,
                norm=norm,
                add_colorbar=True,
                add_labels=False,
                cbar_kwargs={
                    "label": f"Srážky (mm / {window_hours}h)",
                    "boundaries": boundaries,
                    "ticks": boundaries,
                    "format": ticker.FuncFormatter(weather_formatter),
                    "extend": "max"
                }
            )

            ax.add_feature(cfeature.BORDERS, edgecolor="magenta", linewidth=1)
            ax.add_feature(cfeature.COASTLINE, edgecolor="magenta", linewidth=1)

            ax.set_axis_off()

            st.pyplot(fig)

    ds.close()
    del ds, tp


if layer == "Typ srážek" and "ptype_path" in st.session_state:

    ds_ptype = open_grib(st.session_state["ptype_path"])
    ptype = ds_ptype[list(ds_ptype.data_vars)[0]]

    all_times = pd.to_datetime(ptype.valid_time.values)
    run_time = pd.to_datetime(ds_ptype.time.values)

    window_hours = 3

    # =========================================================
    # 1) STRICT SEVERITY MAP (YOUR TABLE - NO CHANGES)
    # =========================================================
    ptype_severity = {
        11: 1,        # mrholení
        1: 2, 201: 2, # déšť

        7: 3, 207: 3, # smíšené srážky

        8: 4,        # zmrzlý déšť

        9: 5,        # krupička, malé kroupy

        5: 6, 205: 6, # suchý sníh

        6: 7, 206: 7, # mokrý sníh

        193: 8, 213: 8, # plískanice

        10: 9,       # kroupy

        12: 10,      # mrznoucí mrholení

        3: 11        # mrznoucí déšť (most dangerous)
    }

    # =========================================================
    # 2) FAST VECTOR LOOKUP (LUT)
    # =========================================================
    lut = np.full(300, np.nan)
    for k, v in ptype_severity.items():
        lut[k] = v

    def to_severity(da):
        da = (da % 200).astype(int)
        return xr.DataArray(
            lut[da],
            dims=da.dims,
            coords=da.coords
        )

    # =========================================================
    # 3) COLORS (LOW → HIGH SEVERITY GRADIENT)
    # =========================================================
    colors = [
        "#3aff3a",  # 1 mrholení
        "#00cc00",  # 2 déšť
        "#009a80",  # 3 smíšené
        "#800080",  # 4 zmrzlý déšť
        "#ffff00",  # 5 krupička
        "#456cff",  # 6 suchý sníh
        "#0000e5",  # 7 mokrý sníh
        "#000080",  # 8 plískanice
        "#ffc000",  # 9 kroupy
        "#ff4040",  # 10 mrznoucí mrholení
        "#ca1718"   # 11 mrznoucí déšť
    ]

    cmap = mcolors.ListedColormap(colors)

    # =========================================================
    # 4) LOOP (3H MAX WINDOW)
    # =========================================================
    # use valid_time as reference (IMPORTANT FIX)
    for idx, t in enumerate(all_times):

        diff_hours = (t - run_time).total_seconds() / 3600
        if diff_hours % 3 != 0:
            continue

        window = []

        for h in range(window_hours):

            target_time = t - pd.Timedelta(hours=h)

            # find closest matching step by valid_time (CRITICAL FIX)
            step_idx = np.argmin(np.abs(all_times - target_time))

            raw = ptype.isel(step=step_idx)
            sev = to_severity(raw)

            window.append(sev)

        if not window:
            continue

        ptype_final = xr.concat(window, dim="t").max(dim="t")

        # =====================================================
        # 5) PLOT
        # =====================================================
        st.markdown(f"## Typ srážek – {t:%d.%m.%Y %H:%M} UTC ({window_hours}h max severity)")
        fig = plt.figure(figsize=(10, 6))
        ax = plt.axes(projection=ccrs.Mercator())
        ax.set_extent([12, 19, 48.3, 51.2], crs=ccrs.PlateCarree())

        ptype_plot = ptype_final[::2, ::2]

        ptype_plot.plot(
            ax=ax,
            transform=ccrs.PlateCarree(),
            cmap=cmap,
            vmin=1,
            vmax=11,
            add_colorbar=False
        )

        ax.set_title("")
        ax.add_feature(cfeature.BORDERS, edgecolor="black")
        ax.add_feature(cfeature.COASTLINE, edgecolor="black")
        ax.set_axis_off()

        # =====================================================
        # 6) LEGEND (STRICT ORDER 1 → 11)
        # =====================================================
        labels = {
            1: "mrholení",
            2: "déšť",
            3: "smíšené srážky",
            4: "zmrzlý déšť",
            5: "malé kroupy",
            6: "suchý sníh",
            7: "mokrý sníh",
            8: "mokrý sníh s deštěm",
            9: "kroupy",
            10: "mrznoucí mrholení",
            11: "mrznoucí déšť"
        }

        import matplotlib.patches as mpatches

        patches = [
            mpatches.Patch(
                color=colors[i - 1],
                label=f"{labels[i]}"
            )
            for i in range(1, 12)
        ]

        ax.legend(
            handles=patches,
            loc="upper right",
            fontsize=8,
            frameon=True
        )

        st.pyplot(fig)

    ds_ptype.close()



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

        st.markdown(f"## Teplota - {t:%d.%m.%y %H:%M} UTC")

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

        ax.add_feature(cfeature.BORDERS, edgecolor="magenta", linewidth=1)
        ax.add_feature(cfeature.COASTLINE, edgecolor="magenta", linewidth=1)

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

        st.markdown(f"## {title} - {valid_time:%d.%m.%y %H:%M} UTC")

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

        ax.add_feature(cfeature.BORDERS, edgecolor="magenta", linewidth=1)
        ax.add_feature(cfeature.COASTLINE, edgecolor="magenta", linewidth=1)

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
        st.markdown(f"## Vítr – {t:%d.%m.%y %H:%M} UTC")

        fig = plt.figure(figsize=(10, 6))
        ax = plt.axes(projection=ccrs.Mercator())

        ax.set_extent([12, 19, 48.3, 51.2], crs=ccrs.PlateCarree())

        # ---- GUST OVERLAY ----
        gust_mask = gust_plot.where(gust_plot >= GUST_MIN)

        gust_cmap = mcolors.LinearSegmentedColormap.from_list(
            "gusts", ["#ffdfdf", "#ffbfbf", "#ff9f9f", "#ff7f7f", "#ff6060", "#ff4040", "#ff2020", "#ff0000"]
        )

        im = gust_mask.plot(
            ax=ax,
            transform=ccrs.PlateCarree(),
            cmap=gust_cmap,
            vmin=GUST_MIN,
            vmax=GUST_MAX,
            alpha=1.0,
            add_colorbar=True,
            cbar_kwargs={
                "label": "Vítr nárazy (m/s)",
                "ticks": [11, 13, 15, 18, 20, 23, 25, 27, 30]
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
        ax.add_feature(cfeature.BORDERS, edgecolor="magenta", linewidth=1)
        ax.add_feature(cfeature.COASTLINE, edgecolor="magenta", linewidth=1)

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

        st.markdown(f"## Oblačnost – {t:%d.%m.%y %H:%M} UTC")

        datasets = [
            (total, cmap_total, "Celková oblačnost"),
            (low, cmap_low, "Nízká"),
            (mid, cmap_mid, "Střední"),
            (high, cmap_high, "Vysoká"),
        ]

        for ds_var, cmap, title in datasets:

            fig = plt.figure(figsize=(10, 6))  # same as precip

            ax = plt.axes(projection=ccrs.Mercator())
            ax.set_extent([12, 19, 48.3, 51.2], crs=ccrs.PlateCarree())

            data = ds_var.isel(step=idx) * 100
            #data = data.where(data > 1)

            data_small = data[::2, ::2]

            data_small.plot(
                ax=ax,
                transform=ccrs.PlateCarree(),
                cmap=cmap,
                norm=cloud_norm,
                add_colorbar=True,
                cbar_kwargs={"label": "%"}
            )

            ax.set_title(title)

            # ✅ borders like precip
            ax.add_feature(cfeature.BORDERS, edgecolor="magenta", linewidth=1)
            ax.add_feature(cfeature.COASTLINE, edgecolor="magenta", linewidth=1)

            ax.set_axis_off()

            st.pyplot(fig)

    ds_total.close()
    ds_low.close()
    ds_mid.close()
    ds_high.close()

