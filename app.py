import tempfile
import zipfile
from pathlib import Path

import branca.colormap as bcm
import folium
import geopandas as gpd
import mapclassify
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium


st.set_page_config(page_title="LST and UHI Viewer", layout="wide")


def normalize_code(series: pd.Series, width: int | None = None) -> pd.Series:
    s = series.astype(str).str.strip().str.replace(".0", "", regex=False)
    s = s.str.replace(r"\s+", "", regex=True)
    if width:
        s = s.str.zfill(width)
    return s


def guess_join_columns(gdf: gpd.GeoDataFrame, df: pd.DataFrame):
    preferred_pairs = [
        ("SEZ21_ID", "SEZ21_ID"),
        ("sez21_id", "SEZ21_ID"),
        ("SEZ_ID", "SEZ21_ID"),
        ("ID_SEZ", "SEZ21_ID"),
        ("SEZ21", "SEZ21"),
        ("SEZ", "SEZ21"),
    ]

    gdf_cols = set(gdf.columns)
    df_cols = set(df.columns)

    for gcol, dcol in preferred_pairs:
        if gcol in gdf_cols and dcol in df_cols:
            return gcol, dcol

    for col in gdf.columns:
        if col in df.columns:
            return col, col

    return None, None


def guess_comune_column(gdf: gpd.GeoDataFrame):
    candidates = [
        "PRO_COM", "PROCOM", "COD_COM", "CODCOM", "COMUNE_ID", "ID_COMUNE",
        "COMUNE", "DENOM_COM", "NOME_COMUNE", "NAME", "MUNICIPIO"
    ]
    for c in candidates:
        if c in gdf.columns:
            return c
    return None


def get_year_columns(df: pd.DataFrame, prefix: str):
    return [c for c in df.columns if c.startswith(prefix)]


def classify_values(values: pd.Series, method: str, k: int = 5):
    values = values.dropna()
    if len(values) == 0:
        return None

    unique_count = len(values.unique())
    k = min(k, max(1, unique_count))

    if method == "Quantili":
        return mapclassify.Quantiles(values, k=k)
    elif method == "Intervalli uguali":
        return mapclassify.EqualInterval(values, k=k)
    elif method == "Natural Breaks (Jenks)":
        return mapclassify.NaturalBreaks(values, k=k)
    elif method == "Deviazione standard":
        return mapclassify.StdMean(values)
    else:
        return mapclassify.Quantiles(values, k=k)


def add_class_column(gdf: gpd.GeoDataFrame, value_col: str, classifier):
    if classifier is None:
        gdf["_class_id"] = None
        return gdf

    def assign_bin(val):
        if pd.isna(val):
            return None
        for i, upper in enumerate(classifier.bins):
            if val <= upper:
                return i
        return len(classifier.bins) - 1

    gdf["_class_id"] = gdf[value_col].apply(assign_bin)
    return gdf


def get_palette_colors(name: str, n: int):
    palettes = {
        "YlOrRd": ["#ffffcc", "#ffeda0", "#feb24c", "#f03b20", "#bd0026"],
        "OrRd": ["#fef0d9", "#fdcc8a", "#fc8d59", "#e34a33", "#b30000"],
        "YlGnBu": ["#ffffd9", "#c7e9b4", "#7fcdbb", "#41b6c4", "#225ea8"],
        "Blues": ["#eff3ff", "#bdd7e7", "#6baed6", "#3182bd", "#08519c"],
        "Viridis": ["#440154", "#3b528b", "#21918c", "#5ec962", "#fde725"],
    }
    base = palettes.get(name, palettes["YlOrRd"])
    if n <= len(base):
        return base[:n]
    return base + [base[-1]] * (n - len(base))


def style_function_factory(color_map: dict):
    def style_function(feature):
        class_id = feature["properties"].get("_class_id")
        fill = "#d3d3d3" if class_id is None else color_map.get(class_id, "#d3d3d3")
        return {
            "fillColor": fill,
            "color": "#444444",
            "weight": 0.5,
            "fillOpacity": 0.75,
        }
    return style_function


def compute_map_center(gdf: gpd.GeoDataFrame):
    centroid = gdf.to_crs(4326).geometry.union_all().centroid
    return [centroid.y, centroid.x]


def load_zipped_shapefile(uploaded_file) -> gpd.GeoDataFrame:
    with tempfile.TemporaryDirectory() as tmpdir:
        zip_path = Path(tmpdir) / "shape.zip"
        zip_path.write_bytes(uploaded_file.getbuffer())

        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(tmpdir)

        shp_files = list(Path(tmpdir).rglob("*.shp"))
        if not shp_files:
            raise FileNotFoundError("Nel file ZIP non è stato trovato alcun .shp")

        gdf = gpd.read_file(shp_files[0])

    return gdf


def load_excel(uploaded_excel):
    xls = pd.ExcelFile(uploaded_excel)
    sheets = xls.sheet_names

    temp_sheet = "Temp_media" if "Temp_media" in sheets else sheets[0]
    uhi_sheet = "UHI" if "UHI" in sheets else sheets[min(1, len(sheets) - 1)]

    temp_df = pd.read_excel(uploaded_excel, sheet_name=temp_sheet)
    uploaded_excel.seek(0)
    uhi_df = pd.read_excel(uploaded_excel, sheet_name=uhi_sheet)
    return temp_df, uhi_df


st.title("Interactive map of Landsat summer LST and UHI")

with st.expander("Instructions", expanded=True):
    st.write("""
    1. Upload the ZIP shapefile of census sections  
    2. Upload the Excel file with the indicator tables  
    3. Choose the municipality  
    4. Select theme, year, and classification
    """)

shape_file = st.file_uploader("Shapefile sections (ZIP)", type="zip")
excel_file = st.file_uploader("Indicator table (XLSX)", type=["xlsx"])

if not shape_file or not excel_file:
    st.stop()

try:
    gdf = load_zipped_shapefile(shape_file)
except Exception as e:
    st.error(f"Error loading shapefile: {e}")
    st.stop()

try:
    temp_df, uhi_df = load_excel(excel_file)
except Exception as e:
    st.error(f"Error loading Excel file: {e}")
    st.stop()

tema = st.sidebar.selectbox("Theme", ["Temp_media", "UHI"])
df = temp_df.copy() if tema == "Temp_media" else uhi_df.copy()
prefix = "Media_" if tema == "Temp_media" else "UHI_"

year_cols = get_year_columns(df, prefix)
if not year_cols:
    st.error("No yearly columns found in the selected sheet.")
    st.stop()

anni = [c.replace(prefix, "") for c in year_cols]
anno = st.sidebar.selectbox("Year", anni, index=len(anni) - 1)
value_col = f"{prefix}{anno}"

g_join, d_join = guess_join_columns(gdf, df)
if g_join is None or d_join is None:
    st.error("Unable to detect a join field between shapefile and table.")
    st.write("Shapefile columns:", list(gdf.columns))
    st.write("Table columns:", list(df.columns))
    st.stop()

gdf[g_join] = normalize_code(gdf[g_join], width=12)
df[d_join] = normalize_code(df[d_join], width=12)

if "PRO_COM" in df.columns:
    df["PRO_COM"] = normalize_code(df["PRO_COM"], width=6)

comune_col_gdf = guess_comune_column(gdf)
if comune_col_gdf and comune_col_gdf.upper() == "PRO_COM":
    gdf[comune_col_gdf] = normalize_code(gdf[comune_col_gdf], width=6)

merged = gdf.merge(df, left_on=g_join, right_on=d_join, how="left")

if "PRO_COM" in merged.columns:
    comune_code_col = "PRO_COM"
elif comune_col_gdf:
    comune_code_col = comune_col_gdf
else:
    st.error("No municipality field found.")
    st.stop()

possible_name_cols = ["COMUNE", "DENOM_COM", "NOME_COMUNE", "NAME"]
name_col = next((c for c in possible_name_cols if c in merged.columns), None)

if name_col:
    comuni_df = merged[[comune_code_col, name_col]].drop_duplicates().sort_values([name_col, comune_code_col])
    comuni_options = {
        f"{row[name_col]} ({row[comune_code_col]})": row[comune_code_col]
        for _, row in comuni_df.iterrows()
    }
    comune_label = st.sidebar.selectbox("Municipality", list(comuni_options.keys()))
    comune_selected = comuni_options[comune_label]
else:
    comuni = sorted(merged[comune_code_col].dropna().astype(str).unique().tolist())
    comune_selected = st.sidebar.selectbox("Municipality", comuni)

filtered = merged[merged[comune_code_col].astype(str) == str(comune_selected)].copy()

if filtered.empty:
    st.warning("No sections found for the selected municipality.")
    st.stop()

classification_method = st.sidebar.selectbox(
    "Classification",
    ["Quantili", "Intervalli uguali", "Natural Breaks (Jenks)", "Deviazione standard"]
)
k_classes = st.sidebar.slider("Number of classes", 3, 7, 5)
palette = st.sidebar.selectbox("Palette", ["YlOrRd", "OrRd", "YlGnBu", "Blues", "Viridis"])
basemap = st.sidebar.selectbox("Basemap", ["OpenStreetMap", "CartoDB positron", "CartoDB dark_matter"])
show_only_valid = st.sidebar.checkbox("Show only valid values", value=True)

if show_only_valid:
    filtered = filtered[filtered[value_col].notna()].copy()

if filtered.empty:
    st.warning("No valid values available for the selected municipality.")
    st.stop()

classifier = classify_values(filtered[value_col], classification_method, k=k_classes)
filtered = add_class_column(filtered, value_col, classifier)

if classifier is None:
    st.warning("Unable to classify values.")
    st.stop()

n_classes = len(classifier.bins)
colors = get_palette_colors(palette, n_classes)
color_map = {i: colors[i] for i in range(n_classes)}

tiles = {
    "OpenStreetMap": "OpenStreetMap",
    "CartoDB positron": "CartoDB positron",
    "CartoDB dark_matter": "CartoDB dark_matter",
}

center = compute_map_center(filtered)
m = folium.Map(location=center, zoom_start=13, tiles=tiles[basemap], control_scale=True)

tooltip_fields = [c for c in [comune_code_col, g_join, "SEZ21", value_col] if c in filtered.columns]
popup_fields = [c for c in [comune_code_col, g_join, "SEZ21", "P1", "P14", "P29", value_col] if c in filtered.columns]

folium.GeoJson(
    filtered.to_crs(4326),
    name="Sections",
    style_function=style_function_factory(color_map),
    tooltip=folium.GeoJsonTooltip(fields=tooltip_fields, aliases=[f"{c}:" for c in tooltip_fields], sticky=False),
    popup=folium.GeoJsonPopup(fields=popup_fields, aliases=[f"{c}:" for c in popup_fields], labels=True),
).add_to(m)

legend = bcm.StepColormap(
    colors=colors,
    index=[float(filtered[value_col].min())] + [float(b) for b in classifier.bins],
    vmin=float(filtered[value_col].min()),
    vmax=float(filtered[value_col].max()),
    caption=f"{tema} - {anno}"
)
legend.add_to(m)

folium.LayerControl().add_to(m)

col1, col2 = st.columns([3, 1])

with col1:
    st.subheader(f"Map - {tema} {anno}")
    st_folium(m, width=None, height=700)

with col2:
    st.subheader("Summary")
    st.write(f"**Selected municipality:** {comune_selected}")
    st.write(f"**Displayed sections:** {len(filtered)}")
    st.write(f"**Minimum:** {filtered[value_col].min():.2f}")
    st.write(f"**Maximum:** {filtered[value_col].max():.2f}")
    st.write(f"**Mean:** {filtered[value_col].mean():.2f}")
    st.write(f"**Median:** {filtered[value_col].median():.2f}")

    cols_show = [c for c in [comune_code_col, g_join, "SEZ21", value_col] if c in filtered.columns]
    st.dataframe(filtered[cols_show].sort_values(value_col, ascending=False), use_container_width=True)

    csv = filtered.drop(columns="geometry", errors="ignore").to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download CSV",
        data=csv,
        file_name=f"{tema}_{anno}_{comune_selected}.csv",
        mime="text/csv"
    )
