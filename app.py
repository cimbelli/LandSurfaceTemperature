import json
from pathlib import Path

import branca.colormap as bcm
import folium
import mapclassify
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium


st.set_page_config(page_title="Land Surface Temperature", layout="wide")


DATA_DIR = Path("data")
TOPO_DIR = DATA_DIR / "topojson"
EXCEL_DIR = DATA_DIR / "excel"


# -----------------------------
# Utility
# -----------------------------
def normalize_code(series: pd.Series, width: int | None = None) -> pd.Series:
    s = series.astype(str).str.strip().str.replace(".0", "", regex=False)
    s = s.str.replace(r"\s+", "", regex=True)
    if width:
        s = s.str.zfill(width)
    return s


def get_available_topojson():
    return sorted(TOPO_DIR.glob("*.json"))


def get_available_excels():
    return sorted(EXCEL_DIR.glob("*.xlsx"))


def infer_code_from_name(path: Path):
    # cerca un codice comune a 6 cifre nel nome file
    parts = path.stem.replace("-", "_").split("_")
    for p in parts:
        if p.isdigit() and len(p) == 6:
            return p
    return None


def index_datasets():
    topo_files = get_available_topojson()
    excel_files = get_available_excels()

    topo_map = {}
    for f in topo_files:
        code = infer_code_from_name(f)
        if code:
            topo_map[code] = f

    excel_map = {}
    for f in excel_files:
        code = infer_code_from_name(f)
        if code:
            excel_map[code] = f

    comuni = sorted(set(topo_map.keys()) & set(excel_map.keys()))
    return comuni, topo_map, excel_map


def load_topojson_as_geojson(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        topo = json.load(f)

    # usa topojson package se disponibile
    try:
        from topojson import Topology
        geo = Topology(topo).to_geojson()
        geojson = json.loads(geo)
    except Exception:
        # fallback: richiede che il package topojson sia installato
        raise RuntimeError(
            "Impossibile leggere il TopoJSON. "
            "Aggiungi 'topojson' al requirements.txt."
        )

    return topo, geojson


def topo_properties_to_dataframe(geojson: dict) -> pd.DataFrame:
    rows = []
    for feat in geojson["features"]:
        props = feat.get("properties", {}).copy()
        rows.append(props)
    return pd.DataFrame(rows)


def load_excel_tables(path: Path):
    xls = pd.ExcelFile(path)
    sheets = xls.sheet_names

    temp_sheet = "Temp_media" if "Temp_media" in sheets else sheets[0]
    uhi_sheet = "UHI" if "UHI" in sheets else sheets[min(1, len(sheets) - 1)]

    temp_df = pd.read_excel(path, sheet_name=temp_sheet)
    uhi_df = pd.read_excel(path, sheet_name=uhi_sheet)

    return temp_df, uhi_df, sheets


def get_year_columns(df: pd.DataFrame, prefix: str):
    return [c for c in df.columns if c.startswith(prefix)]


def classify_values(values: pd.Series, method: str, k: int = 5):
    values = values.dropna()
    if len(values) == 0:
        return None

    unique_count = len(values.unique())
    k = min(k, max(1, unique_count))

    if method == "Quantiles":
        return mapclassify.Quantiles(values, k=k)
    elif method == "Equal intervals":
        return mapclassify.EqualInterval(values, k=k)
    elif method == "Natural Breaks (Jenks)":
        return mapclassify.NaturalBreaks(values, k=k)
    elif method == "Standard deviation":
        return mapclassify.StdMean(values)
    else:
        return mapclassify.Quantiles(values, k=k)


def add_class_column(df: pd.DataFrame, value_col: str, classifier):
    if classifier is None:
        df["_class_id"] = None
        return df

    def assign_bin(val):
        if pd.isna(val):
            return None
        for i, upper in enumerate(classifier.bins):
            if val <= upper:
                return i
        return len(classifier.bins) - 1

    df["_class_id"] = df[value_col].apply(assign_bin)
    return df


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


def build_color_map(class_ids, colors):
    return {i: colors[i] for i in range(len(colors))}


def style_function_factory(color_map: dict):
    def style_function(feature):
        class_id = feature["properties"].get("_class_id")
        fill = "#d3d3d3" if class_id is None else color_map.get(class_id, "#d3d3d3")
        return {
            "fillColor": fill,
            "color": "#444444",
            "weight": 0.4,
            "fillOpacity": 0.75,
        }
    return style_function


def compute_center_from_geojson(geojson):
    # centro grezzo dalla bbox dei vertici
    xs = []
    ys = []

    def walk(coords):
        if isinstance(coords[0], (int, float)):
            xs.append(coords[0])
            ys.append(coords[1])
        else:
            for c in coords:
                walk(c)

    for feat in geojson["features"]:
        walk(feat["geometry"]["coordinates"])

    return [sum(ys) / len(ys), sum(xs) / len(xs)]


def merge_properties_into_geojson(geojson, merged_df, join_key="SEZ21_ID"):
    lookup = {
        str(row[join_key]): row.to_dict()
        for _, row in merged_df.iterrows()
        if pd.notna(row.get(join_key))
    }

    for feat in geojson["features"]:
        props = feat.get("properties", {})
        key = str(props.get(join_key))
        if key in lookup:
            props.update(lookup[key])
        feat["properties"] = props

    return geojson


def build_label(code: str):
    return f"{code}"


# -----------------------------
# App
# -----------------------------
st.title("Interactive map of Landsat summer LST and UHI")

comuni, topo_map, excel_map = index_datasets()

if not comuni:
    st.error(
        "Nessun dataset trovato. "
        "Metti i TopoJSON in data/topojson/ e gli Excel in data/excel/ "
        "usando nel nome il codice comune a 6 cifre."
    )
    st.stop()

with st.sidebar:
    st.header("Controls")

    comune_code = st.selectbox(
        "Municipality",
        comuni,
        format_func=build_label
    )

    theme = st.selectbox("Indicator", ["Temp_media", "UHI"])
    class_method = st.selectbox(
        "Classification",
        ["Quantiles", "Equal intervals", "Natural Breaks (Jenks)", "Standard deviation"]
    )
    k_classes = st.slider("Number of classes", 3, 7, 5)
    palette = st.selectbox("Palette", ["YlOrRd", "OrRd", "YlGnBu", "Blues", "Viridis"])
    basemap = st.selectbox("Basemap", ["OpenStreetMap", "CartoDB positron", "CartoDB dark_matter"])
    show_only_valid = st.checkbox("Show only valid sections", value=True)

topo_path = topo_map[comune_code]
excel_path = excel_map[comune_code]

try:
    topo_raw, geojson = load_topojson_as_geojson(topo_path)
except Exception as e:
    st.error(f"Errore nella lettura del TopoJSON: {e}")
    st.stop()

try:
    temp_df, uhi_df, sheets = load_excel_tables(excel_path)
except Exception as e:
    st.error(f"Errore nella lettura dell'Excel: {e}")
    st.stop()

df = temp_df.copy() if theme == "Temp_media" else uhi_df.copy()
prefix = "Media_" if theme == "Temp_media" else "UHI_"

year_cols = get_year_columns(df, prefix)
if not year_cols:
    st.error(
        f"Nessuna colonna trovata con prefisso '{prefix}' nel file {excel_path.name}."
    )
    st.stop()

years = [c.replace(prefix, "") for c in year_cols]
year = st.sidebar.selectbox("Year", years, index=len(years) - 1)
value_col = f"{prefix}{year}"

# proprietà del topojson
props_df = topo_properties_to_dataframe(geojson)

if "SEZ21_ID" not in props_df.columns:
    st.error("Nel TopoJSON manca la colonna SEZ21_ID.")
    st.stop()

if "SEZ21_ID" not in df.columns:
    st.error("Nell'Excel manca la colonna SEZ21_ID.")
    st.stop()

# normalizzazione join
props_df["SEZ21_ID"] = normalize_code(props_df["SEZ21_ID"], width=12)
df["SEZ21_ID"] = normalize_code(df["SEZ21_ID"], width=12)

if "PRO_COM" in props_df.columns:
    props_df["PRO_COM"] = normalize_code(props_df["PRO_COM"], width=6)
if "PRO_COM" in df.columns:
    df["PRO_COM"] = normalize_code(df["PRO_COM"], width=6)

merged = props_df.merge(df, on="SEZ21_ID", how="left", suffixes=("", "_xls"))

if show_only_valid:
    merged = merged[merged[value_col].notna()].copy()

if merged.empty:
    st.warning("Nessuna sezione valida disponibile per il comune selezionato.")
    st.stop()

classifier = classify_values(merged[value_col], class_method, k=k_classes)
if classifier is None:
    st.warning("Impossibile classificare i valori.")
    st.stop()

merged = add_class_column(merged, value_col, classifier)

n_classes = len(classifier.bins)
colors = get_palette_colors(palette, n_classes)
color_map = build_color_map(merged["_class_id"], colors)

geojson = merge_properties_into_geojson(geojson, merged, join_key="SEZ21_ID")

tiles = {
    "OpenStreetMap": "OpenStreetMap",
    "CartoDB positron": "CartoDB positron",
    "CartoDB dark_matter": "CartoDB dark_matter",
}

center = compute_center_from_geojson(geojson)

m = folium.Map(
    location=center,
    zoom_start=12,
    tiles=tiles[basemap],
    control_scale=True
)

tooltip_fields = [c for c in ["PRO_COM", "SEZ21", "SEZ21_ID", value_col] if c in merged.columns]
popup_fields = [c for c in ["PRO_COM", "SEZ21", "SEZ21_ID", "P1", "P14", "P29", value_col] if c in merged.columns]

folium.GeoJson(
    geojson,
    name="Sections",
    style_function=style_function_factory(color_map),
    tooltip=folium.GeoJsonTooltip(
        fields=tooltip_fields,
        aliases=[f"{c}:" for c in tooltip_fields],
        sticky=False
    ),
    popup=folium.GeoJsonPopup(
        fields=popup_fields,
        aliases=[f"{c}:" for c in popup_fields],
        labels=True
    )
).add_to(m)

legend = bcm.StepColormap(
    colors=colors,
    index=[float(merged[value_col].min())] + [float(b) for b in classifier.bins],
    vmin=float(merged[value_col].min()),
    vmax=float(merged[value_col].max()),
    caption=f"{theme} - {year}"
)
legend.add_to(m)

folium.LayerControl().add_to(m)

col1, col2 = st.columns([3, 1])

with col1:
    st.subheader(f"Map - {theme} {year}")
    st_folium(m, width=None, height=720)

with col2:
    st.subheader("Dataset")
    st.write(f"**Municipality code:** {comune_code}")
    st.write(f"**TopoJSON:** `{topo_path.name}`")
    st.write(f"**Excel:** `{excel_path.name}`")
    st.write(f"**Available sheets:** {', '.join(sheets)}")

    st.subheader("Summary")
    st.write(f"**Displayed sections:** {len(merged)}")
    st.write(f"**Minimum:** {merged[value_col].min():.2f}")
    st.write(f"**Maximum:** {merged[value_col].max():.2f}")
    st.write(f"**Mean:** {merged[value_col].mean():.2f}")
    st.write(f"**Median:** {merged[value_col].median():.2f}")

    cols_show = [c for c in ["PRO_COM", "SEZ21", "SEZ21_ID", "P1", "P14", "P29", value_col] if c in merged.columns]
    st.dataframe(
        merged[cols_show].sort_values(value_col, ascending=False),
        use_container_width=True,
        height=320
    )

    csv = merged.drop(columns=[c for c in merged.columns if c.endswith("_class_id")], errors="ignore")
    csv_bytes = csv.to_csv(index=False).encode("utf-8")

    st.download_button(
        "Download CSV",
        data=csv_bytes,
        file_name=f"{theme}_{year}_{comune_code}.csv",
        mime="text/csv"
    )
