import json
from pathlib import Path

import branca.colormap as bcm
import folium
import mapclassify
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

st.set_page_config(page_title="Temperatura superficiale", layout="wide")

DATA_DIR = Path("data")
TOPO_DIR = DATA_DIR / "topojson"
EXCEL_DIR = DATA_DIR / "excel"
COMUNI_FILE = DATA_DIR / "comuni.csv"


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
    parts = path.stem.replace("-", "_").split("_")
    for p in parts:
        if p.isdigit() and len(p) == 6:
            return p
    return None


def load_comuni_dict():
    if COMUNI_FILE.exists():
        comuni_df = pd.read_csv(COMUNI_FILE, dtype=str)
        comuni_df["PRO_COM"] = normalize_code(comuni_df["PRO_COM"], width=6)
        comuni_df["COMUNE"] = comuni_df["COMUNE"].astype(str).str.strip()
        return comuni_df
    return pd.DataFrame(columns=["PRO_COM", "COMUNE"])


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

    comuni_codes = sorted(set(topo_map.keys()) & set(excel_map.keys()))
    return comuni_codes, topo_map, excel_map


@st.cache_data
def load_excel_tables(path_str: str):
    path = Path(path_str)
    xls = pd.ExcelFile(path)
    sheets = xls.sheet_names

    temp_sheet = "Temp_media" if "Temp_media" in sheets else sheets[0]
    uhi_sheet = "UHI" if "UHI" in sheets else sheets[min(1, len(sheets) - 1)]

    temp_df = pd.read_excel(path, sheet_name=temp_sheet)
    uhi_df = pd.read_excel(path, sheet_name=uhi_sheet)

    return temp_df, uhi_df, sheets


@st.cache_data
def load_topojson(path_str: str):
    path = Path(path_str)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_topology_object_name(topojson: dict) -> str:
    objects = topojson.get("objects", {})
    if not objects:
        raise ValueError("Nel TopoJSON manca il nodo 'objects'.")
    return next(iter(objects))


def topo_properties_to_dataframe(topojson: dict, object_name: str) -> pd.DataFrame:
    rows = []
    geometries = topojson["objects"][object_name].get("geometries", [])
    for geom in geometries:
        rows.append(geom.get("properties", {}).copy())
    return pd.DataFrame(rows)


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
    return mapclassify.Quantiles(values, k=k)


def add_class_column(df: pd.DataFrame, value_col: str, classifier):
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
        "Giallo-Rosso": ["#ffffcc", "#ffeda0", "#feb24c", "#f03b20", "#bd0026"],
        "Arancio-Rosso": ["#fef0d9", "#fdcc8a", "#fc8d59", "#e34a33", "#b30000"],
        "Giallo-Verde-Blu": ["#ffffd9", "#c7e9b4", "#7fcdbb", "#41b6c4", "#225ea8"],
        "Blu": ["#eff3ff", "#bdd7e7", "#6baed6", "#3182bd", "#08519c"],
        "Viridis": ["#440154", "#3b528b", "#21918c", "#5ec962", "#fde725"],
    }
    base = palettes.get(name, palettes["Giallo-Rosso"])
    return base[:n] if n <= len(base) else base + [base[-1]] * (n - len(base))


def style_function_factory(color_map: dict, class_lookup: dict, join_key: str = "SEZ21_ID"):
    def style_function(feature):
        key = normalize_code(pd.Series([feature["properties"].get(join_key)]), width=12).iloc[0]
        class_id = class_lookup.get(key)
        if class_id is None:
            return {
                "fillColor": "#00000000",
                "color": "#00000000",
                "weight": 0,
                "fillOpacity": 0,
            }
        fill = color_map.get(class_id, "#d3d3d3")
        return {
            "fillColor": fill,
            "color": "#444444",
            "weight": 0.4,
            "fillOpacity": 0.75,
        }
    return style_function


def merge_properties_into_topojson(topojson, object_name, merged_df, join_key="SEZ21_ID", only_matched=False):
    normalized_keys = normalize_code(merged_df[join_key], width=None).dropna()
    digit_key_lengths = [len(k) for k in normalized_keys if str(k).isdigit()]
    key_width = max(digit_key_lengths) if digit_key_lengths else None

    def normalize_join_key(value):
        if pd.isna(value):
            return None
        key = normalize_code(pd.Series([value]), width=key_width).iloc[0]
        return key if key else None

    lookup = {
        normalize_join_key(row[join_key]): row.to_dict()
        for _, row in merged_df.iterrows()
        if normalize_join_key(row.get(join_key)) is not None
    }

    source_geometries = topojson["objects"][object_name].get("geometries", [])
    geometries = []
    for geom in source_geometries:
        props = geom.get("properties", {}).copy()
        key = normalize_join_key(props.get(join_key))
        if key in lookup:
            props.update(lookup[key])
            geom_out = geom.copy()
            geom_out["properties"] = props
            geometries.append(geom_out)
        elif not only_matched:
            geom_out = geom.copy()
            geom_out["properties"] = props
            geometries.append(geom_out)

    topo_out = topojson.copy()
    topo_out["objects"] = topojson["objects"].copy()
    obj_out = topojson["objects"][object_name].copy()
    obj_out["geometries"] = geometries if geometries else source_geometries
    topo_out["objects"][object_name] = obj_out

    return topo_out


def get_topojson_centroid(topojson: dict, object_name: str, fallback: list[float] | None = None) -> list[float]:
    fallback = fallback or [42.0, 13.0]
    try:
        bounds = folium.TopoJson(topojson, object_path=f"objects.{object_name}").get_bounds()
        if not bounds or len(bounds) != 2:
            return fallback
        (min_lat, min_lon), (max_lat, max_lon) = bounds
        if min_lat == max_lat and min_lon == max_lon:
            return fallback
        return [(min_lat + max_lat) / 2, (min_lon + max_lon) / 2]
    except Exception:
        return fallback


st.title("Mappa interattiva della temperatura superficiale estiva e UHI")

comuni_codes, topo_map, excel_map = index_datasets()
comuni_dict = load_comuni_dict()

if not comuni_codes:
    st.error("Nessun dataset trovato nelle cartelle data/topojson e data/excel.")
    st.stop()

# costruzione etichette comuni
if not comuni_dict.empty:
    comuni_disponibili = comuni_dict[comuni_dict["PRO_COM"].isin(comuni_codes)].copy()
else:
    comuni_disponibili = pd.DataFrame({
        "PRO_COM": comuni_codes,
        "COMUNE": comuni_codes
    })

if comuni_disponibili.empty:
    st.error("Nessuna corrispondenza trovata tra i dataset disponibili e il file comuni.csv.")
    st.stop()

comuni_disponibili = comuni_disponibili.sort_values("COMUNE")

nome_to_code = {
    row["COMUNE"]: row["PRO_COM"]
    for _, row in comuni_disponibili.iterrows()
}

if "map_loaded" not in st.session_state:
    st.session_state["map_loaded"] = False

with st.sidebar.form("controls_form"):
    nome_comune = st.selectbox("Comune", list(nome_to_code.keys()))
    theme = st.selectbox("Indicatore", ["Temp_media", "UHI"])
    class_method = st.selectbox(
        "Classificazione",
        ["Quantili", "Intervalli uguali", "Natural Breaks (Jenks)", "Deviazione standard"]
    )
    k_classes = st.slider("Numero classi", 3, 7, 5)
    palette = st.selectbox(
        "Palette",
        ["Giallo-Rosso", "Arancio-Rosso", "Giallo-Verde-Blu", "Blu", "Viridis"]
    )
    basemap = st.selectbox(
        "Mappa di base",
        ["OpenStreetMap", "CartoDB positron", "CartoDB dark_matter", "Google Satellite"]
    )
    load_map = st.form_submit_button("Carica mappa")

if load_map:
    st.session_state["map_loaded"] = True

if not st.session_state["map_loaded"]:
    st.info("Seleziona un comune e clicca su 'Carica mappa'.")
    st.stop()

comune_code = nome_to_code[nome_comune]
topo_path = topo_map[comune_code]
excel_path = excel_map[comune_code]

with st.spinner("Caricamento dati geografici e tabellari in corso..."):
    try:
        topojson = load_topojson(str(topo_path))
        object_name = get_topology_object_name(topojson)
    except Exception as e:
        st.error(f"Errore nella lettura del TopoJSON: {e}")
        st.stop()

    try:
        temp_df, uhi_df, sheets = load_excel_tables(str(excel_path))
    except Exception as e:
        st.error(f"Errore nella lettura dell'Excel: {e}")
        st.stop()

df = temp_df.copy() if theme == "Temp_media" else uhi_df.copy()
prefix = "Media_" if theme == "Temp_media" else "UHI_"

year_cols = get_year_columns(df, prefix)
if not year_cols:
    st.error(f"Nessuna colonna trovata con prefisso {prefix}.")
    st.stop()

anno = st.slider("Anno", min_value=2019, max_value=2025, value=2025, step=1)
value_col = f"{prefix}{anno}"
if value_col not in df.columns:
    anni_disponibili = sorted([int(c.replace(prefix, "")) for c in year_cols if c.replace(prefix, "").isdigit()])
    st.warning(f"Anno {anno} non disponibile per {theme}. Anni disponibili: {', '.join(map(str, anni_disponibili))}.")
    st.stop()

props_df = topo_properties_to_dataframe(topojson, object_name)

if "SEZ21_ID" not in props_df.columns:
    st.error("Nel TopoJSON manca la colonna SEZ21_ID.")
    st.stop()

if "SEZ21_ID" not in df.columns:
    st.error("Nell'Excel manca la colonna SEZ21_ID.")
    st.stop()

props_df["SEZ21_ID"] = normalize_code(props_df["SEZ21_ID"], width=12)
df["SEZ21_ID"] = normalize_code(df["SEZ21_ID"], width=12)

if "PRO_COM" in props_df.columns:
    props_df["PRO_COM"] = normalize_code(props_df["PRO_COM"], width=6)
if "PRO_COM" in df.columns:
    df["PRO_COM"] = normalize_code(df["PRO_COM"], width=6)

merged_all = props_df.merge(df, on="SEZ21_ID", how="left", suffixes=("", "_xls"))

topo_cache = st.session_state.setdefault("topo_enriched_cache", {})
topo_cache_key = f"{comune_code}|{theme}"
if topo_cache_key not in topo_cache:
    topo_cache[topo_cache_key] = merge_properties_into_topojson(
        topojson,
        object_name,
        merged_all,
        join_key="SEZ21_ID",
        only_matched=False
    )
topojson_enriched = topo_cache[topo_cache_key]

show_only_valid = True
merged = merged_all[merged_all[value_col].notna()].copy()

if merged.empty:
    st.warning("Nessuna sezione valida disponibile.")
    st.stop()

year_cols_in_range = [
    c for c in year_cols
    if c.replace(prefix, "").isdigit() and 2019 <= int(c.replace(prefix, "")) <= 2025
]
global_values = merged_all[year_cols_in_range].stack().dropna()
if global_values.empty:
    st.warning("Impossibile calcolare la classificazione: non ci sono valori validi nel periodo 2019-2025.")
    st.stop()

classifier_cache = st.session_state.setdefault("classifier_cache", {})
classifier_key = f"{comune_code}|{theme}|{class_method}|{k_classes}"
if classifier_key not in classifier_cache:
    classifier_cache[classifier_key] = classify_values(global_values, class_method, k=k_classes)
classifier = classifier_cache[classifier_key]
if classifier is None:
    st.warning("Impossibile classificare i valori.")
    st.stop()

merged = add_class_column(merged, value_col, classifier)

colors = get_palette_colors(palette, len(classifier.bins))
color_map = {i: colors[i] for i in range(len(classifier.bins))}
class_lookup = {
    row["SEZ21_ID"]: row["_class_id"]
    for _, row in merged[["SEZ21_ID", "_class_id"]].dropna(subset=["SEZ21_ID"]).iterrows()
}

tile_configs = {
    "OpenStreetMap": {"tiles": "OpenStreetMap", "attr": "OpenStreetMap"},
    "CartoDB positron": {"tiles": "CartoDB positron", "attr": "CartoDB"},
    "CartoDB dark_matter": {"tiles": "CartoDB dark_matter", "attr": "CartoDB"},
    "Google Satellite": {
        "tiles": "https://{s}.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
        "attr": "Google",
        "subdomains": ["mt0", "mt1", "mt2", "mt3"],
        "max_zoom": 20,
    },
}

center = get_topojson_centroid(topojson, object_name, fallback=[42.0, 13.0])
map_view_cache = st.session_state.setdefault("map_view_cache", {})
map_view_key = f"{comune_code}|{theme}"
saved_view = map_view_cache.get(map_view_key, {})
saved_center = saved_view.get("center", center)
saved_zoom = saved_view.get("zoom", 12)

tile_cfg = tile_configs[basemap]
m = folium.Map(location=saved_center, zoom_start=saved_zoom, tiles=None, control_scale=True)
folium.TileLayer(
    tiles=tile_cfg["tiles"],
    attr=tile_cfg["attr"],
    name=basemap,
    overlay=False,
    control=False,
    subdomains=tile_cfg.get("subdomains", "abc"),
    max_zoom=tile_cfg.get("max_zoom", 19),
).add_to(m)

tooltip_candidates = ["PRO_COM", "SEZ21", "P1", "P14", "P29", value_col]
tooltip_fields = [c for c in tooltip_candidates if c in merged.columns]
tooltip_aliases = {
    "PRO_COM": "Codice comune:",
    "SEZ21": "Sezione:",
    "P1": "Popolazione totale:",
    "P14": "Pop < 5 anni:",
    "P29": "Pop > 74 anni:",
    value_col: f"{theme} {anno}:",
}

folium.TopoJson(
    topojson_enriched,
    object_path=f"objects.{object_name}",
    name="Sezioni",
    style_function=style_function_factory(color_map, class_lookup, join_key="SEZ21_ID"),
    tooltip=folium.GeoJsonTooltip(
        fields=tooltip_fields,
        aliases=[tooltip_aliases.get(c, f"{c}:") for c in tooltip_fields],
        sticky=False
    )
).add_to(m)

legend = bcm.StepColormap(
    colors=colors,
    index=[float(global_values.min())] + [float(b) for b in classifier.bins],
    vmin=float(global_values.min()),
    vmax=float(global_values.max()),
    caption=f"{theme} - {anno}"
)
legend.add_to(m)

col1, col2 = st.columns([3, 1])

with col1:
    map_state = st_folium(m, width=None, height=720)
    if isinstance(map_state, dict):
        zoom = map_state.get("zoom")
        center_state = map_state.get("center")
        if isinstance(center_state, dict):
            lat = center_state.get("lat")
            lng = center_state.get("lng")
            if lat is not None and lng is not None:
                map_view_cache[map_view_key] = {
                    "center": [float(lat), float(lng)],
                    "zoom": int(zoom) if zoom is not None else saved_zoom,
                }

with col2:
    st.write(f"**Comune:** {nome_comune}")
    st.write(f"**Codice comune:** {comune_code}")
    st.write(f"**TopoJSON:** `{topo_path.name}`")
    st.write(f"**Excel:** `{excel_path.name}`")
    st.write(f"**Fogli disponibili:** {', '.join(sheets)}")
    st.write(f"**Sezioni visualizzate:** {len(merged)}")
    st.write(f"**Minimo:** {merged[value_col].min():.2f}")
    st.write(f"**Massimo:** {merged[value_col].max():.2f}")
    st.write(f"**Media:** {merged[value_col].mean():.2f}")
    st.write(f"**Mediana:** {merged[value_col].median():.2f}")

    cols_show = [c for c in ["PRO_COM", "SEZ21", "SEZ21_ID", "P1", "P14", "P29", value_col] if c in merged.columns]
    st.dataframe(
        merged[cols_show].sort_values(value_col, ascending=False),
        use_container_width=True,
        height=320
    )

    csv_bytes = merged.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Scarica CSV",
        data=csv_bytes,
        file_name=f"{theme}_{anno}_{comune_code}.csv",
        mime="text/csv"
    )
