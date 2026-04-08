# Land Surface Temperature

Interactive web app for visualizing median summer Land Surface Temperature (LST) derived from Landsat data across Italian municipalities.

Users can:
- choose the municipality
- switch between surface temperature and Urban Heat Island (UHI)
- explore data at census section level
- view resident population information
- export filtered data

## Main features
- interactive map
- municipality selector
- year selector
- LST / UHI switch
- dynamic classification
- tooltips and popups
- CSV export

## Tech stack
- Python
- Streamlit
- GeoPandas
- Folium
- Mapclassify

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
