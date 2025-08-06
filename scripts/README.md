# UNR Grid Web Browser

Setup:

```bash
cd scripts/
# EXAMPLE BBOX:
python create-geojson.py --bbox -110 28 -101 36 --start-date 2016-01-01
# Creates geojson_sources/
echo "Visit the URL in your browser:"
echo "http://localhost:8123/browse_unr_grid.html"
python -m http.server 8123
```
