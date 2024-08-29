def save_station_points_kml(station_iter):
    for name, lat, lon, alt in station_iter:
        apertools.kml.create_kml(
            title=name,
            desc="GPS station location",
            lon_lat=(lon, lat),
            kml_out="%s.kml" % name,
            shape="point",
        )
