COPY (
    WITH a AS (
        SELECT
            name,
            ST_POINT(lon, lat) pt
        FROM
            gps_stations
    ),
    b AS (
        SELECT
            frame_id,
            geom
        FROM
            frames_with_geom
        WHERE
            is_north_america
    ),
    c AS (
        SELECT
            *
        FROM
            a
            CROSS JOIN b
    )
    SELECT
        frame_id,
        name,
        pt
    FROM
        c
    WHERE
        st_contains(geom, pt)
) TO 'gps_frame_intersects.csv' (format csv);