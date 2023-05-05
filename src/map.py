import numpy as np
from matplotlib import cm
import xarray
import folium
import folium.plugins
import folium.features

from src.huc import HUC
from src.livneh import LivnehData


def generate_map(huc: HUC, series: str, nday: int):
    bbox = huc.buffered_bbox()
    center = [(bbox[1] + bbox[3]) / 2, (bbox[0] + bbox[2]) / 2]

    map = folium.Map(
        location=center,
        min_lat=bbox[1],
        min_lon=bbox[0],
        max_lon=bbox[2],
        max_lat=bbox[3],
    )

    tl2 = folium.TileLayer()
    tl2.add_to(map)

    basin_fg = folium.FeatureGroup("Basin").add_to(map)

    folium.GeoJson(huc.geometry_as_geodataframe().to_json()).add_to(basin_fg)

    bbox = huc.buffered_bbox(buffer=LivnehData.RESOLUTION / 2)
    bounds = [
        [bbox[1], bbox[0]],
        [bbox[3], bbox[2]],
    ]

    clusters = xarray.open_dataset(
        huc.data_path(f"clusters-weights_{series}_{nday}day.nc")
    )

    for i in range(len(clusters.cluster)):
        raster_val = clusters.w_prec.isel(cluster=i)
        weights = raster_val.values.astype(np.float64)

        # weights = h.weights.weights.values.astype(np.float64)
        weights_scale = 1 / float(raster_val.max())

        def color(val):
            return cm.viridis(val * weights_scale)

        folium.raster_layers.ImageOverlay(
            weights,
            bounds,
            mercator_project=True,
            zindex=1,
            colormap=color,
            name=f"Cluster {i+1}",
            origin="lower",
            show=False,
            overlay=False,
            control=True,
            opacity=0.5,
        ).add_to(map)

    folium.plugins.MousePosition().add_to(map)
    folium.plugins.MiniMap(tl2, position="bottomleft").add_to(map)

    map.add_child(basin_fg)
    layer_ctrl = folium.LayerControl(collapsed=False)
    layer_ctrl.add_to(map)
    fname = huc.data_path(f"map-{series}-{nday}day.html")
    map.save(fname)
    print(fname)
