import numpy as np
from matplotlib import cm
import xarray
import folium
import folium.plugins
import folium.features

from src.huc import HUC
from src.livneh import LivnehData

# from jinja2 import Template


# # to override the default folium template
# with open("./templates/map.htm", "r") as reader:
#     folium.Map._template = Template(reader.read())


def generate_map(huc: HUC, series: str, nday: int):
    bbox = huc.buffered_bbox()
    center = [(bbox[1] + bbox[3]) / 2, (bbox[0] + bbox[2]) / 2]

    map = folium.Map(location=center)
    tl2 = folium.TileLayer()
    tl2.add_to(map)

    basin_fg = folium.FeatureGroup("Basin").add_to(map)
    grid_fg = folium.FeatureGroup("Grid Values", show=False).add_to(map)

    folium.GeoJson(huc.geometry_as_geodataframe().to_json()).add_to(basin_fg)

    min_lat = float(huc.weights.weights.lat.min()) - LivnehData.RESOLUTION / 2
    max_lat = float(huc.weights.weights.lat.max()) + LivnehData.RESOLUTION / 2
    min_lon = float(huc.weights.weights.lon.min()) - LivnehData.RESOLUTION / 2
    max_lon = float(huc.weights.weights.lon.max()) + LivnehData.RESOLUTION / 2
    bounds = [
        [min_lat, min_lon - 360],
        [max_lat, max_lon - 360],
    ]

    map.fit_bounds(bounds)

    ids_df = huc.weights.to_dataframe().dropna()
    for lonlat, row in ids_df.iterrows():
        folium.Marker(
            location=(lonlat[0], lonlat[1] - 360),
            tooltip=f"id={row.ids:.0f}\n w={row.weights:.6f}",
        ).add_to(grid_fg)

    raster_layers = [("Weights", huc.weights.weights)]
    clusters = xarray.open_dataset(
        huc.data_path(f"clusters-weights_{series}_{nday}day.nc")
    )
    for i in range(len(clusters.cluster)):
        raster_layers.append((f"Cluster {i+1}",
                              clusters.w_prec.isel(cluster=i)))

    for name, raster_val in raster_layers:
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
            name=name,
            origin="lower",
            show=False,
            overlay=False,
            control=True,
            opacity=0.5,
        ).add_to(map)

    folium.plugins.MousePosition().add_to(map)
    folium.plugins.MiniMap(tl2, position="bottomleft").add_to(map)

    map.add_child(basin_fg)
    map.add_child(grid_fg)
    layer_ctrl = folium.LayerControl(collapsed=False)
    layer_ctrl.add_to(map)
    fname = huc.data_path(f"map-{series}-{nday}day.html")
    map.save(fname)
    print(fname)
