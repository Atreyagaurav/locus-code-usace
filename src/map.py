import numpy as np
from matplotlib import cm
import xarray
import folium
import folium.plugins
import folium.features
import glob

from src.huc import HUC
from src.livneh import LivnehData

# from jinja2 import Template


# # to override the default folium template
# with open("./templates/map.htm", "r") as reader:
#     folium.Map._template = Template(reader.read())


def generate_map(huc: HUC):
    bbox = huc.buffered_bbox()
    center = [(bbox[1] + bbox[3]) / 2, (bbox[0] + bbox[2]) / 2]

    map = folium.Map(location=center)
    tl2 = folium.TileLayer()
    tl2.add_to(map)

    basin_fg = folium.FeatureGroup("Basin").add_to(map)

    folium.GeoJson(
        huc.geometry_as_geodataframe().to_json(),
        color="black"
    ).add_to(basin_fg)

    min_lat = float(huc.weights.weights.lat.min()) - LivnehData.RESOLUTION / 2
    max_lat = float(huc.weights.weights.lat.max()) + LivnehData.RESOLUTION / 2
    min_lon = float(huc.weights.weights.lon.min()) - LivnehData.RESOLUTION / 2
    max_lon = float(huc.weights.weights.lon.max()) + LivnehData.RESOLUTION / 2
    bounds = [
        [min_lat, min_lon - 360],
        [max_lat, max_lon - 360],
    ]

    map.fit_bounds(bounds)

    def filename_2_layer(filename):
        name = filename.rsplit("/", maxsplit=1)[-1].rsplit(".", maxsplit=1)[0]
        weights = xarray.open_dataset(filename).weights
        return (
            name.split("_", maxsplit=1)[1],
            weights.where(weights > 0)
        )

    raster_layers = [
        filename_2_layer(fn)
        for fn in sorted(
                glob.glob(f"exported-dss/{huc.name.replace(' ', '')}*")
        )
    ]

    for name, raster_val in raster_layers:
        weights = raster_val.values.astype(np.float64)

        def color(val):
            return cm.jet(val / 500)

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
    layer_ctrl = folium.LayerControl(collapsed=False)
    layer_ctrl.add_to(map)
    fname = huc.data_path("folium-map.html")
    map.save(fname)
    print(fname)
