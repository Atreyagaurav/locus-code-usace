import fiona
import shapely

# https://prd-tnm.s3.amazonaws.com/StagedProducts/Hydrography/WBD/National/GPKG/WBD_National_GPKG.zip
# fiona.listlayers("./data/WBD_National_GPKG.gpkg")

class InvalidHUCode(Exception):
    pass


class HUC:
    GPKG_FILE: str = "./data/WBD_National_GPKG.gpkg"

    def __init__(self, code):
        N: int = len(code)
        self.huc_code = code
        with fiona.open(HUC.GPKG_FILE, layer=f"WBDHU{N}") as l:
            try:
                # TODO this filter is slow since it happens on the
                # python side. fiona 2.0 will have a new syntax
                # `l.filter(where="huc2='12'")` it does the filtering
                # from the SQL side and it should be faster. Link:
                # https://github.com/Toblerity/Fiona/issues/1016
                self.feature = next(filter(lambda f: f["properties"][f"huc{N}"] == code, l))
            except StopIteration:
                raise InvalidHUCode(f"No match for {code} in {HUC.GPKG_FILE}")
        self.geometry = shapely.geometry.shape(self.feature["geometry"])

    def __getattr__(self, obj):
        try:
            return super().__getattribute__(obj)
        except AttributeError as e:
            try:
                return self.feature["properties"][obj]
            except KeyError:
                raise AttributeError(e)

    def __repr__(self):
        return f'{self.name} <HUC {self.huc_code}>'
