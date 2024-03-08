import glob
import os.path
from src.huc import HUC
from src.livneh import LivnehData
import json


huc_map = dict()

print("Loading HUC2...")
for huc, name in HUC.all_huc_codes(2):
    huc_map[name.replace(' ', '')] = (huc, name)
print("Loading HUC4...")
for huc, name in HUC.all_huc_codes(4):
    huc_map[name.replace(' ', '')] = (huc, name)
print("Loading HUC8...")
for huc, name in HUC.all_huc_codes(8):
    huc_map[name.replace(' ', '')] = (huc, name)

bounds = dict()


class NoWeightsException(Exception):
    pass


class NcFile:
    def __init__(self, filename):
        self.filename = os.path.basename(filename)
        self.thumbnail = os.path.splitext(self.filename)[0] + ".png"
        name, _ = os.path.splitext(self.filename)
        # format HUC_NAME: HUCNAME_seriesNd_cluster_prec-100mm.nc
        parts = name.split("_")
        self.huc = huc_map[parts[0]][0]

        if self.huc not in bounds:
            huc = HUC(self.huc)
            if huc.weights is None:
                bounds[self.huc] = None
                raise NoWeightsException
            min_lat = float(huc.weights.weights.lat.min()) - LivnehData.RESOLUTION / 2
            max_lat = float(huc.weights.weights.lat.max()) + LivnehData.RESOLUTION / 2
            min_lon = float(huc.weights.weights.lon.min()) - LivnehData.RESOLUTION / 2
            max_lon = float(huc.weights.weights.lon.max()) + LivnehData.RESOLUTION / 2
            bounds[self.huc] = [
                [min_lat, min_lon - 360],
                [max_lat, max_lon - 360],
            ]
        self.bounds = bounds[self.huc]
        if self.bounds is None:
            raise NoWeightsException

        self.huc_name = huc_map[parts[0]][1]
        if parts[1] == "uniform":
            self.series = parts[1]
            self.duration = "NA"
            self.cluster = "NA"
        else:
            self.series = parts[1][:3]
            self.duration = parts[1][3:]
            self.cluster = parts[2]
        self.filesize = os.path.getsize(filename)

    @classmethod
    def table_header(cls):
        return f"""
        <tr>
        <th>Basin Name</th>
        <th>HUC</th>
        <th>Series</th>
        <th>Duration</th>
        <th>Cluster</th>
        <th>Filesize</th>
        <th>Link</th>
        <th>Interactive Map</th>
        </tr>
        """

    def table_row(self):
        return f"""
        <tr>
        <td>{self.huc_name}</td>
        <td>{self.huc}</td>
        <td>{self.series}</td>
        <td>{self.duration}</td>
        <td>{self.cluster}</td>
        <td>{self.filesize / 1024:.1f} kB</td>
        <td><a href="/download/{self.filename}">{self.filename}</a></td>
        <td><a href="/map/{self.huc_name}">Map</a></td>
        </tr>
        """

    def json(self):
        return dict(
            filename=self.filename,
            thumbnail=self.thumbnail,
            huc_name=self.huc_name,
            huc=self.huc,
            series=self.series,
            duration=self.duration,
            cluster=self.cluster,
            filesize=f"{self.filesize / 1024:.1f}",
            bounds=self.bounds,
        )

    def list(self):
        return [
            self.filename,
            self.huc_name,
            self.huc,
            self.series,
            self.duration,
            self.cluster,
            f"{self.filesize / 1024:.1f}",
        ]


files = []

print("Total:", len(glob.glob("./exported-dss/*.nc")))
for i, f in enumerate(sorted(glob.glob("./exported-dss/*.nc"))):
    try:
        print(i, end=', ')
        files.append(NcFile(f).json())
    except NoWeightsException:
        pass


with open("netcdfs-index.json", "w") as w:
    json.dump(
        dict(
            draw=1,
            recordsTotal=len(files),
            recordsFiltered=len(files),
            data=files,
        ),
        w)
