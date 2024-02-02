import glob
import os.path
from src.huc import HUC
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


class NcFile:
    def __init__(self, filename):
        self.filename = os.path.basename(filename)
        name, _ = os.path.splitext(self.filename)
        # format HUC_NAME: HUCNAME_seriesNd_cluster_prec-100mm.nc
        parts = name.split("_")
        self.huc = huc_map[parts[0]][0]
        self.huc_name = huc_map[parts[0]][1]
        if parts[1] == "uniform":
            self.series = parts[1]
            self.duration = "NA"
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
            huc_name=self.huc_name,
            huc=self.huc,
            series=self.series,
            duration=self.duration,
            cluster=self.cluster,
            filesize=f"{self.filesize / 1024:.1f}",
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


files = [
    NcFile(f).json() for f in
    sorted(glob.glob("./exported-dss/*.nc"))
]

with open("netcdfs-index.json", "w") as w:
    json.dump(
        dict(
            draw=1,
            recordsTotal=len(files),
            recordsFiltered=len(files),
            data=files,
        ),
        w)
