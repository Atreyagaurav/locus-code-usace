from unittest import TestCase
from src.waterboundary import Waterboundary

class Test_Waterboundary(TestCase):
    def test_WaterBoundary_default_input_assigned_northbranch_code(self):
        self.assertEqual('02070002', Waterboundary().code)
    def test_filepath_default_Waterboundary_returns_string_to_northbranch_shpfile(self):
        exp = '/Users/johnkucharski/Documents/source/locus/data/input/waterboundary/WBD_02_HU2_Shape/Shape/WBDHU8.shp'
        self.assertEqual(exp, Waterboundary().filepath())
    def test_importwaterboundary_default_Waterboundary_retuns_gdf_with_NorthBranchPotomac(self):
        wbd = Waterboundary().import_waterboundary()
        self.assertEqual(wbd.Name.iloc[0], 'North Branch Potomac')
    def test_area_default_Waterboundary_returns_northbranch_area(self):
        self.assertEqual(Waterboundary().area(), 3478.65)