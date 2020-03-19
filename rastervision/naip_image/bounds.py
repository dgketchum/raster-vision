from math import ceil, floor

from pyproj import Proj
from rasterio import open as rasopen
from rasterio.crs import CRS
from fiona import open as fopen
import fiona

class BBox(object):
    def __init__(self):
        self.west = None
        self.east = None
        self.north = None
        self.south = None

    def as_tuple(self, order='wsen'):
        """ Find 4-tuple of extent
        :param order: order of cardinal directions, default='wsen'
        :return: 4-Tuple
        """
        if order == 'wsen':
            return self.west, self.south, self.east, self.north
        elif order == 'swne':
            return self.south, self.west, self.north, self.east
        elif order == 'nsew':
            return self.north, self.south, self.east, self.west

    def to_web_mercator(self):
        in_proj = Proj({'init': 'epsg:3857'})
        w, s = in_proj(self.west, self.south)
        e, n = in_proj(self.east, self.north)
        return w, s, e, n

    def to_mt_sp(self):
        in_proj = Proj('+proj=lcc +lat_1=45 +lat_2=49 +lat_0=44.25 +'
                       'lon_0=-109.5 +x_0=600000 +y_0=LE07_clip_L1TP_039027_20150529_20160902_01_T1_B1.TIF +ellps=GRS80 +units=m +no_defs')
        w, s = in_proj(self.west, self.south)
        e, n = in_proj(self.east, self.north)
        return w, s, e, n

    def to_epsg(self, epsg):
        in_proj = Proj({'init': 'epsg:{}'.format(epsg)})
        w, s = in_proj(self.west, self.south)
        e, n = in_proj(self.east, self.north)
        return w, s, e, n

    def point_to_epsg(self, epsg, lat, lon):
        in_proj = Proj({'init': 'epsg:{}'.format(epsg)})
        lon, lat = in_proj(lon, lat)
        return lat, lon

    def to_geographic(self, epsg):
        in_proj = Proj({'init': 'epsg:{}'.format(epsg)})
        w, s = in_proj(self.west, self.south, inverse=True)
        e, n = in_proj(self.east, self.north, inverse=True)
        return w, s, e, n

    def lambert_cc_to_geographic(self):
        in_proj = Proj('+proj=lcc +lat_1=45 +lat_2=49 +lat_0=44.25 +lon_0=-109.5 +x_0=600000 '
                       '+y_0=0 +ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs')
        w, s = in_proj(self.west, self.south, inverse=True)
        e, n = in_proj(self.east, self.north, inverse=True)
        return w, s, e, n

    def to_lambert_conformal_conic(self):
        in_proj = Proj('+proj=lcc +lat_1=45 +lat_2=49 +lat_0=44.25 +lon_0=-109.5 +x_0=600000 '
                       '+y_0=0 +ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs')
        w, s = in_proj(self.west, self.south)
        e, n = in_proj(self.east, self.north)
        return w, s, e, n

    def point_to_lambert_conformal_conic(self, lat, lon):
        in_proj = Proj('+proj=lcc +lat_1=45 +lat_2=49 +lat_0=44.25 +lon_0=-109.5 +x_0=600000 '
                       '+y_0=0 +ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs')
        lon, lat = in_proj(lon, lat)
        return lat, lon

    def geographic_to_utm_zone(self, lat, lon):
        utm_band = str((floor((lon + 180) / 6) % 60) + 1)
        if len(utm_band) == 1:
            utm_band = '0' + utm_band
        if lat >= 0:
            epsg_code = '326' + utm_band
        else:
            epsg_code = '327' + utm_band

        return epsg_code

    def expand(self, **delta):

        if not delta:
            if self.west < 0:
                self.west = floor(self.west)
                self.east = ceil(self.east)
            else:
                self.west = ceil(self.west)
                self.east = floor(self.east)

            if self.north > 0:
                self.north = ceil(self.north)
                self.south = floor(self.south)
            else:
                self.north = floor(self.north)
                self.south = ceil(self.south)

        else:

            self.west += delta['west']
            self.east += delta['east']
            self.north += delta['north']
            self.south += delta['south']


class BufferPoint(BBox):
    def __init__(self):
        BBox.__init__(self)

    def buffer_meters(self, lat, lon, distance):
        epsg = self.geographic_to_utm_zone(lat, lon)
        y, x = self.point_to_epsg(epsg, lat, lon)
        self.west = x - distance
        self.south = y - distance
        self.east = x + distance
        self.north = y + distance
        self.west, self.south, self.east, self.north = self.to_geographic(epsg)
        return self.west, self.south, self.east, self.north

class GeoBounds(BBox):
    """Spatial bounding box


    """

    def __init__(self, west=None, south=None, east=None, north=None, wsen=None):
        BBox.__init__(self)

        if wsen:
            self.west = wsen[0]
            self.south = wsen[1]
            self.east = wsen[2]
            self.north = wsen[3]
        else:
            self.west = west
            self.south = south
            self.east = east
            self.north = north


class RasterBounds(BBox):
    """ Spatial bounding box from raster extent.

    :param raster

    """

    def __init__(self, raster=None, affine_transform=None, profile=None, latlon=True):
        BBox.__init__(self)

        if raster:
            with rasopen(raster, 'r') as src:
                profile = src.profile
                affine = profile['transform']

        if affine_transform:
            affine = affine_transform

        col, row = 0, 0
        w, n = affine * (col, row)
        col, row = profile['width'], profile['height']
        e, s = affine * (col, row)

        if latlon and profile['crs'] != CRS({'init': 'epsg:4326'}):
            in_proj = Proj(init=profile['crs']['init'])
            self.west, self.north = in_proj(w, n, inverse=True)
            self.east, self.south = in_proj(e, s, inverse=True)

        else:
            self.north, self.west, self.south, self.east = n, w, s, e

    def get_nwse_tuple(self):
        return self.north, self.west, self.south, self.east

class VectorBounds(BBox):
    """ Spatial bounding box from vector extent.

    :param vector

    """

    def __init__(self, vector=None, profile=None, latlon=True):
        BBox.__init__(self)

        if vector:
            with fopen(vector, 'r') as src:
                self.crs = src.crs
                self.epsg = int(src.crs['init'].split(":")[1])
                self.profile = src.profile
                self.meta = src.meta
                self.west, self.south, self.east, self.north = src.bounds

        if latlon and self.crs != {'init': 'epsg:4326'}:
            in_proj = Proj(init=self.profile['crs']['init'])
            self.west, self.north = in_proj(self.west, self.north, inverse=True)
            self.east, self.south = in_proj(self.east, self.south, inverse=True)

        else:
            pass


if __name__ == '__main__':
    pass
# ========================= EOF ====================================================================