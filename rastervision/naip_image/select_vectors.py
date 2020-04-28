# ===============================================================================
# Copyright 2018 dgketchum
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===============================================================================

import os
import sys

abspath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(abspath)
import shutil

from numpy import uint8
from matplotlib import pyplot as plt
from matplotlib import collections as cplt
import rasterio
import rasterio.plot
from rasterio.features import rasterize
from fiona import open as fiona_open
from descartes import PolygonPatch
from geopandas import read_file, GeoDataFrame
from pandas import DataFrame
from shapely.geometry import Polygon
from naip_image.naip import ApfoNaip

TEMP_TIF = os.path.join(os.path.dirname(__file__), 'temp', 'temp_tile_geo.tif')


def convert_bytes(num):
    for x in ['bytes', 'KB', 'MB', 'GB', 'TB']:
        if num < 1024.0:
            return num, x
        num /= 1024.0


def file_size(file_path):
    if os.path.isfile(file_path):
        file_info = os.stat(file_path)
        return convert_bytes(file_info.st_size)


def get_geometries(shp, n=100):
    features = []
    ct = 0
    with fiona_open(shp, 'r') as src:
        for feat in src:
            if ct > n:
                break
            features.append((int(feat['properties']['ID']), feat['properties']['TYPE'], feat['geometry']))
    print('{} features'.format(len(features)))
    return features


def get_naip_polygon(bbox):
    return Polygon([[bbox[0], bbox[1]], [bbox[0], bbox[3]], [bbox[2], bbox[3]],
                    [bbox[2], bbox[1]]])


def get_training_scenes(geometries, instance_label=False, name_prefix='MT', out_dir=None,
                        year=None, n=10, save_shp=False, feature_range=None):
    ct = 0

    overview = os.path.join(out_dir, 'overview')
    image = os.path.join(out_dir, 'image')
    labels = os.path.join(out_dir, 'labels')

    [os.mkdir(x) for x in [overview, image, labels] if not os.path.exists(x)]

    if feature_range:
        geometries = geometries[feature_range[0]:feature_range[1]]

    for (id_, type_, g) in geometries:
        try:
            print('feature {}'.format(id_))
            g = Polygon(g['coordinates'][0])
            naip_args = dict([('dst_crs', '4326'),
                              ('centroid', (g.centroid.y, g.centroid.x)),
                              ('buffer', 1000),
                              ('year', year)])

            naip = ApfoNaip(**naip_args)
            array, profile = naip.get_image(name_prefix)
            naip.save(array, profile, TEMP_TIF)
            naip_geometry = get_naip_polygon(naip.bbox)
            src = rasterio.open(TEMP_TIF)

            vectors = [(i, t, g) for (i, t, g) in geometries if
                       Polygon(g['coordinates'][0]).centroid.intersects(naip_geometry)]

            fig, ax = plt.subplots()
            rasterio.plot.show((src, 1), cmap='viridis', ax=ax)

            patches = [PolygonPatch(Polygon(geo[2]['coordinates'][0]),
                                    edgecolor="red", facecolor="none",
                                    linewidth=1.) for geo in vectors]

            ax.add_collection(cplt.PatchCollection(patches, match_original=True))
            ax.set_xlim(naip_geometry.bounds[0], naip_geometry.bounds[2])
            ax.set_ylim(naip_geometry.bounds[1], naip_geometry.bounds[3])

            name = '{}_{}'.format(name_prefix, str(id_).rjust(7, '0'))
            if save_shp:
                geos = [Polygon(x[2]['coordinates'][0]) for x in vectors]
                data = [(x[0], x[1]) for x in vectors]
                gpd = GeoDataFrame(data=data, geometry=geos, columns=['id', 'TYPE'])
                shp_name = os.path.join(overview, '{}.shp'.format(name))
                gpd.to_file(shp_name)

            fig_name = os.path.join(overview, '{}.png'.format(name))
            plt.savefig(fig_name)
            plt.close(fig)

            fs, unit = file_size(fig_name)
            if fs < 200. and unit == 'KB':
                print(fs, unit)
                os.remove(fig_name)

            else:
                shutil.move(TEMP_TIF, os.path.join(image, '{}.tif'.format(name)))
                naip_bool_name = os.path.join(labels, '{}.tif'.format(name))

                meta = src.meta.copy()
                meta.update(compress='lzw')
                meta.update(nodata=0)
                meta.update(count=1)

                if instance_label:
                    label_values = [(f[2], i) for i, f in enumerate(vectors)]
                else:
                    label_values = [(f[2], 1) for f in vectors]

                with rasterio.open(naip_bool_name, 'w', **meta) as out:
                    burned = rasterize(shapes=label_values, fill=0, dtype=uint8,
                                       out_shape=(array.shape[1], array.shape[2]), transform=out.transform,
                                       all_touched=False)
                    out.write(burned, 1)
                ct += 1
                plt.close()

            if ct > n:
                break

        except ValueError as e:
            print('error {}'.format(e))
            pass


def clean_out_training_data(parent_dir):
    views = os.path.join(parent_dir, 'overview')
    labels = os.path.join(parent_dir, 'labels')
    image = os.path.join(parent_dir, 'image')

    keep = [x[:16] for x in os.listdir(views)]
    remove = [x for x in os.listdir(labels) if x[:16] not in keep]
    [os.remove(os.path.join(labels, x)) for x in remove]
    remove = [x for x in os.listdir(image) if x[:16] not in keep]
    [os.remove(os.path.join(image, x)) for x in remove]


if __name__ == '__main__':
    out_data = None
    home = os.path.expanduser('~')
    extraction = os.path.join(home, 'data', 'field_extraction')
    if not os.path.exists(extraction):
        extraction = os.path.join(home, 'field_extraction')

    states = [('WA_CMICH.shp', 2017)]
    for file_, year in states:
        name_prefix = file_.strip('.shp')
        out_data = os.path.join(extraction, 'field_data',
                                'raw_data', 'states', name_prefix)
        if not os.path.exists(out_data):
            os.mkdir(out_data)
        shape_dir = os.path.join(extraction, 'field_data', 'raw_shapefiles')
        shapes = os.path.join(shape_dir, file_)
        target_number = 4000
        if not os.path.exists(shapes):
            raise ValueError('{} does not exist'.format(shapes))

        # geos = get_geometries(shapes, n=target_number)
        # get_training_scenes(geos, instance_label=True, name_prefix=name_prefix,
        #                     out_dir=out_data, year=year,
        #                     n=target_number, save_shp=False,
        #                     feature_range=(166809, 170000))
    tables = os.path.join(out_data)
    clean_out_training_data(tables)
# ========================= EOF ====================================================================
