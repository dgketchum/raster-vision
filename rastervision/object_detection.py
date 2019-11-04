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
from os.path import join
import rastervision as rv
import csv
from io import StringIO
import tempfile
import os

import rasterio
from shapely.strtree import STRtree
from shapely.geometry import shape

from rastervision.core import Box
from rastervision.data import RasterioCRSTransformer, GeoJSONVectorSource
from rastervision.utils.files import (
    file_to_str, file_exists, get_local_path, upload_or_copy, make_dir,
    file_to_json)
from rastervision.filesystem import S3FileSystem


if 'home' in os.getcwd():
    home = os.path.expanduser('~')
    ROOT_URI = os.path.join(home, 'field_extraction', 'training_data')
    PROCESSED_URI = os.path.join(ROOT_URI, 'example')
    TMP = os.environ['TMPDIR'] = os.path.join(ROOT_URI, 'tmp')
    os.environ['TORCH_HOME'] = os.path.join(home, 'field_extraction', 'torche-cache')
else:
    ROOT_URI = '/opt/data/training_data'
    PROCESSED_URI = os.path.join(ROOT_URI, 'example')


class CowcObjectDetectionExperiments(rv.ExperimentSet):
    def exp_main(self, raw_uri, processed_uri, root_uri, test=False, use_tf=False):
        """Object detection on COWC (Cars Overhead with Context) Potsdam dataset
        Args:
            raw_uri: (str) directory of raw data
            processed_uri: (str) directory of processed data
            root_uri: (str) root directory for experiment output
            test: (bool) if True, run a very small experiment as a test and generate
                debug output
            use_tf: (bool) if True, use Tensorflow-based backend
        """
        test = str_to_bool(test)
        exp_id = 'cowc-object-detection'
        num_steps = 100000
        batch_size = 8
        debug = False
        train_scene_ids = ['2_10', '2_11', '2_12', '2_14', '3_11',
                           '3_13', '4_10', '5_10', '6_7', '6_9']
        val_scene_ids = ['2_13', '6_8', '3_10']

        if test:
            exp_id += '-test'
            num_steps = 1
            batch_size = 2
            debug = True

            train_scene_ids = train_scene_ids[0:1]
            val_scene_ids = val_scene_ids[0:1]

        task = rv.TaskConfig.builder(rv.OBJECT_DETECTION) \
                            .with_chip_size(300) \
                            .with_classes({'vehicle': (1, 'red')}) \
                            .with_chip_options(neg_ratio=5.0,
                                               ioa_thresh=0.9) \
                            .with_predict_options(merge_thresh=0.5,
                                                  score_thresh=0.9) \
                            .build()

        if use_tf:
            backend = rv.BackendConfig.builder(rv.TF_OBJECT_DETECTION) \
                                        .with_task(task) \
                                        .with_model_defaults(rv.SSD_MOBILENET_V1_COCO) \
                                        .with_debug(debug) \
                                        .with_batch_size(batch_size) \
                                        .with_num_steps(num_steps) \
                                        .build()
        else:
            batch_size = 16
            num_epochs = 10
            if test:
                batch_size = 2
                num_epochs = 2

            backend = rv.BackendConfig.builder(rv.PYTORCH_OBJECT_DETECTION) \
                .with_task(task) \
                .with_train_options(
                    lr=1e-4,
                    one_cycle=True,
                    batch_size=batch_size,
                    num_epochs=num_epochs,
                    model_arch='resnet18',
                    debug=debug) \
                .build()

        def make_scene(id):
            raster_uri = join(
                raw_uri, '4_Ortho_RGBIR/top_potsdam_{}_RGBIR.tif'.format(id))
            label_uri = join(
                processed_uri, 'labels', 'all', 'top_potsdam_{}_RGBIR.json'.format(id))

            if test:
                crop_uri = join(
                    processed_uri, 'crops', os.path.basename(raster_uri))
                save_image_crop(raster_uri, crop_uri, label_uri=label_uri,
                                size=1000, min_features=5)
                raster_uri = crop_uri

            return rv.SceneConfig.builder() \
                                 .with_id(id) \
                                 .with_task(task) \
                                 .with_raster_source(raster_uri, channel_order=[0, 1, 2]) \
                                 .with_label_source(label_uri) \
                                 .build()

        train_scenes = [make_scene(id) for id in train_scene_ids]
        val_scenes = [make_scene(id) for id in val_scene_ids]

        dataset = rv.DatasetConfig.builder() \
                                  .with_train_scenes(train_scenes) \
                                  .with_validation_scenes(val_scenes) \
                                  .build()

        experiment = rv.ExperimentConfig.builder() \
                                        .with_id(exp_id) \
                                        .with_root_uri(root_uri) \
                                        .with_task(task) \
                                        .with_backend(backend) \
                                        .with_dataset(dataset) \
                                        .build()

        return experiment


def save_image_crop(image_uri, crop_uri, label_uri=None, size=600,
                    min_features=10):
    """Save a crop of an image to use for testing.
    If label_uri is set, the crop needs to cover >= min_features.
    Args:
        image_uri: URI of original image
        crop_uri: URI of cropped image to save
        label_uri: optional URI of GeoJSON file
        size: height and width of crop
    Raises:
        ValueError if cannot find a crop satisfying min_features constraint.
    """
    if not file_exists(crop_uri):
        print('Saving test crop to {}...'.format(crop_uri))
        old_environ = os.environ.copy()
        try:
            request_payer = S3FileSystem.get_request_payer()
            if request_payer == 'requester':
                os.environ['AWS_REQUEST_PAYER'] = request_payer
            im_dataset = rasterio.open(image_uri)
            h, w = im_dataset.height, im_dataset.width

            extent = Box(0, 0, h, w)
            windows = extent.get_windows(size, size)
            if label_uri is not None:
                crs_transformer = RasterioCRSTransformer.from_dataset(im_dataset)
                vs = GeoJSONVectorSource(label_uri, crs_transformer)
                geojson = vs.get_geojson()
                geoms = []
                for f in geojson['features']:
                    g = shape(f['geometry'])
                    geoms.append(g)
                tree = STRtree(geoms)

            for w in windows:
                use_window = True
                if label_uri is not None:
                    w_polys = tree.query(w.to_shapely())
                    use_window = len(w_polys) >= min_features

                if use_window:
                    w = w.rasterio_format()
                    im = im_dataset.read(window=w)

                    with tempfile.TemporaryDirectory() as tmp_dir:
                        crop_path = get_local_path(crop_uri, tmp_dir)
                        make_dir(crop_path, use_dirname=True)

                        meta = im_dataset.meta
                        meta['width'], meta['height'] = size, size
                        meta['transform'] = rasterio.windows.transform(
                            w, im_dataset.transform)

                        with rasterio.open(crop_path, 'w', **meta) as dst:
                            dst.colorinterp = im_dataset.colorinterp
                            dst.write(im)

                        upload_or_copy(crop_path, crop_uri)
                    break

            if not use_window:
                raise ValueError('Could not find a good crop.')
        finally:
            os.environ.clear()
            os.environ.update(old_environ)


def str_to_bool(x):
    if type(x) == str:
        if x.lower() == 'true':
            return True
        elif x.lower() == 'false':
            return False
        else:
            raise ValueError('{} is expected to be true or false'.format(x))
    return x


if __name__ == '__main__':
    i = CowcObjectDetectionExperiments().exp_main(test=True, raw_uri=ROOT_URI, processed_uri=PROCESSED_URI)
    rv.cli.main.run(['local', '--tempdir', '{}'.format(TMP)])
    # rv.main()

# ========================= EOF ====================================================================
