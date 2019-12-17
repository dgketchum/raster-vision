import numpy as np
import rasterio
from PIL import Image

import rastervision as rv
from rastervision.data.label import InstanceSegmentationLabels
from rastervision.data.label_store import LabelStore
from rastervision.data.utils import boxes_to_geojson
from rastervision.utils.files import (get_local_path, make_dir, upload_or_copy,
                                      file_exists)
from rastervision.utils.files import json_to_file
from rastervision.viz import display_instances


class InstanceSegmentationLabelStore(LabelStore):
    """A prediction label store for instance segmentation raster files.
    """

    def __init__(self,
                 uri,
                 extent,
                 crs_transformer,
                 tmp_dir,
                 class_map,
                 vector_output=None):
        """Constructor.

        Args:
            uri: (str) URI of GeoTIFF file used for storing predictions as RGB values
            extent: (Box) The extent of the scene
            crs_transformer: (CRSTransformer)
            tmp_dir: (str) temp directory to use
            vector_output: (None or array of dicts) containing vectorifiction
                configuration information
            class_map: (ClassMap) with color values used to convert class ids to
                RGB values

        """
        self.uri = uri
        self.vector_output = vector_output
        self.extent = extent
        self.crs_transformer = crs_transformer
        self.tmp_dir = tmp_dir
        # Note: can't name this class_transformer due to Python using that attribute
        if class_map:
            self.class_trans = None
            # TODO: why use this for instance seg?
            # self.class_trans = SegmentationClassTransformer(class_map)
        else:
            self.class_trans = None

        self.source = None
        if file_exists(uri):
            self.source = rv.RasterSourceConfig.builder(rv.RASTERIO_SOURCE) \
                                               .with_uri(self.uri) \
                                               .build() \
                                               .create_source(self.tmp_dir)

    def _subcomponents_to_activate(self):
        if self.source is not None:
            return [self.source]
        return []

    def get_labels(self, chip_size=1000):
        """Get all labels.

        Returns:
            InstanceSegmentationLabels with windows of size chip_size covering the
                scene with no overlap.
        """

        def label_fn(window):
            raw_labels = self.source.get_raw_chip(window)
            if self.class_trans:
                labels = self.class_trans.rgb_to_class(raw_labels)
            else:
                instances = np.unique(raw_labels)
                labels = raw_labels == instances[:, None, None]

            return labels

        if self.source is None:
            raise Exception('Raster source at {} does not exist'.format(
                self.uri))

        extent = self.source.get_extent()
        windows = extent.get_windows(chip_size, chip_size)
        return InstanceSegmentationLabels(windows, label_fn)

    def save(self, labels, class_map):
        """Save.

        Args:
            labels - (InstanceSegmentationLabels) labels to be saved
        """
        local_path = get_local_path(self.uri, self.tmp_dir)
        make_dir(local_path, use_dirname=True)

        transform = self.crs_transformer.get_affine_transform()
        crs = self.crs_transformer.get_image_crs()

        band_count = 1
        dtype = np.uint8
        if self.class_trans:
            band_count = 3

        if self.vector_output:
            # We need to store the whole output mask to run feature extraction.
            # If the raster is large, this will result in running out of memory, so
            # more work will be needed to get this to work in a scalable way. But this
            # is complicated because of the need to merge features that are split
            # across windows.
            mask = np.zeros(
                (self.extent.ymax, self.extent.xmax), dtype=np.uint8)
        else:
            mask = None

        boxes = labels.get_boxes()
        class_ids = labels.get_class_ids().tolist()
        scores = labels.get_scores().tolist()
        geojson = boxes_to_geojson(
            boxes,
            class_ids,
            self.crs_transformer,
            class_map,
            scores=scores)
        json_to_file(geojson, self.uri)

        # https://github.com/mapbox/rasterio/blob/master/docs/quickstart.rst
        # https://rasterio.readthedocs.io/en/latest/topics/windowed-rw.html
        print(local_path)

        with rasterio.open(
                local_path,
                'w',
                driver='GTiff',
                height=self.extent.ymax,
                width=self.extent.xmax,
                count=band_count,
                dtype=dtype,
                transform=transform,
                crs=crs) as dataset:
            for window in labels.get_windows():
                class_labels = labels.get_label_array(
                    window, clip_extent=self.extent)
                clipped_window = ((window.ymin,
                                   window.ymin + class_labels.shape[0]),
                                  (window.xmin,
                                   window.xmin + class_labels.shape[1]))
                if mask is not None:
                    mask[clipped_window[0][0]:clipped_window[0][1],
                    clipped_window[1][0]:clipped_window[1][
                        1]] = class_labels

                # TODO: why store this as RGB? For visualization?
                # if self.class_trans:
                #     rgb_labels = self.class_trans.class_to_rgb(class_labels)
                #     for chan in range(3):
                #         dataset.write_band(
                #             chan + 1,
                #             rgb_labels[:, :, chan],
                #             window=clipped_window)

                else:
                    img = class_labels.astype(dtype)
                    dataset.write_band(1, img, window=clipped_window)

        # TODO where to put the visualization code/write an RGB representation with instance seg labels?
        im = np.array(Image.open(self.source.imagery_path).convert('RGB'))
        r = rasterio.open(local_path, 'r')
        mask = r.read()
        class_names = [str(x) for x in class_ids]
        display_instances(im, boxes=boxes, masks=masks,
                          class_ids=class_ids,
                          class_names=class_names,
                          show_mask=True, show_bbox=True)

        upload_or_copy(local_path, self.uri)

        if self.vector_output:
            import mask_to_polygons.vectorification as vectorification
            import mask_to_polygons.processing.denoise as denoise

            for vo in self.vector_output:
                denoise_radius = vo['denoise']
                uri = vo['uri']
                mode = vo['mode']
                class_id = vo['class_id']
                class_mask = np.array(mask == class_id, dtype=np.uint8)
                local_geojson_path = get_local_path(uri, self.tmp_dir)

                def transform(x, y):
                    return self.crs_transformer.pixel_to_map((x, y))

                if denoise_radius > 0:
                    class_mask = denoise.denoise(class_mask, denoise_radius)

                if uri and mode == 'buildings':
                    options = vo['building_options']
                    geojson = vectorification.geojson_from_mask(
                        mask=class_mask,
                        transform=transform,
                        mode=mode,
                        min_aspect_ratio=options['min_aspect_ratio'],
                        min_area=options['min_area'],
                        width_factor=options['element_width_factor'],
                        thickness=options['element_thickness'])
                elif uri and mode == 'polygons':
                    geojson = vectorification.geojson_from_mask(
                        mask=class_mask, transform=transform, mode=mode)

                if local_geojson_path:
                    with open(local_geojson_path, 'w') as file_out:
                        file_out.write(geojson)
                        upload_or_copy(local_geojson_path, uri)

    def empty_labels(self):
        """Returns an empty InstanceSegmentationLabels object."""
        return InstanceSegmentationLabels()
