from rastervision.data.label import Labels
from rastervision.data.label.tfod_utils.np_box_list import BoxList
from rastervision.core.box import Box

from copy import deepcopy
import numpy as np
from scipy.stats import mode
from rasterio.features import rasterize
import shapely


class InstanceSegmentationLabels(Labels):
    """A set of spatially referenced instance segmentation labels.

    Since labels are represented as rasters, the labels for a scene can take up a lot of
    memory. Therefore, to avoid running out of memory, labels are computed as needed for
    windows.
    """

    def __init__(self, windows, labels, aoi_polygons=None):
        """Constructor

        Args:
            windows: a list of Box representing the windows covering a scene
            label_fn: a function that takes a window (Box) and returns a label array
                of the same shape with each value a class id.
            aoi_polygons: a list of shapely.geom that contains the AOIs
                (areas of interest) for a scene.

        """
        self.windows = windows
        self.aoi_polygons = aoi_polygons

        if isinstance(labels, dict):
            self.masks = labels['masks']
            self.boxes = labels['boxes']
            self.labels = labels['labels']
            self.scores = labels['scores']

            self.boxlist = BoxList(self.boxes)
            self.boxlist.add_field('classes', self.labels)
            self.boxlist.add_field('scores', self.scores)

    def __add__(self, other):
        """Add labels to these labels.

        Returns a concatenation of this and the other labels.
        """

        self.labels = np.concatenate([self.labels, other.labels], axis=0)
        self.masks = np.concatenate([self.masks, other.masks], axis=0)
        self.boxes = np.concatenate([self.boxes, other.boxes], axis=0)
        self.scores = np.concatenate([self.scores, other.scores], axis=0)
        self.boxlist = BoxList(self.boxes)
        self.boxlist.add_field('classes', self.labels)
        self.boxlist.add_field('scores', self.scores)

        return self

    def __eq__(self, other):
        for window in self.get_windows():
            if not np.array_equal(
                    self.get_label_array(window), other.get_label_array(window)):
                return False
        return True

    def filter_by_aoi(self, aoi_polygons):
        """Returns a new InstanceSegmentationLabels object with aoi_polygons set."""
        return InstanceSegmentationLabels(
            self.windows, self.label_fn, aoi_polygons=aoi_polygons)

    def add_window(self, window):
        self.windows.append(window)

    def get_windows(self):
        return self.windows

    @staticmethod
    def local_to_global(npboxes, window):
        """Convert from local to global coordinates.

        The local coordinates are row/col within the window frame of reference.
        The global coordinates are row/col within the extent of a RasterSource.
        """
        xmin = window.xmin
        ymin = window.ymin
        return npboxes + np.array([[ymin, xmin, ymin, xmin]])

    def get_label_array(self, window, clip_extent=None):
        """Get the label array for a window.

        Note: the window should be kept relatively small to avoid running out of memory.

        Args:
            window: Box
            clip_extent: a Box representing the extent of the corresponding Scene

        Returns:
            np.ndarray of class_ids with zeros filled in outside the AOIs and clipped
                to the clip_extent
        """
        window_geom = window.to_shapely()

        if not self.aoi_polygons:
            masks = deepcopy(self.masks)
            masks[masks == 0] = np.nan

            # TODO: collapse masks here?

            for m in range(masks.shape[0]):
                pass
            label_arr = mode(masks, axis=0, nan_policy='omit')[0][0, 0, :, :]
        else:
            # For each aoi_polygon, intersect with window, and put in window frame of
            # reference.
            window_aois = []
            for aoi in self.aoi_polygons:
                window_aoi = aoi.intersection(window_geom)
                if not window_aoi.is_empty:

                    def transform_shape(x, y, z=None):
                        return (x - window.xmin, y - window.ymin)

                    window_aoi = shapely.ops.transform(transform_shape,
                                                       window_aoi)
                    window_aois.append(window_aoi)

            if window_aois:
                # If window intersects with AOI, set pixels outside the AOI polygon to 0,
                # so they are ignored during eval.
                label_arr = self.label_fn(window)
                mask = rasterize(
                    [(p, 0) for p in window_aois],
                    out_shape=label_arr.shape,
                    fill=1,
                    dtype=np.uint8)
                label_arr[mask.astype(np.bool)] = 0
            else:
                # If window does't overlap with any AOI, then return all zeros.
                label_arr = np.zeros((window.get_height(), window.get_width()))

        if clip_extent is not None:
            clip_window = window.intersection(clip_extent)
            label_arr = label_arr[0:clip_window.get_height(), 0:
                                  clip_window.get_width()]

        return label_arr

    def get_boxes(self):
        """Return list of Boxes."""
        return [Box.from_npbox(npbox) for npbox in self.boxlist.get()]

    def get_class_ids(self):
        return self.boxlist.get_field('classes')

    def get_scores(self):
        return self.boxlist.get_field('scores')
