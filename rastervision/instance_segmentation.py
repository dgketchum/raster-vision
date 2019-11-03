import os

import rastervision as rv

aoi_path = 'AOIs/AOI_1_Rio/srcData/buildingLabels/Rio_OUTLINE_Public_AOI.geojson'

RAW_URI = '/opt/data/training_data'
PROCESSED_URI = os.path.join(RAW_URI, 'example')
ROOT_URI = '/opt/data/training_data'


class SemanticSegmentationExperiments(rv.ExperimentSet):
    def exp_main(self):

        train_scene_info = get_scene_info('train')
        val_scene_info = get_scene_info('val')

        exp_id = 'idaho-espa-semseg'
        classes = {'field': (1, 'green'), 'background': (0, 'white')}

        task = rv.TaskConfig.builder(rv.SEMANTIC_SEGMENTATION) \
            .with_chip_size(300) \
            .with_chip_options(chips_per_scene=50) \
            .with_classes(classes) \
            .build()

        backend = rv.BackendConfig.builder(rv.PYTORCH_SEMANTIC_SEGMENTATION) \
            .with_task(task) \
            .with_train_options(
            batch_size=8,
            lr=1e-4,
            num_epochs=100,
            model_arch='resnet50',
            debug=True) \
            .build()

        train_scenes = [make_scene(x, y, task) for x, y in train_scene_info]
        val_scenes = [make_scene(x, y, task) for x, y in val_scene_info]

        dataset = rv.DatasetConfig.builder() \
            .with_train_scenes(train_scenes) \
            .with_validation_scenes(val_scenes) \
            .build()

        experiment = rv.ExperimentConfig.builder() \
            .with_id(exp_id) \
            .with_root_uri(ROOT_URI) \
            .with_task(task) \
            .with_backend(backend) \
            .with_dataset(dataset) \
            .with_stats_analyzer() \
            .build()

        return experiment


def make_scene(raster_uri, label_uri, task):

    _id = os.path.splitext(os.path.basename(raster_uri))[0]

    label_raster_source = rv.RasterSourceConfig.builder(rv.RASTERIO_SOURCE) \
        .with_uri(label_uri)\
        .build()

    label_source = rv.LabelSourceConfig.builder(rv.SEMANTIC_SEGMENTATION) \
        .with_raster_source(label_raster_source) \
        .build()

    return rv.SceneConfig.builder() \
        .with_task(task) \
        .with_id(_id) \
        .with_raster_source(raster_uri, channel_order=[0, 1, 2]) \
        .with_label_source(label_source) \
        .build()


def get_scene_info(_type='train'):
    img_dir_train = os.path.join(PROCESSED_URI, 'image_{}'.format(_type))
    img_uris_train = [os.path.join(img_dir_train, x) for x in os.listdir(img_dir_train) if x.endswith('.tif')]
    labels_train = img_dir_train.replace('image_{}'.format(_type), 'label_{}'.format(_type))
    labels_uris_train = [os.path.join(labels_train, x) for x in os.listdir(img_dir_train) if x.endswith('.tif')]
    return [(x, y) for x, y in zip(img_uris_train, labels_uris_train)]


if __name__ == '__main__':
    rv.main()
