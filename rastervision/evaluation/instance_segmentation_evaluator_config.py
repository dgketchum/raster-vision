import rastervision as rv
from rastervision.evaluation import InstanceSegmentationEvaluator
from rastervision.evaluation \
    import (ClassificationEvaluatorConfig, ClassificationEvaluatorConfigBuilder)


class InstanceSegmentationEvaluatorConfig(ClassificationEvaluatorConfig):
    def __init__(self, class_map, output_uri=None, vector_output_uri=None):
        super().__init__(rv.INSTANCE_SEGMENTATION_EVALUATOR, class_map,
                         output_uri, vector_output_uri)

    def create_evaluator(self):
        return InstanceSegmentationEvaluator(self.class_map, self.output_uri,
                                             self.vector_output_uri)


class InstanceSegmentationEvaluatorConfigBuilder(
        ClassificationEvaluatorConfigBuilder):
    def __init__(self, prev=None):
        super().__init__(InstanceSegmentationEvaluatorConfig, prev)
