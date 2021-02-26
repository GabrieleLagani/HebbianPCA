import torch.nn as nn

from ..optimization import MetricManager


# Criterion manager for cross entropy loss
class CrossEntMetricManager(MetricManager):
	def __init__(self, config):
		super().__init__(config)
	
	def get_metric(self):
		return nn.CrossEntropyLoss()
	
	def higher_is_better(self):
		return False
	
	def get_name(self):
		return "cross-entropy"