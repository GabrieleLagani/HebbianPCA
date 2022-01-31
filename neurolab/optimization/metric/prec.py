import torch

from ..optimization import MetricManager
from neurolab import params as P


# Evaluate outputs on a retrieval experiment basis: outputs must contain list of retrieved elements for each query/input.
# Return average precision score @k (number of retrieved elements), renormalized to 0, 1, averaged over all the queries
# (Mean Average Precision - MAP score) over the batch.
class PrecMetric:
	def __call__(self, outputs, targets):
		k = outputs.size(1)
		outputs = (outputs == targets.view(-1, 1)).float()
		prec_at_i = outputs.cumsum(dim=1)/torch.arange(1, k + 1, device=P.DEVICE).float().view(1, -1)
		aps = (outputs * prec_at_i).sum(dim=1)/k
		return aps.mean().item()

# Criterion manager for MAP
class PrecMetricManager(MetricManager):
	def __init__(self, config):
		super().__init__(config)
	
	def get_metric(self):
		return PrecMetric()

	def higher_is_better(self):
		return True
	
	def get_name(self):
		return "precision"