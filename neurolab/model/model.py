import torch
import torch.nn as nn

from .. import params as P


# Base class for network models
class Model(nn.Module):
	def __init__(self, config, input_shape=None):
		super(Model, self).__init__()
		self.config = config
		
		# Shape of the tensors that we expect to receive as input
		self.INPUT_SHAPE = input_shape if input_shape is not None else P.GLB_PARAMS[P.KEY_DATASET_METADATA][P.KEY_DS_INPUT_SHAPE]
	
	# Return module input shape
	def get_input_shape(self):
		return self.INPUT_SHAPE
	
	# This function forwards an input through the convolutional layers and computes the resulting output
	def get_conv_output(self, x):
		raise NotImplementedError
	
	# Compute the shape of the output feature map from any layer of a network. This is useful to correctly set the size
	# of the successive layers. By default, the method considers only convolutional layers. If fwd is true, all layers
	# are considered.
	def get_output_fmap_shape(self, layer, fwd=False):
		training = self.training
		self.eval()
		# In order to compute the shape of the output of the network convolutional layers, we can feed the network with
		# a simulated input and return the resulting output shape
		with torch.no_grad():
			fake_input = torch.ones(1, *self.get_input_shape())
			res = tuple((self(fake_input) if fwd else self.get_conv_output(fake_input))[layer].size())[1:]
		self.train(training)
		return res
	
	# Function for setting teacher signal for supervised hebbian learning
	def set_teacher_signal(self, y):
		pass
	
	def local_updates(self):
		pass
	
	def get_param_groups(self):
		return [{'params': self.parameters()}]

