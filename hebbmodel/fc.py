import torch.nn as nn
import params as P
import hebbmodel.hebb as H


class Net(nn.Module):
	# Layer names
	FC = 'fc'
	CLASS_SCORES = FC  # Symbolic name of the layer providing the class scores as output
	
	def __init__(self, config, input_shape=P.INPUT_SHAPE):
		super(Net, self).__init__()
		
		# Shape of the tensors that we expect to receive as input
		self.input_shape = input_shape
		if len(input_shape) != 3: self.input_shape = (input_shape[0], 1, 1)
		
		# Here we define the layers of our network
		
		# FC Layers
		self.fc = H.HebbianMap2d(
			in_channels=self.input_shape[0],
			out_size=P.NUM_CLASSES,
			kernel_size=(self.input_shape[1], self.input_shape[2]),
			reconstruction=H.HebbianMap2d.REC_QNT_SGN,
			reduction=H.HebbianMap2d.RED_W_AVG,
			lrn_sim=H.raised_cos2d_pow(2),
			lrn_act=H.identity,
			out_sim=H.vector_proj2d,
			out_act=H.identity,
			weight_upd_rule=H.HebbianMap2d.RULE_BASE,
			eta=config.LEARNING_RATE,
		)  # conv kernels with the same height, width depth as input (equivalent to a FC layer), 10 kernels (one per class)
	
	# Here we define the flow of information through the network
	def forward(self, x):
		out = {}
		
		# Linear FC layer, outputs are the class scores
		fc_out = self.fc(x.view(-1, *self.input_shape)).view(-1, P.NUM_CLASSES)
		
		# Build dictionary containing outputs from convolutional and FC layers
		out[self.FC] = fc_out
		return out
	
	# Function for setting teacher signal for supervised hebbian learning
	def set_teacher_signal(self, y, set_deep=False):
		self.fc.set_teacher_signal(y)
