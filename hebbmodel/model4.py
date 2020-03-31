import torch
import torch.nn as nn
import torch.nn.functional as F
import hebbmodel.hebb as H
import params as P
import utils


class Net(nn.Module):
	# Layer names
	CONV1 = 'conv1'
	POOL1 = 'pool1'
	BN1 = 'bn1'
	CONV2 = 'conv2'
	BN2 = 'bn2'
	CONV3 = 'conv3'
	POOL3 = 'pool3'
	BN3 = 'bn3'
	CONV4 = 'conv4'
	BN4 = 'bn4'
	CONV_OUTPUT = BN4  # Symbolic name for the last convolutional layer providing extracted features
	FC5 = 'fc5'
	CLASS_SCORES = FC5  # Symbolic name of the layer providing the class scores as output
	
	def __init__(self, config, input_shape=P.INPUT_SHAPE):
		super(Net, self).__init__()
		
		# Shape of the tensors that we expect to receive as input
		self.input_shape = input_shape
		
		# Here we define the layers of our network
		
		# First convolutional layer
		self.conv1 = H.HebbianMap2d(
			in_channels=3,
			out_size=(8, 12),
			kernel_size=5,
			eta=config.LEARNING_RATE,
		) # 3 input channels, 8x12=96 output channels, 5x5 convolutions
		self.bn1 = nn.BatchNorm2d(96)  # Batch Norm layer
		
		# Second convolutional layer
		self.conv2 = H.HebbianMap2d(
			in_channels=96,
			out_size=(8, 16),
			kernel_size=3,
			eta=config.LEARNING_RATE,
		)  # 96 input channels, 8x16=128 output channels, 3x3 convolutions
		self.bn2 = nn.BatchNorm2d(128)  # Batch Norm layer
		
		# Third convolutional layer
		self.conv3 = H.HebbianMap2d(
			in_channels=128,
			out_size=(12, 16),
			kernel_size=3,
			eta=config.LEARNING_RATE,
		)  # 128 input channels, 12x16=192 output channels, 3x3 convolutions
		self.bn3 = nn.BatchNorm2d(192)  # Batch Norm layer
		
		# Fourth convolutional layer
		self.conv4 = H.HebbianMap2d(
			in_channels=192,
			out_size=(16, 16),
			kernel_size=3,
			eta=config.LEARNING_RATE,
		)  # 192 input channels, 16x16=256 output channels, 3x3 convolutions
		self.bn4 = nn.BatchNorm2d(256)  # Batch Norm layer
		
		self.conv_output_shape = utils.get_conv_output_shape(self)

		# FC Layers (convolution with kernel size equal to the entire feature map size is like a fc layer)

		self.fc5 = H.HebbianMap2d(
			in_channels=self.conv_output_shape[0],
			out_size=P.NUM_CLASSES,
			kernel_size=(self.conv_output_shape[1], self.conv_output_shape[2]),
			reconstruction=H.HebbianMap2d.REC_QNT_SGN,
			reduction=H.HebbianMap2d.RED_W_AVG,
			lrn_sim=H.raised_cos2d_pow(2),
			lrn_act=H.identity,
			out_sim=H.vector_proj2d,
			out_act=H.identity,
			weight_upd_rule=H.HebbianMap2d.RULE_BASE,
			eta=config.LEARNING_RATE,
		)  # conv_output_shape-shaped input, 10-dimensional output (one per class)
	
	# This function forwards an input through the convolutional layers and computes the resulting output
	def get_conv_output(self, x):
		# Layer 1: Convolutional + 2x2 Max Pooling + Batch Norm
		conv1_out = self.conv1(x)
		pool1_out = F.max_pool2d(conv1_out, 2)
		bn1_out = self.bn1(pool1_out)
		
		# Layer 2: Convolutional + Batch Norm
		conv2_out = self.conv2(bn1_out)
		bn2_out = self.bn2(conv2_out)
		
		# Layer 3: Convolutional + 2x2 Max Pooling + Batch Norm
		conv3_out = self.conv3(bn2_out)
		pool3_out = F.max_pool2d(conv3_out, 2)
		bn3_out = self.bn3(pool3_out)
		
		# Layer 4: Convolutional + Batch Norm
		conv4_out = self.conv4(bn3_out)
		bn4_out = self.bn4(conv4_out)
		
		# Build dictionary containing outputs of each layer
		conv_out = {
			self.CONV1: conv1_out,
			self.POOL1: pool1_out,
			self.BN1: bn1_out,
			self.CONV2: conv2_out,
			self.BN2: bn2_out,
			self.CONV3: conv3_out,
			self.POOL3: pool3_out,
			self.BN3: bn3_out,
			self.CONV4: conv4_out,
			self.BN4: bn4_out,
		}
		return conv_out
	
	# Here we define the flow of information through the network
	def forward(self, x):
		# Compute the output feature map from the convolutional layers
		out = self.get_conv_output(x)

		# Linear FC layer, outputs are the class scores
		fc5_out = self.fc5(out[self.CONV_OUTPUT]).view(-1, P.NUM_CLASSES)

		# Build dictionary containing outputs from convolutional and FC layers
		out[self.FC5] = fc5_out
		return out
	
	# Function for setting teacher signal for supervised hebbian learning
	def set_teacher_signal(self, y, set_deep=False):
		self.fc5.set_teacher_signal(y)
		
		if y is None:
			self.conv2.set_teacher_signal(y)
			self.conv3.set_teacher_signal(y)
			self.conv4.set_teacher_signal(y)
		elif set_deep:
			# Extend teacher signal for layer 2, 3, 4
			l2_knl_per_class = 5
			l3_knl_per_class = 12
			l4_knl_per_class = 20
			l5_knl_per_class = 25
			self.conv2.set_teacher_signal(
				torch.cat((
					torch.ones(y.size(0), self.conv2.weight.size(0) - l2_knl_per_class * P.NUM_CLASSES, device=y.device),
					y.view(y.size(0), y.size(1), 1).repeat(1, 1, l2_knl_per_class).view(y.size(0), -1),
				), dim=1)
			)
			self.conv3.set_teacher_signal(
				torch.cat((
					torch.ones(y.size(0), self.conv3.weight.size(0) - l3_knl_per_class * P.NUM_CLASSES, device=y.device),
					y.view(y.size(0), y.size(1), 1).repeat(1, 1, l3_knl_per_class).view(y.size(0), -1),
				), dim=1)
			)
			self.conv4.set_teacher_signal(
				torch.cat((
					torch.ones(y.size(0), self.conv4.weight.size(0) - l4_knl_per_class * P.NUM_CLASSES, device=y.device),
					y.view(y.size(0), y.size(1), 1).repeat(1, 1, l4_knl_per_class).view(y.size(0), -1),
				), dim=1)
			)

