import torch
import torch.nn as nn
import torch.nn.functional as F

from neurolab import params as P
import params as PP
from neurolab import utils
from neurolab.model import Model
import hebb as H
from hebb import functional as HF


class Net(Model):
	# Layer names
	CONV8 = 'conv8'
	RELU8 = 'relu8'
	BN8 = 'bn8'
	CONV_OUTPUT = BN8 # Symbolic name for the last convolutional layer providing extracted features
	FLAT = 'flat'
	FC9 = 'fc9'
	RELU9 = 'relu9'
	BN9 = 'bn9'
	FC10 = 'fc10'
	CLASS_SCORES = FC10 # Symbolic name of the layer providing the class scores as output
	
	def __init__(self, config, input_shape=None):
		super(Net, self).__init__(config, input_shape)
		
		self.NUM_CLASSES = P.GLB_PARAMS[P.KEY_DATASET_METADATA][P.KEY_DS_NUM_CLASSES]
		self.DEEP_TEACHER_SIGNAL = config.CONFIG_OPTIONS.get(P.KEY_DEEP_TEACHER_SIGNAL, False)
		self.COMPETITIVE = False
		self.K = 0
		self.RECONSTR = H.HebbianConv2d.REC_LIN_CMB
		self.RED = H.HebbianConv2d.RED_AVG
		self.LRN_SIM = HF.kernel_mult2d
		self.LRN_ACT = F.relu
		self.OUT_SIM = HF.kernel_mult2d
		self.OUT_ACT = F.relu
		self.WEIGHT_UPD_RULE = H.HebbianConv2d.RULE_HEBB
		self.LOC_LRN_RULE = config.CONFIG_OPTIONS.get(P.KEY_LOCAL_LRN_RULE, 'hpca')
		if self.LOC_LRN_RULE == 'hwta':
			self.COMPETITIVE = True
			self.K = config.CONFIG_OPTIONS.get(PP.KEY_WTA_K, 1)
			self.RECONSTR = H.HebbianConv2d.REC_QNT_SGN
			self.RED = H.HebbianConv2d.RED_W_AVG
			self.LRN_SIM = HF.raised_cos2d_pow(2)
			self.LRN_ACT = HF.identity
			self.OUT_SIM = HF.vector_proj2d
			self.OUT_ACT = F.relu
			self.WEIGHT_UPD_RULE = H.HebbianConv2d.RULE_BASE
		self.ALPHA = config.CONFIG_OPTIONS.get(P.KEY_ALPHA, 1.)
		
		# Here we define the layers of our network
		
		# Eighth convolutional layer
		self.conv8 = H.HebbianConv2d(
			in_channels=self.get_input_shape()[0],
			out_size=(16, 32),
			kernel_size=3,
			competitive=self.COMPETITIVE,
			reconstruction=self.RECONSTR,
			reduction=self.RED,
			lrn_sim=self.LRN_SIM,
			lrn_act=F.relu,
			out_sim=self.OUT_SIM,
			out_act=F.relu,
			weight_upd_rule=self.WEIGHT_UPD_RULE,
			alpha=self.ALPHA,
		)  # 384 input channels, 16x32=512 output channels, 3x3 convolutions
		self.bn8 = nn.BatchNorm2d(512)  # Batch Norm layer
		
		self.CONV_OUTPUT_SHAPE = self.get_output_fmap_shape(self.CONV_OUTPUT)
		
		# FC Layers (convolution with kernel size equal to the entire feature map size is like a fc layer)
		
		self.fc9 = H.HebbianConv2d(
			in_channels=self.CONV_OUTPUT_SHAPE[0],
			out_size=(64, 64),
			kernel_size=(self.CONV_OUTPUT_SHAPE[1], self.CONV_OUTPUT_SHAPE[2]),
			competitive=self.COMPETITIVE,
			reconstruction=self.RECONSTR,
			reduction=self.RED,
			lrn_sim=self.LRN_SIM,
			lrn_act=F.relu,
			out_sim=self.OUT_SIM,
			out_act=F.relu,
			weight_upd_rule=self.WEIGHT_UPD_RULE,
			alpha=self.ALPHA,
		)  # conv_output_shape-shaped input, 64x64=4096 output channels
		self.bn9 = nn.BatchNorm2d(4096)  # Batch Norm layer
		
		self.fc10 = H.HebbianConv2d(
			in_channels=4096,
			out_size=self.NUM_CLASSES,
			kernel_size=1,
			reconstruction=H.HebbianConv2d.REC_QNT_SGN,
			reduction=H.HebbianConv2d.RED_W_AVG,
			lrn_sim=HF.raised_cos2d_pow(2),
			lrn_act=HF.identity,
			out_sim=HF.vector_proj2d,
			out_act=HF.identity,
			weight_upd_rule=H.HebbianConv2d.RULE_BASE,
			alpha=self.ALPHA,
		)  # 4096-dimensional input, NUM_CLASSES-dimensional output (one per class)
	
	def get_conv_output(self, x):
		# Layer 8: Convolutional + Batch Norm
		conv8_out = self.conv8(x)
		bn8_out = HF.modified_bn(self.bn8, conv8_out)
		
		# Build dictionary containing outputs of each layer
		conv_out = {
			self.CONV8: conv8_out,
			self.BN8: bn8_out,
		}
		return conv_out
	
	# Here we define the flow of information through the network
	def forward(self, x):
		# Compute the output feature map from the convolutional layers
		out = self.get_conv_output(x)
		
		# Layer 9: FC + Batch Norm
		fc9_out = self.fc9(out[self.CONV_OUTPUT])
		bn9_out = HF.modified_bn(self.bn9, fc9_out)
		
		# Linear FC layer, outputs are the class scores
		fc10_out = self.fc10(bn9_out).view(-1, self.NUM_CLASSES)
		
		# Build dictionary containing outputs from convolutional and FC layers
		out[self.FC9] = fc9_out
		out[self.BN9] = bn9_out
		out[self.FC10] = fc10_out
		return out
	
	def set_teacher_signal(self, y):
		if y is not None: y = utils.dense2onehot(y, self.NUM_CLASSES)
		
		self.fc10.set_teacher_signal(y)
		if y is None:
			self.conv8.set_teacher_signal(y)
			self.fc9.set_teacher_signal(y)
		elif self.DEEP_TEACHER_SIGNAL:
			# Extend teacher signal for deep layers
			l8_knl_per_class = 500 // self.NUM_CLASSES
			l9_knl_per_class = 4000 // self.NUM_CLASSES
			if self.NUM_CLASSES <= 20:
				self.conv8.set_teacher_signal(
					torch.cat((
						torch.ones(y.size(0), self.conv8.weight.size(0) - l8_knl_per_class * self.NUM_CLASSES, device=y.device),
						y.view(y.size(0), y.size(1), 1).repeat(1, 1, l8_knl_per_class).view(y.size(0), -1),
					), dim=1)
				)
			self.fc9.set_teacher_signal(
				torch.cat((
					torch.ones(y.size(0), self.fc9.weight.size(0) - l9_knl_per_class * self.NUM_CLASSES, device=y.device),
					y.view(y.size(0), y.size(1), 1).repeat(1, 1, l9_knl_per_class).view(y.size(0), -1),
				), dim=1)
			)

	def local_updates(self):
		self.conv8.local_update()
		self.fc9.local_update()
		self.fc10.local_update()

