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
	FC1 = 'fc1'
	BN1 = 'bn1'
	FC2 = 'fc2'
	CLASS_SCORES = FC2  # Symbolic name of the layer providing the class scores as output
	
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
		
		# FC Layers (convolution with kernel size equal to the entire feature map size is like a fc layer)
		
		self.fc1 = H.HebbianConv2d(
			in_channels=self.get_input_shape()[0],
			out_size=(64, 64),
			kernel_size=(self.get_input_shape()[1], self.get_input_shape()[2]) if len(self.get_input_shape()) >= 3 else 1,
			competitive=self.COMPETITIVE,
			reconstruction=self.RECONSTR,
			reduction=self.RED,
			lrn_sim=self.LRN_SIM,
			lrn_act=F.relu,
			out_sim=self.OUT_SIM,
			out_act=F.relu,
			weight_upd_rule=self.WEIGHT_UPD_RULE,
			alpha=self.ALPHA,
		)  # input_shape-shaped input, 64x64=4096 output channels
		self.bn1 = nn.BatchNorm2d(4096)  # Batch Norm layer
		
		self.fc2 = H.HebbianConv2d(
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
		)  # 4000-dimensional input, NUM_CLASSES-dimensional output (one per class)
	
	# Here we define the flow of information through the network
	def forward(self, x):
		out = {}
		
		# Hidden Layer: FC + Batch Norm
		fc1_out = self.fc1(x if len(self.get_input_shape()) >= 3 else x.view(x.size(0), x.size(1), 1, 1))
		bn1_out = HF.modified_bn(self.bn1, fc1_out)
		
		# Output Layer, outputs are the class scores
		fc2_out = self.fc2(bn1_out).view(-1, self.NUM_CLASSES)
		
		# Build dictionary containing outputs from convolutional and FC layers
		out[self.FC1] = fc1_out
		out[self.BN1] = bn1_out
		out[self.FC2] = fc2_out
		return out
	
	def set_teacher_signal(self, y):
		if y is not None: y = utils.dense2onehot(y, self.NUM_CLASSES)
		
		self.fc2.set_teacher_signal(y)
		if y is None:
			self.fc1.set_teacher_signal(y)
		elif self.DEEP_TEACHER_SIGNAL:
			# Extend teacher signal for deep layers
			l1_knl_per_class = 4000 // self.NUM_CLASSES
			self.fc1.set_teacher_signal(
				torch.cat((
					torch.ones(y.size(0), self.fc1.weight.size(0) - l1_knl_per_class * self.NUM_CLASSES, device=y.device),
					y.view(y.size(0), y.size(1), 1).repeat(1, 1, l1_knl_per_class).view(y.size(0), -1),
				), dim=1)
			)

	def local_updates(self):
		self.fc1.local_update()
		self.fc2.local_update()

