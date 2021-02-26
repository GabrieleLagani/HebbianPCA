import torch.nn as nn

from .functional import *
import params as P


# TODO: Implement Hebbian update directly on gpu. Alternatively, use matmul and the exp-log trick to switch multiplications to sums.
# TODO: Add other functionalities to Hebbian module.

# This module represents a layer of convolutional neurons that are trained with a Hebbian-WTA rule
class HebbianConv2d(nn.Module):
	# delta_w = alpha * r * (x - reconst)
	
	# Types of learning rules
	RULE_BASE = 'base'  # r = lfb
	RULE_HEBB = 'hebb'  # r = y * lfb
	RULE_DIFF = 'diff'  # r = y' * (lfb - y)
	RULE_SMX = 'smx'  # r = y' * (lfb - softmax(y))
	
	# Type of reconstruction scheme
	REC_QNT = 'qnt'  # reconst = w
	REC_QNT_SGN = 'qnt_sgn'  # reconst = sign(lfb) * w
	REC_LIN_CMB = 'lin_cmb'  # reconst = sum_i y_i w_i
	
	# Types of LFB kernels
	LFB_GAUSS = 'gauss'
	LFB_DoG = 'DoG'
	LFB_EXP = 'exp'
	LFB_DoE = 'DoE'
	
	# Types of weight initialization schemes
	INIT_BASE = 'base'
	INIT_NORM = 'norm'
	
	# Type of update reduction scheme
	RED_AVG = 'avg'  # average
	RED_W_AVG = 'w_avg'  # weighted average
	
	def __init__(self,
	             in_channels,
	             out_size,
	             kernel_size,
	             competitive=False,
	             reconstruction=REC_LIN_CMB,
	             reduction=RED_AVG,
	             random_abstention=False,
	             lfb_value=0,
	             lrn_sim=kernel_mult2d,
	             lrn_act=identity,
	             out_sim=kernel_mult2d,
	             out_act=identity,
	             weight_init=INIT_BASE,
	             weight_upd_rule=RULE_HEBB,
	             alpha=1.,
	             tau=1000):
		super(HebbianConv2d, self).__init__()
		
		# Init weights
		out_size_list = [out_size] if not hasattr(out_size, '__len__') else out_size
		self.out_size = torch.tensor(out_size_list[0:min(len(out_size_list), 3)])
		out_channels = self.out_size.prod().item()
		if hasattr(kernel_size, '__len__') and len(kernel_size) == 1: kernel_size = kernel_size[0]
		if not hasattr(kernel_size, '__len__'): kernel_size = [kernel_size, kernel_size]
		stdv = 1 / (in_channels * kernel_size[0] * kernel_size[1]) ** 0.5
		#self.register_buffer('weight', torch.empty(out_channels, in_channels, kernel_size[0], kernel_size[1]))
		self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size[0], kernel_size[1]), requires_grad=True)
		nn.init.uniform_(self.weight, -stdv, stdv)  # Same initialization used by default pytorch conv modules (the one from the paper "Efficient Backprop, LeCun")
		if weight_init == self.INIT_NORM: self.weight = self.weight / self.weight.view(self.weight.size(0), -1).norm(dim=1, p=2).view(-1, 1, 1, 1)  # normalize weights
		
		# Enable/disable features as random abstention, competitive learning, lateral feedback, type of reconstruction
		self.competitive = competitive
		self.reconstruction = reconstruction
		self.reduction = reduction
		self.random_abstention = competitive and random_abstention
		self.lfb_on = competitive and isinstance(lfb_value, str)
		self.lfb_value = lfb_value
		
		# Set output function, similarity function and learning rule
		self.lrn_sim = lrn_sim
		self.lrn_act = lrn_act
		self.out_sim = out_sim
		self.out_act = out_act
		self.teacher_signal = None  # Teacher signal for supervised training
		self.weight_upd_rule = weight_upd_rule
		
		# Alpha is the constant which determines the trade off between global and local updates
		self.alpha = alpha
		
		# Buffer where the weight update is stored
		self.register_buffer('delta_w', torch.zeros_like(self.weight))
		
		# Set parameters related to the lateral feedback feature
		if self.lfb_on:
			# Prepare the variables to generate the kernel that will be used to apply lateral feedback
			map_radius = (self.out_size - 1) // 2
			sigma_lfb = map_radius.max().item()
			x = torch.abs(torch.arange(0, self.out_size[0].item()) - map_radius[0])
			for i in range(1, self.out_size.size(0)):
				x_new = torch.abs(torch.arange(0, self.out_size[i].item()) - map_radius[i])
				for j in range(i): x_new = x_new.unsqueeze(j)
				x = torch.max(x.unsqueeze(-1),
				              x_new)  # max gives L_infinity distance, sum would give L_1 distance, root_p(sum x^p) for L_p
			# Store the kernel that will be used to apply lateral feedback in a registered buffer
			if lfb_value == self.LFB_EXP or lfb_value == self.LFB_DoE:
				self.register_buffer('lfb_kernel', torch.exp(-x.float() / sigma_lfb))
			else:
				self.register_buffer('lfb_kernel', torch.exp(-x.pow(2).float() / (2 * (sigma_lfb ** 2))))
			# Padding that will pad the inputs before applying the lfb kernel
			pad_pre = map_radius.unsqueeze(1)
			pad_post = (self.out_size - 1 - map_radius).unsqueeze(1)
			self.pad = tuple(torch.cat((pad_pre, pad_post), dim=1).flip(0).view(-1))
			# LFB kernel shrinking parameter
			self.gamma = torch.exp(torch.log(torch.tensor(sigma_lfb).float()) / tau).item()
			if lfb_value == self.LFB_GAUSS or lfb_value == self.LFB_DoG: self.gamma = self.gamma ** 2
		else:
			self.register_buffer('lfb_kernel', None)
		
		# Init variables for statistics collection
		if self.random_abstention:
			self.register_buffer('victories_count', torch.zeros(out_channels))
		else:
			self.register_buffer('victories_count', None)
	
	def set_teacher_signal(self, y):
		self.teacher_signal = y
	
	def forward(self, x):
		y = self.out_act(self.out_sim(x, self.weight))
		if self.training and self.alpha != 0: self.compute_update(x)
		return y
	
	def compute_update(self, x):
		# Store previous gradient computation flag and disable gradient computation before computing update
		prev_grad_enabled = torch.is_grad_enabled()
		torch.set_grad_enabled(False)
		
		# Prepare the inputs
		s = self.lrn_sim(x, self.weight)
		# Compute y and y'
		torch.set_grad_enabled(True)
		s.requires_grad = True
		y = self.lrn_act(s)
		y.backward(torch.ones_like(y), retain_graph=prev_grad_enabled)
		y_prime = s.grad
		torch.set_grad_enabled(False)
		# Prepare other inputs
		t = self.teacher_signal
		if t is not None: t = t.unsqueeze(2).unsqueeze(3) * torch.ones_like(s, device=s.device)
		s = s.permute(0, 2, 3, 1).contiguous().view(-1, self.weight.size(0))
		y = y.permute(0, 2, 3, 1).contiguous().view(-1, self.weight.size(0))
		y_prime = y_prime.permute(0, 2, 3, 1).contiguous().view(-1, self.weight.size(0))
		if t is not None: t = t.permute(0, 2, 3, 1).contiguous().view(-1, self.weight.size(0))
		x_unf = unfold_map2d(x, self.weight.size(2), self.weight.size(3))
		x_unf = x_unf.permute(0, 2, 3, 1, 4).contiguous().view(s.size(0), 1, -1)
		
		# Random abstention
		if self.random_abstention:
			abst_prob = self.victories_count / (self.victories_count.max() + y.size(0) / y.size(1)).clamp(1)
			scores = y * (torch.rand_like(abst_prob, device=y.device) >= abst_prob).float().unsqueeze(0)
		else: scores = y
		
		# Competition. The returned winner_mask is a bitmap telling where a neuron won and where one lost.
		if self.competitive:
			if t is not None: scores = scores * t
			winner_mask = (scores == scores.max(1, keepdim=True)[0]).float()
			if self.random_abstention:  # Update statistics if using random abstension
				winner_mask_sum = winner_mask.sum(0)  # Number of inputs over which a neuron won
				self.victories_count += winner_mask_sum
				self.victories_count -= self.victories_count.min().item()
		else: winner_mask = torch.ones_like(y, device=y.device)
		
		# Lateral feedback
		if self.lfb_on:
			lfb_kernel = self.lfb_kernel
			if self.lfb_value == self.LFB_DoG or self.lfb_value == self.LFB_DoE: lfb_kernel = 2 * lfb_kernel - lfb_kernel.pow(0.5)  # Difference of Gaussians/Exponentials (mexican hat shaped function)
			lfb_in = F.pad(winner_mask.view(-1, *self.out_size), self.pad)
			if self.out_size.size(0) == 1: lfb_out = torch.conv1d(lfb_in.unsqueeze(1), lfb_kernel.unsqueeze(0).unsqueeze(1))
			elif self.out_size.size(0) == 2: lfb_out = torch.conv2d(lfb_in.unsqueeze(1), lfb_kernel.unsqueeze(0).unsqueeze(1))
			else: lfb_out = torch.conv3d(lfb_in.unsqueeze(1), lfb_kernel.unsqueeze(0).unsqueeze(1))
			lfb_out = lfb_out.clamp(-1, 1).view_as(y)
		else:
			lfb_out = winner_mask
			if self.competitive: lfb_out[lfb_out == 0] = self.lfb_value
			elif t is not None: lfb_out = t
		
		# Compute step modulation coefficient
		r = lfb_out  # RULE_BASE
		if self.weight_upd_rule == self.RULE_HEBB: r = r * y
		if self.weight_upd_rule == self.RULE_DIFF: r = y_prime * (r - y)
		if self.weight_upd_rule == self.RULE_SMX: r = y_prime * (r - torch.softmax(y, dim=1))
		r_abs = r.abs()
		r_sign = r.sign()
		
		# Compute delta_w (serialized version for computation of delta_w using less memory)
		w = self.weight.view(1, self.weight.size(0), -1)
		delta_w_avg = torch.zeros_like(self.weight.view(self.weight.size(0), -1))
		x_bar = None
		for i in range((self.weight.size(0) // P.HEBB_UPD_GRP) +
		               (1 if self.weight.size(0) % P.HEBB_UPD_GRP != 0 else 0)):
			start = i * P.HEBB_UPD_GRP
			end = min((i + 1) * P.HEBB_UPD_GRP, self.weight.size(0))
			w_i = w[:, start:end, :]
			r_i = r.unsqueeze(2)[:, start:end, :]
			r_abs_i = r_abs.unsqueeze(2)[:, start:end, :]
			r_sign_i = r_sign.unsqueeze(2)[:, start:end, :]
			if self.reconstruction == self.REC_QNT: x_bar = w_i
			if self.reconstruction == self.REC_QNT_SGN: x_bar = r_sign_i * w_i
			if self.reconstruction == self.REC_LIN_CMB: x_bar = torch.cumsum(r_i * w_i, dim=1) + (x_bar[:, -1, :].unsqueeze(1) if x_bar is not None else 0.)
			x_bar = x_bar if x_bar is not None else 0.
			delta_w_i = r_i * (x_unf - x_bar)
			# Since we use batches of inputs, we need to aggregate the different update steps of each kernel in a unique
			# update. We do this by taking the weighted average of the steps, the weights being the r coefficients that
			# determine the length of each step
			if self.reduction == self.RED_W_AVG:
				r_sum = r_abs_i.sum(0)
				r_sum = r_sum + (r_sum == 0).float()  # Prevent divisions by zero
				delta_w_avg[start:end, :] = (delta_w_i * r_abs_i).sum(0) / r_sum
			else:
				delta_w_avg[start:end, :] = delta_w_i.mean(dim=0)  # RED_AVG
		
		# Apply delta
		self.delta_w = delta_w_avg.view_as(self.weight)
		
		# LFB kernel shrinking schedule
		if self.lfb_on: self.lfb_kernel = self.lfb_kernel.pow(self.gamma)
		
		# Restore gradient computation
		torch.set_grad_enabled(prev_grad_enabled)
		
	# Takes local update from self.delta_w, global update from self.weight.grad, and combines them using the parameter alpha.
	def local_update(self):
		if self.alpha != 0:
			# NB: self.delta_w has a minus sign in front because the optimizer will take update steps in the opposite direction.
			self.weight.grad = self.alpha * (-self.delta_w) + (1 - self.alpha) * (self.weight.grad if self.weight.grad is not None else 0.)

