import torch
import torch.nn as nn
import torch.nn.functional as F
import utils

# Apply unfold operation to input in order to prepare it to be processed against a sliding kernel whose shape
# is passed as argument.
def unfold_map2d(input, kernel_height, kernel_width):
	# Before performing an operation between an input and a sliding kernel we need to unfold the input, i.e. the
	# windows on which the kernel is going to be applied are extracted and set apart. For this purpose, the kernel
	# shape is passed as argument to the operation. The single extracted windows are reshaped by the unfold operation
	# to rank 1 vectors. The output of F.unfold(input, (kernel_height, kernel_width)).transpose(1, 2) is a
	# tensor structured as follows: the first dimension is the batch dimension; the second dimension is the slide
	# dimension, i.e. each element is a window extracted at a different offset (and reshaped to a rank 1 vector);
	# the third dimension is a scalar within said vector.
	inp_unf = F.unfold(input, (kernel_height, kernel_width)).transpose(1, 2)
	# Now we need to reshape our tensors to the actual shape that we want in output, which is the following: the
	# first dimension is the batch dimension, the second dimension is the output channels dimension, the third and
	# fourth are height and width dimensions (obtained by splitting the former third dimension, the slide dimension,
	# representing a linear offset within the input map, into two new dimensions representing height and width), the
	# fifth is the window components dimension, corresponding to the elements of a window extracted from the input with
	# the unfold operation (reshaped to rank 1 vectors). The resulting tensor is then returned.
	inp_unf = inp_unf.view(
		input.size(0),  # Batch dimension
		1,  # Output channels dimension
		input.size(2) - kernel_height + 1,  # Height dimension
		input.size(3) - kernel_width + 1,  # Width dimension
		-1  # Filter/window dimension
	)
	return inp_unf

# Custom vectorial function representing sum of an input with a sliding kernel, just like convolution is multiplication
# by a sliding kernel (as an analogy think convolution as a kernel_mult2d)
def kernel_sum2d(input, kernel):
	# In order to perform the sum with the sliding kernel we first need to unfold the input. The resulting tensor will
	# have the following structure: the first dimension is the batch dimension, the second dimension is the output
	# channels dimension, the third and fourth are height and width dimensions, the fifth is the filter/window
	# components dimension, corresponding to the elements of a window extracted from the input with the unfold
	# operation and equivalently to the elements of a filter (reshaped to rank 1 vectors)
	inp_unf = unfold_map2d(input, kernel.size(2), kernel.size(3))
	# At this point the two tensors can be summed. The kernel is reshaped by unsqueezing singleton dimensions along
	# the batch dimension and the height and width dimensions. By exploiting broadcasting, it happens that the inp_unf
	# tensor is broadcast over the output channels dimension (since its shape along this dimension is 1) and therefore
	# it is automatically processed against the different filters of the kernel. In the same way, the kernel is
	# broadcast along the first dimension (and thus automatically processed against the different inputs along
	# the batch dimension) and along the third and fourth dimensions (and thus automatically processed against
	# different windows extracted from the image at different height and width offsets).
	out = inp_unf + kernel.view(1, kernel.size(0), 1, 1, -1)
	return out

# Test the implementation of the kernel_sum2d function
def test_kernelsum():
	x = torch.randn(
		8,  # Batch dimension
		3,  # Input channels dimension
		10, # Height dimension
		12  # Width dimension
	)
	w = torch.randn(
		6,  # Output channels dimension
		3,  # Input channels dimension
		4,  # Height dimension
		5   # Width dimension
	)
	output = torch.empty(
		x.shape[0],  # Batch dimension
		w.shape[0],  # Output channels dimension
		x.shape[2] - w.shape[2] + 1,  # Height dimension
		x.shape[3] - w.shape[3] + 1,  # Width dimension
		w.shape[1] * w.shape[2] * w.shape[3]  # Filter dimension
	)
	
	# Cross-validate vectorial implementation with for-loop implementation
	for batch in range(0, x.shape[0]):  # Loop over batch dimension
		for outchn in range(0, w.shape[0]):  # Loop over output channel dimension
			for i in range(0, x.shape[2] - w.shape[2] + 1):  # Loop over height dimension
				for j in range(0, x.shape[3] - w.shape[3] + 1):  # Loop over width dimension
					output[batch, outchn, i, j, :] = (x[batch, :, i:i + w.shape[2], j:j + w.shape[3]] + w[outchn, :, :, :]).view(-1)
	
	out = kernel_sum2d(x, w)
	
	print((output.equal(out)))  # Should print out True


# The identity function
def identity(x):
	return x

# Compute product between input and sliding kernel
def kernel_mult2d(x, w, b=None):
	return F.conv2d(x, w, b)

# Projection of input on weight vectors
def vector_proj2d(x, w, bias=None):
	# Compute scalar product with sliding kernel
	prod = kernel_mult2d(x, w)
	# Divide by the norm of the weight vector to obtain the projection
	norm_w = torch.norm(w.view(w.size(0), -1), p=2, dim=1).view(1, -1, 1, 1)
	norm_w += (norm_w == 0).float()  # Prevent divisions by zero
	if bias is None: return prod / norm_w
	return prod / norm_w + bias.view(1, -1, 1, 1)

# Cosine similarity between an input map and a sliding kernel
def cos_sim2d(x, w, bias=None):
	proj = vector_proj2d(x, w)
	# Divide by the norm of the input to obtain the cosine similarity
	x_unf = unfold_map2d(x, w.size(2), w.size(3))
	norm_x = torch.norm(x_unf, p=2, dim=4)
	norm_x += (norm_x == 0).float()  # Prevent divisions by zero
	if bias is None: return proj / norm_x
	return (proj / norm_x + bias.view(1, -1, 1, 1)).clamp(-1, 1)

# Cosine similarity remapped to 0, 1
def raised_cos2d(x, w, bias=None):
	return (cos_sim2d(x, w, bias) + 1) / 2

# Returns function that computes raised cosine power p
def raised_cos2d_pow(p=2):
	def raised_cos2d_pow_p(x, w, bias=None):
		if bias is None: return raised_cos2d(x, w).pow(p)
		return (raised_cos2d(x, w).pow(p) + bias.view(1, -1, 1, 1)).clamp(0, 1)
	return raised_cos2d_pow_p

# Response of a gaussian activation function
def gauss(x, w, sigma=None):
	try:
		d = torch.norm(kernel_sum2d(x, -w), p=2, dim=4)
	except (MemoryError, RuntimeError):
		# Serialized version of distance computation using less memory
		x_unf = unfold_map2d(x, w.size(2), w.size(3)).squeeze(1) # Squeeze out-channel dimension, because we are going to do serial processing of 1 channel at a time
		d = torch.zeros(x_unf.size(0), w.size(0), x_unf.size(1), x_unf.size(2), device=x.device) # batch, out-channel, height, width
		for i in range(w.size(0)):
			w_i = w[i, :, :, :].view(1, 1, 1, -1) # batch, height, width, filter
			d[:, i, :, :] = torch.norm(x_unf - w_i, p=2, dim=3) # w_i broadcast over x_unf batch, height and width dimensions
	if sigma is None: return torch.exp(-d.pow(2) / (2*utils.shape2size(tuple(w[0].size())))) # heuristic: use number of dimensions as variance
	#if sigma is None: return torch.exp(-d.pow(2) / (2 * torch.norm(w.view(w.size(0), 1, -1) - w.view(1, w.size(0), -1), p=2, dim=2).max().pow(2)/w.size(0))) # heuristic: normalization condition
	#if sigma is None: return torch.exp(-d.pow(2) / (2 * d.mean().pow(2)))
	return torch.exp(-d.pow(2) / (2 * (sigma.view(1, -1, 1, 1).pow(2))))

# Returns lambda function for exponentially decreasing learning rate scheduling
def sched_exp(tau=1000, eta_min=0.01):
	gamma = torch.exp(torch.tensor(-1./tau)).item()
	return lambda eta: (eta * gamma).clamp(eta_min)


# This module represents a layer of convolutional neurons that are trained with a Hebbian-WTA rule
class HebbianMap2d(nn.Module):
	# Types of learning rules
	RULE_BASE = 'base' # delta_w = eta * y' * lfb * (x - reconst)
	RULE_HEBB = 'hebb' # delta_w = eta * y' * y * lfb * (x - reconst)
	RULE_DIFF = 'diff' # delta_w = eta * y' * (lfb - y) * (x - reconst)
	RULE_SMX = 'smx' # delta_w = eta * y' * (lfb - softmax(y)) * (x - reconst)
	
	# Types of LFB kernels
	LFB_GAUSS = 'gauss'
	LFB_DoG = 'DoG'
	LFB_EXP = 'exp'
	LFB_DoE = 'DoE'

	# Type of reconstruction scheme
	REC_QNT = 'qnt' # reconst = w
	REC_QNT_SGN = 'qnt_sgn' # reconst = sign(lfb) * w
	REC_LIN_CMB = 'lin_cmb' # reconst = sum_i y_i w_i

	# Type of update reduction scheme
	RED_AVG = 'avg' # average
	RED_W_AVG = 'w_avg' # weighted average

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
				 lrn_act=F.relu,
				 out_sim=kernel_mult2d,
				 out_act=F.relu,
				 weight_upd_rule=RULE_HEBB,
				 eta=1e-2,
				 lr_schedule=None,
				 tau=1000):
		super(HebbianMap2d, self).__init__()
		
		# Init weights
		out_size_list = [out_size] if not hasattr(out_size, '__len__') else out_size
		self.out_size = torch.tensor(out_size_list[0:min(len(out_size_list), 3)])
		out_channels = self.out_size.prod().item()
		if hasattr(kernel_size, '__len__') and len(kernel_size) == 1: kernel_size = kernel_size[0]
		if not hasattr(kernel_size, '__len__'): kernel_size = [kernel_size, kernel_size]
		stdv = 1 / (in_channels * kernel_size[0] * kernel_size[1]) ** 0.5
		self.register_buffer('weight', torch.empty(out_channels, in_channels, kernel_size[0], kernel_size[1]))
		nn.init.uniform_(self.weight, -stdv, stdv) # Same initialization used by default pytorch conv modules (the one from the paper "Efficient Backprop, LeCun")
		
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
		self.teacher_signal = None # Teacher signal for supervised training
		self.weight_upd_rule = weight_upd_rule
		
		# Initial learning rate and lR scheduling policy. LR wrapped into a registered buffer so that we can save/load it
		self.register_buffer('eta', torch.tensor(eta))
		self.lr_schedule = lr_schedule # LR scheduling policy
		
		# Set parameters related to the lateral feedback feature
		if self.lfb_on:
			# Prepare the variables to generate the kernel that will be used to apply lateral feedback
			map_radius = (self.out_size - 1) // 2
			sigma_lfb = map_radius.max().item()
			x = torch.abs(torch.arange(0, self.out_size[0].item()) - map_radius[0])
			for i in range(1, self.out_size.size(0)):
				x_new = torch.abs(torch.arange(0, self.out_size[i].item()) - map_radius[i])
				for j in range(i): x_new = x_new.unsqueeze(j)
				x = torch.max(x.unsqueeze(-1), x_new) # max gives L_infinity distance, sum would give L_1 distance, root_p(sum x^p) for L_p
			# Store the kernel that will be used to apply lateral feedback in a registered buffer
			if lfb_value == self.LFB_EXP or lfb_value == self.LFB_DoE: self.register_buffer('lfb_kernel', torch.exp(-x.float() / sigma_lfb))
			else: self.register_buffer('lfb_kernel', torch.exp(-x.pow(2).float() / (2 * (sigma_lfb ** 2))))
			# Padding that will pad the inputs before applying the lfb kernel
			pad_pre = map_radius.unsqueeze(1)
			pad_post = (self.out_size - 1 - map_radius).unsqueeze(1)
			self.pad = tuple(torch.cat((pad_pre, pad_post), dim=1).flip(0).view(-1))
			# LFB kernel shrinking parameter
			self.alpha = torch.exp( torch.log(torch.tensor(sigma_lfb).float()) / tau ).item()
			if lfb_value == self.LFB_GAUSS or lfb_value == self.LFB_DoG: self.alpha = self.alpha ** 2
		else: self.register_buffer('lfb_kernel', None)
		
		# Init variables for statistics collection
		if self.random_abstention: self.register_buffer('victories_count', torch.zeros(out_channels))
		else: self.register_buffer('victories_count', None)
	
	def set_teacher_signal(self, y):
		self.teacher_signal = y
	
	def forward(self, x):
		y = self.out_act(self.out_sim(x, self.weight))
		if self.training: self.update(x)
		return y
	
	def update(self, x):
		# Prepare the inputs
		s = self.lrn_sim(x, self.weight)
		t = self.teacher_signal
		if t is not None: t = t.unsqueeze(2).unsqueeze(3) * torch.ones_like(s, device=s.device)
		s = s.permute(0, 2, 3, 1).contiguous().view(-1, self.weight.size(0))
		if t is not None: t = t.permute(0, 2, 3, 1).contiguous().view(-1, self.weight.size(0))
		x_unf = unfold_map2d(x, self.weight.size(2), self.weight.size(3))
		x_unf = x_unf.permute(0, 2, 3, 1, 4).contiguous().view(s.size(0), 1, -1)
		torch.set_grad_enabled(True)
		s.requires_grad = True
		y = self.lrn_act(s)
		z = y.sum((0, 1))
		z.backward()
		y_prime = s.grad
		s.requires_grad = False
		torch.set_grad_enabled(False)
		
		# Random abstention
		if self.random_abstention:
			abst_prob = self.victories_count / (self.victories_count.max() + y.size(0) / y.size(1)).clamp(1)
			scores = y * (torch.rand_like(abst_prob, device=y.device) >= abst_prob).float().unsqueeze(0)
		else: scores = y
		
		# Competition. The returned winner_mask is a bitmap telling where a neuron won and where one lost.
		if self.competitive:
			if t is not None: scores *= t
			winner_mask = (scores == scores.max(1, keepdim=True)[0]).float()
			if self.random_abstention: # Update statistics if using random abstension
				winner_mask_sum = winner_mask.sum(0)  # Number of inputs over which a neuron won
				self.victories_count += winner_mask_sum
				self.victories_count -= self.victories_count.min().item()
		else: winner_mask = torch.ones_like(y, device=y.device)
		
		# Lateral feedback
		if self.lfb_on:
			lfb_kernel = self.lfb_kernel
			if self.lfb_value == self.LFB_DoG or self.lfb_value == self.LFB_DoE: lfb_kernel = 2 * lfb_kernel - lfb_kernel.pow(0.5) # Difference of Gaussians/Exponentials (mexican hat shaped function)
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
		r = lfb_out # RULE_BASE
		if self.weight_upd_rule == self.RULE_HEBB: r *= y
		if self.weight_upd_rule == self.RULE_DIFF: r = (r - y)
		if self.weight_upd_rule == self.RULE_SMX: r = (r - torch.softmax(y, dim=1))
		r *= y_prime
		r_abs = r.abs()
		r_sign = r.sign()

		# Compute delta
		w = self.weight.view(1, self.weight.size(0), -1)
		try:
			reconstr = 0.
			if self.reconstruction == self.REC_QNT: reconstr = w
			if self.reconstruction == self.REC_QNT_SGN: reconstr = r_sign.unsqueeze(2) * w
			if self.reconstruction == self.REC_LIN_CMB: reconstr = torch.cumsum(r.unsqueeze(2) * w, dim=1)
			delta_w = r.unsqueeze(2) * (x_unf - reconstr)
	
			# Since we use batches of inputs, we need to aggregate the different update steps of each kernel in a unique
			# update. We do this by taking the weighted average of the steps, the weights being the r coefficients that
			# determine the length of each step
			if self.reduction == self.RED_W_AVG:
				r_sum = r_abs.sum(0)
				r_sum += (r_sum == 0).float()  # Prevent divisions by zero
				delta_w_avg = (delta_w * r_abs.unsqueeze(2)).sum(0) / r_sum.unsqueeze(1)
			else: delta_w_avg = delta_w.mean(dim=0) # RED_AVG
		except (MemoryError, RuntimeError):
			# Serialized version for computation of delta_w using less memory
			delta_w_avg = torch.zeros_like(self.weight.view(self.weight.size(0), -1))
			reconstr = 0.
			for i in range(self.weight.size(0)):
				w_i = w[:, i, :]
				r_i = r.unsqueeze(2)[:, i, :]
				r_abs_i = r_abs.unsqueeze(2)[:, i, :]
				r_sign_i = r_sign.unsqueeze(2)[:, i, :]
				if self.reconstruction == self.REC_QNT: reconstr = w_i
				if self.reconstruction == self.REC_QNT_SGN: reconstr = r_sign_i * w_i
				if self.reconstruction == self.REC_LIN_CMB: reconstr += r_i * w_i
				delta_w_i = r_i * (x_unf.squeeze(1) - reconstr)
				if self.reduction == self.RED_W_AVG:
					r_sum = r_abs_i.sum(0)
					r_sum += (r_sum == 0).float()  # Prevent divisions by zero
					delta_w_avg[i, :] = (delta_w_i * r_abs_i).sum(0) / r_sum
				else: delta_w_avg[i, :] = delta_w_i.mean(dim=0) # RED_AVG

		# Apply delta
		self.weight += self.eta * delta_w_avg.view_as(self.weight)
		
		# LFB kernel shrinking and LR schedule
		if self.lfb_on: self.lfb_kernel = self.lfb_kernel.pow(self.alpha)
		if self.lr_schedule is not None: self.eta = self.lr_schedule(self.eta)
	

# Generate a batch of random inputs for testing
def gen_batch(centers, batch_size, win_height, win_width):
	# Generate an input "image" by first generating patches as random perturbations on the cluster centers and then
	# concatenating them in the horizontal and vertical dimensions. Repeat to generate a batch.
	batch = torch.empty(0)
	for j in range(batch_size):  # Loop to generate batch
		image = torch.empty(0)
		for k in range(win_height):  # Loop to concat image rows vertically
			row = torch.empty(0)
			for l in range(win_width):  # Loop to concat patches horizontally
				# Generate an input patch by perturbing a cluster center
				index = int(torch.floor(torch.rand(1) * centers.size(0)).item())
				patch = centers[index] + 0.1 * torch.randn_like(centers[index])
				# Concatenate patch horizonally to the image row
				row = torch.cat((row, patch), 2)
			# Concatenate row to the image vertically
			image = torch.cat((image, row), 1)
		# Concatenate the image to the batch
		batch = torch.cat((batch, image.unsqueeze(0)), 0)
	return batch

# Test for the batch generation function
def test_genbatch():
	# Generate centers around which clusters are built
	centers = torch.randn(6, 3, 4, 5)
	# Generate a batch of inputs around the centers
	batch = gen_batch(centers, 10, 2, 2)
	# Check that the batch size is correct (just to be sure)
	print(batch.size()) # Should print 10x3x8x10

# Test the implementation of the HebbianMap2d
def test_hebbianmap():
	# Function for printing summary information
	def print_results(model, centers):
		print('\n' + '#'*79 + '\n')
		responses = model(centers).squeeze()
		top_act, closest_neurons = responses.max(1)
		for i in range(responses.size(0)): print("Closest neuron to center " + str(i) + ": " + str(closest_neurons[i].item()) + ", output: " + str(top_act[i].item()))
		print()
		top_act, closest_centers = responses.max(0)
		for i in range(responses.size(1)): print("Closest center to neuron " + str(i) + ": " + str(closest_centers[i].item()) + ", output: " + str(top_act[i].item()))
		print('\n' + '#' * 79 + '\n')
	
	torch.random.manual_seed(3)
	kernel_shape = (6, 3, 4, 5)
	num_centers = 6
	num_iter = 2000
	batch_size = 10
	win_height = 2
	win_width = 2
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	model = HebbianMap2d(
		in_channels=kernel_shape[1],
		out_size=kernel_shape[0],
		kernel_size=[kernel_shape[2], kernel_shape[3]],
		competitive=True,
		random_abstention=False,
		lfb_value=0,
		lrn_sim=raised_cos2d_pow(2),
		out_sim=cos_sim2d,
		weight_upd_rule=HebbianMap2d.RULE_BASE,
		eta=0.1,
		lr_schedule=sched_exp(1000, 0.01),
		tau=1000
	)
	model.eval()
	model.to(device)
	
	# Generate centers around which clusters are built
	centers = torch.randn(num_centers, *kernel_shape[1:4])
	# Check the distance between the centers and the randomly initialized weight vectors
	print_results(model, centers)
	
	# Train the model: generate a batch of inputs and feed it to the model, repeat for the desired number of iterations
	model.train()
	for i in range(num_iter):
		batch = gen_batch(centers, batch_size, win_height, win_width)
		batch = batch.to(device)
		model(batch)
	model.eval()
	
	# Verify that the weight vectors of the model have converged to the cluster centers
	print_results(model, centers)


if __name__=='__main__':
	test_kernelsum()
	test_genbatch()
	test_hebbianmap()
