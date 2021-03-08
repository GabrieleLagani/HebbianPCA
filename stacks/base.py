from neurolab import params as P


# TODO: Write documentation, add checks on user provided parameters, print exception in try-except-pass blocks, publish pip package.
# TODO: (as and when needed)
#  Add adversarial examples.
#  Add deep layer visualization.
#  Add other dataset configurations.
#  Add transfer learning configurations.
#  Add other tasks.
# TODO: Enable dispatch of different seeds/different hyperparam configs to different gpus.
# TODO: Enable dispatch of different layers to different gpus.
# TODO: Add additional data augmentation transformations (random blur, random noise, random occlusion).


stack_base = [{
	P.KEY_STACK_CONFIG: 'configs.base.multiexp_fc_on_gdes_layer',
	P.KEY_STACK_MODE: P.MODE_TRN,
	P.KEY_STACK_DEVICE: 'cuda:0',
	P.KEY_STACK_SEEDS: [0],
	P.KEY_STACK_TOKENS: None,
	P.KEY_STACK_HPSEARCH: False,
	P.KEY_STACK_HPSEEDS: [100],
	P.KEY_STACK_DATASEEDS: [200],
	P.KEY_STACK_CHECKPOINT: None,
	P.KEY_STACK_RESTART: False,
	P.KEY_STACK_CLEARHIST: True,
	P.KEY_STACK_BRANCH: None,
}]

