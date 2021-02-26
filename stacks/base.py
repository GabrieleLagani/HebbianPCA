from neurolab import params as P


# TODO:
#  Add a utility script to compute mean and ci from csv files and convergence epochs.
#  Add tinyimagenet 32 and 96 and simple transfer learning experiments.
#  Add experiment sequence for hyperparam config.
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
	P.KEY_STACK_CONFIG: 'configs.base.config_1l',
	P.KEY_STACK_MODE: P.MODE_TRN,
	P.KEY_STACK_DEVICE: 'cuda:0',
	P.KEY_STACK_SEEDS: [0],
	P.KEY_STACK_TOKENS: None,
	P.KEY_STACK_HPSEARCH: True,
	P.KEY_STACK_HPSEEDS: [100],
	P.KEY_STACK_DATASEEDS: [200],
	P.KEY_STACK_CHECKPOINT: None,
	P.KEY_STACK_RESTART: True,
	P.KEY_STACK_CLEARHIST: True,
	P.KEY_STACK_BRANCH: None,
}]

