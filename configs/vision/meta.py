lrn_rules = ['hpca', 'hpca_dts', '1wta', '1wta_dts']
lrn_rule_keys = {'hpca': 'hpca', 'hpca_dts': 'hpca', '1wta': 'hwta', '1wta_dts': 'hwta'}
lrn_rule_k = {'hpca': 0, 'hpca_dts': 0, '1wta': 1, '1wta_dts': 1}
lrn_rule_dts = {'hpca': False, 'hpca_dts': True, '1wta': False, '1wta_dts': True}
datasets = ['mnist', 'cifar10', 'cifar100', 'tinyimagenet', 'imagenet']
data_managers = {'mnist': 'MNISTDataManager', 'cifar10': 'CIFAR10DataManager', 'cifar100': 'CIFAR100DataManager', 'tinyimagenet': 'TinyImageNetDataManager', 'imagenet': 'ImageNetDataManager'}
tot_trn_samples = {'mnist': 50000, 'cifar10': 40000, 'cifar100': 40000, 'tinyimagenet': 90000, 'imagenet': 1200000}
input_shapes = {'mnist': (3, 32, 32), 'cifar10': (3, 32, 32), 'cifar100': (3, 32, 32), 'tinyimagenet': (3, 32, 32), 'imagenet': (3, 210, 210)}
num_layers = {'mnist': 6, 'cifar10': 6, 'cifar100': 6, 'tinyimagenet': 6, 'imagenet': 10}
net_outputs = {'mnist': 'fc6', 'cifar10': 'fc6', 'cifar100': 'fc6', 'tinyimagenet': 'fc6', 'imagenet': 'fc10'}
l2_penalties = {'mnist': 5e-2, 'cifar10': 5e-2, 'cifar100': 1e-2, 'tinyimagenet': 5e-3, 'imagenet': 1e-3}
samples_per_class = {'mnist': 5000, 'cifar10': 4000, 'cifar100': 400, 'tinyimagenet': 450, 'imagenet': 1200}
retr_k = {'mnist': [100, 5000], 'cifar10': [100, 4000], 'cifar100': [100, 400], 'tinyimagenet': [100, 450], 'imagenet': [100, 1200]}
smpleff_regimes = {
	'mnist': [500, 1000, 1500, 2000, 2500, 5000, 12500],
	'cifar10': [400, 800, 1200, 1600, 2000, 4000, 10000],
	'cifar100': [400, 800, 1200, 1600, 2000, 4000, 10000],
	'tinyimagenet': [900, 1800, 2700, 3600, 4500, 9000, 22500],
	'imagenet': [12000, 24000, 36000, 48000, 60000, 120000, 300000],
}