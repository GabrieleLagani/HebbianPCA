import neurolab.params as P
from configs.vision.meta import *


lrn_rules = ['hpca', '1wta']

SEEDS1 = range(0, 5)
SEEDS2 = range(300, 305)
SEEDS3 = range(400, 405)
DATASEEDS = range(200, 205)

gdes = {ds + ('_da' if da else ''): [] for ds in datasets for da in [False, True]}
hebb = {lrn_rule + '_' + ds + ('_da' if da else ''): [] for lrn_rule in lrn_rules for ds in datasets for da in [False, True]}
vae = {ds + ('_da' if da else ''): [] for ds in datasets for da in [False, True]}
hybrid = {lrn_rule + '_' + ds + ('_da' if da else ''): [] for lrn_rule in lrn_rules for ds in datasets for da in [False, True]}
smpleff_gdes = {ds + ('_da' if da else ''): [] for ds in datasets for da in [False, True]}
smpleff_hebb = {lrn_rule + '_' + ds + ('_da' if da else ''): [] for lrn_rule in lrn_rules for ds in datasets for da in [False, True]}
smpleff_vae = {ds + ('_da' if da else ''): [] for ds in datasets for da in [False, True]}
sk_gdes = {ds + ('_da' if da else ''): [] for ds in datasets for da in [False, True]}
sk_hebb = {lrn_rule + '_' + ds + ('_da' if da else ''): [] for lrn_rule in lrn_rules for ds in datasets for da in [False, True]}
sk_vae = {ds + ('_da' if da else ''): [] for ds in datasets for da in [False, True]}
smpleff_sk_gdes = {ds + ('_da' if da else ''): [] for ds in datasets for da in [False, True]}
smpleff_sk_hebb = {lrn_rule + '_' + ds + ('_da' if da else ''): [] for lrn_rule in lrn_rules for ds in datasets for da in [False, True]}
smpleff_sk_vae = {ds + ('_da' if da else ''): [] for ds in datasets for da in [False, True]}
all = {ds + ('_da' if da else ''): [] for ds in datasets for da in [False, True]}


for ds in datasets:
	for da in [False, True]:
		gdes[ds + ('_da' if da else '')] += [{P.KEY_STACK_CONFIG: 'configs.vision.gdes.config_base_gdes[' + ds + ('_da' if da else '') + ']', P.KEY_STACK_SEEDS: SEEDS1, P.KEY_STACK_DATASEEDS: DATASEEDS}]
		gdes[ds + ('_da' if da else '')] += [{P.KEY_STACK_CONFIG: 'configs.vision.gdes.gdes_fc_on_gdes_layer[' + str(l) + '_' + ds + ('_da' if da else '') + ']', P.KEY_STACK_SEEDS: SEEDS2, P.KEY_STACK_TOKENS: [str(i) for i in SEEDS1], P.KEY_STACK_DATASEEDS: DATASEEDS} for l in range(1, num_layers[ds])]
		#gdes[ds + ('_da' if da else '')] += [{P.KEY_STACK_CONFIG: 'configs.vision.gdes.gdes_fc_on_gdes_layer_ft[' + str(l) + '_' + ds + ('_da' if da else '') + ']', P.KEY_STACK_SEEDS: SEEDS2, P.KEY_STACK_TOKENS: [str(i) for i in SEEDS1], P.KEY_STACK_DATASEEDS: DATASEEDS} for l in range(1, num_layers[ds])]
		gdes[ds + ('_da' if da else '')] += [{P.KEY_STACK_CONFIG: 'configs.vision.gdes.hebb_fc_on_gdes_layer[' + str(l) + '_' + ds + ('_da' if da else '') + ']', P.KEY_STACK_SEEDS: SEEDS2, P.KEY_STACK_TOKENS: [str(i) for i in SEEDS1], P.KEY_STACK_DATASEEDS: DATASEEDS} for l in range(1, num_layers[ds])]
		#gdes[ds + ('_da' if da else '')] += [{P.KEY_STACK_CONFIG: 'configs.vision.gdes.hebb_fc_on_gdes_layer_ft[' + str(l) + '_' + ds + ('_da' if da else '') + ']', P.KEY_STACK_SEEDS: SEEDS3, P.KEY_STACK_TOKENS: [str(i) for i in SEEDS2], P.KEY_STACK_DATASEEDS: DATASEEDS} for l in range(1, num_layers[ds])]
		#gdes[ds + ('_da' if da else '')] += [{P.KEY_STACK_CONFIG: 'configs.vision.gdes.gdes_fc2_on_gdes_layer[' + str(l) + '_' + ds + ('_da' if da else '') + ']', P.KEY_STACK_SEEDS: SEEDS2, P.KEY_STACK_TOKENS: [str(i) for i in SEEDS1], P.KEY_STACK_DATASEEDS: DATASEEDS} for l in range(1, num_layers[ds])]
		for lrn_rule in lrn_rules:
			gdes[ds + ('_da' if da else '')] += [{P.KEY_STACK_CONFIG: 'configs.vision.gdes.hebb_fc2_on_gdes_layer[' + str(l) + '_' + lrn_rule + '_' + ds + ('_da' if da else '') + ']', P.KEY_STACK_SEEDS: SEEDS2, P.KEY_STACK_TOKENS: [str(i) for i in SEEDS1], P.KEY_STACK_DATASEEDS: DATASEEDS} for l in range(num_layers[ds] - 2, num_layers[ds] - 1)] #for l in range(1, num_layers[ds])]
		sk_gdes[ds + ('_da' if da else '')] += [{P.KEY_STACK_CONFIG: 'configs.vision.gdes.prec_on_gdes_layer[' + str(l) + '_' + ds + ('_da' if da else '') + ']', P.KEY_STACK_SEEDS: SEEDS2, P.KEY_STACK_TOKENS: [str(i) for i in SEEDS1], P.KEY_STACK_DATASEEDS: DATASEEDS} for l in range(1, num_layers[ds])]
		#sk_gdes[ds + ('_da' if da else '')] += [{P.KEY_STACK_CONFIG: 'configs.vision.gdes.prec_on_gdes_layer_ft[' + str(l) + '_' + ds + ('_da' if da else '') + ']', P.KEY_STACK_SEEDS: SEEDS3, P.KEY_STACK_TOKENS: [str(i) for i in SEEDS2], P.KEY_STACK_DATASEEDS: DATASEEDS} for l in range(1, num_layers[ds])]
		sk_gdes[ds + ('_da' if da else '')] += [{P.KEY_STACK_CONFIG: 'configs.vision.gdes.knn_on_gdes_layer[' + str(l) + '_' + ds + ('_da' if da else '') + ']', P.KEY_STACK_SEEDS: SEEDS2, P.KEY_STACK_TOKENS: [str(i) for i in SEEDS1], P.KEY_STACK_DATASEEDS: DATASEEDS} for l in range(1, num_layers[ds])]
		#sk_gdes[ds + ('_da' if da else '')] += [{P.KEY_STACK_CONFIG: 'configs.vision.gdes.knn_on_gdes_layer_ft[' + str(l) + '_' + ds + ('_da' if da else '') + ']', P.KEY_STACK_SEEDS: SEEDS3, P.KEY_STACK_TOKENS: [str(i) for i in SEEDS2], P.KEY_STACK_DATASEEDS: DATASEEDS} for l in range(1, num_layers[ds])]
		sk_gdes[ds + ('_da' if da else '')] += [{P.KEY_STACK_CONFIG: 'configs.vision.gdes.svm_on_gdes_layer[' + str(l) + '_' + ds + ('_da' if da else '') + ']', P.KEY_STACK_SEEDS: SEEDS2, P.KEY_STACK_TOKENS: [str(i) for i in SEEDS1], P.KEY_STACK_DATASEEDS: DATASEEDS} for l in range(1, num_layers[ds])]
		#sk_gdes[ds + ('_da' if da else '')] += [{P.KEY_STACK_CONFIG: 'configs.vision.gdes.svm_on_gdes_layer_ft[' + str(l) + '_' + ds + ('_da' if da else '') + ']', P.KEY_STACK_SEEDS: SEEDS3, P.KEY_STACK_TOKENS: [str(i) for i in SEEDS2], P.KEY_STACK_DATASEEDS: DATASEEDS} for l in range(1, num_layers[ds])]

		for lrn_rule in lrn_rules:
			hebb[lrn_rule + '_' + ds + ('_da' if da else '')] += [{P.KEY_STACK_CONFIG: 'configs.vision.hebb.config_base_hebb[' + lrn_rule + '_' + ds + ('_da' if da else '') + ']', P.KEY_STACK_SEEDS: SEEDS1, P.KEY_STACK_DATASEEDS: DATASEEDS}]
			hebb[lrn_rule + '_' + ds + ('_da' if da else '')] += [{P.KEY_STACK_CONFIG: 'configs.vision.hebb.gdes_fc_on_hebb_layer[' + str(l) + '_' + lrn_rule + '_' + ds + ('_da' if da else '') + ']', P.KEY_STACK_SEEDS: SEEDS2, P.KEY_STACK_TOKENS: [str(i) for i in SEEDS1], P.KEY_STACK_DATASEEDS: DATASEEDS} for l in range(1, num_layers[ds])]
			hebb[lrn_rule + '_' + ds + ('_da' if da else '')] += [{P.KEY_STACK_CONFIG: 'configs.vision.hebb.gdes_fc_on_hebb_layer_ft[' + str(l) + '_' + lrn_rule + '_' + ds + ('_da' if da else '') + ']', P.KEY_STACK_SEEDS: SEEDS2, P.KEY_STACK_TOKENS: [str(i) for i in SEEDS1], P.KEY_STACK_DATASEEDS: DATASEEDS} for l in range(1, num_layers[ds])]
			hebb[lrn_rule + '_' + ds + ('_da' if da else '')] += [{P.KEY_STACK_CONFIG: 'configs.vision.hebb.hebb_fc_on_hebb_layer[' + str(l) + '_' + lrn_rule + '_' + ds + ('_da' if da else '') + ']', P.KEY_STACK_SEEDS: SEEDS2, P.KEY_STACK_TOKENS: [str(i) for i in SEEDS1], P.KEY_STACK_DATASEEDS: DATASEEDS} for l in range(1, num_layers[ds])]
			hebb[lrn_rule + '_' + ds + ('_da' if da else '')] += [{P.KEY_STACK_CONFIG: 'configs.vision.hebb.hebb_fc_on_hebb_layer_ft[' + str(l) + '_' + lrn_rule + '_' + ds + ('_da' if da else '') + ']', P.KEY_STACK_SEEDS: SEEDS3, P.KEY_STACK_TOKENS: [str(i) for i in SEEDS2], P.KEY_STACK_DATASEEDS: DATASEEDS} for l in range(1, num_layers[ds])]
			hebb[lrn_rule + '_' + ds + ('_da' if da else '')] += [{P.KEY_STACK_CONFIG: 'configs.vision.hebb.gdes_fc2_on_hebb_layer[' + str(l) + '_' + lrn_rule + '_' + ds + ('_da' if da else '') + ']', P.KEY_STACK_SEEDS: SEEDS2, P.KEY_STACK_TOKENS: [str(i) for i in SEEDS1], P.KEY_STACK_DATASEEDS: DATASEEDS} for l in range(num_layers[ds] - 2, num_layers[ds] - 1)] #for l in range(1, num_layers[ds])]
			#hebb[lrn_rule + '_' + ds + ('_da' if da else '')] += [{P.KEY_STACK_CONFIG: 'configs.vision.hebb.hebb_fc2_on_hebb_layer[' + str(l) + '_' + lrn_rule + '_' + ds + ('_da' if da else '') + ']', P.KEY_STACK_SEEDS: SEEDS2, P.KEY_STACK_TOKENS: [str(i) for i in SEEDS1], P.KEY_STACK_DATASEEDS: DATASEEDS} for l in range(1, num_layers[ds])]
			sk_hebb[lrn_rule + '_' + ds + ('_da' if da else '')] += [{P.KEY_STACK_CONFIG: 'configs.vision.hebb.prec_on_hebb_layer[' + str(l) + '_' + lrn_rule + '_' + ds + ('_da' if da else '') + ']', P.KEY_STACK_SEEDS: SEEDS2, P.KEY_STACK_TOKENS: [str(i) for i in SEEDS1], P.KEY_STACK_DATASEEDS: DATASEEDS} for l in range(1, num_layers[ds])]
			sk_hebb[lrn_rule + '_' + ds + ('_da' if da else '')] += [{P.KEY_STACK_CONFIG: 'configs.vision.hebb.prec_on_hebb_layer_ft[' + str(l) + '_' + lrn_rule + '_' + ds + ('_da' if da else '') + ']', P.KEY_STACK_SEEDS: SEEDS3, P.KEY_STACK_TOKENS: [str(i) for i in SEEDS2], P.KEY_STACK_DATASEEDS: DATASEEDS} for l in range(1, num_layers[ds])]
			sk_hebb[lrn_rule + '_' + ds + ('_da' if da else '')] += [{P.KEY_STACK_CONFIG: 'configs.vision.hebb.knn_on_hebb_layer[' + str(l) + '_' + lrn_rule + '_' + ds + ('_da' if da else '') + ']', P.KEY_STACK_SEEDS: SEEDS2, P.KEY_STACK_TOKENS: [str(i) for i in SEEDS1], P.KEY_STACK_DATASEEDS: DATASEEDS} for l in range(1, num_layers[ds])]
			sk_hebb[lrn_rule + '_' + ds + ('_da' if da else '')] += [{P.KEY_STACK_CONFIG: 'configs.vision.hebb.knn_on_hebb_layer_ft[' + str(l) + '_' + lrn_rule + '_' + ds + ('_da' if da else '') + ']', P.KEY_STACK_SEEDS: SEEDS3, P.KEY_STACK_TOKENS: [str(i) for i in SEEDS2], P.KEY_STACK_DATASEEDS: DATASEEDS} for l in range(1, num_layers[ds])]
			sk_hebb[lrn_rule + '_' + ds + ('_da' if da else '')] += [{P.KEY_STACK_CONFIG: 'configs.vision.hebb.svm_on_hebb_layer[' + str(l) + '_' + lrn_rule + '_' + ds + ('_da' if da else '') + ']', P.KEY_STACK_SEEDS: SEEDS2, P.KEY_STACK_TOKENS: [str(i) for i in SEEDS1], P.KEY_STACK_DATASEEDS: DATASEEDS} for l in range(1, num_layers[ds])]
			sk_hebb[lrn_rule + '_' + ds + ('_da' if da else '')] += [{P.KEY_STACK_CONFIG: 'configs.vision.hebb.svm_on_hebb_layer_ft[' + str(l) + '_' + lrn_rule + '_' + ds + ('_da' if da else '') + ']', P.KEY_STACK_SEEDS: SEEDS3, P.KEY_STACK_TOKENS: [str(i) for i in SEEDS2], P.KEY_STACK_DATASEEDS: DATASEEDS} for l in range(1, num_layers[ds])]
			
		vae[ds + ('_da' if da else '')] += [{P.KEY_STACK_CONFIG: 'configs.vision.vae.config_base_vae[' + ds + ('_da' if da else '') + ']', P.KEY_STACK_SEEDS: SEEDS1, P.KEY_STACK_DATASEEDS: DATASEEDS}]
		vae[ds + ('_da' if da else '')] += [{P.KEY_STACK_CONFIG: 'configs.vision.vae.gdes_fc_on_vae_layer[' + str(l) + '_' + ds + ('_da' if da else '') + ']', P.KEY_STACK_SEEDS: SEEDS2, P.KEY_STACK_TOKENS: [str(i) for i in SEEDS1], P.KEY_STACK_DATASEEDS: DATASEEDS} for l in range(1, num_layers[ds])]
		vae[ds + ('_da' if da else '')] += [{P.KEY_STACK_CONFIG: 'configs.vision.vae.gdes_fc_on_vae_layer_ft[' + str(l) + '_' + ds + ('_da' if da else '') + ']', P.KEY_STACK_SEEDS: SEEDS2, P.KEY_STACK_TOKENS: [str(i) for i in SEEDS1], P.KEY_STACK_DATASEEDS: DATASEEDS} for l in range(1, num_layers[ds])]
		vae[ds + ('_da' if da else '')] += [{P.KEY_STACK_CONFIG: 'configs.vision.vae.hebb_fc_on_vae_layer[' + str(l) + '_' + ds + ('_da' if da else '') + ']', P.KEY_STACK_SEEDS: SEEDS2, P.KEY_STACK_TOKENS: [str(i) for i in SEEDS1], P.KEY_STACK_DATASEEDS: DATASEEDS} for l in range(1, num_layers[ds])]
		vae[ds + ('_da' if da else '')] += [{P.KEY_STACK_CONFIG: 'configs.vision.vae.hebb_fc_on_vae_layer_ft[' + str(l) + '_' + ds + ('_da' if da else '') + ']', P.KEY_STACK_SEEDS: SEEDS3, P.KEY_STACK_TOKENS: [str(i) for i in SEEDS2], P.KEY_STACK_DATASEEDS: DATASEEDS} for l in range(1, num_layers[ds])]
		#vae[ds + ('_da' if da else '')] += [{P.KEY_STACK_CONFIG: 'configs.vision.vae.gdes_fc2_on_vae_layer[' + str(l) + '_' + ds + ('_da' if da else '') + ']', P.KEY_STACK_SEEDS: SEEDS2, P.KEY_STACK_TOKENS: [str(i) for i in SEEDS1], P.KEY_STACK_DATASEEDS: DATASEEDS} for l in range(1, num_layers[ds])]
		#for lrn_rule in lrn_rules:
		#   vae[ds + ('_da' if da else '')] += [{P.KEY_STACK_CONFIG: 'configs.vision.vae.hebb_fc2_on_vae_layer[' + str(l) + '_' + lrn_rule + '_' + ds + ('_da' if da else '') + ']', P.KEY_STACK_SEEDS: SEEDS2, P.KEY_STACK_TOKENS: [str(i) for i in SEEDS1], P.KEY_STACK_DATASEEDS: DATASEEDS} for l in range(1, num_layers[ds])]
		sk_vae[ds + ('_da' if da else '')] += [{P.KEY_STACK_CONFIG: 'configs.vision.vae.prec_on_vae_layer[' + str(l) + '_' + ds + ('_da' if da else '') + ']', P.KEY_STACK_SEEDS: SEEDS2, P.KEY_STACK_TOKENS: [str(i) for i in SEEDS1], P.KEY_STACK_DATASEEDS: DATASEEDS} for l in range(1, num_layers[ds])]
		sk_vae[ds + ('_da' if da else '')] += [{P.KEY_STACK_CONFIG: 'configs.vision.vae.prec_on_vae_layer_ft[' + str(l) + '_' + ds + ('_da' if da else '') + ']', P.KEY_STACK_SEEDS: SEEDS3, P.KEY_STACK_TOKENS: [str(i) for i in SEEDS2], P.KEY_STACK_DATASEEDS: DATASEEDS} for l in range(1, num_layers[ds])]
		sk_vae[ds + ('_da' if da else '')] += [{P.KEY_STACK_CONFIG: 'configs.vision.vae.knn_on_vae_layer[' + str(l) + '_' + ds + ('_da' if da else '') + ']', P.KEY_STACK_SEEDS: SEEDS2, P.KEY_STACK_TOKENS: [str(i) for i in SEEDS1], P.KEY_STACK_DATASEEDS: DATASEEDS} for l in range(1, num_layers[ds])]
		sk_vae[ds + ('_da' if da else '')] += [{P.KEY_STACK_CONFIG: 'configs.vision.vae.knn_on_vae_layer_ft[' + str(l) + '_' + ds + ('_da' if da else '') + ']', P.KEY_STACK_SEEDS: SEEDS3, P.KEY_STACK_TOKENS: [str(i) for i in SEEDS2], P.KEY_STACK_DATASEEDS: DATASEEDS} for l in range(1, num_layers[ds])]
		sk_vae[ds + ('_da' if da else '')] += [{P.KEY_STACK_CONFIG: 'configs.vision.vae.svm_on_vae_layer[' + str(l) + '_' + ds + ('_da' if da else '') + ']', P.KEY_STACK_SEEDS: SEEDS2, P.KEY_STACK_TOKENS: [str(i) for i in SEEDS1], P.KEY_STACK_DATASEEDS: DATASEEDS} for l in range(1, num_layers[ds])]
		sk_vae[ds + ('_da' if da else '')] += [{P.KEY_STACK_CONFIG: 'configs.vision.vae.svm_on_vae_layer_ft[' + str(l) + '_' + ds + ('_da' if da else '') + ']', P.KEY_STACK_SEEDS: SEEDS3, P.KEY_STACK_TOKENS: [str(i) for i in SEEDS2], P.KEY_STACK_DATASEEDS: DATASEEDS} for l in range(1, num_layers[ds])]
		
		for lrn_rule in lrn_rules:
			hybrid[lrn_rule + '_' + ds + ('_da' if da else '')] += [{P.KEY_STACK_CONFIG: 'configs.vision.hybrid.gdes_on_hebb_layer[' + str(l) + '_' + lrn_rule + '_' + ds + ('_da' if da else '') + ']', P.KEY_STACK_SEEDS: SEEDS2, P.KEY_STACK_TOKENS: [str(i) for i in SEEDS1], P.KEY_STACK_DATASEEDS: DATASEEDS} for l in range(1, num_layers[ds] - 2)]
			hybrid[lrn_rule + '_' + ds + ('_da' if da else '')] += [{P.KEY_STACK_CONFIG: 'configs.vision.hybrid.hebb_on_gdes_layer[' + str(l) + '_' + lrn_rule + '_' + ds + ('_da' if da else '') + ']', P.KEY_STACK_SEEDS: SEEDS2, P.KEY_STACK_TOKENS: [str(i) for i in SEEDS1], P.KEY_STACK_DATASEEDS: DATASEEDS} for l in range(1, num_layers[ds] - 2)]
			for l1 in range(1, num_layers[ds] - 1):
				for l2 in range(l1 + 1, num_layers[ds]):
					hybrid[lrn_rule + '_' + ds + ('_da' if da else '')] += [{P.KEY_STACK_CONFIG: 'configs.vision.hybrid.ghg[' + str(l1) + '_' + str(l2) + '_' + lrn_rule + '_' + ds + ('_da' if da else '') + ']', P.KEY_STACK_SEEDS: SEEDS3, P.KEY_STACK_TOKENS: ['{},{}'.format(i, j) for i, j in zip(SEEDS1, SEEDS2)], P.KEY_STACK_DATASEEDS: DATASEEDS}]
			
		smpleff_gdes[ds + ('_da' if da else '')] += [{P.KEY_STACK_CONFIG: 'configs.vision.smpleff.config_base_gdes[' + ds + '_' + str(n) + ('_da' if da else '') + ']', P.KEY_STACK_SEEDS: SEEDS1, P.KEY_STACK_DATASEEDS: DATASEEDS} for n in smpleff_regimes[ds]]
		for l in range(1, num_layers[ds]):
			smpleff_gdes[ds + ('_da' if da else '')] += [{P.KEY_STACK_CONFIG: 'configs.vision.smpleff.gdes_fc_on_gdes_layer[' + str(l) + '_' + ds + '_' + str(n) + ('_da' if da else '') + ']', P.KEY_STACK_SEEDS: SEEDS2, P.KEY_STACK_TOKENS: [str(i) for i in SEEDS1], P.KEY_STACK_DATASEEDS: DATASEEDS} for n in smpleff_regimes[ds]]
			#smpleff_gdes[ds + ('_da' if da else '')] += [{P.KEY_STACK_CONFIG: 'configs.vision.smpleff.gdes_fc_on_gdes_layer_ft[' + str(l) + '_' + ds + '_' + str(n) + ('_da' if da else '') + ']', P.KEY_STACK_SEEDS: SEEDS2, P.KEY_STACK_TOKENS: [str(i) for i in SEEDS1], P.KEY_STACK_DATASEEDS: DATASEEDS} for n in smpleff_regimes[ds]]
			#smpleff_gdes[ds + ('_da' if da else '')] += [{P.KEY_STACK_CONFIG: 'configs.vision.smpleff.hebb_fc_on_gdes_layer[' + str(l) + '_' + ds + '_' + str(n) + ('_da' if da else '') + ']', P.KEY_STACK_SEEDS: SEEDS2, P.KEY_STACK_TOKENS: [str(i) for i in SEEDS1], P.KEY_STACK_DATASEEDS: DATASEEDS} for n in smpleff_regimes[ds]]
			#smpleff_gdes[ds + ('_da' if da else '')] += [{P.KEY_STACK_CONFIG: 'configs.vision.smpleff.hebb_fc_on_gdes_layer_ft[' + str(l) + '_' + ds + '_' + str(n) + ('_da' if da else '') + ']', P.KEY_STACK_SEEDS: SEEDS3, P.KEY_STACK_TOKENS: [str(i) for i in SEEDS2], P.KEY_STACK_DATASEEDS: DATASEEDS} for n in smpleff_regimes[ds]]
			smpleff_sk_gdes[ds + ('_da' if da else '')] += [{P.KEY_STACK_CONFIG: 'configs.vision.smpleff.prec_on_gdes_layer[' + str(l) + '_' + ds + '_' + str(n) + ('_da' if da else '') + ']', P.KEY_STACK_SEEDS: SEEDS2, P.KEY_STACK_TOKENS: [str(i) for i in SEEDS1], P.KEY_STACK_DATASEEDS: DATASEEDS} for n in smpleff_regimes[ds]]
			#smpleff_sk_gdes[ds + ('_da' if da else '')] += [{P.KEY_STACK_CONFIG: 'configs.vision.smpleff.prec_on_gdes_layer_ft[' + str(l) + '_' + ds + '_' + str(n) + ('_da' if da else '') + ']', P.KEY_STACK_SEEDS: SEEDS3, P.KEY_STACK_TOKENS: [str(i) for i in SEEDS2], P.KEY_STACK_DATASEEDS: DATASEEDS} for n in smpleff_regimes[ds]]
			smpleff_sk_gdes[ds + ('_da' if da else '')] += [{P.KEY_STACK_CONFIG: 'configs.vision.smpleff.knn_on_gdes_layer[' + str(l) + '_' + ds + '_' + str(n) + ('_da' if da else '') + ']', P.KEY_STACK_SEEDS: SEEDS2, P.KEY_STACK_TOKENS: [str(i) for i in SEEDS1], P.KEY_STACK_DATASEEDS: DATASEEDS} for n in smpleff_regimes[ds]]
			#smpleff_sk_gdes[ds + ('_da' if da else '')] += [{P.KEY_STACK_CONFIG: 'configs.vision.smpleff.knn_on_gdes_layer_ft[' + str(l) + '_' + ds + '_' + str(n) + ('_da' if da else '') + ']', P.KEY_STACK_SEEDS: SEEDS3, P.KEY_STACK_TOKENS: [str(i) for i in SEEDS2], P.KEY_STACK_DATASEEDS: DATASEEDS} for n in smpleff_regimes[ds]]
			smpleff_sk_gdes[ds + ('_da' if da else '')] += [{P.KEY_STACK_CONFIG: 'configs.vision.smpleff.svm_on_gdes_layer[' + str(l) + '_' + ds + '_' + str(n) + ('_da' if da else '') + ']', P.KEY_STACK_SEEDS: SEEDS2, P.KEY_STACK_TOKENS: [str(i) for i in SEEDS1], P.KEY_STACK_DATASEEDS: DATASEEDS} for n in smpleff_regimes[ds]]
			#smpleff_sk_gdes[ds + ('_da' if da else '')] += [{P.KEY_STACK_CONFIG: 'configs.vision.smpleff.svm_on_gdes_layer_ft[' + str(l) + '_' + ds + '_' + str(n) + ('_da' if da else '') + ']', P.KEY_STACK_SEEDS: SEEDS3, P.KEY_STACK_TOKENS: [str(i) for i in SEEDS2], P.KEY_STACK_DATASEEDS: DATASEEDS} for n in smpleff_regimes[ds]]
		for lrn_rule in lrn_rules:
			for l in range(1, num_layers[ds]):
				smpleff_hebb[lrn_rule + '_' + ds + ('_da' if da else '')] += [{P.KEY_STACK_CONFIG: 'configs.vision.smpleff.gdes_fc_on_hebb_layer[' + str(l) + '_' + lrn_rule + '_' + ds + '_' + str(n) + ('_da' if da else '') + ']', P.KEY_STACK_SEEDS: SEEDS2, P.KEY_STACK_TOKENS: [str(i) for i in SEEDS1], P.KEY_STACK_DATASEEDS: DATASEEDS} for n in smpleff_regimes[ds]]
				smpleff_hebb[lrn_rule + '_' + ds + ('_da' if da else '')] += [{P.KEY_STACK_CONFIG: 'configs.vision.smpleff.gdes_fc_on_hebb_layer_ft[' + str(l) + '_' + lrn_rule + '_' + ds + '_' + str(n) + ('_da' if da else '') + ']', P.KEY_STACK_SEEDS: SEEDS2, P.KEY_STACK_TOKENS: [str(i) for i in SEEDS1], P.KEY_STACK_DATASEEDS: DATASEEDS} for n in smpleff_regimes[ds]]
				#smpleff_hebb[lrn_rule + '_' + ds + ('_da' if da else '')] += [{P.KEY_STACK_CONFIG: 'configs.vision.smpleff.hebb_fc_on_hebb_layer[' + str(l) + '_' + lrn_rule + '_' + ds + '_' + str(n) + ('_da' if da else '') + ']', P.KEY_STACK_SEEDS: SEEDS2, P.KEY_STACK_TOKENS: [str(i) for i in SEEDS1], P.KEY_STACK_DATASEEDS: DATASEEDS} for n in smpleff_regimes[ds]]
				#smpleff_hebb[lrn_rule + '_' + ds + ('_da' if da else '')] += [{P.KEY_STACK_CONFIG: 'configs.vision.smpleff.hebb_fc_on_hebb_layer_ft[' + str(l) + '_' + lrn_rule + '_' + ds + '_' + str(n) + ('_da' if da else '') + ']', P.KEY_STACK_SEEDS: SEEDS3, P.KEY_STACK_TOKENS: [str(i) for i in SEEDS2], P.KEY_STACK_DATASEEDS: DATASEEDS} for n in smpleff_regimes[ds]]
				smpleff_sk_hebb[lrn_rule + '_' + ds + ('_da' if da else '')] += [{P.KEY_STACK_CONFIG: 'configs.vision.smpleff.prec_on_hebb_layer_ft[' + str(l) + '_' + lrn_rule + '_' + ds + '_' + str(n) + ('_da' if da else '') + ']', P.KEY_STACK_SEEDS: SEEDS3, P.KEY_STACK_TOKENS: [str(i) for i in SEEDS2], P.KEY_STACK_DATASEEDS: DATASEEDS} for n in smpleff_regimes[ds]]
				smpleff_sk_hebb[lrn_rule + '_' + ds + ('_da' if da else '')] += [{P.KEY_STACK_CONFIG: 'configs.vision.smpleff.knn_on_hebb_layer[' + str(l) + '_' + lrn_rule + '_' + ds + '_' + str(n) + ('_da' if da else '') + ']', P.KEY_STACK_SEEDS: SEEDS2, P.KEY_STACK_TOKENS: [str(i) for i in SEEDS1], P.KEY_STACK_DATASEEDS: DATASEEDS} for n in smpleff_regimes[ds]]
				smpleff_sk_hebb[lrn_rule + '_' + ds + ('_da' if da else '')] += [{P.KEY_STACK_CONFIG: 'configs.vision.smpleff.knn_on_hebb_layer_ft[' + str(l) + '_' + lrn_rule + '_' + ds + '_' + str(n) + ('_da' if da else '') + ']', P.KEY_STACK_SEEDS: SEEDS3, P.KEY_STACK_TOKENS: [str(i) for i in SEEDS2], P.KEY_STACK_DATASEEDS: DATASEEDS} for n in smpleff_regimes[ds]]
				smpleff_sk_hebb[lrn_rule + '_' + ds + ('_da' if da else '')] += [{P.KEY_STACK_CONFIG: 'configs.vision.smpleff.svm_on_hebb_layer[' + str(l) + '_' + lrn_rule + '_' + ds + '_' + str(n) + ('_da' if da else '') + ']', P.KEY_STACK_SEEDS: SEEDS2, P.KEY_STACK_TOKENS: [str(i) for i in SEEDS1], P.KEY_STACK_DATASEEDS: DATASEEDS} for n in smpleff_regimes[ds]]
				smpleff_sk_hebb[lrn_rule + '_' + ds + ('_da' if da else '')] += [{P.KEY_STACK_CONFIG: 'configs.vision.smpleff.svm_on_hebb_layer_ft[' + str(l) + '_' + lrn_rule + '_' + ds + '_' + str(n) + ('_da' if da else '') + ']', P.KEY_STACK_SEEDS: SEEDS3, P.KEY_STACK_TOKENS: [str(i) for i in SEEDS2], P.KEY_STACK_DATASEEDS: DATASEEDS} for n in smpleff_regimes[ds]]
		for l in range(1, num_layers[ds]):
			smpleff_vae[ds + ('_da' if da else '')] += [{P.KEY_STACK_CONFIG: 'configs.vision.smpleff.gdes_fc_on_vae_layer[' + str(l) + '_' + ds + '_' + str(n) + ('_da' if da else '') + ']', P.KEY_STACK_SEEDS: SEEDS2, P.KEY_STACK_TOKENS: [str(i) for i in SEEDS1], P.KEY_STACK_DATASEEDS: DATASEEDS} for n in smpleff_regimes[ds]]
			smpleff_vae[ds + ('_da' if da else '')] += [{P.KEY_STACK_CONFIG: 'configs.vision.smpleff.gdes_fc_on_vae_layer_ft[' + str(l) + '_' + ds + '_' + str(n) + ('_da' if da else '') + ']', P.KEY_STACK_SEEDS: SEEDS2, P.KEY_STACK_TOKENS: [str(i) for i in SEEDS1], P.KEY_STACK_DATASEEDS: DATASEEDS} for n in smpleff_regimes[ds]]
			#smpleff_vae[ds + ('_da' if da else '')] += [{P.KEY_STACK_CONFIG: 'configs.vision.smpleff.hebb_fc_on_vae_layer[' + str(l) + '_' + ds + '_' + str(n) + ('_da' if da else '') + ']', P.KEY_STACK_SEEDS: SEEDS2, P.KEY_STACK_TOKENS: [str(i) for i in SEEDS1], P.KEY_STACK_DATASEEDS: DATASEEDS} for n in smpleff_regimes[ds]]
			#smpleff_vae[ds + ('_da' if da else '')] += [{P.KEY_STACK_CONFIG: 'configs.vision.smpleff.hebb_fc_on_vae_layer_ft[' + str(l) + '_' + ds + '_' + str(n) + ('_da' if da else '') + ']', P.KEY_STACK_SEEDS: SEEDS3, P.KEY_STACK_TOKENS: [str(i) for i in SEEDS2], P.KEY_STACK_DATASEEDS: DATASEEDS} for n in smpleff_regimes[ds]]
			smpleff_sk_vae[ds + ('_da' if da else '')] += [{P.KEY_STACK_CONFIG: 'configs.vision.smpleff.prec_on_vae_layer_ft[' + str(l) + '_' + ds + '_' + str(n) + ('_da' if da else '') + ']', P.KEY_STACK_SEEDS: SEEDS3, P.KEY_STACK_TOKENS: [str(i) for i in SEEDS2], P.KEY_STACK_DATASEEDS: DATASEEDS} for n in smpleff_regimes[ds]]
			smpleff_sk_vae[ds + ('_da' if da else '')] += [{P.KEY_STACK_CONFIG: 'configs.vision.smpleff.knn_on_vae_layer[' + str(l) + '_' + ds + '_' + str(n) + ('_da' if da else '') + ']', P.KEY_STACK_SEEDS: SEEDS2, P.KEY_STACK_TOKENS: [str(i) for i in SEEDS1], P.KEY_STACK_DATASEEDS: DATASEEDS} for n in smpleff_regimes[ds]]
			smpleff_sk_vae[ds + ('_da' if da else '')] += [{P.KEY_STACK_CONFIG: 'configs.vision.smpleff.knn_on_vae_layer_ft[' + str(l) + '_' + ds + '_' + str(n) + ('_da' if da else '') + ']', P.KEY_STACK_SEEDS: SEEDS3, P.KEY_STACK_TOKENS: [str(i) for i in SEEDS2], P.KEY_STACK_DATASEEDS: DATASEEDS} for n in smpleff_regimes[ds]]
			smpleff_sk_vae[ds + ('_da' if da else '')] += [{P.KEY_STACK_CONFIG: 'configs.vision.smpleff.svm_on_vae_layer[' + str(l) + '_' + ds + '_' + str(n) + ('_da' if da else '') + ']', P.KEY_STACK_SEEDS: SEEDS2, P.KEY_STACK_TOKENS: [str(i) for i in SEEDS1], P.KEY_STACK_DATASEEDS: DATASEEDS} for n in smpleff_regimes[ds]]
			smpleff_sk_vae[ds + ('_da' if da else '')] += [{P.KEY_STACK_CONFIG: 'configs.vision.smpleff.svm_on_vae_layer_ft[' + str(l) + '_' + ds + '_' + str(n) + ('_da' if da else '') + ']', P.KEY_STACK_SEEDS: SEEDS3, P.KEY_STACK_TOKENS: [str(i) for i in SEEDS2], P.KEY_STACK_DATASEEDS: DATASEEDS} for n in smpleff_regimes[ds]]
	
	
		all[ds + ('_da' if da else '')] += gdes[ds + ('_da' if da else '')]
		all[ds + ('_da' if da else '')] += vae[ds + ('_da' if da else '')]
		for lrn_rule in lrn_rules: all[ds + ('_da' if da else '')] += hebb[lrn_rule + '_' + ds + ('_da' if da else '')]
		for lrn_rule in lrn_rules: all[ds + ('_da' if da else '')] += hybrid[lrn_rule + '_' + ds + ('_da' if da else '')]
		all[ds + ('_da' if da else '')] += smpleff_gdes[ds + ('_da' if da else '')]
		all[ds + ('_da' if da else '')] += smpleff_vae[ds + ('_da' if da else '')]
		for lrn_rule in lrn_rules: all[ds + ('_da' if da else '')] += smpleff_hebb[lrn_rule + '_' + ds + ('_da' if da else '')]
		#all[ds + ('_da' if da else '')] += sk_gdes[ds + ('_da' if da else '')]
		#all[ds + ('_da' if da else '')] += sk_vae[ds + ('_da' if da else '')]
		#for lrn_rule in lrn_rules: all[ds + ('_da' if da else '')] += sk_hebb[lrn_rule + '_' + ds + ('_da' if da else '')]
		#all[ds + ('_da' if da else '')] += smpleff_sk_gdes[ds + ('_da' if da else '')]
		#all[ds + ('_da' if da else '')] += smpleff_sk_vae[ds + ('_da' if da else '')]
		#for lrn_rule in lrn_rules: all[ds + ('_da' if da else '')] += smpleff_sk_hebb[lrn_rule + '_' + ds + ('_da' if da else '')]
