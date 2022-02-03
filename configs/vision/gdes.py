from neurolab import params as P
import params as PP
from .meta import *


config_base_gdes = {}
gdes_fc_on_gdes_layer = {}
gdes_fc_on_gdes_layer_ft = {}
hebb_fc_on_gdes_layer = {}
hebb_fc_on_gdes_layer_ft = {}
gdes_fc2_on_gdes_layer = {}
hebb_fc2_on_gdes_layer = {}
prec_on_gdes_layer = {}
prec_on_gdes_layer_ft = {}
knn_on_gdes_layer = {}
knn_on_gdes_layer_ft = {}
svm_on_gdes_layer = {}
svm_on_gdes_layer_ft = {}



for ds in datasets:
	for da in [False, True]:
		config_base_gdes[ds + ('_da' if da else '')] = {
			P.KEY_EXPERIMENT: 'neurolab.experiment.VisionExperiment',
			P.KEY_NET_MODULES: 'models.gdes.model_' + str(num_layers[ds]) + 'l.Net',
			P.KEY_NET_OUTPUTS: net_outputs[ds],
			P.KEY_DATA_MANAGER: 'neurolab.data.' + data_managers[ds],
			P.KEY_AUGMENT_MANAGER: None if not da else 'neurolab.data.LightCustomAugmentManager',
			P.KEY_AUGM_BEFORE_STATS: True,
			P.KEY_AUGM_STAT_PASSES: 2,
			P.KEY_WHITEN: None,
			P.KEY_TOT_TRN_SAMPLES: tot_trn_samples[ds],
			P.KEY_BATCHSIZE: 64,
			P.KEY_INPUT_SHAPE: input_shapes[ds],
			P.KEY_NUM_EPOCHS: 20 if not da else 40,
			P.KEY_OPTIM_MANAGER: 'neurolab.optimization.optim.SGDOptimManager',
			P.KEY_SCHED_MANAGER: 'neurolab.optimization.sched.MultiStepSchedManager',
			P.KEY_LOSS_METRIC_MANAGER: 'neurolab.optimization.metric.CrossEntMetricManager',
			P.KEY_CRIT_METRIC_MANAGER: ['neurolab.optimization.metric.TopKAccMetricManager', 'neurolab.optimization.metric.TopKAccMetricManager'],
			P.KEY_TOPKACC_K: [1, 5],
		    P.KEY_LEARNING_RATE: 1e-3,
		    P.KEY_LR_DECAY: 0.5 if not da else 0.1,
		    P.KEY_MILESTONES: range(10, 20) if not da else [20, 30],
		    P.KEY_MOMENTUM: 0.9,
		    P.KEY_L2_PENALTY: l2_penalties[ds],
			P.KEY_DROPOUT_P: 0.5,
		}
		
		for l in range(1, num_layers[ds]):
			gdes_fc_on_gdes_layer[str(l) + '_' + ds + ('_da' if da else '')] = {
				P.KEY_EXPERIMENT: 'neurolab.experiment.VisionExperiment',
				P.KEY_NET_MODULES: 'models.gdes.fc.Net',
				P.KEY_NET_OUTPUTS: 'fc',
				P.KEY_DATA_MANAGER: 'neurolab.data.' + data_managers[ds],
				P.KEY_AUGMENT_MANAGER: None if not da else 'neurolab.data.LightCustomAugmentManager',
				P.KEY_AUGM_BEFORE_STATS: True,
				P.KEY_AUGM_STAT_PASSES: 2,
				P.KEY_WHITEN: None,
				P.KEY_TOT_TRN_SAMPLES: tot_trn_samples[ds],
				P.KEY_BATCHSIZE: 64,
				P.KEY_INPUT_SHAPE: input_shapes[ds],
				P.KEY_NUM_EPOCHS: 20 if not da else 40,
				P.KEY_OPTIM_MANAGER: 'neurolab.optimization.optim.SGDOptimManager',
				P.KEY_SCHED_MANAGER: 'neurolab.optimization.sched.MultiStepSchedManager',
				P.KEY_LOSS_METRIC_MANAGER: 'neurolab.optimization.metric.CrossEntMetricManager',
				P.KEY_CRIT_METRIC_MANAGER: ['neurolab.optimization.metric.TopKAccMetricManager', 'neurolab.optimization.metric.TopKAccMetricManager'],
				P.KEY_TOPKACC_K: [1, 5],
			    P.KEY_LEARNING_RATE: 1e-3,
			    P.KEY_LR_DECAY: 0.5 if not da else 0.1,
		        P.KEY_MILESTONES: range(10, 20) if not da else [20, 30],
			    P.KEY_MOMENTUM: 0.9,
			    P.KEY_L2_PENALTY: 5e-4,
				P.KEY_DROPOUT_P: 0.5,
				P.KEY_PRE_NET_MODULES: ['models.gdes.model_' + str(num_layers[ds]) + 'l.Net'],
				P.KEY_PRE_NET_MDL_PATHS: [P.PROJECT_ROOT + '/results/configs/vision/gdes/config_base_gdes[' + ds + ('_da' if da else '') + ']/iter_' + P.STR_TOKEN + '/models/model0.pt'],
				P.KEY_PRE_NET_OUTPUTS: ['bn' + str(l)],
			}
			
			gdes_fc_on_gdes_layer_ft[str(l) + '_' + ds + ('_da' if da else '')] = {
				P.KEY_EXPERIMENT: 'neurolab.experiment.VisionExperiment',
				P.KEY_NET_MODULES: ['models.gdes.model_6l.Net', 'models.gdes.fc.Net'],
				P.KEY_NET_MDL_PATHS: [P.PROJECT_ROOT + '/results/configs/vision/gdes/config_base_gdes[' + ds + ('_da' if da else '') + ']/iter_' + P.STR_TOKEN + '/models/model0.pt'],
				P.KEY_NET_OUTPUTS: ['bn' + str(l), 'fc'],
				P.KEY_DATA_MANAGER: 'neurolab.data.' + data_managers[ds],
				P.KEY_AUGMENT_MANAGER: None if not da else 'neurolab.data.LightCustomAugmentManager',
				P.KEY_AUGM_BEFORE_STATS: True,
				P.KEY_AUGM_STAT_PASSES: 2,
				P.KEY_WHITEN: None,
				P.KEY_TOT_TRN_SAMPLES: tot_trn_samples[ds],
				P.KEY_BATCHSIZE: 64,
				P.KEY_INPUT_SHAPE: input_shapes[ds],
				P.KEY_NUM_EPOCHS: 20 if not da else 40,
				P.KEY_OPTIM_MANAGER: 'neurolab.optimization.optim.SGDOptimManager',
				P.KEY_SCHED_MANAGER: 'neurolab.optimization.sched.MultiStepSchedManager',
				P.KEY_LOSS_METRIC_MANAGER: 'neurolab.optimization.metric.CrossEntMetricManager',
				P.KEY_CRIT_METRIC_MANAGER: ['neurolab.optimization.metric.TopKAccMetricManager', 'neurolab.optimization.metric.TopKAccMetricManager'],
				P.KEY_TOPKACC_K: [1, 5],
			    P.KEY_LEARNING_RATE: 1e-3,
			    P.KEY_LR_DECAY: 0.5 if not da else 0.1,
		        P.KEY_MILESTONES: range(10, 20) if not da else [20, 30],
			    P.KEY_MOMENTUM: 0.9,
			    P.KEY_L2_PENALTY: 5e-4,
				P.KEY_DROPOUT_P: 0.5,
			}
			
			hebb_fc_on_gdes_layer[str(l) + '_' + ds + ('_da' if da else '')] = {
				P.KEY_EXPERIMENT: 'neurolab.experiment.VisionExperiment',
				P.KEY_NET_MODULES: 'models.hebb.fc.Net',
				P.KEY_NET_OUTPUTS: 'fc',
				P.KEY_DATA_MANAGER: 'neurolab.data.' + data_managers[ds],
				P.KEY_AUGMENT_MANAGER: None if not da else 'neurolab.data.LightCustomAugmentManager',
				P.KEY_AUGM_BEFORE_STATS: True,
				P.KEY_AUGM_STAT_PASSES: 2,
				P.KEY_WHITEN: None,
				P.KEY_TOT_TRN_SAMPLES: tot_trn_samples[ds],
				P.KEY_BATCHSIZE: 64,
				P.KEY_INPUT_SHAPE: input_shapes[ds],
				P.KEY_NUM_EPOCHS: 20,
				P.KEY_OPTIM_MANAGER: 'neurolab.optimization.optim.SGDOptimManager',
				P.KEY_CRIT_METRIC_MANAGER: ['neurolab.optimization.metric.TopKAccMetricManager', 'neurolab.optimization.metric.TopKAccMetricManager'],
				P.KEY_TOPKACC_K: [1, 5],
			    P.KEY_LEARNING_RATE: 1e-3,
				P.KEY_PRE_NET_MODULES: ['models.gdes.model_' + str(num_layers[ds]) + 'l.Net'],
				P.KEY_PRE_NET_MDL_PATHS: [P.PROJECT_ROOT + '/results/configs/vision/gdes/config_base_gdes[' + ds + ('_da' if da else '') + ']/iter_' + P.STR_TOKEN + '/models/model0.pt'],
				P.KEY_PRE_NET_OUTPUTS: ['bn' + str(l)],
			}
			
			hebb_fc_on_gdes_layer_ft[str(l) + '_' + ds + ('_da' if da else '')] = {
				P.KEY_EXPERIMENT: 'neurolab.experiment.VisionExperiment',
				P.KEY_NET_MODULES: 'models.hebb.fc.Net',
				P.KEY_NET_OUTPUTS: 'fc',
				P.KEY_DATA_MANAGER: 'neurolab.data.' + data_managers[ds],
				P.KEY_AUGMENT_MANAGER: None if not da else 'neurolab.data.LightCustomAugmentManager',
				P.KEY_AUGM_BEFORE_STATS: True,
				P.KEY_AUGM_STAT_PASSES: 2,
				P.KEY_WHITEN: None,
				P.KEY_TOT_TRN_SAMPLES: tot_trn_samples[ds],
				P.KEY_BATCHSIZE: 64,
				P.KEY_INPUT_SHAPE: input_shapes[ds],
				P.KEY_NUM_EPOCHS: 20,
				P.KEY_OPTIM_MANAGER: 'neurolab.optimization.optim.SGDOptimManager',
				P.KEY_CRIT_METRIC_MANAGER: ['neurolab.optimization.metric.TopKAccMetricManager', 'neurolab.optimization.metric.TopKAccMetricManager'],
				P.KEY_TOPKACC_K: [1, 5],
			    P.KEY_LEARNING_RATE: 1e-3,
				P.KEY_PRE_NET_MODULES: ['models.gdes.model_' + str(num_layers[ds]) + 'l.Net'],
				P.KEY_PRE_NET_MDL_PATHS: [P.PROJECT_ROOT + '/results/configs/vision/gdes/gdes_fc_on_gdes_layer_ft[' + str(l) + '_' + ds + ('_da' if da else '') + ']/iter_' + P.STR_TOKEN + '/models/model0.pt'],
				P.KEY_PRE_NET_OUTPUTS: ['bn' + str(l)],
			}
			
			gdes_fc2_on_gdes_layer[str(l) + '_' + ds + ('_da' if da else '')] = {
				P.KEY_EXPERIMENT: 'neurolab.experiment.VisionExperiment',
				P.KEY_NET_MODULES: 'models.gdes.fc2.Net',
				P.KEY_NET_OUTPUTS: 'fc2',
				P.KEY_DATA_MANAGER: 'neurolab.data.' + data_managers[ds],
				P.KEY_AUGMENT_MANAGER: None if not da else 'neurolab.data.LightCustomAugmentManager',
				P.KEY_AUGM_BEFORE_STATS: True,
				P.KEY_AUGM_STAT_PASSES: 2,
				P.KEY_WHITEN: None,
				P.KEY_TOT_TRN_SAMPLES: tot_trn_samples[ds],
				P.KEY_BATCHSIZE: 64,
				P.KEY_INPUT_SHAPE: input_shapes[ds],
				P.KEY_NUM_EPOCHS: 20 if not da else 40,
				P.KEY_OPTIM_MANAGER: 'neurolab.optimization.optim.SGDOptimManager',
				P.KEY_SCHED_MANAGER: 'neurolab.optimization.sched.MultiStepSchedManager',
				P.KEY_LOSS_METRIC_MANAGER: 'neurolab.optimization.metric.CrossEntMetricManager',
				P.KEY_CRIT_METRIC_MANAGER: ['neurolab.optimization.metric.TopKAccMetricManager', 'neurolab.optimization.metric.TopKAccMetricManager'],
				P.KEY_TOPKACC_K: [1, 5],
			    P.KEY_LEARNING_RATE: 1e-3,
			    P.KEY_LR_DECAY: 0.5 if not da else 0.1,
		        P.KEY_MILESTONES: range(10, 20) if not da else [20, 30],
			    P.KEY_MOMENTUM: 0.9,
			    P.KEY_L2_PENALTY: 5e-4,
				P.KEY_DROPOUT_P: 0.5,
				P.KEY_PRE_NET_MODULES: ['models.gdes.model_' + str(num_layers[ds]) + 'l.Net'],
				P.KEY_PRE_NET_MDL_PATHS: [P.PROJECT_ROOT + '/results/configs/vision/gdes/config_base_gdes[' + ds + ('_da' if da else '') + ']/iter_' + P.STR_TOKEN + '/models/model0.pt'],
				P.KEY_PRE_NET_OUTPUTS: ['bn' + str(l)],
			}
			
			for lrn_rule in lrn_rules:
				hebb_fc2_on_gdes_layer[str(l) + '_' + lrn_rule + '_' + ds + ('_da' if da else '')] = {
					P.KEY_EXPERIMENT: 'neurolab.experiment.VisionExperiment',
					P.KEY_NET_MODULES: 'models.hebb.fc2.Net',
					P.KEY_NET_OUTPUTS: 'fc2',
					P.KEY_DATA_MANAGER: 'neurolab.data.' + data_managers[ds],
					P.KEY_AUGMENT_MANAGER: None if not da else 'neurolab.data.LightCustomAugmentManager',
					P.KEY_AUGM_BEFORE_STATS: True,
					P.KEY_AUGM_STAT_PASSES: 2,
					P.KEY_WHITEN: None,
					P.KEY_TOT_TRN_SAMPLES: tot_trn_samples[ds],
					P.KEY_BATCHSIZE: 64,
					P.KEY_INPUT_SHAPE: input_shapes[ds],
					P.KEY_NUM_EPOCHS: 20,
					P.KEY_OPTIM_MANAGER: 'neurolab.optimization.optim.SGDOptimManager',
					P.KEY_CRIT_METRIC_MANAGER: ['neurolab.optimization.metric.TopKAccMetricManager', 'neurolab.optimization.metric.TopKAccMetricManager'],
					P.KEY_TOPKACC_K: [1, 5],
				    P.KEY_LEARNING_RATE: 1e-3,
					P.KEY_LOCAL_LRN_RULE: lrn_rule_keys[lrn_rule],
					PP.KEY_WTA_K: lrn_rule_k[lrn_rule],
					P.KEY_DEEP_TEACHER_SIGNAL: lrn_rule_dts[lrn_rule],
					P.KEY_PRE_NET_MODULES: ['models.gdes.model_' + str(num_layers[ds]) + 'l.Net'],
					P.KEY_PRE_NET_MDL_PATHS: [P.PROJECT_ROOT + '/results/configs/vision/gdes/config_base_gdes[' + ds + ('_da' if da else '') + ']/iter_' + P.STR_TOKEN + '/models/model0.pt'],
					P.KEY_PRE_NET_OUTPUTS: ['bn' + str(l)],
				}
			
			prec_on_gdes_layer[str(l) + '_' + ds + ('_da' if da else '')] = {
				P.KEY_EXPERIMENT: 'neurolab.experiment.VisionExperiment',
				P.KEY_NET_MODULES: 'neurolab.model.skclassif.Retriever',
				P.KEY_NET_OUTPUTS: 'clf',
				P.KEY_DATA_MANAGER: 'neurolab.data.' + data_managers[ds],
				P.KEY_AUGMENT_MANAGER: None,
				P.KEY_AUGM_BEFORE_STATS: True,
				P.KEY_AUGM_STAT_PASSES: 2,
				P.KEY_WHITEN: None,
				P.KEY_TOT_TRN_SAMPLES: tot_trn_samples[ds],
				P.KEY_BATCHSIZE: 64,
				P.KEY_INPUT_SHAPE: input_shapes[ds],
				P.KEY_NUM_EPOCHS: 1,
				P.KEY_CRIT_METRIC_MANAGER: 'neurolab.optimization.metric.PrecMetricManager',
				P.KEY_SKCLF_NUM_SAMPLES: tot_trn_samples[ds],
				P.KEY_NYSTROEM_N_COMPONENTS: 100,
				P.KEY_KNN_N_NEIGHBORS: samples_per_class[ds],
				P.KEY_PRE_NET_MODULES: ['models.gdes.model_' + str(num_layers[ds]) + 'l.Net'],
				P.KEY_PRE_NET_MDL_PATHS: [P.PROJECT_ROOT + '/results/configs/vision/gdes/config_base_gdes[' + ds + ('_da' if da else '') + ']/iter_' + P.STR_TOKEN + '/models/model0.pt'],
				P.KEY_PRE_NET_OUTPUTS: ['bn' + str(l)],
			}
			
			prec_on_gdes_layer_ft[str(l) + '_' + ds + ('_da' if da else '')] = {
				P.KEY_EXPERIMENT: 'neurolab.experiment.VisionExperiment',
				P.KEY_NET_MODULES: 'neurolab.model.skclassif.Retriever',
				P.KEY_NET_OUTPUTS: 'clf',
				P.KEY_DATA_MANAGER: 'neurolab.data.' + data_managers[ds],
				P.KEY_AUGMENT_MANAGER: None,
				P.KEY_AUGM_BEFORE_STATS: True,
				P.KEY_AUGM_STAT_PASSES: 2,
				P.KEY_WHITEN: None,
				P.KEY_TOT_TRN_SAMPLES: tot_trn_samples[ds],
				P.KEY_BATCHSIZE: 64,
				P.KEY_INPUT_SHAPE: input_shapes[ds],
				P.KEY_NUM_EPOCHS: 1,
				P.KEY_CRIT_METRIC_MANAGER: 'neurolab.optimization.metric.PrecMetricManager',
				P.KEY_SKCLF_NUM_SAMPLES: tot_trn_samples[ds],
				P.KEY_NYSTROEM_N_COMPONENTS: 100,
				P.KEY_KNN_N_NEIGHBORS: samples_per_class[ds],
				P.KEY_PRE_NET_MODULES: ['models.gdes.model_' + str(num_layers[ds]) + 'l.Net'],
				P.KEY_PRE_NET_MDL_PATHS: [P.PROJECT_ROOT + '/results/configs/vision/gdes/gdes_fc_on_gdes_layer_ft[' + str(l) + '_' + ds + ('_da' if da else '') + ']/iter_' + P.STR_TOKEN + '/models/model0.pt'],
				P.KEY_PRE_NET_OUTPUTS: ['bn' + str(l)],
			}
			
			knn_on_gdes_layer[str(l) + '_' + ds + ('_da' if da else '')] = {
				P.KEY_EXPERIMENT: 'neurolab.experiment.VisionExperiment',
				P.KEY_NET_MODULES: 'neurolab.model.skclassif.KNNClassifier',
				P.KEY_NET_OUTPUTS: 'clf',
				P.KEY_DATA_MANAGER: 'neurolab.data.' + data_managers[ds],
				P.KEY_AUGMENT_MANAGER: None if not da else 'neurolab.data.LightCustomAugmentManager',
				P.KEY_AUGM_BEFORE_STATS: True,
				P.KEY_AUGM_STAT_PASSES: 2,
				P.KEY_WHITEN: None,
				P.KEY_TOT_TRN_SAMPLES: tot_trn_samples[ds],
				P.KEY_BATCHSIZE: 64,
				P.KEY_INPUT_SHAPE: input_shapes[ds],
				P.KEY_NUM_EPOCHS: 1 if not da else 2,
				P.KEY_CRIT_METRIC_MANAGER: ['neurolab.optimization.metric.TopKAccMetricManager', 'neurolab.optimization.metric.TopKAccMetricManager'],
				P.KEY_TOPKACC_K: [1, 5],
				P.KEY_SKCLF_NUM_SAMPLES: tot_trn_samples[ds] if not da else 2*tot_trn_samples[ds],
				P.KEY_NYSTROEM_N_COMPONENTS: 100,
				P.KEY_KNN_N_NEIGHBORS: 10,
				P.KEY_PRE_NET_MODULES: ['models.gdes.model_' + str(num_layers[ds]) + 'l.Net'],
				P.KEY_PRE_NET_MDL_PATHS: [P.PROJECT_ROOT + '/results/configs/vision/gdes/config_base_gdes[' + ds + ('_da' if da else '') + ']/iter_' + P.STR_TOKEN + '/models/model0.pt'],
				P.KEY_PRE_NET_OUTPUTS: ['bn' + str(l)],
			}
			
			knn_on_gdes_layer_ft[str(l) + '_' + ds + ('_da' if da else '')] = {
				P.KEY_EXPERIMENT: 'neurolab.experiment.VisionExperiment',
				P.KEY_NET_MODULES: 'neurolab.model.skclassif.KNNClassifier',
				P.KEY_NET_OUTPUTS: 'clf',
				P.KEY_DATA_MANAGER: 'neurolab.data.' + data_managers[ds],
				P.KEY_AUGMENT_MANAGER: None if not da else 'neurolab.data.LightCustomAugmentManager',
				P.KEY_AUGM_BEFORE_STATS: True,
				P.KEY_AUGM_STAT_PASSES: 2,
				P.KEY_WHITEN: None,
				P.KEY_TOT_TRN_SAMPLES: tot_trn_samples[ds],
				P.KEY_BATCHSIZE: 64,
				P.KEY_INPUT_SHAPE: input_shapes[ds],
				P.KEY_NUM_EPOCHS: 1 if not da else 2,
				P.KEY_CRIT_METRIC_MANAGER: ['neurolab.optimization.metric.TopKAccMetricManager', 'neurolab.optimization.metric.TopKAccMetricManager'],
				P.KEY_TOPKACC_K: [1, 5],
				P.KEY_SKCLF_NUM_SAMPLES: tot_trn_samples[ds] if not da else 2*tot_trn_samples[ds],
				P.KEY_NYSTROEM_N_COMPONENTS: 100,
				P.KEY_KNN_N_NEIGHBORS: 10,
				P.KEY_PRE_NET_MODULES: ['models.gdes.model_' + str(num_layers[ds]) + 'l.Net'],
				P.KEY_PRE_NET_MDL_PATHS: [P.PROJECT_ROOT + '/results/configs/vision/gdes/gdes_fc_on_gdes_layer_ft[' + str(l) + '_' + ds + ('_da' if da else '') + ']/iter_' + P.STR_TOKEN + '/models/model0.pt'],
				P.KEY_PRE_NET_OUTPUTS: ['bn' + str(l)],
			}
			
			svm_on_gdes_layer[str(l) + '_' + ds + ('_da' if da else '')] = {
				P.KEY_EXPERIMENT: 'neurolab.experiment.VisionExperiment',
				P.KEY_NET_MODULES: 'neurolab.model.skclassif.SVMClassifier',
				P.KEY_NET_OUTPUTS: 'clf',
				P.KEY_DATA_MANAGER: 'neurolab.data.' + data_managers[ds],
				P.KEY_AUGMENT_MANAGER: None if not da else 'neurolab.data.LightCustomAugmentManager',
				P.KEY_AUGM_BEFORE_STATS: True,
				P.KEY_AUGM_STAT_PASSES: 2,
				P.KEY_WHITEN: None,
				P.KEY_TOT_TRN_SAMPLES: tot_trn_samples[ds],
				P.KEY_BATCHSIZE: 64,
				P.KEY_INPUT_SHAPE: input_shapes[ds],
				P.KEY_NUM_EPOCHS: 1 if not da else 2,
				P.KEY_CRIT_METRIC_MANAGER: ['neurolab.optimization.metric.TopKAccMetricManager', 'neurolab.optimization.metric.TopKAccMetricManager'],
				P.KEY_TOPKACC_K: [1, 5],
				P.KEY_SKCLF_NUM_SAMPLES: tot_trn_samples[ds] if not da else 2*tot_trn_samples[ds],
				P.KEY_NYSTROEM_N_COMPONENTS: 100,
				P.KEY_PRE_NET_MODULES: ['models.gdes.model_' + str(num_layers[ds]) + 'l.Net'],
				P.KEY_PRE_NET_MDL_PATHS: [P.PROJECT_ROOT + '/results/configs/vision/gdes/config_base_gdes[' + ds + ('_da' if da else '') + ']/iter_' + P.STR_TOKEN + '/models/model0.pt'],
				P.KEY_PRE_NET_OUTPUTS: ['bn' + str(l)],
			}
			
			svm_on_gdes_layer_ft[str(l) + '_' + ds + ('_da' if da else '')] = {
				P.KEY_EXPERIMENT: 'neurolab.experiment.VisionExperiment',
				P.KEY_NET_MODULES: 'neurolab.model.skclassif.SVMClassifier',
				P.KEY_NET_OUTPUTS: 'clf',
				P.KEY_DATA_MANAGER: 'neurolab.data.' + data_managers[ds],
				P.KEY_AUGMENT_MANAGER: None if not da else 'neurolab.data.LightCustomAugmentManager',
				P.KEY_AUGM_BEFORE_STATS: True,
				P.KEY_AUGM_STAT_PASSES: 2,
				P.KEY_WHITEN: None,
				P.KEY_TOT_TRN_SAMPLES: tot_trn_samples[ds],
				P.KEY_BATCHSIZE: 64,
				P.KEY_INPUT_SHAPE: input_shapes[ds],
				P.KEY_NUM_EPOCHS: 1 if not da else 2,
				P.KEY_CRIT_METRIC_MANAGER: ['neurolab.optimization.metric.TopKAccMetricManager', 'neurolab.optimization.metric.TopKAccMetricManager'],
				P.KEY_TOPKACC_K: [1, 5],
				P.KEY_SKCLF_NUM_SAMPLES: tot_trn_samples[ds] if not da else 2*tot_trn_samples[ds],
				P.KEY_NYSTROEM_N_COMPONENTS: 100,
				P.KEY_PRE_NET_MODULES: ['models.gdes.model_' + str(num_layers[ds]) + 'l.Net'],
				P.KEY_PRE_NET_MDL_PATHS: [P.PROJECT_ROOT + '/results/configs/vision/gdes/gdes_fc_on_gdes_layer_ft[' + str(l) + '_' + ds + ('_da' if da else '') + ']/iter_' + P.STR_TOKEN + '/models/model0.pt'],
				P.KEY_PRE_NET_OUTPUTS: ['bn' + str(l)],
			}
			
			