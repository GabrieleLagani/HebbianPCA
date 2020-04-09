import params as P
import basemodel.model, basemodel.model1, basemodel.model2, basemodel.model3, basemodel.model4, basemodel.fc, \
	basemodel.top1, basemodel.top2, basemodel.top3, basemodel.top4
import hebbmodel.model, hebbmodel.top1, hebbmodel.top2, hebbmodel.top3, hebbmodel.top4, hebbmodel.fc, \
	hebbmodel.g1h2_6, hebbmodel.g1_2h3_6, hebbmodel.g1_3h4_6, hebbmodel.g1_4h5_6


class Configuration:
	def __init__(self,
				 config_family,
				 config_name,
				 net_class,
				 batch_size,
				 num_epochs,
				 iteration_ids,
				 val_set_split,
				 augment_data,
				 whiten_data,
				 learning_rate=None,
				 lr_decay=None,
				 milestones=None,
				 momentum=None,
				 l2_penalty=None,
				 deep_teacher_signal=False,
				 pre_net_class=None,
				 pre_net_mdl_path=None,
				 pre_net_out=None):
		self.CONFIG_FAMILY = config_family
		self.CONFIG_NAME = config_name
		self.CONFIG_ID = self.CONFIG_FAMILY + '/' + self.CONFIG_NAME
		
		self.Net = net_class
		
		self.BATCH_SIZE = batch_size
		self.NUM_EPOCHS = num_epochs
		self.ITERATION_IDS = iteration_ids
		
		# Paths where to save the model
		self.MDL_PATH = {}
		# Path where to save accuracy plot
		self.ACC_PLT_PATH = {}
		# Path where to save kernel images
		self.KNL_PLT_PATH = {}
		for iter_id in self.ITERATION_IDS:
			# Path where to save the model
			self.MDL_PATH[iter_id] = P.RESULT_FOLDER + '/' + self.CONFIG_ID + '/save/model' + str(iter_id) + '.pt'
			# Path where to save accuracy plot
			self.ACC_PLT_PATH[iter_id] = P.RESULT_FOLDER + '/' + self.CONFIG_ID + '/figures/accuracy' + str(iter_id) + '.png'
			# Path where to save kernel images
			self.KNL_PLT_PATH[iter_id] = P.RESULT_FOLDER + '/' + self.CONFIG_ID + '/figures/kernels' + str(iter_id) + '.png'
		# Path to the CSV where test results are saved
		self.CSV_PATH = P.RESULT_FOLDER + '/' + self.CONFIG_ID + '/test_results.csv'
		
		# Define the splitting point of the training batches between training and validation datasets
		self.VAL_SET_SPLIT = val_set_split
		
		# Define whether to apply data augmentation or whitening
		self.AUGMENT_DATA = augment_data
		self.WHITEN_DATA = whiten_data
		
		self.LEARNING_RATE = learning_rate # Initial learning rate, periodically decreased by a lr_scheduler
		self.LR_DECAY = lr_decay # LR decreased periodically by a factor of 10
		self.MILESTONES = milestones # Epochs at which LR is decreased
		self.MOMENTUM = momentum
		self.L2_PENALTY = l2_penalty
		self.DEEP_TEACHER_SIGNAL = deep_teacher_signal
		
		self.PreNet = pre_net_class
		self.PRE_NET_MDL_PATH = pre_net_mdl_path
		self.PRE_NET_OUT = pre_net_out
	

CONFIG_LIST = [
	
	################################################################################################################
	####										GDES CONFIGURATIONS												####
	################################################################################################################
	
	Configuration(
		config_family=P.CONFIG_FAMILY_GDES,
		config_name='config_base_91', #91.55
		net_class=basemodel.model.Net,
		batch_size=64,
		num_epochs=100,
		iteration_ids=[0],
		val_set_split=50000,
		augment_data=True,
		whiten_data=False,
		learning_rate=1e-3,
		lr_decay=0.1,
		milestones=[40, 70, 90],
		momentum=0.9,
		l2_penalty=3e-2,
	),
	
	Configuration(
		config_family=P.CONFIG_FAMILY_GDES,
		config_name='config_base', # 84.95
		net_class=basemodel.model.Net,
		batch_size=64,
		num_epochs=20,
		iteration_ids=[0],
		val_set_split=40000,
		augment_data=False,
		whiten_data=False,
		learning_rate=1e-3,
		lr_decay=0.5,
		milestones=range(10, 20),
		momentum=0.9,
		l2_penalty=6e-2,
	),
	
	Configuration(
		config_family=P.CONFIG_FAMILY_GDES,
		config_name='config_1l', # 69.04
		net_class=basemodel.model1.Net,
		batch_size=64,
		num_epochs=20,
		iteration_ids=[0],
		val_set_split=40000,
		augment_data=False,
		whiten_data=False,
		learning_rate=1e-3,
		lr_decay=0.5,
		milestones=range(10, 20),
		momentum=0.9,
		l2_penalty=2e-2,
	),
	
	Configuration(
		config_family=P.CONFIG_FAMILY_GDES,
		config_name='config_2l', # 76.91
		net_class=basemodel.model2.Net,
		batch_size=64,
		num_epochs=20,
		iteration_ids=[0],
		val_set_split=40000,
		augment_data=False,
		whiten_data=False,
		learning_rate=1e-3,
		lr_decay=0.5,
		milestones=range(10, 20),
		momentum=0.9,
		l2_penalty=3e-2
	),
	
	Configuration(
		config_family=P.CONFIG_FAMILY_GDES,
		config_name='config_3l', # 81.37
		net_class=basemodel.model3.Net,
		batch_size=64,
		num_epochs=20,
		iteration_ids=[0],
		val_set_split=40000,
		augment_data=False,
		whiten_data=False,
		learning_rate=1e-3,
		lr_decay=0.5,
		milestones=range(10, 20),
		momentum=0.9,
		l2_penalty=4e-2
	),
	
	Configuration(
		config_family=P.CONFIG_FAMILY_GDES,
		config_name='config_4l', # 83.97
		net_class=basemodel.model4.Net,
		batch_size=64,
		num_epochs=20,
		iteration_ids=[0],
		val_set_split=40000,
		augment_data=False,
		whiten_data=False,
		learning_rate=1e-3,
		lr_decay=0.5,
		milestones=range(10, 20),
		momentum=0.9,
		l2_penalty=5e-2
	),

	################################################################################################################
	####					CONFIGS: GDES CLASSIFIER ON FEATURES EXTRACTED FROM GDES LAYERS						####
	################################################################################################################

	Configuration(
		config_family=P.CONFIG_FAMILY_GDES,
		config_name='fc_on_gdes_conv1',  # 60.71
		net_class=basemodel.fc.Net,
		batch_size=64,
		num_epochs=20,
		iteration_ids=[0],
		val_set_split=40000,
		augment_data=False,
		whiten_data=False,
		learning_rate=1e-3,
		lr_decay=0.5,
		milestones=range(10, 20),
		momentum=0.9,
		l2_penalty=5e-4,
		pre_net_class=basemodel.model.Net,
		pre_net_mdl_path=P.PROJECT_ROOT + '/results/gdes/config_base/save/model0.pt',
		pre_net_out=basemodel.model.Net.BN1
	),

	Configuration(
		config_family=P.CONFIG_FAMILY_GDES,
		config_name='fc_on_gdes_conv2',  # 66.30
		net_class=basemodel.fc.Net,
		batch_size=64,
		num_epochs=20,
		iteration_ids=[0],
		val_set_split=40000,
		augment_data=False,
		whiten_data=False,
		learning_rate=1e-3,
		lr_decay=0.5,
		milestones=range(10, 20),
		momentum=0.9,
		l2_penalty=5e-4,
		pre_net_class=basemodel.model.Net,
		pre_net_mdl_path=P.PROJECT_ROOT + '/results/gdes/config_base/save/model0.pt',
		pre_net_out=basemodel.model.Net.BN2
	),

	Configuration(
		config_family=P.CONFIG_FAMILY_GDES,
		config_name='fc_on_gdes_conv3',  # 72.39
		net_class=basemodel.fc.Net,
		batch_size=64,
		num_epochs=20,
		iteration_ids=[0],
		val_set_split=40000,
		augment_data=False,
		whiten_data=False,
		learning_rate=1e-3,
		lr_decay=0.5,
		milestones=range(10, 20),
		momentum=0.9,
		l2_penalty=5e-4,
		pre_net_class=basemodel.model.Net,
		pre_net_mdl_path=P.PROJECT_ROOT + '/results/gdes/config_base/save/model0.pt',
		pre_net_out=basemodel.model.Net.BN3
	),

	Configuration(
		config_family=P.CONFIG_FAMILY_GDES,
		config_name='fc_on_gdes_conv4', # 82.69
		net_class=basemodel.fc.Net,
		batch_size=64,
		num_epochs=20,
		iteration_ids=[0],
		val_set_split=40000,
		augment_data=False,
		whiten_data=False,
		learning_rate=1e-3,
		lr_decay=0.5,
		milestones=range(10, 20),
		momentum=0.9,
		l2_penalty=5e-4,
		pre_net_class=basemodel.model.Net,
		pre_net_mdl_path=P.PROJECT_ROOT + '/results/gdes/config_base/save/model0.pt',
		pre_net_out=basemodel.model.Net.BN4
	),

	################################################################################################################
	####					CONFIGS: GDES CLASSIFIER ON FEATURES EXTRACTED FROM HEBB LAYERS						####
	################################################################################################################
	
	Configuration(
		config_family=P.CONFIG_FAMILY_GDES,
		config_name='fc_on_hebb_conv1',
		net_class=basemodel.fc.Net,
		batch_size=64,
		num_epochs=20,
		iteration_ids=[0],
		val_set_split=40000,
		augment_data=False,
		whiten_data=False,
		learning_rate=1e-3,
		lr_decay=0.5,
		milestones=range(10, 20),
		momentum=0.9,
		l2_penalty=5e-4,
		pre_net_class=hebbmodel.model.Net,
		pre_net_mdl_path=P.PROJECT_ROOT + '/results/hebb/config_base/save/model0.pt',
		pre_net_out=hebbmodel.model.Net.BN1
	),
	
	Configuration(
		config_family=P.CONFIG_FAMILY_GDES,
		config_name='fc_on_hebb_conv2',
		net_class=basemodel.fc.Net,
		batch_size=64,
		num_epochs=20,
		iteration_ids=[0],
		val_set_split=40000,
		augment_data=False,
		whiten_data=False,
		learning_rate=1e-3,
		lr_decay=0.5,
		milestones=range(10, 20),
		momentum=0.9,
		l2_penalty=5e-4,
		pre_net_class=hebbmodel.model.Net,
		pre_net_mdl_path=P.PROJECT_ROOT + '/results/hebb/config_base/save/model0.pt',
		pre_net_out=hebbmodel.model.Net.BN2
	),
	
	Configuration(
		config_family=P.CONFIG_FAMILY_GDES,
		config_name='fc_on_hebb_conv3',
		net_class=basemodel.fc.Net,
		batch_size=64,
		num_epochs=20,
		iteration_ids=[0],
		val_set_split=40000,
		augment_data=False,
		whiten_data=False,
		learning_rate=1e-3,
		lr_decay=0.5,
		milestones=range(10, 20),
		momentum=0.9,
		l2_penalty=5e-4,
		pre_net_class=hebbmodel.model.Net,
		pre_net_mdl_path=P.PROJECT_ROOT + '/results/hebb/config_base/save/model0.pt',
		pre_net_out=hebbmodel.model.Net.BN3
	),
	
	Configuration(
		config_family=P.CONFIG_FAMILY_GDES,
		config_name='fc_on_hebb_conv4',
		net_class=basemodel.fc.Net,
		batch_size=64,
		num_epochs=20,
		iteration_ids=[0],
		val_set_split=40000,
		augment_data=False,
		whiten_data=False,
		learning_rate=1e-3,
		lr_decay=0.5,
		milestones=range(10, 20),
		momentum=0.9,
		l2_penalty=5e-4,
		pre_net_class=hebbmodel.model.Net,
		pre_net_mdl_path=P.PROJECT_ROOT + '/results/hebb/config_base/save/model0.pt',
		pre_net_out=hebbmodel.model.Net.BN4
	),
	
	Configuration(
		config_family=P.CONFIG_FAMILY_GDES,
		config_name='fc_on_hebb_fc5',
		net_class=basemodel.fc.Net,
		batch_size=64,
		num_epochs=20,
		iteration_ids=[0],
		val_set_split=40000,
		augment_data=False,
		whiten_data=False,
		learning_rate=1e-3,
		lr_decay=0.5,
		milestones=range(10, 20),
		momentum=0.9,
		l2_penalty=5e-4,
		pre_net_class=hebbmodel.model.Net,
		pre_net_mdl_path=P.PROJECT_ROOT + '/results/hebb/config_base/save/model0.pt',
		pre_net_out=hebbmodel.model.Net.BN5
	),
	
	Configuration(
		config_family=P.CONFIG_FAMILY_GDES,
		config_name='top1',
		net_class=basemodel.top1.Net,
		batch_size=64,
		num_epochs=20,
		iteration_ids=[0],
		val_set_split=40000,
		augment_data=False,
		whiten_data=False,
		learning_rate=1e-3,
		lr_decay=0.5,
		milestones=range(10, 20),
		momentum=0.9,
		l2_penalty=7e-2,
		pre_net_class=hebbmodel.model.Net,
		pre_net_mdl_path=P.PROJECT_ROOT + '/results/hebb/config_base/save/model0.pt',
		pre_net_out=hebbmodel.model.Net.BN1
	),
	
	Configuration(
		config_family=P.CONFIG_FAMILY_GDES,
		config_name='top2',
		net_class=basemodel.top2.Net,
		batch_size=64,
		num_epochs=20,
		iteration_ids=[0],
		val_set_split=40000,
		augment_data=False,
		whiten_data=False,
		learning_rate=1e-3,
		lr_decay=0.5,
		milestones=range(10, 20),
		momentum=0.9,
		l2_penalty=6e-2,
		pre_net_class=hebbmodel.model.Net,
		pre_net_mdl_path=P.PROJECT_ROOT + '/results/hebb/config_base/save/model0.pt',
		pre_net_out=hebbmodel.model.Net.BN2
	),
	
	Configuration(
		config_family=P.CONFIG_FAMILY_GDES,
		config_name='top3',
		net_class=basemodel.top3.Net,
		batch_size=64,
		num_epochs=20,
		iteration_ids=[0],
		val_set_split=40000,
		augment_data=False,
		whiten_data=False,
		learning_rate=1e-3,
		lr_decay=0.5,
		milestones=range(10, 20),
		momentum=0.9,
		l2_penalty=5e-2,
		pre_net_class=hebbmodel.model.Net,
		pre_net_mdl_path=P.PROJECT_ROOT + '/results/hebb/config_base/save/model0.pt',
		pre_net_out=hebbmodel.model.Net.BN3
	),
	
	Configuration(
		config_family=P.CONFIG_FAMILY_GDES,
		config_name='top4',
		net_class=basemodel.top4.Net,
		batch_size=64,
		num_epochs=20,
		iteration_ids=[0],
		val_set_split=40000,
		augment_data=False,
		whiten_data=False,
		learning_rate=1e-3,
		lr_decay=0.5,
		milestones=range(10, 20),
		momentum=0.9,
		l2_penalty=5e-4,
		pre_net_class=hebbmodel.model.Net,
		pre_net_mdl_path=P.PROJECT_ROOT + '/results/hebb/config_base/save/model0.pt',
		pre_net_out=hebbmodel.model.Net.BN4
	),
	
	################################################################################################################
	####										HEBB CONFIGURATIONS												####
	################################################################################################################

	Configuration(
		config_family=P.CONFIG_FAMILY_HEBB,
		config_name='config_base',
		net_class=hebbmodel.model.Net,
		batch_size=64,
		num_epochs=20,
		iteration_ids=[0],
		val_set_split=40000,
		augment_data=False,
		whiten_data=False,
		learning_rate=1e-3,
		deep_teacher_signal=True
	),

	################################################################################################################
	####					CONFIGS: HEBB CLASSIFIER ON FEATURES EXTRACTED FROM GDES LAYERS 					####
	################################################################################################################

	Configuration(
		config_family=P.CONFIG_FAMILY_HEBB,
		config_name='fc_on_gdes_fc5', # Val: 85.76, Test:84.88
		net_class=hebbmodel.fc.Net,
		batch_size=64,
		num_epochs=2,
		iteration_ids=[0],
		val_set_split=40000,
		augment_data=False,
		whiten_data=False,
		pre_net_class=basemodel.model.Net,
		pre_net_mdl_path=P.PROJECT_ROOT + '/results/gdes/config_base/save/model0.pt',
		pre_net_out=basemodel.model.Net.BN5,
		learning_rate=1e-1,
	),
	
	Configuration(
		config_family=P.CONFIG_FAMILY_HEBB,
		config_name='top1',
		net_class=hebbmodel.top1.Net,
		batch_size=64,
		num_epochs=20,
		iteration_ids=[0],
		val_set_split=40000,
		augment_data=False,
		whiten_data=False,
		pre_net_class=basemodel.model1.Net,
		pre_net_mdl_path=P.PROJECT_ROOT + '/results/gdes/config_1l/save/model0.pt',
		pre_net_out=basemodel.model1.Net.BN1,
		learning_rate=1e-3,
		deep_teacher_signal=True,
	),
	
	Configuration(
		config_family=P.CONFIG_FAMILY_HEBB,
		config_name='top2',
		net_class=hebbmodel.top2.Net,
		batch_size=64,
		num_epochs=20,
		iteration_ids=[0],
		val_set_split=40000,
		augment_data=False,
		whiten_data=False,
		pre_net_class=basemodel.model2.Net,
		pre_net_mdl_path=P.PROJECT_ROOT + '/results/gdes/config_2l/save/model0.pt',
		pre_net_out=basemodel.model2.Net.BN2,
		learning_rate=1e-3,
		deep_teacher_signal=True,
	),
	
	Configuration(
		config_family=P.CONFIG_FAMILY_HEBB,
		config_name='top3',
		net_class=hebbmodel.top3.Net,
		batch_size=64,
		num_epochs=20,
		iteration_ids=[0],
		val_set_split=40000,
		augment_data=False,
		whiten_data=False,
		pre_net_class=basemodel.model3.Net,
		pre_net_mdl_path=P.PROJECT_ROOT + '/results/gdes/config_3l/save/model0.pt',
		pre_net_out=basemodel.model3.Net.BN3,
		learning_rate=1e-3,
		deep_teacher_signal=True,
	),
	
	Configuration(
		config_family=P.CONFIG_FAMILY_HEBB,
		config_name='top4',
		net_class=hebbmodel.top4.Net,
		batch_size=64,
		num_epochs=20,
		iteration_ids=[0],
		val_set_split=40000,
		augment_data=False,
		whiten_data=False,
		pre_net_class=basemodel.model4.Net,
		pre_net_mdl_path=P.PROJECT_ROOT + '/results/gdes/config_4l/save/model0.pt',
		pre_net_out=basemodel.model4.Net.BN4,
		learning_rate=1e-3,
		deep_teacher_signal=True,
	),
	
	################################################################################################################
	####											CONFIGS G-H-G 												####
	################################################################################################################
	
	Configuration(
		config_family=P.CONFIG_FAMILY_GDES,
		config_name='g1h2g3_6',
		net_class=basemodel.top2.Net,
		batch_size=64,
		num_epochs=20,
		iteration_ids=[0],
		val_set_split=40000,
		augment_data=False,
		whiten_data=False,
		learning_rate=1e-3,
		lr_decay=0.5,
		milestones=range(10, 20),
		momentum=0.9,
		l2_penalty=6e-2,
		pre_net_class=hebbmodel.g1h2_6.Net,
		pre_net_mdl_path=P.PROJECT_ROOT + '/results/hebb/g1_2h3_6/save/model0.pt', # Path to non-exising file, because in this case the model is already loaded in the constructor of the pre-net
		pre_net_out=hebbmodel.g1h2_6.Net.BN2
	),
	
	Configuration(
		config_family=P.CONFIG_FAMILY_GDES,
		config_name='g1_2h3g4_6',
		net_class=basemodel.top3.Net,
		batch_size=64,
		num_epochs=20,
		iteration_ids=[0],
		val_set_split=40000,
		augment_data=False,
		whiten_data=False,
		learning_rate=1e-3,
		lr_decay=0.5,
		milestones=range(10, 20),
		momentum=0.9,
		l2_penalty=5e-2,
		pre_net_class=hebbmodel.g1_2h3_6.Net,
		pre_net_mdl_path=P.PROJECT_ROOT + '/results/hebb/g1_2h3_6/save/model0.pt', # Path to non-exising file, because in this case the model is already loaded in the constructor of the pre-net
		pre_net_out=hebbmodel.g1_2h3_6.Net.BN3
	),
	
	Configuration(
		config_family=P.CONFIG_FAMILY_GDES,
		config_name='g1_3h4g5_6',
		net_class=basemodel.top4.Net,
		batch_size=64,
		num_epochs=20,
		iteration_ids=[0],
		val_set_split=40000,
		augment_data=False,
		whiten_data=False,
		learning_rate=1e-3,
		lr_decay=0.5,
		milestones=range(10, 20),
		momentum=0.9,
		l2_penalty=5e-4,
		pre_net_class=hebbmodel.g1_3h4_6.Net,
		pre_net_mdl_path=P.PROJECT_ROOT + '/results/hebb/g1_3h4_6/save/model0.pt', # Path to non-exising file, because in this case the model is already loaded in the constructor of the pre-net
		pre_net_out=hebbmodel.g1_3h4_6.Net.BN4
	),
	
	Configuration(
		config_family=P.CONFIG_FAMILY_GDES,
		config_name='g1_4h5g6',
		net_class=basemodel.fc.Net,
		batch_size=64,
		num_epochs=20,
		iteration_ids=[0],
		val_set_split=40000,
		augment_data=False,
		whiten_data=False,
		learning_rate=1e-3,
		lr_decay=0.5,
		milestones=range(10, 20),
		momentum=0.9,
		l2_penalty=5e-4,
		pre_net_class=hebbmodel.g1_4h5_6.Net,
		pre_net_mdl_path=P.PROJECT_ROOT + '/results/hebb/g1_4h5_6/save/model0.pt', # Path to non-exising file, because in this case the model is already loaded in the constructor of the pre-net
		pre_net_out=hebbmodel.g1_4h5_6.Net.BN5
	),
	
	Configuration(
		config_family=P.CONFIG_FAMILY_GDES,
		config_name='g1h2_3g4_6',
		net_class=basemodel.top3.Net,
		batch_size=64,
		num_epochs=20,
		iteration_ids=[0],
		val_set_split=40000,
		augment_data=False,
		whiten_data=False,
		learning_rate=1e-3,
		lr_decay=0.5,
		milestones=range(10, 20),
		momentum=0.9,
		l2_penalty=5e-2,
		pre_net_class=hebbmodel.g1h2_6.Net,
		pre_net_mdl_path=P.PROJECT_ROOT + '/results/hebb/g1h2_6/save/model0.pt', # Path to non-exising file, because in this case the model is already loaded in the constructor of the pre-net
		pre_net_out=hebbmodel.g1h2_6.Net.BN3
	),
	
	Configuration(
		config_family=P.CONFIG_FAMILY_GDES,
		config_name='g1_2h3_4g5_6',
		net_class=basemodel.top4.Net,
		batch_size=64,
		num_epochs=20,
		iteration_ids=[0],
		val_set_split=40000,
		augment_data=False,
		whiten_data=False,
		learning_rate=1e-3,
		lr_decay=0.5,
		milestones=range(10, 20),
		momentum=0.9,
		l2_penalty=5e-4,
		pre_net_class=hebbmodel.g1_2h3_6.Net,
		pre_net_mdl_path=P.PROJECT_ROOT + '/results/hebb/g1_2h3_6/save/model0.pt', # Path to non-exising file, because in this case the model is already loaded in the constructor of the pre-net
		pre_net_out=hebbmodel.g1_2h3_6.Net.BN4
	),
	
	Configuration(
		config_family=P.CONFIG_FAMILY_GDES,
		config_name='g1_3h4_5g6',
		net_class=basemodel.fc.Net,
		batch_size=64,
		num_epochs=20,
		iteration_ids=[0],
		val_set_split=40000,
		augment_data=False,
		whiten_data=False,
		learning_rate=1e-3,
		lr_decay=0.5,
		milestones=range(10, 20),
		momentum=0.9,
		l2_penalty=5e-4,
		pre_net_class=hebbmodel.g1_3h4_6.Net,
		pre_net_mdl_path=P.PROJECT_ROOT + '/results/hebb/g1_3h4_6/save/model0.pt', # Path to non-exising file, because in this case the model is already loaded in the constructor of the pre-net
		pre_net_out=hebbmodel.g1_3h4_6.Net.BN5
	),
	
	Configuration(
		config_family=P.CONFIG_FAMILY_GDES,
		config_name='g1h2_4g5_6',
		net_class=basemodel.top4.Net,
		batch_size=64,
		num_epochs=20,
		iteration_ids=[0],
		val_set_split=40000,
		augment_data=False,
		whiten_data=False,
		learning_rate=1e-3,
		lr_decay=0.5,
		milestones=range(10, 20),
		momentum=0.9,
		l2_penalty=5e-4,
		pre_net_class=hebbmodel.g1h2_6.Net,
		pre_net_mdl_path=P.PROJECT_ROOT + '/results/hebb/g1h2_6/save/model0.pt', # Path to non-exising file, because in this case the model is already loaded in the constructor of the pre-net
		pre_net_out=hebbmodel.g1h2_6.Net.BN4
	),
	
	Configuration(
		config_family=P.CONFIG_FAMILY_GDES,
		config_name='g1_2h3_5g6',
		net_class=basemodel.fc.Net,
		batch_size=64,
		num_epochs=20,
		iteration_ids=[0],
		val_set_split=40000,
		augment_data=False,
		whiten_data=False,
		learning_rate=1e-3,
		lr_decay=0.5,
		milestones=range(10, 20),
		momentum=0.9,
		l2_penalty=5e-4,
		pre_net_class=hebbmodel.g1_2h3_6.Net,
		pre_net_mdl_path=P.PROJECT_ROOT + '/results/hebb/g1_2h3_6/save/model0.pt', # Path to non-exising file, because in this case the model is already loaded in the constructor of the pre-net
		pre_net_out=hebbmodel.g1_2h3_6.Net.BN5
	),
	
	Configuration(
		config_family=P.CONFIG_FAMILY_GDES,
		config_name='g1h2_5g6',
		net_class=basemodel.fc.Net,
		batch_size=64,
		num_epochs=20,
		iteration_ids=[0],
		val_set_split=40000,
		augment_data=False,
		whiten_data=False,
		learning_rate=1e-3,
		lr_decay=0.5,
		milestones=range(10, 20),
		momentum=0.9,
		l2_penalty=5e-4,
		pre_net_class=hebbmodel.g1h2_6.Net,
		pre_net_mdl_path=P.PROJECT_ROOT + '/results/hebb/g1h2_6/save/model0.pt', # Path to non-exising file, because in this case the model is already loaded in the constructor of the pre-net
		pre_net_out=hebbmodel.g1h2_6.Net.BN5
	),
	
]


CONFIGURATIONS = {}
for c in CONFIG_LIST: CONFIGURATIONS[c.CONFIG_ID] = c
