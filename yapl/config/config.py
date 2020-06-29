import os
import yaml
import yapl

class Config:
    '''
        Extend this Base class to avoid Re-writing most of the things

        example:
            class config(Config):
                def __init__(self):
                    super().__init__()
                    self.EXPERIMENT = {
                        'NAME':"Exp1",
                        'DATE' : datetime.datetime.now(),
                    }
                    self.DATA_FILE = '../hsjdhfs/sdfjsk.zip'
                    self.DEVICE = 'cpu'

            config = config()
            config.DEVICE
            config.BATCHSIZE
    '''

    def __init__(self):
        # General Experiment Params
        self.EXPERIMENT_NAME = 'img_class_resner50_trail25'
        self.DATATYPE = 'img' #img: if images; text: if textual; tabular: if csv tabular
        self.PROBLEM_TYPE = 'classification' #classification, regression
        self.TRAIN_CSV = '../input/<dataset-name>/train.csv'
        self.TEST_CSV = '../input/<dataset-name>/test.csv'
        self.VALIDATION_CSV = ""
        
        self.TOTAL_TRAIN_IMG = 0
        self.TOTAL_TEST_IMG = 0
        
        #Data specific information
        self.INPUT_SHAPE = (128,128,3) #exam: image shape
        self.NUM_CLASSES = 1 #one for binary classfication and regreesion
        self.IMG_SIZE = [1024, 1024]
        self.IMG_SHAPE = (512, 512, 3)

        #model specific
        self.DO_FINETUNE = True
        
        
        #Training information
        self.BATCH_SIZE = 8
        self.BUFFER_SIZE = 100
        self.EPOCHES = 10 
        
        # self.LOSS = tf.keras.losses.BinaryCrossentropy()
        # self.OPTIMIZER = tf.keras.optimizers.Adam(learning_rate=0.01)
        self.ACCURACY = ['accuracy']
        
        self.STRATEGY = None
        
        self.LOG_DIR = './log'
        self.CHECKPOINT_DIR = './log/checkpoint/cp.cpkt'


    def dumpconfig(self, location, toyml=False):
        if yapl.config.backend == 'torch':
            LOSS_PARAM = {}
            for key, values in  yapl.config.LOSS.__dict__.items():
                try:
                    LOSS_PARAM[key] = dict(values)
                except:
                    LOSS_PARAM[key] = values
            
            yapl.config.LOSS = LOSS_PARAM

        if toyml:
            with open(location, 'w') as dumpfile:
                yaml.dump(self.__dict__, dumpfile)
        else:
            raise Exception("yet to implement")

        return "you file has successfully saved"

    def make_global(self):
        yapl.config = self

    def setupTPU(self):
        if yapl.backend == 'tf':
            try:
                tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
                print('Running on TPU ', tpu.master())
            except ValueError:
                tpu = None

            if tpu:
                tf.config.experimental_connect_to_cluster(tpu)
                tf.tpu.experimental.initialize_tpu_system(tpu)
                strategy = tf.distribute.experimental.TPUStrategy(tpu)
                self.STRATEGY = strategy
                self.BATCH_SIZE = 32 * strategy.num_replicas_in_sync 
            else:
                strategy = tf.distribute.get_strategy() 

        if yapl.backend == 'torch':
            # write some thing for assertion
            try:
                os.system('curl https://raw.githubusercontent.com/pytorch/xla/master/contrib/scripts/env-setup.py -o pytorch-xla-env-setup.py')
                os.system('python pytorch-xla-env-setup.py --version nightly --apt-packages libomp5 libopenblas-dev')

                print("ADD THESE IMPORTS IF NOT ALREADY")
                print("import torch_xla.core.xla_model as xm\nimport torch_xla.distributed.parallel_loader as pl\nimport torch_xla.distributed.xla_multiprocessing as xmp")
                import torch_xla.core.xla_model as xm

                self.DEVICE = xm.xla_device()
                self.STRATEGY = tpu
            except:
                self.STRATEGY = None