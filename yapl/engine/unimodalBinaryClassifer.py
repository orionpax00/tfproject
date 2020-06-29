import torch
from torch.utils.tensorboard import SummaryWriter
import tensorflow as tf

import pandas as pd
import numpy as np

import yapl

from yapl.utils.accuracy import AverageBinaryAccuracyTorch
from yapl.utils.loss import AverageLossTorch

try:
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.parallel_loader as pl
    _is_xla = True
except ImportError:
    _is_xla = False

def reduce_fn(vals):
    return sum(vals) / len(vals)


class Engine:
    def __init__(self, model, traindataloader, valdataloder = None, testdataloader=None, logger=None):
        self.config = yapl.config
        self.history = {
            'LOSS' : [],
            'ACCURACY' : [],
            'VAL_LOSS' : []
            }
        self.history_logger = None
        self.writer = None
        if logger != None:
            self._activate_logger(logger['loggers'], logger['folder_name'])
            self.writer.add_graph(model)
        self.predictions = None

    def loop(self):
        if yapl.backend == 'tf':
            for epoch in range(self.config.EPOCHES):
                epoch_loss_avg = tf.keras.metrics.Mean()
                epoch_accuracy_avg = tf.keras.metrics.AUC()

                for it, (data_batch, label_batch) in enumerate(traindataloader):
                    with tf.GradientTape() as tape:
                        output = model(data_batch, training=True)
                        losses = self.config.LOSS(y_true = label_batch, y_pred=output)
                        grads = tape.gradient(losses, model.trainable_variables)
                        self.config.OPTIMIZER.apply_gradients(zip(grads, model.trainable_variables))
                        epoch_loss_avg.update_state(losses)
                        epoch_accuracy_avg.update_state(label_batch, tf.squeeze(output))

                print('{} - Loss: {} | Accuracy: {}'.format(epoch, epoch_loss_avg.result(), epoch_accuracy_avg.result()))
                self.history['LOSS'].append(epoch_loss_avg.result())
                self.history['ACCURACY'].append(epoch_accuracy_avg.result())
            
                if valdataloder != None:
                    # Do validation Loop

            if testdataloader != None:
                # Do prediction

        elif yapl.backend == 'torch':
            if yapl.config.STRATEGY != None and _is_xla == False :
                raise Exception("TPUs are not setup properly for TPU training")
                
            if yapl.config.STRATEGY != None and _is_xla == True:
                traindataloader = pl.ParallelLoader(traindataloader, [yapl.config.DEVICE])

            model.train()
            for epoch in range(self.config.EPOCHES):
                loss_avg = AverageLossTorch()
                for (data_batch, label_batch) in traindataloader:
                    data_batch = data_batch.to(self.config.DEVICE, dtype=torch.float)
                    label_batch = label_batch.to(self.config.DEVICE, dtype=torch.float)
                    
                    self.config.OPTIMIZER.zero_grad()
                    output = model(data_batch)
                    losses = self.config.LOSS(output, label_batch.unsqueeze(1))

                    if yapl.config.STRATEGY != None and _is_xla == True: 
                        losses.backward()
                        xm.optimizer_step(self.config.OPTIMIZER)

                        reduced_loss = xm.mesh_reduce('loss_reduce', losses, reduce_fn)
                        loss_avg.update(reduced_loss.item(), traindataloader.batch_size)

                    else:
                        losses.backward()
                        self.config.OPTIMIZER.step()
                        loss_avg.update(losses.item(), traindataloader.batch_size)

                    # TODO: implement AUC metrics
                    
                    del data_batch
                    del label_batch
                    
                print("{} - LOSS: {}".format(epoch, loss_avg.avg))
                self.writer.add_scalar('Loss/train', loss_avg.avg, epoch)
                self.history['LOSS'].append(loss_avg.avg)

                if valdataloder != None:
                    model.eval()
                    loss_avg = AverageLossTorch()
                    with torch.no_grad():
                        if yapl.config.STRATEGY != None and _is_xla == True:
                            traindataloader = pl.ParallelLoader(traindataloader, [yapl.config.DEVICE])
                        
                        for (data_batch, label_batch) in valdataloder:
                            data_batch = data_batch.to(self.config.DEVICE, dtype=torch.float)
                            label_batch = label_batch.to(self.config.DEVICE, dtype=torch.float)
                            
                            output = model(data_batch)
                            losses = self.config.LOSS(output, label_batch.unsqueeze(1))

                            if yapl.config.STRATEGY != None and _is_xla == True: 
                                reduced_loss = xm.mesh_reduce('loss_reduce', losses, reduce_fn)
                                loss_avg.update(reduced_loss.item(), traindataloader.batch_size)

                            else:
                                loss_avg.update(losses.item(), traindataloader.batch_size)

                            # TODO: implement AUC metrics
                            
                            del data_batch
                            del label_batch
                            
                        print("{} - VAL LOSS: {}".format(epoch, loss_avg.avg))
                        self.writer.add_scalar('Loss/Val', loss_avg.avg, epoch)
                        self.history['VAL_LOSS'].append(loss_avg.avg)

            if testdataloader != None:
                predictions = []
                ids = []
                model.eval()
                with torch.no_grad():
                    for data_batch, id_batch in traindataloader:
                        data_batch = data_batch.to(self.config.DEVICE, dtype=torch.float)
                        pred = model(data_batch)
                        predictions.append(pred.cpu().numpy())
                        ids.append(id_batch.numpy())

                self.predictions = {
                    "id" : np.stack(ids),
                    "predictions" : np.stack(predictions)
                }
                
        if self.writer != None:
            self.writer.close()
        if self.history_logger != None:
            pd.DataFrame({self.history}).to_csv(self.history_logger, index=False)

    def fit(self, istraining = True):
        if yapl.backend == 'tf':
            model.compile(
                optimizer=self.config.OPTIMIZER, 
                loss=self.config.LOSS, 
                metrics=self.config.ACCURACY
            )
            history = model.fit(
                self.dataloader, 
                epochs=self.config.EPOCHES, 
                steps_per_epoch=(self.config.TOTAL_TRAIN_IMG//self.config.BATCH_SIZE),
                # TODO: callbacks
            )

            return history
        else:
            raise Exception('Fit_engine is only available for Tensorflow')
        
        self.writer.close()

    
    def _activate_logger(self, loggers, folder_loc):
        LOGGERS_AVL = ['tensorboard', 'checkpoint', 'history']
        if yapl.config.STRATEGY != None and _is_xla == True:
            raise Exception("Logger in not yet implemented for TPU Training")

        for x in loggers:
            if x is not in LOGGERS_AVL:
                raise Exception("{} is not Not Available | please checkout for typo".format(x))
        
        for logger in loggers:
            if logger == 'tensorboard':
                self.writer = SummaryWriter(folder_loc, comment= "___" + yapl.config.EXPERIMENT_NAME)
        
        for logger in loggers:
            if logger == 'history':
                self.history_logger = folder_loc+'/history.csv'

