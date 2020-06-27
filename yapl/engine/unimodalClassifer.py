import torch
import tensorflow as tf

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
    def __init__(self, model, traindataloader, valdataloder = None, testdataloader=None):
        self.model = model
        self.traindataloader = traindataloader
        self.config = yapl.config

    def loop(self):
        if yapl.backend == 'tf':
            history = []
            for epoch in range(self.config.EPOCHES):
                epoch_loss_avg = tf.keras.metrics.Mean()
                epoch_accuracy_avg = tf.keras.metrics.AUC()

                for it, (data_batch, label_batch) in enumerate(self.traindataloader):
                    with tf.GradientTape() as tape:
                        output = self.model(data_batch, training=True)
                        losses = self.config.LOSS(y_true = label_batch, y_pred=output)
                        grads = tape.gradient(losses, model.trainable_variables)
                        self.config.OPTIMIZER.apply_gradients(zip(grads, model.trainable_variables))
                        epoch_loss_avg.update_state(losses)
                        epoch_accuracy_avg.update_state(label_batch, tf.squeeze(output))

                print('{} - Loss: {} | Accuracy: {}'.format(epoch, epoch_loss_avg.result(), epoch_accuracy_avg.result()))
                history.append((epoch_loss_avg.result(), epoch_accuracy_avg.result()))
            
                if valdataloder != None:
                    # Do validation Loop

            if testdataloader != None:
                # Do prediction

        elif yapl.backend == 'torch':
            if yapl.config.STRATEGY != None and _is_xla == False :
                raise Exception("TPUs are not setup properly for TPU training")
                
            if yapl.config.STRATEGY != None and _is_xla == True:
                traindataloader = pl.ParallelLoader(self.traindataloader, [yapl.config.DEVICE])

            history = []
            self.model.train()
            for epoch in self.config.EPOCHES:
                loss_avg = AverageLossTorch()
                for (data_batch, label_batch) in self.traindataloader:
                    data_batch = data_batch.to(self.config.DEVICE, dtype=torch.float)
                    label_batch = label_batch.to(self.config.DEVICE, dtype=torch.float)
                    
                    self.config.OPTIMIZER.zero_grad()
                    output = self.model(data_batch)
                    losses = self.config.LOSS(output, label_batch.unsqueeze(1))

                    if yapl.config.STRATEGY != None and _is_xla == True: 
                        losses.backward()
                        xm.optimizer_step(self.config.OPTIMIZER)

                        reduced_loss = xm.mesh_reduce('loss_reduce', losses, reduce_fn)
                        loss_avg.update(reduced_loss.item(), self.traindataloader.batch_size)

                    else:
                        losses.backward()
                        self.config.OPTIMIZER.step()
                        loss_avg.update(losses.item(), self.traindataloader.batch_size)

                    # TODO: implement AUC metrics
                    
                    del data_batch
                    del label_batch
                    
                print("{} - LOSS: {}".format(epoch, loss_avg.avg))
                history.append(loss_avg.avg)

                if valdataloder != None:
                # Do validation Loop

            if testdataloader != None:
                # Do prediction


    def fit(self, istraining = True):
        if yapl.backend == 'tf':
            self.model.compile(
                optimizer=self.config.OPTIMIZER, 
                loss=self.config.LOSS, 
                metrics=self.config.ACCURACY
            )
            history = self.model.fit(
                self.dataloader, 
                epochs=self.config.EPOCHES, 
                steps_per_epoch=(self.config.TOTAL_TRAIN_IMG//self.config.BATCH_SIZE),
                # TODO: callbacks
            )

            return history
        else:
            raise Exception('Fit_engine is only available for Tensorflow')


