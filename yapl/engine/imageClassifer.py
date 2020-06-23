import torch
import tensorflow as tf

import yapl

from yapl.utils.accuracy import AverageBinaryAccuracyTorch
from yapl.utils.loss import AverageLossTorch

class Engine:
    def __init__(self, model, dataloader):
        self.model = model
        self.dataloader = dataloader
        self.config = yapl.config

    def loopengine(self, istraining = True):
        if yapl.backend == 'tf':
            history = []
            for epoch in range(self.config.EPOCHES):
                epoch_loss_avg = tf.keras.metrics.Mean()
                epoch_accuracy_avg = tf.keras.metrics.BinaryAccuracy()

                for it, (data_batch, label_batch) in enumerate(dataloader):
                    with tf.GradientTape() as tape:
                        output = self.model(data_batch, training=True)
                        losses = self.config.LOSS(y_true = label_batch, y_pred=output)
                        grads = tape.gradient(losses, model.trainable_variables)
                        self.config.OPTIMIZER.apply_gradients(zip(grads, model.trainable_variables))
                        epoch_loss_avg.update_state(losses)
                        epoch_accuracy_avg.update_state(label_batch, tf.squeeze(output))

                print('Loss: {} | Accuracy: {}'.format(epoch_loss_avg.result(), epoch_accuracy_avg.result()))
                history.append((epoch_loss_avg.result(), epoch_accuracy_avg.result()))
            
            return history

        elif yapl.backend == 'torch':
            if istraining:
                self.model.train()
                loss_avg = AverageLossTorch()

                for (data_batch, label_batch) in dataloader:
                    print(".", end='')
                    data_batch = data_batch.to(self.config.DEVICE, dtype=torch.float)
                    label_batch = label_batch.to(self.config.DEVICE, dtype=torch.float)
                    
                    self.config.OPTIMIZER.zero_grad()
                    output = self.model(data_batch)
                    
                    losses = self.config.LOSS(output, label_batch.unsqueeze(1))
                    losses.backward()
                    self.config.OPTIMIZER.step()
                    
                    loss_avg.update(losses.item(), self.config.BATCH_SIZE)
                    
                    del data_batch
                    del label_batch
                    
                    torch.cuda.empty_cache()

                return loss_avg.avg


    def fitengine(self, istraining = True):
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
        #         callbacks=[
        #             Callbacks().checkpoint_logger(),
        #             Callbacks().tesorboard_logger()
        #         ]
            )

            return history
        else:
            raise Exception('Fit_engine is only available for Tensorflow')


