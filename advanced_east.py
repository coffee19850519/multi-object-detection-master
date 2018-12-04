import os
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam
import keras.layers
import cfg
from network import East
from losses import quad_loss
from data_generator import gen



if __name__ == '__main__':

  os.environ["CUDA_VISIBLE_DEVICES"] = "1"
  east = East()
  east_network = east.east_network()
  east_network.summary()

  for layer in east_network.layers:
    if layer.name not in ['inside_score','side_vertex_code',
                             'side_vertex_coord']:
      layer.trainable = False


  east_network.compile(loss=quad_loss, optimizer=Adam(lr=cfg.lr,
                                                    # clipvalue=cfg.clipvalue,
                                                    decay=cfg.decay))
  if cfg.load_weights and os.path.exists(cfg.load_model_weights_file):
    east_network.load_weights(cfg.load_model_weights_file, by_name=True,
                              skip_mismatch= True)


  east_network.fit_generator(generator=gen(),
                           steps_per_epoch=cfg.steps_per_epoch,
                           epochs=cfg.epoch_num,
                           validation_data=gen(is_val=True),
                           validation_steps=cfg.validation_steps,
                           verbose=1,
                           initial_epoch=cfg.initial_epoch,
                           callbacks=[
                               EarlyStopping(patience=cfg.patience, verbose=1),
                               ModelCheckpoint(
                                   filepath=cfg.saved_checkpoint_weights_file,
                                               save_best_only=True,
                                               save_weights_only=True,
                                               verbose=1)])
  east_network.save(cfg.saved_last_model_file)
  east_network.save_weights(cfg.saved_last_weight_file)
