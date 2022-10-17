# pylint: disable=g-bad-file-header
# Copyright 2022 Riksi
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
import pickle
from absl import app
from absl import flags
from absl import logging
import numpy as np
import time
import tensorflow as tf
# TODO: update the code for these 
# from meshgraphnets import cfd_eval
# from meshgraphnets import cfd_model
from meshgraphnets import cloth_eval
from meshgraphnets import cloth_model
from meshgraphnets import core_model
from meshgraphnets import dataset


FLAGS = flags.FLAGS
flags.DEFINE_enum('mode', 'train', ['train', 'eval'],
                  'Train model, or run evaluation.')
flags.DEFINE_enum('model', None, ['cfd', 'cloth'],
                  'Select model to run.')
flags.DEFINE_string('restore_dir', None, 'Directory from which restore checkpoint')
flags.DEFINE_string('checkpoint_dir', None, 'Directory to save checkpoint')
flags.DEFINE_string('dataset_dir', None, 'Directory to load dataset from.')
flags.DEFINE_string('rollout_path', None,
                    'Pickle file to save eval trajectories')
flags.DEFINE_enum('rollout_split', 'valid', ['train', 'test', 'valid'],
                  'Dataset split to use for rollouts.')
flags.DEFINE_integer('num_rollouts', 10, 'No. of rollout trajectories')
flags.DEFINE_integer('num_training_steps', int(10e6), 'No. of training steps')
flags.DEFINE_integer('save_summaries_steps', 10000, 'Save summaries every N steps')
flags.DEFINE_integer('n_gpus', 1, 'Number of GPUs to use')

PARAMETERS = {
    # 'cfd': dict(noise=0.02, gamma=1.0, field='velocity', history=False,
    #             size=2, batch=2, model=cfd_model, evaluator=cfd_eval),
    'cloth': dict(noise=0.003, gamma=0.1, field='world_pos', history=True,
                  size=3, batch=1, model=cloth_model, evaluator=cloth_eval)
}

def learner(model, params, strategy=None) -> None:
    ds = dataset.load_dataset(FLAGS.dataset_dir, 'train')
    ds = dataset.add_targets(ds, [params['field']], add_history=params['history'])
    ds = dataset.split_and_preprocess(ds, noise_field=params['field'],
                                noise_scale=params['noise'],
                                noise_gamma=params['gamma'])
    n_gpus = get_num_gpus()
    if strategy is not None:
        # TODO: allow for batch size > 1
        ds = ds.batch(n_gpus)

    optim_kwargs = dict(learning_rate=1e-4 * n_gpus,
                        # TODO: adjust decay steps for batch size > 1
                        decay_steps=int(5e6) // n_gpus,
                        decay_rate=0.1,
                        alpha_final=1e-6)

    def _setup_ckpt_and_optim():
        model.add_optimizer(**optim_kwargs)
        ckpt = tf.train.Checkpoint(
                model=model,
                optimizer=model.optimizer
            )
        ckpt_manager = tf.train.CheckpointManager(
                    ckpt, FLAGS.checkpoint_dir, max_to_keep=10
                )
        if FLAGS.restore_dir is not None:
            # Restore latest checkpoint
            logging.info('Restoring from checkpoint', FLAGS.restore_dir)
            ckpt.restore(ckpt_manager.latest_checkpoint)

        return ckpt, ckpt_manager


    if strategy is not None:
        ds = strategy.experimental_distribute_dataset(ds)
        with strategy.scope():
            ckpt, ckpt_manager = _setup_ckpt_and_optim()
            
    else:
        ckpt, ckpt_manager = _setup_ckpt_and_optim()

    start = time.time()
    save_checkpoint_secs = 600
    factor = 1
    # TODO: adjust steps for batch size > 1
    for step, inputs in enumerate(ds):
        if step < (1000 // n_gpus):
            loss = (model.distributed_update_norm(inputs) 
                if strategy is not None else model.update_norm(inputs)
            ) 
        else:
            loss = (model.distributed_train_step(inputs) 
                if strategy is not None else model.train_step(inputs)
            )
        if step % 1000 == 0:
            logging.info('Step %d: Loss %g', step, loss.numpy())

        # TODO: adjust number of steps for batch size > 1
        if step == (FLAGS.num_training_steps // n_gpus - 1):
            break
        # TODO: maybe save on a different basis
        if ((time.time() - start) > (factor * save_checkpoint_secs)):
            logging.info('Saving checkpoint')
            factor += 1
            ckpt_manager.save()
    logging.info('Training complete')

def evaluator(model, params):
    """Run a model rollout trajectory."""
    ds = dataset.load_dataset(FLAGS.dataset_dir, FLAGS.rollout_split)
    ds = dataset.add_targets(ds, [params['field']], add_history=params['history'])

    trajectories = []
    scalars = []

    ckpt = tf.train.Checkpoint(model=model)
    ckpt_manager = tf.train.CheckpointManager(
        ckpt, FLAGS.checkpoint_dir, max_to_keep=10
    )
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)

    for traj_idx, inputs in zip(range(FLAGS.num_rollouts), ds):
        logging.info('Rollout trajectory %d', traj_idx)
        scalar_data, traj_data = params['evaluator'].evaluate(model, inputs)
        trajectories.append(traj_data)
        scalars.append(scalar_data)
    for key in scalars[0]:
        logging.info('%s: %g', key, np.mean([x[key] for x in scalars]))
    with open(FLAGS.rollout_path, 'wb') as fp:
        pickle.dump(trajectories, fp)

def get_model(params):
    learned_model = core_model.EncodeProcessDecode(
        output_size=params['size'],
        latent_size=128,
        num_layers=2,
        message_passing_steps=15)
    model = params['model'].Model(learned_model, log_dir=FLAGS.checkpoint_dir,
                                  # TODO: update 
                                  # global_batch_size, max_accumulations,
                                  # to allow for batch size > 1
                                  global_batch_size=get_num_gpus(),
                                  max_accumulations=10**6 // get_num_gpus(),
                                  save_summaries_steps=FLAGS.save_summaries_steps)
    return model

def get_num_gpus():
    return min(FLAGS.n_gpus, len(tf.config.experimental.list_physical_devices('GPU')))

def main(argv):
    del argv
    n_gpus = get_num_gpus()
    logging.info('Using %d GPUs', n_gpus)

    params = PARAMETERS[FLAGS.model]

    if n_gpus > 1 and FLAGS.mode == 'train':
        # TODO: add distributed eval
        strategy = tf.distribute.MirroredStrategy()
        with strategy.scope():
            model = get_model(params)
    else:
        strategy = None
        model = get_model(params)

    
    if FLAGS.mode == 'train':
        learner(model, params, strategy)

    elif FLAGS.mode == 'eval':
        # TODO: add distributed eval
        evaluator(model, params)

if __name__ == '__main__':
  app.run(main)