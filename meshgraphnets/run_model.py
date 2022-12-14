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
flags.DEFINE_string('checkpoint_dir', None, 'Directory to save checkpoint')
flags.DEFINE_string('dataset_dir', None, 'Directory to load dataset from.')
flags.DEFINE_string('rollout_path', None,
                    'Pickle file to save eval trajectories')
flags.DEFINE_enum('rollout_split', 'valid', ['train', 'test', 'valid'],
                  'Dataset split to use for rollouts.')
flags.DEFINE_integer('num_rollouts', 10, 'No. of rollout trajectories')
flags.DEFINE_integer('num_training_steps', int(10e6), 'No. of training steps')
flags.DEFINE_integer('save_summaries_steps', 10000, 'Save summaries every N steps')

PARAMETERS = {
    # 'cfd': dict(noise=0.02, gamma=1.0, field='velocity', history=False,
    #             size=2, batch=2, model=cfd_model, evaluator=cfd_eval),
    'cloth': dict(noise=0.003, gamma=0.1, field='world_pos', history=True,
                  size=3, batch=1, model=cloth_model, evaluator=cloth_eval)
}

def learner(model, params) -> None:
    ds = dataset.load_dataset(FLAGS.dataset_dir, 'train')
    ds = dataset.add_targets(ds, [params['field']], add_history=params['history'])
    ds = dataset.split_and_preprocess(ds, noise_field=params['field'],
                                noise_scale=params['noise'],
                                noise_gamma=params['gamma'])
    model.add_optimizer(learning_rate=1e-4,
                        decay_steps=int(5e6),
                        decay_rate=0.1,
                        alpha_final=1e-6)
    start = time.time()
    ckpt = tf.train.Checkpoint(
        model=model,
        optimizer=model.optimizer
    )
    ckpt_manager = tf.train.CheckpointManager(
            ckpt, FLAGS.checkpoint_dir, max_to_keep=10
        )
    save_checkpoint_secs = 600
    for step, inputs in enumerate(ds):
        loss = model.train_step(inputs)
        if step % 1000 == 0:
            logging.info('Step %d: Loss %g', step, loss.numpy())

        if step == (FLAGS.num_training_steps - 1):
            break
        # TODO: maybe save on a different basis
        if ((time.time() - start) % save_checkpoint_secs) == 0:
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

def main(argv):
    del argv
    # TODO: what is this
    # tf.enable_resource_variables()
    params = PARAMETERS[FLAGS.model]
    learned_model = core_model.EncodeProcessDecode(
        output_size=params['size'],
        latent_size=128,
        num_layers=2,
        message_passing_steps=15)
    model = params['model'].Model(learned_model, log_dir=FLAGS.checkpoint_dir,
                                  save_summaries_steps=FLAGS.save_summaries_steps)
    if FLAGS.mode == 'train':
        learner(model, params)
    elif FLAGS.mode == 'eval':
        evaluator(model, params)

if __name__ == '__main__':
  app.run(main)