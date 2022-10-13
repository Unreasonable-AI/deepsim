# pylint: disable=g-bad-file-header
# Copyright 2022 Riksi
# Copyright 2020 DeepMind Technologies Limited. All Rights Reserved.
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
"""Functions to build evaluation metrics for cloth data."""

import tensorflow as tf

from meshgraphnets.common import NodeType

def evaluate(model, inputs):
    """Performs model rollouts and create stats."""
    initial_state = {k: v[0] for k, v in inputs.items()}
    num_steps = inputs['cells'].shape[0]
    prediction = model.rollout(initial_state, num_steps)

    error = tf.reduce_mean((prediction - inputs['world_pos'])**2, axis=-1)
    scalars = {'mse_%d_steps' % horizon: tf.reduce_mean(error[1:horizon+1])
                for horizon in [1, 10, 20, 50, 100, 200]}
    scalars['mse_all_steps'] = tf.reduce_mean(error[1:])
    mse_keys = list(scalars.keys())
    for key in mse_keys:
        scalars[f'r{key}'] = scalars[key] ** 0.5

    traj_ops = {
        'faces': inputs['cells'],
        'mesh_pos': inputs['mesh_pos'],
        'gt_pos': inputs['world_pos'],
        'pred_pos': prediction
    }
    return scalars, traj_ops
