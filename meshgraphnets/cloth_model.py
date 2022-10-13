# pylint: disable=g-bad-file-header
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
"""Model for FlagSimple."""

import tensorflow as tf

from meshgraphnets import common
from meshgraphnets import core_model
from meshgraphnets import normalization
from meshgraphnets.common import NodeType


class Model(core_model.BaseModel):
  """Model for static cloth simulation."""

  def __init__(self, learned_model, log_dir=None, save_summaries_steps=10000, name='Model'):
    super(Model, self).__init__(name=name, log_dir=log_dir, save_summaries_steps=save_summaries_steps)
    self._learned_model = learned_model
    self._output_normalizer = normalization.Normalizer(
        size=3, name=f'{self.name}/output_normalizer')
    self._node_normalizer = normalization.Normalizer(
        size=3+common.NodeType.SIZE, name=f'{self.name}/node_normalizer')
    self._edge_normalizer = normalization.Normalizer(
        size=7, name=f'{self.name}/edge_normalizer')  # 2D coord + 3D coord + 2*length = 7

  def _build_graph(self, inputs, is_training):
    """Builds input graph."""
    # construct graph nodes
    velocity = inputs['world_pos'] - inputs['prev|world_pos']
    node_type = tf.one_hot(inputs['node_type'][:, 0], common.NodeType.SIZE)
    node_features = tf.concat([velocity, node_type], axis=-1)

    # construct graph edges
    senders, receivers = common.triangles_to_edges(inputs['cells'])
    relative_world_pos = (tf.gather(inputs['world_pos'], senders) -
                          tf.gather(inputs['world_pos'], receivers))
    relative_mesh_pos = (tf.gather(inputs['mesh_pos'], senders) -
                         tf.gather(inputs['mesh_pos'], receivers))
    edge_features = tf.concat([
        relative_world_pos,
        tf.norm(relative_world_pos, axis=-1, keepdims=True),
        relative_mesh_pos,
        tf.norm(relative_mesh_pos, axis=-1, keepdims=True)], axis=-1)

    mesh_edges = core_model.EdgeSet(
        name='mesh_edges',
        features=self._edge_normalizer(edge_features, is_training),
        receivers=receivers,
        senders=senders)
    return core_model.MultiGraph(
        node_features=self._node_normalizer(node_features, is_training),
        edge_sets=[mesh_edges])

  def predict_step(self, inputs):
    graph = self._build_graph(inputs, is_training=False)
    per_node_network_output = self._learned_model(graph, training=False)
    return self._update(inputs, per_node_network_output)

  def loss(self, inputs):
    # TODO: add option for validation
    """L2 loss on position."""
    graph = self._build_graph(inputs, is_training=True)
    network_output = self._learned_model(graph, training=True)

    # build target acceleration
    cur_position = inputs['world_pos']
    prev_position = inputs['prev|world_pos']
    target_position = inputs['target|world_pos']
    target_acceleration = target_position - 2*cur_position + prev_position
    target_normalized = self._output_normalizer(target_acceleration)

    # build loss
    loss_mask = tf.equal(inputs['node_type'][:, 0], common.NodeType.NORMAL)
    error = tf.reduce_sum((target_normalized - network_output)**2, axis=1)
    loss = tf.reduce_mean(error[loss_mask])
    return loss

  def _update(self, inputs, per_node_network_output):
    """Integrate model outputs."""
    acceleration = self._output_normalizer.inverse(per_node_network_output)
    # integrate forward
    cur_position = inputs['world_pos']
    prev_position = inputs['prev|world_pos']
    position = 2*cur_position + acceleration - prev_position
    return position

  @tf.function
  def rollout(self, initial_state, num_steps):
    """Rolls out a model trajectory."""
    mask = tf.equal(initial_state['node_type']#[:, 0]
                    , NodeType.NORMAL)

    def step_fn(step, prev_pos, cur_pos, trajectory):
      prediction = self.predict_step({**initial_state,
                          'prev|world_pos': prev_pos,
                          'world_pos': cur_pos})
      # don't update kinematic nodes
      next_pos = tf.where(mask, prediction, cur_pos)
      trajectory = trajectory.write(step, cur_pos)
      return step+1, cur_pos, next_pos, trajectory

    _, _, _, output = tf.while_loop(
        cond=lambda step, last, cur, traj: tf.less(step, num_steps),
        body=step_fn,
        loop_vars=(0, initial_state['prev|world_pos'], initial_state['world_pos'],
                  tf.TensorArray(tf.float32, num_steps)),
        parallel_iterations=1)

    return output.stack()
