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
"""Core learned graph net model."""

import collections
import functools
import tensorflow as tf
from meshgraphnets.normalization import Normalizer

EdgeSet = collections.namedtuple('EdgeSet', ['name', 'features', 'senders',
                                             'receivers'])
MultiGraph = collections.namedtuple('Graph', ['node_features', 'edge_sets'])


class GraphNetBlock(tf.keras.layers.Layer):
  """Multi-Edge Interaction Network with residual connections."""

  def __init__(self, model_fn, edge_set_names=['mesh_edges'], name='GraphNetBlock'):
    super(GraphNetBlock, self).__init__(name=name)
    self.edge_models = {name: model_fn(name=f'{self.name}/{name}_edge_fn') 
                        for name in edge_set_names}
    self.node_model = model_fn(name=f'{self.name}/node_fn')
    self.edge_set_names = edge_set_names

  def _update_edge_features(self, node_features, edge_set, training=None):
    """Aggregrates node features, and applies edge function."""
    sender_features = tf.gather(node_features, edge_set.senders)
    receiver_features = tf.gather(node_features, edge_set.receivers)
    features = [sender_features, receiver_features, edge_set.features]
    return self.edge_models[edge_set.name](tf.concat(features, axis=-1), training=training)

  def _update_node_features(self, node_features, edge_sets, training=None):
    """Aggregrates edge features, and applies node function."""
    num_nodes = tf.shape(node_features)[0]
    features = [node_features]
    for edge_set in edge_sets:
      features.append(tf.math.unsorted_segment_sum(edge_set.features,
                                                   edge_set.receivers,
                                                   num_nodes))
    return self.node_model(tf.concat(features, axis=-1), training=training)

  def call(self, graph, training=None):
    """Applies GraphNetBlock and returns updated MultiGraph."""

    # apply edge functions
    new_edge_sets = []
    for edge_set in graph.edge_sets:
      updated_features = self._update_edge_features(graph.node_features,
                                                    edge_set, training=training)
      new_edge_sets.append(edge_set._replace(features=updated_features))

    # apply node function
    new_node_features = self._update_node_features(graph.node_features,
                                                   new_edge_sets, training=training)

    # add residual connections
    new_node_features += graph.node_features
    new_edge_sets = [es._replace(features=es.features + old_es.features)
                     for es, old_es in zip(new_edge_sets, graph.edge_sets)]
    return MultiGraph(new_node_features, new_edge_sets)


class MLP(tf.keras.models.Model):
    def __init__(self, widths, activate_final=False, name='MLP'):
      super(MLP, self).__init__(name=name)
      self.network = tf.keras.models.Sequential(
      [
          tf.keras.layers.Dense(units=width, 
          activation='relu', name=f'{self.name}/linear_{idx}') 
          for idx, width in enumerate(widths[:-1])
      ]
      + [
          tf.keras.layers.Dense(units=widths[-1], 
          activation='relu' if activate_final else None,
          name=f'{self.name}/linear_{len(widths)-1}')
      ]
  )

    def call(self, inputs, training=None):
        return self.network(inputs, training=training)

class ExponentialDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(
        self, 
        learning_rate=1e-4,
        decay_steps=5e6, 
        decay_rate=0.1,
        alpha_final=1e-6,
      ):
        self.alpha_final = alpha_final
        self.lr = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=learning_rate,
            decay_steps=int(decay_steps),
            decay_rate=decay_rate,
        )

    def get_config(self):
        return dict(alpha_final=self.alpha_final,
                    **self.lr.get_config())

    def __call__(self, step):
        return self.lr(step) + self.alpha_final

class EncodeProcessDecode(tf.keras.models.Model):
  """Encode-Process-Decode GraphNet model."""

  def __init__(self,
               output_size,
               latent_size,
               num_layers,
               message_passing_steps,
               name='EncodeProcessDecode',
               edge_set_names=['mesh_edges']):
    super(EncodeProcessDecode, self).__init__(name=name)
    self._latent_size = latent_size
    self._output_size = output_size
    self._num_layers = num_layers
    self._message_passing_steps = message_passing_steps
    self.node_encoder_mlp = self._make_mlp(self._latent_size, name=f'{self.name}/node_encoder')
    self.edge_encoder_mlps = [
      self._make_mlp(self._latent_size, name=f'{self.name}/{name}_edge_encoder') 
      for name in edge_set_names
    ]
    model_fn = functools.partial(self._make_mlp, output_size=self._latent_size, name=f'{self.name}/decoder')
    self.message_passing_blocks = [
      GraphNetBlock(model_fn, edge_set_names, 
        name=f"{self.name}/GraphNetBlock{'_' if idx > 0 else ''}{idx if idx > 0 else ''}"
      ) for idx in range(self._message_passing_steps)
    ]
    self.decoder = self._make_mlp(self._output_size, layer_norm=False, name=f'{self.name}/decoder')

  def _make_mlp(self, output_size, layer_norm=True, name=None):
    """Builds an MLP."""
    widths = [self._latent_size] * self._num_layers + [output_size]
    network = MLP(
      widths, activate_final=False,
      name=(f'{name}/' if  name is not None else '') + 'mlp'
    )
    if layer_norm:
      network = tf.keras.models.Sequential([network, tf.keras.layers.LayerNormalization(
        name=(f'{name}/' if  name is not None else '') + 'layer_norm'
      )])
    return network

  def _encode(self, graph, training=None):
    """Encodes node and edge features into latent features."""
    with tf.name_scope('encoder'):
      node_latents = self.node_encoder_mlp(graph.node_features, training=training)
      new_edges_sets = []
      for edge_set, edge_encoder_mlp in zip(graph.edge_sets, self.edge_encoder_mlps):
        latent = edge_encoder_mlp(edge_set.features, training=training)
        new_edges_sets.append(edge_set._replace(features=latent))
    return MultiGraph(node_latents, new_edges_sets)

  def _decode(self, graph, training=None):
    """Decodes node features from graph."""
    # TODO: make sure naming works
    with tf.name_scope('decoder'):
      return self.decoder(graph.node_features, training=training)

  def call(self, graph, training=None):
    """Encodes and processes a multigraph, and returns node features."""
    latent_graph = self._encode(graph, training=training)
    for block in self.message_passing_blocks:
      latent_graph = block(latent_graph, training=training)
    return self._decode(latent_graph, training=training)


class BaseModel(tf.keras.models.Model):
  def __init__(self, log_dir=None, save_summaries_steps=10000, name='BaseModel'):
    super(BaseModel, self).__init__(name=name)
    self.save_summaries = log_dir is not None
    if self.save_summaries:
      self.writers = {'train': tf.summary.create_file_writer(f'{log_dir}/train'),
                            'val': tf.summary.create_file_writer(f'{log_dir}/val')}
      self.save_summaries_steps = save_summaries_steps
    self.strategy = tf.distribute.get_strategy()
    

  def add_optimizer(self, **decay_kwargs):
    self.lr = ExponentialDecay(**decay_kwargs)
    self.optimizer = tf.optimizers.Adam(self.lr)

  def loss(self, inputs):
    raise NotImplementedError

  def _maybe_add_summary(self, loss):
    step = self.optimizer.iterations
    if self.save_summaries:
      if tf.logical_or(
        step == 2, # wait just a bit for training to get going
        tf.logical_and(
          (step % self.save_summaries_steps) == 0,
          step > 0
        )
      ):
        with self.writers['train'].as_default():
          with tf.name_scope(''):
            tf.summary.scalar('loss', loss, step=step)


  @tf.function
  def rollout(self, inputs):
    raise NotImplementedError

  def _update_norm(self, inputs):
    loss = self.loss(inputs)
    # self.sync_normalizers()
    return loss

  def _dist_update_norm(self, inputs):
    inputs = {k: v[0] for k, v in inputs.items()}
    return self._update_norm(inputs)

  @tf.function
  def update_norm(self, inputs):
    loss = self._update_norm(inputs)
    self._maybe_add_summary(loss)
    return loss
      

  @tf.function
  def distributed_update_norm(self, inputs):
    per_replica_losses = self.strategy.run(self._dist_update_norm, args=(inputs,))
    loss = self.strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,
                         axis=None)
    if tf.distribute.get_replica_context().replica_id_in_sync_group == 0:
      self._maybe_add_summary(loss)
    return loss
    

  
  def _train_step(self, inputs):
    with tf.GradientTape() as tape:
      loss_val = self.loss(inputs)
    variables = self.trainable_variables
    gradients = tape.gradient(loss_val, variables)
    self.optimizer.apply_gradients(zip(gradients, variables))
    # self.sync_normalizers()
    return loss_val

  def _dist_train_step(self, inputs):
    inputs = {k: v[0] for k, v in inputs.items()}
    return self._train_step(
      inputs, 
    )
    

  @tf.function
  def train_step(self, inputs):
    loss = self._train_step(
      inputs,
    )
    self._maybe_add_summary(loss)

  @tf.function
  def distributed_train_step(self, inputs):
    per_replica_losses = self.strategy.run(self._dist_train_step, args=(inputs,))
    loss = self.strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,
                         axis=None)
    if tf.distribute.get_replica_context().replica_id_in_sync_group == 0:
      self._maybe_add_summary(loss)

    return loss

  def sync_normalizers(self):
    for layer in self.layers:
      if isinstance(layer, Normalizer):
        layer.sync_variables()





