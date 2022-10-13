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
"""Online data normalization."""

import tensorflow as tf


class Normalizer(tf.keras.layers.Layer):
  """Feature normalizer that accumulates statistics online."""

  def __init__(self, size, max_accumulations=10**6, std_epsilon=1e-8,
               name='Normalizer'):
    super(Normalizer, self).__init__(name=name)
    self._max_accumulations = max_accumulations
    self._std_epsilon = std_epsilon
    self._acc_count = tf.Variable(0, dtype=tf.float32, trainable=False,
      name=f'{self.name}/Variable' 
    )
    self._num_accumulations = tf.Variable(0, dtype=tf.float32,
                                          trainable=False,
                                          name=f'{self.name}/Variable_1')
    self._acc_sum = tf.Variable(tf.zeros(size, tf.float32), trainable=False,
                  name=f'{self.name}/Variable_2' )
    self._acc_sum_squared = tf.Variable(tf.zeros(size, tf.float32),
                                        trainable=False,
                                        name=f'{self.name}/Variable_3' )

  def call(self, batched_data, accumulate=True):
    """Normalizes input data and accumulates statistics."""
    if accumulate:
        # stop accumulating after a million updates, to prevent accuracy issues
        if tf.less(self._num_accumulations, self._max_accumulations):
            self._accumulate(batched_data)

    return (batched_data - self._mean()) / self._std_with_epsilon()

  def inverse(self, normalized_batch_data):
    """Inverse transformation of the normalizer."""
    return normalized_batch_data * self._std_with_epsilon() + self._mean()

  def _accumulate(self, batched_data):
    """Function to perform the accumulation of the batch_data statistics."""
    count = tf.cast(tf.shape(batched_data)[0], tf.float32)
    data_sum = tf.reduce_sum(batched_data, axis=0)
    squared_data_sum = tf.reduce_sum(batched_data**2, axis=0)
    self._acc_sum.assign_add(data_sum),
    self._acc_sum_squared.assign_add(squared_data_sum),
    self._acc_count.assign_add(count),
    self._num_accumulations.assign_add(1.)
    

  def _mean(self):
    safe_count = tf.maximum(self._acc_count, 1.)
    return self._acc_sum / safe_count

  def _std_with_epsilon(self):
    safe_count = tf.maximum(self._acc_count, 1.)
    std = tf.sqrt(self._acc_sum_squared / safe_count - self._mean()**2)
    return tf.math.maximum(std, self._std_epsilon)
