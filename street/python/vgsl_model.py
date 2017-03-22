# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""String network description language to define network layouts."""
import re
import time
import vgslspecs
import decoder
import errorcounter as ec
import shapes
import tensorflow as tf
import vgsl_input
import tensorflow.contrib.slim as slim
from tensorflow.core.framework import summary_pb2
from tensorflow.python.platform import tf_logging as logging


# Parameters for rate decay.
# We divide the learning_rate_halflife by DECAY_STEPS_FACTOR and use DECAY_RATE
# as the decay factor for the learning rate, ie we use the DECAY_STEPS_FACTORth
# root of 2 as the decay rate every halflife/DECAY_STEPS_FACTOR to achieve the
# desired halflife.
DECAY_STEPS_FACTOR = 16
DECAY_RATE = pow(0.5, 1.0 / DECAY_STEPS_FACTOR)


def Train(train_dir,
          train_data,
          max_steps,
          master='',
          task=0,
          ps_tasks=0,
          initial_learning_rate=0.001,
          final_learning_rate=0.001,
          learning_rate_halflife=160000,
          optimizer_type='Adam',
          num_preprocess_threads=1,
          reader=None):
  """Testable trainer with no dependence on FLAGS.

  Args:
    train_dir: Directory to write checkpoints.
    model_str: Network specification string.
    train_data: Training data file pattern.
    max_steps: Number of training steps to run.
    master: Name of the TensorFlow master to use.
    task: Task id of this replica running the training. (0 will be master).
    ps_tasks: Number of tasks in ps job, or 0 if no ps job.
    initial_learning_rate: Learing rate at start of training.
    final_learning_rate: Asymptotic minimum learning rate.
    learning_rate_halflife: Number of steps over which to halve the difference
      between initial and final learning rate.
    optimizer_type: One of 'GradientDescent', 'AdaGrad', 'Momentum', 'Adam'.
    num_preprocess_threads: Number of input threads.
    reader: Function that returns an actual reader to read Examples from input
      files. If None, uses tf.TFRecordReader().
  """
  if master.startswith('local'):
    device = tf.ReplicaDeviceSetter(ps_tasks)
  else:
    device = '/cpu:0'
  with tf.Graph().as_default():
    with tf.device(device):

      model = InitNetwork(train_data, 'train', initial_learning_rate,
                          final_learning_rate, learning_rate_halflife,
                          optimizer_type, num_preprocess_threads, reader)

      # Create a Supervisor.  It will take care of initialization, summaries,
      # checkpoints, and recovery.
      #
      # When multiple replicas of this program are running, the first one,
      # identified by --task=0 is the 'chief' supervisor.  It is the only one
      # that takes case of initialization, etc.
      sv = tf.train.Supervisor(
          logdir=train_dir,
          is_chief=(task == 0),
          saver=model.saver,
          save_summaries_secs=10,
          save_model_secs=30,
          recovery_wait_secs=5)

      step = 0
      while step < max_steps:
        try:
          # Get an initialized, and possibly recovered session.  Launch the
          # services: Checkpointing, Summaries, step counting.
          with sv.managed_session(master) as sess:
            while step < max_steps:
              start = time.time()              
              loss_, step = model.TrainAStep(sess)
              end = time.time()
              print "Step %d, Loss = %f, %.3f seconds used."%(step, loss_, end - start)
              print "Step:", step, ", Loss =", loss_
              if sv.coord.should_stop():
                break
        except tf.errors.AbortedError as e:
          logging.error('Received error:%s', e)
          continue


def Eval(train_dir,
         eval_dir,
         eval_data,
         decoder_file,
         num_steps,
         graph_def_file=None,
         eval_interval_secs=0,
         reader=None):
  """Restores a model from a checkpoint and evaluates it.

  Args:
    train_dir: Directory to find checkpoints.
    eval_dir: Directory to write summary events.
    model_str: Network specification string.
    eval_data: Evaluation data file pattern.
    decoder_file: File to read to decode the labels.
    num_steps: Number of eval steps to run.
    graph_def_file: File to write graph definition to for freezing.
    eval_interval_secs: How often to run evaluations, or once if 0.
    reader: Function that returns an actual reader to read Examples from input
      files. If None, uses tf.TFRecordReader().
  Returns:
    (char error rate, word recall error rate, sequence error rate) as percent.
  Raises:
    ValueError: If unimplemented feature is used.
  """
  decode = None
  if decoder_file:
    decode = decoder.Decoder(decoder_file)

  # Run eval.
  rates = ec.ErrorRates(
      label_error=None,
      word_recall_error=None,
      word_precision_error=None,
      sequence_error=None)
  with tf.Graph().as_default():
    model = InitNetwork(eval_data, 'eval', reader=reader)
    sw = tf.summary.FileWriter(eval_dir)

    while True:
      sess = tf.Session('')
      if graph_def_file is not None:
        # Write the eval version of the graph to a file for freezing.
        if not tf.gfile.Exists(graph_def_file):
          with tf.gfile.FastGFile(graph_def_file, 'w') as f:
            f.write(
                sess.graph.as_graph_def(add_shapes=True).SerializeToString())
      ckpt = tf.train.get_checkpoint_state(train_dir)
      if ckpt and ckpt.model_checkpoint_path:
        step = model.Restore(ckpt.model_checkpoint_path, sess)
        if decode:
          rates = decode.SoftmaxEval(sess, model, num_steps)
          _AddRateToSummary('Label error rate', rates.label_error, step, sw)
          _AddRateToSummary('Word recall error rate', rates.word_recall_error,
                            step, sw)
          _AddRateToSummary('Word precision error rate',
                            rates.word_precision_error, step, sw)
          _AddRateToSummary('Sequence error rate', rates.sequence_error, step,
                            sw)
          sw.flush()
          print 'Error rates=', rates
        else:
          raise ValueError('Non-softmax decoder evaluation not implemented!')
      if eval_interval_secs:
        time.sleep(eval_interval_secs)
      else:
        break
  return rates


def InitNetwork(input_pattern,
                mode='eval',
                initial_learning_rate=0.00005,
                final_learning_rate=0.00005,
                halflife=1600000,
                optimizer_type='Adam',
                num_preprocess_threads=1,
                reader=None):
  """Constructs a python tensor flow model defined by model_spec.

  Args:
    input_pattern: File pattern of the data in tfrecords of Example.
    mode: One of 'train', 'eval'
    initial_learning_rate: Initial learning rate for the network.
    final_learning_rate: Final learning rate for the network.
    halflife: Number of steps over which to halve the difference between
              initial and final learning rate for the network.
    optimizer_type: One of 'GradientDescent', 'AdaGrad', 'Momentum', 'Adam'.
    num_preprocess_threads: Number of threads to use for image processing.
    reader: Function that returns an actual reader to read Examples from input
      files. If None, uses tf.TFRecordReader().
    Eval tasks need only specify input_pattern and model_spec.

  Returns:
    A VGSLImageModel class.

  Raises:
    ValueError: if the model spec syntax is incorrect.
  """
  model = VGSLImageModel(mode, initial_learning_rate, final_learning_rate, halflife)
  model.Build(input_pattern, optimizer_type, num_preprocess_threads, reader)
  return model


class VGSLImageModel(object):
  """Class that builds a tensor flow model for training or evaluation.
  """
  def __init__(self, mode, initial_learning_rate, final_learning_rate, halflife):
    """Constructs a VGSLImageModel.

    Args:
      mode:        One of "train", "eval"
      initial_learning_rate: Initial learning rate for the network.
      final_learning_rate: Final learning rate for the network.
      halflife: Number of steps over which to halve the difference between
                initial and final learning rate for the network.
    """
    # The layers between input and output.
    self.layers = None
    # The train/eval mode.
    self.mode = mode
    # The initial learning rate.
    self.initial_learning_rate = initial_learning_rate
    self.final_learning_rate = final_learning_rate
    self.decay_steps = halflife / DECAY_STEPS_FACTOR
    self.decay_rate = DECAY_RATE
    # Tensor for the labels.
    self.labels = None
    self.sparse_labels = None
    # Debug data containing the truth text.
    self.truths = None
    # Tensor for loss
    self.loss = None
    # Train operation
    self.train_op = None
    # Tensor for the global step counter
    self.global_step = None
    # Tensor for the output predictions (usually softmax)
    self.output = None
    # True if we are using CTC training mode.
    self.using_ctc = True
    # Saver object to load or restore the variables.
    self.saver = None

  def Build(self, input_pattern, optimizer_type, num_preprocess_threads, reader):
    batch_size = 1
    y_size = 150
    x_size = 600
    depth = 3
    out_dims = 1
    out_func = 'c'
    num_classes = 134

    self.global_step = tf.Variable(0, name='global_step', trainable=False)
    shape = vgsl_input.ImageShape(batch_size, y_size, x_size, depth);
    self.using_ctc = True
    images, heights, widths, labels, sparse, _ = vgsl_input.ImageInput(input_pattern, num_preprocess_threads, shape, self.using_ctc, reader)
    self.labels = labels
    self.sparse_labels = sparse
    
    # reshape S2(4x150)0,2: 150x600x3 -> 4x150x150x3
    reshaped_image = shapes.transposing_reshape(images, 2, 4, 150, 0, 2, name = "reshape_image_into_4");# (4, 150, 150, 3)
    conv1 = slim.conv2d(reshaped_image, 16, [5, 5], activation_fn = tf.nn.relu, scope= "conv1");#(4, 150, 150, 16)
    pool1 = slim.max_pool2d(conv1, [2, 2], [2, 2],  padding='SAME',  scope="pool1"); #(4, 75, 75, 16)
    conv2 = slim.conv2d(pool1, 64, [5, 5], activation_fn = tf.nn.relu, scope= "conv2");#(4, 75, 75, 64)
    pool2 = slim.max_pool2d(conv2, [3, 3], [3, 3],  padding='SAME',  scope="pool2");
    self.layers = vgslspecs.VGSLSpecs(tf.constant([25]), tf.constant([25]), self.mode == 'train') 
    model_spec = '[([Lrys64 Lbx128][Lbys64 Lbx128][Lfys64 Lbx128])S3(3x0)2,3 Lfx128 Lrx128 S0(1x4)0,3 Do Lfx256]'
    lstm_after_dropout = self.layers.Build(pool2, model_spec)
    import pdb
    pdb.set_trace()
    ## output: O1c134
    self._AddOutputs(lstm_after_dropout, out_dims, out_func, num_classes)
    if self.mode == 'train':
      self._AddOptimizer(optimizer_type)

    # For saving the model across training and evaluation
    self.saver = tf.train.Saver(max_to_keep=1000)

  def TrainAStep(self, sess):
    """Runs a training step in the session.

    Args:
      sess: Session in which to train the model.
    Returns:
      loss, global_step.
    """
    _, loss, step = sess.run([self.train_op, self.loss, self.global_step])
    return loss, step

  def Restore(self, checkpoint_path, sess):
    """Restores the model from the given checkpoint path into the session.

    Args:
      checkpoint_path: File pathname of the checkpoint.
      sess:            Session in which to restore the model.
    Returns:
      global_step of the model.
    """
    self.saver.restore(sess, checkpoint_path)
    return tf.train.global_step(sess, self.global_step)

  def RunAStep(self, sess):
    """Runs a step for eval in the session.

    Args:
      sess:            Session in which to run the model.
    Returns:
      output tensor result, labels tensor result.
    """
    return sess.run([self.output, self.labels])

  def _AddOutputs(self, prev_layer, out_dims, out_func, num_classes):
    """Adds the output layer and loss function.

    Args:
      prev_layer:  Output of last layer of main network.
      out_dims:    Number of output dimensions, 0, 1 or 2.
      out_func:    Output non-linearity. 's' or 'c'=softmax, 'l'=logistic.
      num_classes: Number of outputs/size of last output dimension.
    """
    height_in = shapes.tensor_dim(prev_layer, dim=1)
    logits, outputs = self._AddOutputLayer(prev_layer, out_dims, out_func,
                                           num_classes)
    if self.mode == 'train':
      # Setup loss for training.
      self.loss = self._AddLossFunction(logits, height_in, out_dims, out_func)
      tf.summary.scalar('loss', self.loss)
    elif out_dims == 0:
      # Be sure the labels match the output, even in eval mode.
      self.labels = tf.slice(self.labels, [0, 0], [-1, 1])
      self.labels = tf.reshape(self.labels, [-1])

    logging.info('Final output=%s', outputs)
    logging.info('Labels tensor=%s', self.labels)
    self.output = outputs

  def _AddOutputLayer(self, prev_layer, out_dims, out_func, num_classes):
    """Add the fully-connected logits and SoftMax/Logistic output Layer.

    Args:
      prev_layer:  Output of last layer of main network.
      out_dims:    Number of output dimensions, 0, 1 or 2.
      out_func:    Output non-linearity. 's' or 'c'=softmax, 'l'=logistic.
      num_classes: Number of outputs/size of last output dimension.

    Returns:
      logits:  Pre-softmax/logistic fully-connected output shaped to out_dims.
      outputs: Post-softmax/logistic shaped to out_dims.

    Raises:
      ValueError: if syntax is incorrect.
    """
    # Reduce dimensionality appropriate to the output dimensions.
    batch_in = shapes.tensor_dim(prev_layer, dim=0)
    height_in = shapes.tensor_dim(prev_layer, dim=1)
    width_in = shapes.tensor_dim(prev_layer, dim=2)
    depth_in = shapes.tensor_dim(prev_layer, dim=3)
    if out_dims:
      # Combine any remaining height and width with batch and unpack after.
      shaped = tf.reshape(prev_layer, [-1, depth_in])
    else:
      # Everything except batch goes to depth, and therefore has to be known.
      shaped = tf.reshape(prev_layer, [-1, height_in * width_in * depth_in])
    logits = slim.fully_connected(shaped, num_classes, activation_fn=None)
    if out_func == 'l':
      raise ValueError('Logistic not yet supported!')
    else:
      output = tf.nn.softmax(logits)
    # Reshape to the dessired output.
    if out_dims == 2:
      output_shape = [batch_in, height_in, width_in, num_classes]
    elif out_dims == 1:
      output_shape = [batch_in, height_in * width_in, num_classes]
    else:
      output_shape = [batch_in, num_classes]
    output = tf.reshape(output, output_shape, name='Output')
    logits = tf.reshape(logits, output_shape)
    return logits, output

  def _AddLossFunction(self, logits, height_in, out_dims, out_func):
    """Add the appropriate loss function.

    Args:
      logits:  Pre-softmax/logistic fully-connected output shaped to out_dims.
      height_in:  Height of logits before going into the softmax layer.
      out_dims:   Number of output dimensions, 0, 1 or 2.
      out_func:   Output non-linearity. 's' or 'c'=softmax, 'l'=logistic.

    Returns:
      loss: That which is to be minimized.

    Raises:
      ValueError: if logistic is used.
    """
    if out_func == 'c':
      # Transpose batch to the middle.
      ctc_input = tf.transpose(logits, [1, 0, 2])
      # Compute the widths of each batch element from the input widths.
      widths = self.layers.GetLengths(dim=2, factor=height_in)
      cross_entropy = tf.nn.ctc_loss(self.sparse_labels, ctc_input, widths)
    elif out_func == 's':
      if out_dims == 2:
        self.labels = _PadLabels3d(logits, self.labels)
      elif out_dims == 1:
        self.labels = _PadLabels2d(
            shapes.tensor_dim(
                logits, dim=1), self.labels)
      else:
        self.labels = tf.slice(self.labels, [0, 0], [-1, 1])
        self.labels = tf.reshape(self.labels, [-1])
      cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
          logits=logits, labels=self.labels, name='xent')
    else:
      # TODO(rays) Labels need an extra dimension for logistic, so different
      # padding functions are needed, as well as a different loss function.
      raise ValueError('Logistic not yet supported!')
    return tf.reduce_sum(cross_entropy)

  def _AddOptimizer(self, optimizer_type):
    """Adds an optimizer with learning rate decay to minimize self.loss.

    Args:
      optimizer_type: One of 'GradientDescent', 'AdaGrad', 'Momentum', 'Adam'.
    Raises:
      ValueError: if the optimizer type is unrecognized.
    """
    learn_rate_delta = self.initial_learning_rate - self.final_learning_rate
    learn_rate_dec = tf.add(
        tf.train.exponential_decay(learn_rate_delta, self.global_step,
                                   self.decay_steps, self.decay_rate),
        self.final_learning_rate)
    if optimizer_type == 'GradientDescent':
      opt = tf.train.GradientDescentOptimizer(learn_rate_dec)
    elif optimizer_type == 'AdaGrad':
      opt = tf.train.AdagradOptimizer(learn_rate_dec)
    elif optimizer_type == 'Momentum':
      opt = tf.train.MomentumOptimizer(learn_rate_dec, momentum=0.9)
    elif optimizer_type == 'Adam':
      opt = tf.train.AdamOptimizer(learning_rate=learn_rate_dec)
    else:
      raise ValueError('Invalid optimizer type: ' + optimizer_type)
    tf.summary.scalar('learn_rate', learn_rate_dec)

    self.train_op = opt.minimize(
        self.loss, global_step=self.global_step, name='train')

  def _LSTMLayer(self, prev_layer, direction, dim, summarize, depth, name):
    """Adds an LSTM layer with the given pre-parsed attributes.

    Always maps 4-D to 4-D regardless of summarize.
    Args:
      prev_layer: Input tensor.
      direction:  'forward' 'backward' or 'bidirectional'
      dim:        'x' or 'y', dimension to consider as time.
      summarize:  True if we are to return only the last timestep.
      depth:      Output depth.
      name:       Some string naming the op.

    Returns:
      Output tensor.
    """
    # If the target dimension is y, we need to transpose.
    if dim == 'x':
      lengths = self.GetLengths(2, 1)
      inputs = prev_layer
    else:
      lengths = self.GetLengths(1, 1)
      inputs = tf.transpose(prev_layer, [0, 2, 1, 3], name=name + '_ytrans_in')
    input_batch = shapes.tensor_dim(inputs, 0)
    num_slices = shapes.tensor_dim(inputs, 1)
    num_steps = shapes.tensor_dim(inputs, 2)
    input_depth = shapes.tensor_dim(inputs, 3)
    # Reshape away the other dimension.
    inputs = tf.reshape(
        inputs, [-1, num_steps, input_depth], name=name + '_reshape_in')
    # We need to replicate the lengths by the size of the other dimension, and
    # any changes that have been made to the batch dimension.
    tile_factor = tf.to_float(input_batch *
                              num_slices) / tf.to_float(tf.shape(lengths)[0])
    lengths = tf.tile(lengths, [tf.cast(tile_factor, tf.int32)])
    lengths = tf.cast(lengths, tf.int64)
    outputs = nn_ops.rnn_helper(
        inputs,
        lengths,
        cell_type='lstm',
        num_nodes=depth,
        direction=direction,
        name=name,
        stddev=0.1)
    # Output depth is doubled if bi-directional.
    if direction == 'bidirectional':
      output_depth = depth * 2
    else:
      output_depth = depth
    # Restore the other dimension.
    if summarize:
      outputs = tf.slice(
          outputs, [0, num_steps - 1, 0], [-1, 1, -1], name=name + '_sum_slice')
      outputs = tf.reshape(
          outputs, [input_batch, num_slices, 1, output_depth],
          name=name + '_reshape_out')
    else:
      outputs = tf.reshape(
          outputs, [input_batch, num_slices, num_steps, output_depth],
          name=name + '_reshape_out')
    if dim == 'y':
      outputs = tf.transpose(outputs, [0, 2, 1, 3], name=name + '_ytrans_out')
    return outputs

def _PadLabels3d(logits, labels):
  """Pads or slices 3-d labels to match logits.

  Covers the case of 2-d softmax output, when labels is [batch, height, width]
  and logits is [batch, height, width, onehot]
  Args:
    logits: 4-d Pre-softmax fully-connected output.
    labels: 3-d, but not necessarily matching in size.

  Returns:
    labels: Resized by padding or clipping to match logits.
  """
  logits_shape = shapes.tensor_shape(logits)
  labels_shape = shapes.tensor_shape(labels)
  labels = tf.reshape(labels, [-1, labels_shape[2]])
  labels = _PadLabels2d(logits_shape[2], labels)
  labels = tf.reshape(labels, [labels_shape[0], -1])
  labels = _PadLabels2d(logits_shape[1] * logits_shape[2], labels)
  return tf.reshape(labels, [labels_shape[0], logits_shape[1], logits_shape[2]])


def _PadLabels2d(logits_size, labels):
  """Pads or slices the 2nd dimension of 2-d labels to match logits_size.

  Covers the case of 1-d softmax output, when labels is [batch, seq] and
  logits is [batch, seq, onehot]
  Args:
    logits_size: Tensor returned from tf.shape giving the target size.
    labels:      2-d, but not necessarily matching in size.

  Returns:
    labels: Resized by padding or clipping the last dimension to logits_size.
  """
  pad = logits_size - tf.shape(labels)[1]

  def _PadFn():
    return tf.pad(labels, [[0, 0], [0, pad]])

  def _SliceFn():
    return tf.slice(labels, [0, 0], [-1, logits_size])

  return tf.cond(tf.greater(pad, 0), _PadFn, _SliceFn)





def _AddRateToSummary(tag, rate, step, sw):
  """Adds the given rate to the summary with the given tag.

  Args:
    tag:   Name for this value.
    rate:  Value to add to the summary. Perhaps an error rate.
    step:  Global step of the graph for the x-coordinate of the summary.
    sw:    Summary writer to which to write the rate value.
  """
  sw.add_summary(
      summary_pb2.Summary(value=[summary_pb2.Summary.Value(
          tag=tag, simple_value=rate)]), step)
