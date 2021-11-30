from tensorflow.keras import Model
import tensorflow as tf
import tensorflow.experimental.numpy as tnp
import numpy as np
from tensorflow.keras.utils import to_categorical
from src.networks.utils import scale_gradient, scale_target, scalar_loss, encode_support, decode_support, unscale_target, ce_loss
from pprint import pprint
import io
from datetime import datetime
from src.data_classes import NetworkOutput
from src.prof import p, fl


class Network(Model):

    def __init(self):
        super(Model, self).__init__()

    def compile(self, optimizer):
        # super(Model, self).compile()
        self.optimizer = optimizer

    def train_step(self, batch, weight_decay):

        # h3 = p.start('update_weights')

        # stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        # logdir = 'logs/func/%s' % stamp
        # writer = tf.summary.create_file_writer(logdir)

        # Bracket the function call with
        # tf.summary.trace_on() and tf.summary.trace_export()
        # tf.summary.trace_on(graph=True, profiler=True)

        batch_rewards = batch.rewards
        batch_values = batch.values
        batch_policy_probs = batch.policy_probs
        observations = tf.Variable(tnp.array(batch.observations), name="observations")
        actions = tf.Variable(tnp.array(batch.actions), name="actions")

        metrics = self.update_weights(
                observations,
                actions, 
                batch_rewards, 
                batch_values,
                batch_policy_probs,
                weight_decay
                )

        """
        cf = self.update_weights.get_concrete_function(batch_size, 
                unroll_steps, 
                weight_decay,
                batch.observations,
                batch.actions, 
                batch.rewards, 
                batch.values,
                batch.policy_probs)
        """

        # graph = cf.graph
        #for node in graph.as_graph_def().node:
        #    print(f'{node.input} -> {node.name}')

        # with writer.as_default():
        #   tf.summary.trace_export(
        #           name="my_func_trace",
        #           step=0,
        #           profiler_outdir=logdir)

        print(">>> [TRAIN] loss={} value_loss={} reward_loss={} policy_loss={} reg_loss={}".format(
               metrics['loss'], metrics['value_loss'], metrics['reward_loss'], metrics['policy_loss'], metrics['reg_loss']
               ))

        # p.stop(h3)
        # p.dump_log('./prof1.prof')

        return metrics

    def update_weights(self, observations_batch, action_batch, reward_batch, value_batch, policy_batch, weight_decay):

        batch_size = action_batch.shape[0]
        unroll_steps = action_batch.shape[1]

        trainable_vars = self.trainable_variables
        
        # Note: from here variables will not be named with "batch" but every caclulation is done 
        # batch-wise so each variable has first dimension `batch_size`
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(trainable_vars)

            # h = p.start('initial_inference')
            value_logits, _, policy_logits, hidden_state = self._initial_inference(observations_batch)
            # p.stop(h)

            grad_weights = [1]

            predicted_value_logits = [value_logits]
            # we ignore the first reward as initial_inference doesn't produce a reward
            # TODO would it help to pass a real value through from the env here since this is essentially
            predicted_reward_logits = [tnp.zeros((batch_size, self.support_dim), dtype="float32")]
            predicted_policy_logits = [policy_logits]

            for i in range(unroll_steps):
                
                # (batch_size x unroll_steps x actions_size) 
                # -[slice]-> (batch_size x 1 x actions_size) 
                # -[squeeze]-> (batch_size x actions_size)
                current_action = tf.squeeze(tf.slice(action_batch, [0, i], [batch_size, 1]), axis=1)

                value_logits, reward_logits, policy_logits, hidden_state = self._recurrent_inference(hidden_state, current_action)

                grad_weights.append(1/unroll_steps)
                predicted_value_logits.append(value_logits)
                predicted_reward_logits.append(reward_logits)
                predicted_policy_logits.append(policy_logits)

                hidden_state = scale_gradient(hidden_state, 0.5)

            grad_weights = tnp.array(grad_weights)

            # (unroll_steps x batch_size x ...) -[transpose]-> (batch_size x unroll_steps x ...)
            predicted_value_logits = tf.transpose(tnp.array(predicted_value_logits), perm=[1, 0, 2])
            predicted_reward_logits = tf.transpose(tnp.array(predicted_reward_logits), perm=[1, 0, 2])
            predicted_policy_logits = tf.transpose(tnp.array(predicted_policy_logits), perm=[1, 0, 2])
         
            loss, value_loss, reward_loss, policy_loss = self._loss(
                    (predicted_value_logits, predicted_reward_logits, predicted_policy_logits),
                    (value_batch, reward_batch, policy_batch),
                    grad_weights,
                    trainable_vars
                    )

            reg_loss = weight_decay * tf.add_n([ 
                tf.nn.l2_loss(v) for v in trainable_vars
                if 'bias' not in v.name 
                ])

            loss = loss + reg_loss
        
        gradients = tape.gradient(loss, trainable_vars)
        # self.optimizer.minimize(loss, trainable_vars, tape=tape)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        return {
                'loss': loss / batch_size,
                'value_loss': value_loss / batch_size,
                'reward_loss': reward_loss / batch_size,
                'policy_loss': policy_loss / batch_size,
                'reg_loss': reg_loss,
                }


    def _loss(self, predicted, targets, grad_weights, network_weights):

        value_logits, reward_logits, policy_logits = predicted
        target_values, target_rewards, target_policies = targets

        # print(f"[PRED] values={value_logits[0]}, rewards={reward_logits[0]}")
        # print(f"[TARG] values={target_values[0]}, rewards={target_rewards[0]}")

        scaled_target_values = scale_target(target_values)

        value_loss = tf.nn.softmax_cross_entropy_with_logits(
            encode_support(scale_target(target_values), self.support_size),
            value_logits
        )
        value_loss = scale_gradient(value_loss, grad_weights)
        value_loss = tf.reduce_sum(value_loss)

        reward_loss = tf.nn.softmax_cross_entropy_with_logits(
            encode_support(scale_target(target_rewards), self.support_size),
            reward_logits
        )
        reward_loss = scale_gradient(reward_loss, grad_weights)
        reward_loss = tf.reduce_sum(reward_loss)

        policy_loss = tf.nn.softmax_cross_entropy_with_logits(
            target_policies,
            policy_logits
        )
        policy_loss = scale_gradient(policy_loss, grad_weights)
        policy_loss = tf.reduce_sum(policy_loss)

        loss = value_loss + reward_loss + policy_loss

        return loss, value_loss, reward_loss, policy_loss


    def _decode_output(self, value_logits, reward_logits, policy_logits, hidden_state=None):
        """
        Given a raw network output tuple decodes the output
        """
        value_probs = tf.nn.softmax(value_logits)
        value = unscale_target(decode_support(value_probs))

        if reward_logits is None:
            # If output was a result of initial_inference we set the predicted rewards as 
            # zeros to ensure no contribution to the loss
            batch_size = value_logits.shape[0]
            reward = tnp.zeros((batch_size,))
        else:
            reward_probs = tf.nn.softmax(reward_logits)
            reward = unscale_target(decode_support(reward_probs))

        policy_probs = tf.nn.softmax(policy_logits)

        return (value, reward, policy_probs, hidden_state)


    def recurrent_inference(self, hidden_state, action):
        """
        Runs recurrent_inference for a single input then formats and processes the output
        """
        hidden_state_batch = np.array([hidden_state])
        action_batch = np.array([action])

        output_batch = self._recurrent_inference(hidden_state_batch, action_batch)
        
        value, reward, policy, hidden_state = self._decode_output(*output_batch)

        return NetworkOutput(
                value=np.array(value)[0], 
                reward=np.array(reward)[0], 
                policy=np.array(policy)[0], 
                hidden_state=np.array(hidden_state)[0]) 


    def initial_inference(self, observation):
        """
        Runs initial_inference for a single input then formats and processes the output
        """
        observation_batch = np.array([observation])

        output_batch = self._initial_inference(observation_batch) 
        value, reward, policy, hidden_state = self._decode_output(*output_batch)

        return NetworkOutput(
                value=np.array(value)[0], 
                reward=np.array(reward)[0], 
                policy=np.array(policy)[0], 
                hidden_state=np.array(hidden_state)[0]) 


    def _initial_inference(self, observation_batch):
        """
        Runs inital_inference step on batch of inputs and retuns raw output
        for use in training
        """

        hidden_state = self.representation(observation_batch)
        policy_logits, value_logits = self.prediction(hidden_state)

        # Note: we return None in place of reward here so the shape matches the 
        # return type of _recurrent_inference. However, we do not calculate a 
        # reward in the network execution here so we return None
        return (value_logits, None, policy_logits, hidden_state)


    def _recurrent_inference(self, hidden_state_batch, action_batch):
        """
        Runs recurrent_inference step on batch of inputs and retuns raw output
        for use in training
        """

        action_one_hot = tf.one_hot(action_batch, self.action_space_size)

        next_hidden_state, reward_logits = self.dynamics([hidden_state_batch, action_one_hot])
        policy_logits, value_logits = self.prediction(next_hidden_state)

        return (value_logits, reward_logits, policy_logits, next_hidden_state)


    def get_model_summary(self):
        summary_string = ""
        for n in [self.dynamics, self.representation, self.prediction]:
            stream = io.StringIO()
            n.summary(print_fn=lambda x: stream.write(x + '\n'))
            summary_string += stream.getvalue()
            stream.close()
        return summary_string
