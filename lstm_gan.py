import tensorflow as tf
import numpy as np

class LSTMGAN(object):
    """ LSTM-GAN implementation in tensorflow.

        Args:
            seq_size (int): Max size of sentence.
            vocab_size (int): Size of words vocabulary.
            first_input (int): Code of first word in generator.
            hidden_size_gen (int, optional): Size of hidden layer
                in generator.
            hidden_size_disc (int, optional): Size of hidden layer
                in discriminator.
            input_noise_size (int, optional): Size of input noise
                in generator input.
            batch_size (int, optional): Batch size.
            dropout (float, optional): dropout rate.
            lr (float, optional): learning rate in Adam optimizer.
            grad_cap (float, optional): gradient cap value.

    """
    def __init__(self, seq_size, vocab_size, first_input,
                 hidden_size_gen = 512, hidden_size_disc = 512,
                 input_noise_size = 32, batch_size = 128, dropout = 0.2,
                 lr = 1e-4, grad_cap = 1.):

        self.seq_size = seq_size
        self.vocab_size = vocab_size
        self.hidden_size_gen = hidden_size_gen
        self.hidden_size_disc = hidden_size_disc
        self.input_noise_size = input_noise_size
        self.batch_size = batch_size
        self.first_input = first_input
        self.keep_prob = 1 - dropout
        self.lr = lr
        self.grad_cap = grad_cap

        self.build_model()
        self.build_trainers()

    def train_gen_on_batch(self, session, batch):
        """Train generator on given `batch` in current `session`"""
        feed = {
            self.input_noise: batch
        }
        ret_values = [self.gen_cost, self.gen_train]
        cost, _ = session.run(ret_values, feed_dict = feed)
        return cost

    def train_disc_on_batch(self, session, noise_batch, real_batch):
        """Train discriminator on given `noise_batch` and `real_batch`
        in current `session`

        """
        feed = {
            self.input_noise: noise_batch,
            self.real_sent : real_batch
        }
        ret_values = [self.disc_cost, self.disc_train]
        cost, _ = session.run(ret_values, feed_dict = feed)
        return cost

    def generate_sent(self, session, noise):
        """Generate one sentence in current `session` with given `noise`"""
        feed_dict = {self.input_noise_one_sent: [noise]}
        generated = session.run(self.sent_generator, feed_dict = feed_dict)
        return np.argmax(generated[0], axis=1)

    def build_model(self):
        batch_size, input_noise_size, seq_size, vocab_size = \
            self.batch_size, self.input_noise_size, \
            self.seq_size, self.vocab_size

        embedding = tf.diag(np.ones((vocab_size, ), dtype=np.float32))
        self.embedding = embedding

        input_noise = tf.placeholder(tf.float32, [batch_size, input_noise_size])
        input_noise_one_sent = tf.placeholder(tf.float32, [1, input_noise_size])
        self.input_noise = input_noise
        self.input_noise_one_sent = input_noise_one_sent

        real_sent = tf.placeholder(tf.int32, [batch_size, seq_size])
        input_sentence = tf.nn.embedding_lookup(embedding, real_sent)
        self.real_sent = real_sent

        _, gen_vars = self.build_generator(input_noise, is_train = True)
        generated_sent, _ = self.build_generator(input_noise, reuse = True)
        sent_generator, _ = self.build_generator(input_noise_one_sent, reuse = True)
        self.gen_vars = gen_vars
        self.generated_sent = generated_sent
        self.sent_generator = sent_generator

        _, disc_vars = self.build_discriminator(input_sentence, is_train = True)
        desc_decision_fake, _ = self.build_discriminator(generated_sent, reuse = True)
        disc_decision_real, _ = self.build_discriminator(input_sentence, reuse = True)
        self.disc_vars = disc_vars
        self.desc_decision_fake = desc_decision_fake
        self.disc_decision_real = disc_decision_real

        self.gen_cost = 1. - desc_decision_fake
        self.disc_cost = 1. - disc_decision_real*(1. - desc_decision_fake)


    def build_trainers(self):
        cap, lr, disc_cost, disc_vars, gen_cost, gen_vars = \
            self.grad_cap, self.lr, \
            self.disc_cost, self.disc_vars, \
            self.gen_cost, self.gen_vars

        optimizer_disc = tf.train.AdamOptimizer(lr)
        gvs = optimizer_disc.compute_gradients(disc_cost, disc_vars)
        capped_grads_and_vars = [(tf.clip_by_value(grad, -cap, cap), var) \
                                 for grad, var in gvs]
        optimizer_disc.apply_gradients(capped_grads_and_vars)

        optimizer_gen = tf.train.AdamOptimizer(lr)
        gvs = optimizer_gen.compute_gradients(gen_cost, gen_vars)
        capped_grads_and_vars = [(tf.clip_by_value(grad, -cap, cap), var) \
                                 for grad, var in gvs]
        optimizer_gen.apply_gradients(capped_grads_and_vars)

        self.disc_train = optimizer_disc.minimize(disc_cost)
        self.gen_train = optimizer_gen.minimize(gen_cost)

    def build_generator(self, input_, reuse = False, is_train = False):
        vocab_size, hidden_size_gen, input_noise_size, seq_size, keep_prob = \
            self.vocab_size, self.hidden_size_gen, \
            self.input_noise_size, self.seq_size, \
            self.keep_prob
        embedding, first_input = self.embedding, self.first_input

        with tf.variable_scope('generator_model', reuse = reuse):
            input_noise_w = tf.get_variable(
                "input_noise_w",
                [input_noise_size, hidden_size_gen],
                initializer=tf.random_normal_initializer(0, stddev=1/np.sqrt(vocab_size))
            )
            input_noise_b = tf.get_variable(
                "input_noise_b",
                [hidden_size_gen],
                initializer=tf.constant_initializer(1e-4)
            )
            
            first_hidden_state = tf.nn.relu(tf.matmul(input_, input_noise_w) + input_noise_b)

            cell = tf.nn.rnn_cell.GRUCell(hidden_size_gen)
            if is_train:
                cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=keep_prob)

            input_w = tf.get_variable(
                "input_w",
                [vocab_size, hidden_size_gen],
                initializer=tf.random_normal_initializer(0, stddev=1/np.sqrt(vocab_size))
            )
            input_b = tf.get_variable(
                "input_b",
                [hidden_size_gen],
                initializer=tf.constant_initializer(1e-4)
            )

            softmax_w = tf.get_variable(
                "softmax_w",
                [hidden_size_gen, vocab_size],
                initializer = tf.random_normal_initializer(0, stddev=1/np.sqrt(hidden_size_gen))
            )
            softmax_b = tf.get_variable(
                "softmax_b",
                [vocab_size],
                initializer=tf.constant_initializer(1e-4)
            )

            state = first_hidden_state

            labels = tf.fill([tf.shape(input_)[0], 1], tf.cast(first_input, tf.int32))
            input_ = tf.nn.embedding_lookup(embedding, labels)

            outputs = []
            with tf.variable_scope("GRU_generator"):
                for time_step in range(seq_size):
                    if time_step > 0: tf.get_variable_scope().reuse_variables()
                    inp = tf.nn.relu(tf.matmul(input_[:, 0, :], input_w) + input_b)

                    cell_output, state = cell(inp, state)
                    logits = tf.nn.softmax(tf.matmul(cell_output, softmax_w) + softmax_b)
                    labels = tf.expand_dims(tf.argmax(logits, 1), 1)
                    input_ = tf.nn.embedding_lookup(embedding, labels)
                    outputs.append(tf.expand_dims(logits, 1))

            output = tf.concat(1, outputs)
        variables = [v for v in tf.all_variables() if 'generator_model' in v.name]

        return output, variables

    def build_discriminator(self, input_, is_train = False, reuse = False):
        vocab_size, hidden_size_disc, batch_size, seq_size,  keep_prob = \
            self.vocab_size, self.hidden_size_disc, \
            self.batch_size, self.seq_size, self.keep_prob

        with tf.variable_scope('discriminator_model', reuse = reuse):
            cell = tf.nn.rnn_cell.GRUCell(hidden_size_disc)
            if is_train:
                cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=keep_prob)

            if is_train:
                input_ = tf.nn.dropout(input_, keep_prob)

            state = cell.zero_state(batch_size, tf.float32)

            input_w = tf.get_variable(
                "input_w",
                [vocab_size, hidden_size_disc],
                initializer=tf.random_normal_initializer(0, stddev=1/np.sqrt(vocab_size))
            )
            input_b = tf.get_variable(
                "input_b",
                [hidden_size_disc],
                initializer=tf.constant_initializer(1e-4)
            )

            with tf.variable_scope("GRU_discriminator"):
                for time_step in range(seq_size):
                    if time_step > 0: tf.get_variable_scope().reuse_variables()
                    inp = tf.nn.relu(tf.matmul(input_[:, time_step, :], input_w) + input_b)
                    cell_output, state = cell(inp, state)

            out_w = tf.get_variable(
                "discriminator_output_w",
                [hidden_size_disc, 1],
                initializer=tf.random_normal_initializer(0, 1./np.sqrt(hidden_size_disc))
            )
            out_b = tf.get_variable(
                "discriminator_output_b",
                [1],
                initializer=tf.constant_initializer(1e-4)
            )

            output = tf.reduce_mean(tf.sigmoid(tf.matmul(cell_output, out_w) + out_b))

        variables = [v for v in tf.all_variables() if 'discriminator_model' in v.name]

        return output, variables