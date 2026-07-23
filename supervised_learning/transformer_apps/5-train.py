#!/usr/bin/env python3
"""Train a Transformer for Portuguese-to-English translation."""

import tensorflow as tf

Dataset = __import__('3-dataset').Dataset
create_masks = __import__('4-create_masks').create_masks
Transformer = __import__('5-transformer').Transformer


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Implement the Transformer learning-rate schedule."""

    def __init__(self, dm, warmup_steps=4000):
        """Initialize the schedule."""
        super(CustomSchedule, self).__init__()
        self.dm = tf.cast(dm, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        """Calculate the learning rate for a training step."""
        step = tf.cast(step, tf.float32)
        first = tf.math.rsqrt(step)
        second = step * self.warmup_steps ** -1.5

        return tf.math.rsqrt(self.dm) * tf.math.minimum(first, second)


def train_transformer(N, dm, h, hidden, max_len, batch_size, epochs):
    """Create and train a Portuguese-to-English Transformer."""
    data = Dataset(batch_size, max_len)
    input_vocab = data.tokenizer_pt.vocab_size + 2
    target_vocab = data.tokenizer_en.vocab_size + 2

    transformer = Transformer(
        N,
        dm,
        h,
        hidden,
        input_vocab,
        target_vocab,
        max_len,
        max_len
    )

    learning_rate = CustomSchedule(dm)
    optimizer = tf.keras.optimizers.Adam(
        learning_rate,
        beta_1=0.9,
        beta_2=0.98,
        epsilon=1e-9
    )
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True,
        reduction='none'
    )
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
        name='train_accuracy'
    )

    def loss_function(real, prediction):
        """Calculate loss while ignoring padding tokens."""
        mask = tf.cast(tf.not_equal(real, 0), tf.float32)
        loss = loss_object(real, prediction)
        loss *= mask

        return tf.math.divide_no_nan(
            tf.reduce_sum(loss),
            tf.reduce_sum(mask)
        )

    @tf.function
    def train_step(inputs, target):
        """Perform one gradient update."""
        target_input = target[:, :-1]
        target_real = target[:, 1:]
        encoder_mask, combined_mask, decoder_mask = create_masks(
            inputs,
            target_input
        )

        with tf.GradientTape() as tape:
            prediction = transformer(
                inputs,
                target_input,
                training=True,
                encoder_mask=encoder_mask,
                look_ahead_mask=combined_mask,
                decoder_mask=decoder_mask
            )
            loss = loss_function(target_real, prediction)

        gradients = tape.gradient(loss, transformer.trainable_variables)
        optimizer.apply_gradients(
            zip(gradients, transformer.trainable_variables)
        )
        train_loss.update_state(loss)
        train_accuracy.update_state(target_real, prediction)

    for epoch in range(epochs):
        train_loss.reset_state()
        train_accuracy.reset_state()

        for batch, (inputs, target) in enumerate(data.data_train):
            train_step(inputs, target)

            if batch % 50 == 0:
                print(
                    'Epoch {}, Batch {}: Loss {}, Accuracy {}'.format(
                        epoch + 1,
                        batch,
                        train_loss.result(),
                        train_accuracy.result()
                    )
                )

        print(
            'Epoch {}: Loss {}, Accuracy {}'.format(
                epoch + 1,
                train_loss.result(),
                train_accuracy.result()
            )
        )

    return transformer
