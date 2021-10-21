import tensorflow as tf
import warnings
warnings.filterwarnings("ignore")


class TwoTowersModel:
    def __init__(self,
                 seq_length,
                 vocab_size,
                 embedding_dim
                 ):
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

    def model(self):
        # input
        # user sequence feature
        self.inputs_item_seq = tf.keras.Input(shape=(self.seq_length,), dtype=tf.int32)
        self.inputs_time_seq = tf.keras.Input(shape=(self.seq_length,), dtype=tf.float32)
        # other user feature
        '''self.inputs_user_age = tf.keras.Input(shape=(1,), dtype=tf.int32)
        self.inputs_user_gender = tf.keras.Input(shape=(1,), dtype=tf.int32)
        self.inputs_user_level = tf.keras.Input(shape=(1,), dtype=tf.int32)'''
        # the next item the user interact with
        self.inputs_item_next = tf.keras.Input(shape=(1,), dtype=tf.int32)
        self.embedding_ = tf.keras.layers.Embedding(input_dim=self.vocab_size,
                                                    output_dim=self.embedding_dim
                                                    )
        self.cat_embedding = self.embedding_(self.inputs_item_seq)
        self.item_embedding = tf.squeeze(self.embedding_(self.inputs_item_next), axis=1)

        # float embedding
        list_float_embedding = []
        split_ = tf.split(self.inputs_time_seq, num_or_size_splits=self.seq_length, axis=-1)
        dense_ = tf.keras.layers.Dense(units=self.embedding_dim, activation=tf.nn.relu)
        for s in split_:
            float_embedding_ = tf.expand_dims(dense_(s), 1)
            list_float_embedding.append(float_embedding_)
        self.float_embedding = tf.concat(list_float_embedding, axis=1)

        # print(self.cat_embedding,self.float_embedding)
        # concat
        self.concat = tf.keras.layers.concatenate([self.cat_embedding, self.float_embedding], axis=-1)

        # print(self.concat)
        #         self.concat = tf.Tensor(self.concat,dtype=tf.float32)
        # define LSTM
        self.lstm = tf.keras.layers.LSTM(units=64)(self.concat)

        # user DNN
        self.dense_user = tf.keras.layers.Dense(units=64, activation=tf.nn.relu)(self.lstm)

        # item DNN
        self.dense_item = tf.keras.layers.Dense(units=64, activation=tf.nn.relu)(self.item_embedding)

        # a * b
        self.similar = tf.reduce_sum(tf.multiply(self.dense_user, self.dense_item), axis=-1)

        # softmax
        self.outputs = tf.nn.sigmoid(self.similar)

        # output
        self.model_user = tf.keras.Model(inputs=[self.inputs_item_seq, self.inputs_time_seq], outputs=self.dense_user)
        self.model_item = tf.keras.Model(inputs=[self.inputs_item_next], outputs=self.dense_item)

        self.model = tf.keras.Model(inputs=[self.inputs_item_seq, self.inputs_time_seq, self.inputs_item_next],
                                    outputs=self.outputs)

        return self.model, self.model_user, self.model_item


if __name__ == "__main__":
    twoTower = TwoTowersModel(seq_length=56, vocab_size=1858338, embedding_dim=128)
    model2, user_tower, item_tower = twoTower.model()
    print(user_tower.summary())
