import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, ReLU, MultiHeadAttention, LayerNormalization
from tensorflow.keras import Model, Sequential, layers
from data_generation import Environment


class OpenWorld(Model):
    def __init__(self):
        super().__init__()
        self.source_embed = layers.Embedding(2, 128)
        self.end_embed = layers.Embedding(2, 128)
        self.action_embed = layers.Embedding(6, 768)

    def call(self, x):
        start = tf.concat([self.source_embed(x[:, :, i]) for i in range(0, 6)], axis=-1)
        #start[:, -1, :] = start[:, -1, :] * mask[:, :, 0]
        action = self.action_embed(x[:, :, 6])
        #action[:, -1, :] = action[:, -1, :] * mask[:, :, 1]
        end = tf.concat([self.end_embed(x[:, :, i]) for i in range(7, 13)], axis=-1)
        #end[:, -1, :] = end[:, -1, :] * mask[:, :, 2]
        return tf.reduce_mean(tf.stack([start, action, end], axis=0), axis=0)

class EPN(Model):
    def __init__(self,sa_iterations,input_dim):
        super().__init__()
        self.attention_iterations = sa_iterations
        self.norm = LayerNormalization()
        self.mha = MultiHeadAttention(num_heads=1,key_dim=64)
        self.add = layers.Add()

        self.add_state = layers.Concatenate(axis=1)
        self.planner = Sequential([
            ReLU(),
            Dense(64),
            Dense(64),
            Dense(input_dim)
        ])

        self.pool = layers.Maximum()
        self.policy_net = Sequential([
            Dense(64),Dense(64),ReLU()
        ])

        self.logits = Dense(64)
        self.baseline = Dense(64)

    def call(self,x,state):
        b= []
        for i in range(self.attention_iterations):
            att = self.norm(x)
            att= self.mha(att,att)
            x = self.add([x,att])
            xc = self.add_state([x,state])
            x= self.planner(x)
            b.append(xc)

        x = self.pool(b)
        x = self.policy_net(x)
        logits = self.logits(x)
        baseline = self.baseline(x)
        return logits, baseline

def get_batch(env, batch_size):
    x, m = env.generate_memory_bank(batch_size)
    x = tf.convert_to_tensor(x)
    print(x)
    #y = x[:,-1,:]
    #xmask  = tf.expand_dims(tf.one_hot(tf.convert_to_tensor(m),3),axis=1)

    #act = tf.one_hot(y[:,6],6,dtype=tf.int64)
    #y = tf.stack([y[:,:6],act,y[:,7:]])

    '''mask = tf.ones_like(y)
    for i in range(mask.shape[1]):
        mask[m[i],i,:] = 0
    mask = mask ==1'''
    return x#y, #xmask,mask

if __name__ == "__main__":
    env =Environment()
    model = EPN(6,768)
    embed = OpenWorld()
    src = get_batch(env, 10)
    em = embed(src)
    state = tf.expand_dims(em[:, -1, :],axis=1)
    out = model(em[:,:-1,:],state)

