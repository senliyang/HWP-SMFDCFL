import numpy as np
import pandas as pd
import scipy.sparse as sp
import time
import tensorflow.compat.v1 as tf
from model import GCNModelAE, GCNModelVAE
from optimizer import OptimizerAE, OptimizerVAE


# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 250, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 100, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2',64, 'Number of units in hidden layer 2.')
flags.DEFINE_float('weight_decay', 0, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_float('dropout', 0 , 'Dropout rate (1 - keep probability).')
flags.DEFINE_integer('features', 1, 'Whether to use features (1) or not (0).')



#W is the matrix which needs to be normalized
def new_normalization (w):
    m = w.shape[0]
    p = np.zeros([m,m])
    for i in range(m):
        for j in range(m):
            if i == j:
                p[i][j] = 1/2
            elif np.sum(w[i,:])-w[i,i]>0:
                p[i][j] = w[i,j]/(2*(np.sum(w[i,:])-w[i,i]))
    return p

def set_digo_zero(sim, z):
    sim_new = sim.copy()
    n = sim.shape[0]
    for i in range(n):
        sim_new[i][i] = z
    return sim_new

def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape


def preprocess_graph(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return sparse_to_tuple(adj_normalized)

def construct_feed_dict(adj_normalized, adj, features, placeholders):
    # construct feed dictionary
    feed_dict = dict()
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['adj']: adj_normalized})
    feed_dict.update({placeholders['adj_orig']: adj})
    return feed_dict


A =  pd.read_csv('MD_A.csv', header=0, index_col=0).values
print("the number of LncRNAs and diseases", A.shape)
print("the number of associations", sum(sum(A)))
x,y = A.shape

sim_d =  pd.read_csv('d_sim_final.txt', header=None, index_col=None,sep=' ').values

sim_d_0 = set_digo_zero(new_normalization(sim_d), 0)

print("the maxmum of simd network",np.max(np.max(sim_d_0, axis = 0)), "the minimum of simd network", np.min(np.min(sim_d_0, axis=0)))

#getting features by adjacency matrix
features_d = A.transpose()

# Define placeholders
tf.disable_eager_execution()
placeholders = {
    'features': tf.sparse_placeholder(tf.float32),
    'adj': tf.sparse_placeholder(tf.float32),
    'adj_orig': tf.sparse_placeholder(tf.float32),
    'dropout': tf.placeholder_with_default(0., shape=())
}

#training disease by VGAE
features_d = sp.coo_matrix(features_d)

# Some preprocessing
adj_norm = preprocess_graph(sim_d_0)
num_nodes = sim_d.shape[0]

features = sparse_to_tuple(features_d.tocoo())
num_features = features[2][1]
features_nonzero = features[1].shape[0]

# Create model
model = GCNModelVAE(placeholders, num_features, num_nodes, features_nonzero)
pos_weight = 5
norm = 0.1
# Optimizer
with tf.name_scope('optimizer'):
        opt = OptimizerVAE(preds=model.reconstructions,
                           labels=tf.reshape(tf.sparse_tensor_to_dense(placeholders['adj_orig'],
                                                                       validate_indices=False), [-1]),
                           model=model, num_nodes=num_nodes,
                           pos_weight=pos_weight,
                           norm=norm)

# Initialize session
sess = tf.Session()
sess.run(tf.global_variables_initializer())
cost_val = []
acc_val = []
val_roc_score = []

sim_d_0 = sp.coo_matrix(sim_d_0)
sim_d_0.eliminate_zeros()
adj_label = sim_d_0 + sp.eye(sim_d_0.shape[0])
adj_label = sparse_to_tuple(adj_label)

# Train model
for epoch in range(FLAGS.epochs):

    t = time.time()
    # Construct feed dictionary
    feed_dict = construct_feed_dict(adj_norm, adj_label, features, placeholders)
    feed_dict.update({placeholders['dropout']: FLAGS.dropout})
    # Run single weight update
    outs = sess.run([opt.opt_op, opt.cost, opt.accuracy], feed_dict=feed_dict)
    # Compute average loss
    avg_cost = outs[1]
    avg_accuracy = outs[2]
    print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(avg_cost),
          "time=", "{:.5f}".format(time.time() - t))
print("Optimization Finished!")
#getting the feature vectors for miRNAs
Z = sess.run(model.z, feed_dict=feed_dict)
Z =np.array(Z)
feature_d = Z
print(feature_d.shape)
np.savetxt('VGAE_64_disease_feature.csv', feature_d, delimiter=',')