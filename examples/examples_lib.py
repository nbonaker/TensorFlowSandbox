import numpy as np
import tensorflow as tf


######################
# Quadratic

def get_quadratic_obj(x, a, b):
    return 0.5 * tf.einsum('i,j,ij', x, x, a) + tf.tensordot(x, b, axes=1)


def get_quadratic_params(dim):
    a = np.random.random((dim, dim))
    a = a @ a.T
    b = np.random.random(dim)

    x = np.random.random(dim)

    return a, b, x


######################
# Normal

def get_norm_ll_vec(x, mu, sigma):
    mu = tf.reshape(mu, (1, ))
    sigma = tf.reshape(sigma, (1, ))
    return \
        -0.5 * (x - mu) ** 2 / (sigma ** 2) \
        - tf.math.log(sigma)


def get_norm_ll(x, mu, sigma):
    return tf.reduce_sum(get_norm_ll_vec(x, mu, sigma))


def get_norm_ll_infl(x, mu, sigma, w):
    return tf.reduce_sum(w * get_norm_ll_vec(x, mu, sigma))


def get_normal_params(num_obs):
    mu = 2.0
    sigma = 1.5
    x = np.random.normal(loc=mu, scale=sigma, size=num_obs)
    w = np.ones(num_obs)
    return mu, sigma, x, w


######################
# Normal clustering

def get_norm_clustering_mstep_mat(x, mu_vec, sigma_vec, pi_vec, e_z):
    num_clusters = pi_vec.shape[0]
    ll_mat = tf.stack(
        [ get_norm_ll_vec(x, mu_vec[k], sigma_vec[k]) \
          for k in range(num_clusters) ], axis=1)

    prior_mat = tf.math.log(tf.expand_dims(pi_vec, 0) * e_z)

    return ll_mat + prior_mat

def get_norm_clustering_mstep(x, mu_vec, sigma_vec, pi_vec, e_z):
    return tf.reduce_sum(get_norm_clustering_mstep_mat(x, mu_vec, sigma_vec, pi_vec, e_z))


def get_norm_cluster_params(num_clusters, num_obs):
    mu_vec = np.linspace(1, num_clusters, num=num_clusters)
    sigma_vec = np.linspace(0.1, 0.2, num=num_clusters)
    pi_vec = np.linspace(1, num_clusters, num=num_clusters)
    pi_vec = pi_vec / np.sum(pi_vec)

    z_true = np.random.multinomial(1, pi_vec, size=(num_obs, ))
    k_true = np.argwhere(z_true == 1)[:, 1]
    x = np.full(num_obs, float('nan'))
    for n in range(num_obs):
        x[n] = np.random.normal(
            loc=mu_vec[k_true[n]],
            scale=sigma_vec[k_true[n]],
            size=1)

    return x, mu_vec, sigma_vec, pi_vec, z_true


######################
# Logistic

def get_logistic_ll_vec(y, x_mat, theta):
    z = x_mat @ theta
    return y * z + tf.math.log1p(z)

def get_logistic_ll(y, x_mat, theta):
    return tf.reduce_sum(get_logistic_ll_vec(y, x_mat, theta))


def get_logistic_params(num_obs, dim):
    x_mat = np.hstack([ np.ones((num_obs, 1)), np.random.random((num_obs, dim - 1)) ])
    theta = np.random.random(dim) / (10 * dim)
    y = (np.random.random(num_obs) < x_mat @ theta).astype(np.float64)
    return theta, x_mat, y
