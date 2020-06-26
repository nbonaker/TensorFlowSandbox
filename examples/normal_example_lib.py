import numpy as np
import tensorflow as tf


@tf.function
def normal_lp_suff(mu, mu2, tau, log_tau, xsum, x2sum, num_obs):
    mu_dim = len(xsum)
    quad_term = \
        tf.linalg.trace(x2sum) - \
        2.0 * tf.tensordot(xsum, mu, 1) + \
        num_obs * tf.linalg.trace(mu2)
    lp = \
        -0.5 * tau * quad_term + \
        0.5 * mu_dim * num_obs * log_tau
    return lp


@tf.function
def normal_lp(par, data):
    mu = tf.convert_to_tensor(par['mu'], dtype=tf.float64)
    tau = tf.convert_to_tensor(par['tau'], dtype=tf.float64)
    return normal_lp_suff(
        mu=mu,
        mu2=tf.tensordot(mu, mu, 0),
        tau=tau,
        log_tau=tf.math.log(tau),
        **data)


@tf.function
def flatten_par(par):
    tau = tf.convert_to_tensor(par['tau'], dtype=tf.float64)
    tau = tf.reshape(tau, (1, ))
    return tf.concat([ par['mu'], tf.math.log(tau) ], axis=0)


@tf.function
def fold_par(par_flat):
    par_flat = tf.convert_to_tensor(par_flat)
    par_len = par_flat.get_shape()[0]
    mu_dim = len(par_flat) - 1
    return { 'mu': par_flat[0:mu_dim], 'tau': tf.math.exp(par_flat[mu_dim]) }


def get_objectives(data):
    # These cannot be decorated with @tf.function because they need to
    # be able to return numpy for use in scipy optimize.
    def normal_objective(par_flat, to_numpy=True):
        lp = -1 * normal_lp(fold_par(par_flat), data)
        print(lp.numpy())
        if to_numpy:
            return lp.numpy()
        else: return lp

    def normal_objective_grad(par_flat, to_numpy=True):
        par_flat_tf = tf.Variable(par_flat)
        with tf.GradientTape() as tape:
            lp = normal_objective(par_flat_tf, to_numpy=False)
        grad = tape.gradient(lp, par_flat_tf)
        if to_numpy:
            return grad.numpy()
        else:
            return grad

    def normal_objective_hessian(par_flat, to_numpy=True):
        par_flat_tf = tf.Variable(par_flat)
        with tf.GradientTape() as tape:
            with tf.GradientTape() as gtape:
                lp = normal_objective(par_flat_tf, to_numpy=False)
            grad = gtape.gradient(lp, par_flat_tf)
        hess = tape.jacobian(grad, par_flat_tf)
        if to_numpy:
            return hess.numpy()
        else:
            return hess

    return normal_objective, normal_objective_grad, normal_objective_hessian
