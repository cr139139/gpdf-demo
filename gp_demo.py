import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import jax.numpy as jp
from jax import vmap
from jax.config import config

config.update("jax_enable_x64", False)

# parameter settings for Matern 1/2 GPIS
l = 1.


def pair_dist(x1, x2):
    return jp.linalg.norm(x2 - x1)


def cdist(x, y):
    return vmap(lambda x1: vmap(lambda y1: pair_dist(x1, y1))(y))(x)


def pair_diff(x1, x2):
    return x1 - x2


def cdiff(x, y):
    return vmap(lambda x1: vmap(lambda y1: pair_diff(x1, y1))(y))(x)


def cov(x1, x2):
    d = cdist(x1, x2)
    return jp.exp(-d / l)


def cov_grad(x1, x2):
    pair_diff = cdiff(x1, x2)
    d = jp.linalg.norm(pair_diff, axis=2)
    return (-jp.exp(-d / l) / l / d).reshape((x1.shape[0], x2.shape[0], 1)) * pair_diff


def cov_hessian(x1, x2):
    pair_diff = cdiff(x1, x2)
    d = jp.linalg.norm(pair_diff, axis=2)
    h = (pair_diff.reshape((x1.shape[0], x2.shape[0], 3, 1)) @ pair_diff.reshape((x1.shape[0], x2.shape[0], 1, 3)))
    hessian_constant_1 = (jp.exp(-d / l)).reshape((x1.shape[0], x2.shape[0], 1, 1))
    hessian_constant_2 = (1 / (l ** 2 * d ** 2) + 1 / (l * d ** 3)).reshape((x1.shape[0], x2.shape[0], 1, 1))
    hessian_constant_3 = (1 / (l * d)).reshape((x1.shape[0], x2.shape[0], 1, 1)) * jp.eye(3).reshape((1, 1, 3, 3))
    return hessian_constant_1 * (hessian_constant_2 * h - hessian_constant_3)


# sampling the surface of a sphere
def fibonacci_sphere(num_samples):
    phi = jp.pi * (3. - jp.sqrt(5.))  # golden angle in radians
    y = jp.linspace(1, -1, num_samples)
    radius = jp.sqrt(1 - y * y)
    theta = phi * jp.arange(num_samples)
    x = jp.cos(theta) * radius
    z = jp.sin(theta) * radius
    return x, y, z


def reverting_function(x):
    return -l * jp.log(x)


def reverting_function_derivative(x):
    return -l / x


def reverting_function_second_derivative(x):
    return l / x ** 2


if __name__ == "__main__":
    # creating a sphere point cloud
    N_obs = 1000  # number of observations
    sphereRadius = 1
    xa, yb, zc = fibonacci_sphere(N_obs)
    sphere = sphereRadius * jp.concatenate([xa.reshape(-1, 1), yb.reshape(-1, 1), zc.reshape(-1, 1)], axis=1)

    # using a 2D plane to query the distances
    start = -1.5
    end = 1.5
    samples = int((end - start) / 0.02) + 1
    xg, yg = jp.meshgrid(jp.linspace(start, end, samples), jp.linspace(start, end, samples));
    querySlice = jp.concatenate([xg.reshape(-1, 1), yg.reshape(-1, 1), jp.zeros(xg.shape).reshape(-1, 1)], axis=1)

    K = cov(sphere, sphere)
    k = cov(querySlice, sphere)
    y = jp.ones((N_obs, 1))
    model = jp.linalg.solve(K, y)

    # distance inference
    mu = k @ model
    mean = reverting_function(mu)

    # gradient inference
    covariance_grad = cov_grad(querySlice, sphere)
    mu_grad = jp.moveaxis(covariance_grad, -1, 0) @ model
    grad = reverting_function_derivative(mu) * mu_grad

    # hessian inference
    hessian = (jp.moveaxis(mu_grad, 0, 1)
               @ jp.moveaxis(mu_grad, 0, 2)
               * reverting_function_second_derivative(mu)[:, :, None])
    covariance_hessian = cov_hessian(querySlice, sphere)
    mu_hessian = ((jp.moveaxis(covariance_hessian, 1, -1) @ model)[..., 0]
                  * reverting_function_derivative(mu)[:, :, None])
    hessian += mu_hessian

    # gradient normalization
    grad_orig = jp.copy(grad)
    norms = jp.linalg.norm(grad, axis=0, keepdims=True)
    grad = jp.where(norms != 0, grad, grad / jp.min(jp.abs(grad), axis=0))
    grad /= jp.linalg.norm(grad, axis=0, keepdims=True)

    # GPIS visualization
    xg = querySlice[:, 0].reshape(xg.shape)
    yg = querySlice[:, 1].reshape(yg.shape)
    zg = jp.zeros(xg.shape)

    xd = grad[0].reshape(xg.shape)
    yd = grad[1].reshape(yg.shape)
    zd = grad[2].reshape(xg.shape)
    colors = jp.arctan2(xd, yd)

    fig = plt.figure(figsize=(18, 3))
    fig.suptitle('GPDF result')

    # First subplot
    ax = fig.add_subplot(1, 5, 1, projection='3d')
    ax.scatter(sphere[:, 0], sphere[:, 1], sphere[:, 2], alpha=0.1, c='k')
    ax.set_box_aspect([1.0, 1.0, 1.0])
    ax.set_title('Point cloud')

    ax = fig.add_subplot(1, 5, 2, projection='3d')
    surf = ax.plot_surface(xg, yg, mean.reshape(xg.shape), cmap=cm.magma,
                           linewidth=0, antialiased=False)
    ax.set_box_aspect([1.0, 1.0, 1.0])
    plt.colorbar(surf, shrink=0.8, ax=ax)
    ax.set_title('Distance field')

    # Second subplot
    ax = fig.add_subplot(1, 5, 3)
    gradient = ax.scatter(xg, yg,
                          c=colors, cmap=cm.twilight,
                          s=5)
    ax.set_aspect('equal')
    ax.set_xlim([start, end])
    ax.set_ylim([start, end])
    plt.colorbar(gradient, shrink=0.8, ax=ax)
    ax.set_title('Gradient field')
    ax.set_facecolor('black')

    ax = fig.add_subplot(1, 5, 4)
    K = jp.tile(jp.eye(4), (hessian.shape[0], 1, 1))
    K = K.at[:, :3, :3].set(hessian)
    K = K.at[:, :3, 3].set(grad_orig[:, :, 0].T)
    K = K.at[:, 3, :3].set(grad_orig[:, :, 0].T)
    gaussian_curvature = -(jp.linalg.det(K) / norms.flatten() ** 4).reshape((xg.shape[0], xg.shape[1]))
    curvature = ax.scatter(xg, yg,
                           c=gaussian_curvature,
                           cmap=cm.magma,
                           s=5, vmin=-0.5-1, vmax=-0.5+1)
    ax.set_aspect('equal')
    ax.set_xlim([start, end])
    ax.set_ylim([start, end])
    plt.colorbar(curvature, shrink=0.8, ax=ax)
    ax.set_title('Gaussian curvature field')
    ax.set_facecolor('black')

    ax = fig.add_subplot(1, 5, 5)
    mean_curvature = (jp.moveaxis(grad_orig, 0, 2)
                      @ hessian @ jp.moveaxis(grad_orig, 0, 1)).flatten()
    mean_curvature -= jp.trace(hessian, axis1=1, axis2=2) * norms.flatten() ** 2
    mean_curvature /= 2 * norms.flatten() ** 3
    mean_curvature = mean_curvature.reshape((xg.shape[0], xg.shape[1]))
    curvature = ax.scatter(xg, yg,
                           c=mean_curvature,
                           cmap=cm.magma,
                           s=5, vmin=-0.5-1, vmax=-0.5+1)
    ax.set_aspect('equal')
    ax.set_xlim([start, end])
    ax.set_ylim([start, end])
    plt.colorbar(curvature, shrink=0.8, ax=ax)
    ax.set_title('Mean curvature field')
    ax.set_facecolor('black')

    plt.show()
