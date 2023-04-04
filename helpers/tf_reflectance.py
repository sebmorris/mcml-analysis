import tensorflow as tf
import numpy as np
from julia import Main, LightPropagation

"""
accepts numpy arguments

n layers
mu_a    [bulk, n]
mu_s    [bulk, n]
n       [bulk, n]
h       [bulk, n]
z       [bulk, 1]
rho     [bulk, 1]
"""

pack_call = lambda mu_a, mu_s, n, h, z, rho: mu_a + mu_s + n + h + [z] + [rho]
unpack_result = lambda e, n: (e[..., :n], e[..., n:2*n], e[..., 2*n:3*n], e[..., 3*n:4*n], e[..., -2, None], e[..., -1, None])

# julia indexing starts at 1
julia_unpack_call = lambda n_ext, a, MaxIter: f"""
function unpack_call(x)
    len = length(x)
    n_layers = (len - 2) ÷ 4

    mu_a = x[1:n_layers]
    mu_s = x[n_layers+1:2*n_layers]
    n = x[2*n_layers+1:3*n_layers]
    h = x[3*n_layers+1:4*n_layers]
    z = x[end-1]
    rho = x[end]

    LightPropagation.flux_DA_Nlay_cylinder_CW(
        rho, mu_a, mu_s; n_ext={n_ext}, n_med=n, l=h, a={a}, z=z, MaxIter={MaxIter}
    )
end
"""

#Main.eval(julia_unpack_call)
Main.eval("using LightPropagation; using ForwardDiff")

def reflectance(mu_as, mu_ss, ns, hs, zs, rhos, n_ext=1.4, a=100.0, MaxIter=10000):
    result = np.zeros(rhos.shape)
    for i in np.ndindex(mu_as.shape[:-1]):
        z, rho = [e[i][0].item() for e in [zs, rhos]]
        mu_a, mu_s, n, h = [e[i].tolist() for e in [mu_as, mu_ss, ns, hs]]
        result[i][0] = LightPropagation.flux_DA_Nlay_cylinder_CW(
            rho, mu_a, mu_s, n_ext=n_ext, n_med=n, l=h, a=a, z=z, MaxIter=MaxIter
        )

    return result

def reflectance_gradient(mu_as, mu_ss, ns, hs, zs, rhos, n_ext=1.4, a=100.0, MaxIter=10000):
    Main.eval(julia_unpack_call(n_ext, a, MaxIter))
    calc_result = Main.eval("combined -> ForwardDiff.gradient(unpack_call, combined)")

    n_layers = mu_as.shape[-1]
    grad_dim = n_layers*4 + 2
    grad_shape = np.array(rhos.shape)
    grad_shape[-1] = grad_dim

    gradient_result = np.zeros(grad_shape)
    for i in np.ndindex(mu_as.shape[:-1]):
        z, rho = [e[i][0].item() for e in [zs, rhos]]
        mu_a, mu_s, n, h = [e[i].tolist() for e in [mu_as, mu_ss, ns, hs]]

        combined = pack_call(mu_a, mu_s, n, h, z, rho)

        gradient_result[i] = calc_result(combined)
    
    result = unpack_result(gradient_result, n_layers)
    return [e.astype(np.single) for e in result]

@tf.custom_gradient
def tf_reflectance(mu_as, mu_ss, ns, hs, zs, rhos, n_ext=1.4, a=100.0, MaxIter=10000):
    kwargs = dict(n_ext=n_ext, a=a, MaxIter=MaxIter)
    wrap_ref = lambda *args: reflectance(*args, **kwargs)
    wrap_grad_ref = lambda *args: reflectance_gradient(*args, **kwargs)

    args = [mu_as, mu_ss, ns, hs, zs, rhos]
    result = tf.numpy_function(wrap_ref, args, Tout=tf.float32)

    def gradient(dy):
        grad = tf.numpy_function(wrap_grad_ref, args, Tout=tf.float32)
        
        dy = tf.cast(dy, tf.float32)
        mu_a = grad[0]*dy
        mu_s = grad[1]*dy
        n = grad[2]*dy
        h = grad[3]*dy
        z = grad[4]*dy
        rho = grad[5]*dy
        
        return mu_a, mu_s, n, h, z, rho
    
    return result, gradient
