import tensorflow as tf
from helpers.tf_reflectance import tf_reflectance

class Layer(tf.keras.layers.Layer):
    def __init__(self, infinite=False, **kwargs):
        super().__init__(**kwargs)

        self.n = tf.constant(1.4, dtype=tf.float32)

        if infinite:
            self.infinite = True
        else:
            self.h = tf.Variable(10.0)
    
    def mu_a(self):
        pass
    
    def mu_s(self):
        pass

class ConstLayer(Layer):
    def __init__(self, mu_a, mu_s, **kwargs):
        super().__init__(**kwargs)

        self._mu_a = tf.constant(mu_a)
        self._mu_s = tf.constant(mu_s)

    def mu_s(self):
        return self._mu_s

    def mu_a(self):
        return self._mu_a

class VariableLayer(Layer):
    def __init__(self, wavelengths, coeffs, **kwargs):
        super().__init__(**kwargs)

        self.wavelengths = tf.constant(wavelengths, dtype=tf.float32)

        # Scattering
        self.a = tf.Variable(1.0)
        self.b = tf.Variable(1.0)

        # Absorption
        self.coeffs = tf.constant(coeffs, dtype=tf.float32)
        self.n_concs = self.coeffs.shape[1] # maybe the wrong one
        self.concs = tf.Variable(tf.ones(self.n_concs), dtype=tf.float32)

    def mu_s(self):
        return (self.a*(self.wavelengths / 500)**(-self.b))[:, 0]

    def mu_a(self):
        return tf.transpose(self.coeffs @ self.concs[:, None])[0, ...]

"""
Material has a forward model which spits out the reflectance
"""
class Material(tf.keras.layers.Layer):
    def __init__(self, layers, n_ext=1.4, a=100, MaxIter=10000):
        super().__init__()

        self.layers = layers

        self.n_ext = tf.constant(n_ext)
        self.z = tf.constant(0.0)
        self.a = tf.constant(a)
        self.MaxIter = tf.constant(10000)
    
    def reflectance(self, distance):
        mu_as, mu_ss = self._build_coeffs()

        size = mu_as.shape[0]

        hs = [l.h for l in self.layers]
        hs = self._build_constant_scalar_part(size, hs)

        ns = [l.n for l in self.layers]
        ns = self._build_constant_scalar_part(size, ns)

        shape = ns.shape[:-1]
        shape = tf.constant(shape + [1])

        zs = tf.broadcast_to(self.z, shape)
        rhos = tf.broadcast_to(distance, shape)

        ref = tf_reflectance(mu_as, mu_ss, ns, hs, zs, rhos)
        return ref

    def _build_coeffs(self):
        mu_as = [l.mu_a() for l in self.layers]
        mu_ss = [l.mu_s() for l in self.layers]

        max_size = tf.reduce_max([tf.size(e) for e in mu_as])

        mu_as = [tf.broadcast_to(e[..., None], (max_size, 1)) for e in mu_as]
        mu_ss = [tf.broadcast_to(e[..., None], (max_size, 1)) for e in mu_ss]

        mu_as = tf.concat(mu_as, -1)
        mu_ss = tf.concat(mu_ss, -1)

        return mu_as, mu_ss
        
    @staticmethod
    def _build_constant_scalar_part(size, vals):
        result = [tf.broadcast_to(e[..., None, None], (size, 1)) for e in vals]
        result = tf.concat(result, -1)

        return result

    def grad_reflectance():
        pass

    def call(self, i):
        return self.reflectance(i)