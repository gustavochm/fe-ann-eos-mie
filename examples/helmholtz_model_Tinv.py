import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as tfl

PRECISSION = 'float64'

if PRECISSION == 'float32':
    type_np = np.float32
    type_tf = tf.float32
elif PRECISSION == 'float64':
    type_np = np.float64
    type_tf = tf.float64

tf.keras.backend.set_floatx(PRECISSION)

class HelmholtzModel_Tinv(tf.keras.Model):

    def __init__(self, hidden_layers=1, neurons=1, dropout_rate=0.0,
                 seed=None, penalty_cv=1., min_cv=1.5):

        super().__init__()

        self.initializer = tf.keras.initializers.GlorotUniform(seed=seed)
        self.concatenate = tfl.Concatenate()

        self.hidden_layers = []
        #  self.dropout_layers = []
        for i in range(hidden_layers):
            self.hidden_layers.append(tfl.Dense(units=neurons, activation='tanh',
                                                kernel_initializer=self.initializer))
            # self.hidden_layers.append(tfl.Dropout(rate=dropout_rate))

        self.Aad_layer = tfl.Dense(units=1, activation=None, use_bias=False,
                                   kernel_initializer=self.initializer)
        # Add penalty to the loss function
        # self.add_penalty = tf.Variable(add_penalty, name='add_penalty',
        #                                trainable=False, dtype=tf.bool)
        self.penalty_cv = tf.Variable(penalty_cv, name='penalty_cv',
                                      trainable=False, dtype=type_tf)
        self.min_cv = tf.Variable(min_cv, name='min_cv', trainable=False,
                                  dtype=type_tf)


        # Saving values for config
        self.hidden_layers_config = hidden_layers
        self.neurons_config = neurons
        self.dropout_rate_config = dropout_rate
        self.seed_config = seed
        self.penalty_cv_config = penalty_cv
        self.min_cv_config = min_cv


    def get_config(self):
        return {"hidden_layers": self.hidden_layers_config, "neurons" :self.neurons_config,
                "dropout_rate":  self.dropout_rate_config, "seed": self.seed_config, 
                "penalty_cv": self.penalty_cv_config,  "min_cv": self.min_cv_config}

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        if len(data) == 3:
            x, y, sample_weight = data
        else:
            sample_weight = None
            x, y = data
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            # Compute the loss value.
            # The loss function is configured in `compile()`.
            loss = self.compiled_loss(y, y_pred,
                                      sample_weight=sample_weight,
                                      regularization_losses=self.losses)
            # Penalty for Cv < 1.5
            cv_unphysical = tf.math.minimum(y_pred[2], self.min_cv)
            non_zero_cv = tf.cast(cv_unphysical < self.min_cv, type_tf)
            count_non_zero_cv = tf.reduce_sum(non_zero_cv) + 1. # The +1 is to avoid division by 0.
            loss_penalty_cv = tf.reduce_sum(tf.math.square(cv_unphysical - self.min_cv)) / count_non_zero_cv
            loss += self.penalty_cv * loss_penalty_cv

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Update the metrics.
        # Metrics are configured in `compile()`.
        self.compiled_metrics.update_state(y, y_pred, sample_weight=sample_weight)

        # Return a dict mapping metric names to current value.
        # Note that it will include the loss (tracked in self.metrics).
        return {m.name: m.result() for m in self.metrics}

    def call(self, inputs, training=False):
        alpha_in, rhoad_in, Tad_in = inputs

        alpha_in = tf.reshape(alpha_in, [-1, 1])
        rhoad_in = tf.reshape(rhoad_in, [-1, 1])
        Tad_in = tf.reshape(Tad_in, [-1, 1])

        rho0 = tf.zeros_like(rhoad_in)

        with tf.GradientTape(persistent=True) as model_tape2:
            model_tape2.watch(rhoad_in)
            model_tape2.watch(Tad_in)
            with tf.GradientTape(persistent=False) as model_tape1:
                model_tape1.watch(rhoad_in)
                model_tape1.watch(Tad_in)
                inv_Tad = tf.math.reciprocal(Tad_in)
                f_helmholtz = self.concatenate([alpha_in, rhoad_in, inv_Tad])
                f_helmholtz0 = self.concatenate([alpha_in, rho0, inv_Tad])
                for layer in self.hidden_layers:
                    f_helmholtz = layer(f_helmholtz, training=training)
                    f_helmholtz0 = layer(f_helmholtz0, training=training)
                f_helmholtz = self.Aad_layer(f_helmholtz, training=training)
                f_helmholtz0 = self.Aad_layer(f_helmholtz0, training=training)
                helmholtz = f_helmholtz - f_helmholtz0
                # helmholtz = tf.math.multiply(Tad_in, f_helmholtz - f_helmholtz0)

            dhelmholtz_drhoad, dhelmholtz_dTad = model_tape1.gradient(helmholtz, [rhoad_in, Tad_in])
        # d2helmholtz_dTad2 = model_tape2.gradient(dhelmholtz_dTad, Tad_in)
        d2helmholtz_drhoadTad, d2helmholtz_dTad2 = model_tape2.gradient(dhelmholtz_dTad, [rhoad_in, Tad_in])
        d2helmholtz_drhoad2 = model_tape2.gradient(dhelmholtz_drhoad, rhoad_in)

        del model_tape2
        del model_tape1

        # compressibility factor
        Pad_byrhoad = tf.math.multiply(rhoad_in, dhelmholtz_drhoad)
        Pad_byrhoad += Tad_in  # ideal gas Pad/rhoad
        Z = tf.math.divide(Pad_byrhoad, Tad_in)
        Z = tf.reshape(Z, [-1, 1])

        # internal energy
        internal_aux1 = tf.math.multiply(Tad_in, dhelmholtz_dTad)
        internal_aux1 = tf.reshape(internal_aux1, [-1, 1])
        internal = tf.math.subtract(helmholtz, internal_aux1)
        internal += 1.5 * Tad_in  # ideal gas contribution
        internal = tf.reshape(internal, [-1, 1])
        # computing isochoric heat capacity
        Cv = - tf.math.multiply(Tad_in, d2helmholtz_dTad2)
        Cv += 1.5  # ideal gas contribution
        Cv = tf.reshape(Cv, [-1, 1])

        # dP_drho + Tad_in == dPres_drho + dPideal_drho
        # dP_dT + rhoad_in == dPres_dT + dPideal_dT
        dP_dT = tf.math.multiply(tf.math.square(rhoad_in), d2helmholtz_drhoadTad)
        dP_dT += rhoad_in  # ideal gas contribution
        dP_dT = tf.reshape(dP_dT, [-1, 1])

        dP_drho = 2. * tf.math.multiply(rhoad_in, dhelmholtz_drhoad)
        dP_drho += tf.math.multiply(tf.math.square(rhoad_in), d2helmholtz_drhoad2)
        dP_drho += Tad_in  # ideal gas contribution
        dP_drho = tf.reshape(dP_drho, [-1, 1])

        # Thermal Pressure coefficient
        alphap_aux = tf.math.divide(dP_dT, rhoad_in)
        alphap = tf.math.divide(alphap_aux, dP_drho)
        alphap = tf.reshape(alphap, [-1, 1])
        # Isothermal Compressibility
        rho_kappaT = tf.math.reciprocal(dP_drho)
        rho_kappaT = tf.reshape(rho_kappaT, [-1, 1])
        kappaT = tf.math.divide(rho_kappaT, rhoad_in)
        # inv_rhokappaT = tf.math.reciprocal(rho_kappaT)

        # Thermal Pressure Coefficient
        GammaV = tf.math.divide(alphap, kappaT)
        
        # Isobaric Heat Capacity
        Cp_aux1 = tf.math.multiply(Tad_in, tf.math.square(alphap))
        Cp_aux2 = tf.math.multiply(rhoad_in, kappaT)
        Cp = tf.math.add(Cv, tf.math.divide(Cp_aux1, Cp_aux2))
        """
        Both expressions are equal
        Cp_aux = - tf.math.multiply(Tad_in, tf.math.square(dP_dT))
        Cp = tf.math.divide(Cp_aux, -tf.math.multiply(tf.math.square(rhoad_in), dP_drho))
        Cp += Cv
        """
        # Joule Thompson Coefficient
        JT_aux1 = tf.math.subtract(tf.math.multiply(Tad_in, alphap), 1.)
        JT_aux2 = tf.math.multiply(rhoad_in, Cp)
        JT = tf.math.divide(JT_aux1, JT_aux2)

        # Abiabatic expansion coefficient
        Cp_Cv = tf.math.divide(Cp, Cv)

        out = (Z, internal, Cv, Cp, alphap, rho_kappaT, JT, Cp_Cv, GammaV)

        return out

    @tf.function(input_signature=[tf.TensorSpec(shape=(None,), dtype=type_tf),
                                  tf.TensorSpec(shape=(None,), dtype=type_tf),
                                  tf.TensorSpec(shape=(None,), dtype=type_tf)])
    def helmholtz(self, alpha_in, rhoad_in, Tad_in):

        alpha_in = tf.reshape(alpha_in, [-1, 1])
        rhoad_in = tf.reshape(rhoad_in, [-1, 1])
        Tad_in = tf.reshape(Tad_in, [-1, 1])
        rho0 = tf.zeros_like(rhoad_in)

        inv_Tad = tf.math.reciprocal(Tad_in)
        f_helmholtz = self.concatenate([alpha_in, rhoad_in, inv_Tad])
        f_helmholtz0 = self.concatenate([alpha_in, rho0, inv_Tad])
        for layer in self.hidden_layers:
            f_helmholtz = layer(f_helmholtz)
            f_helmholtz0 = layer(f_helmholtz0)
        f_helmholtz = self.Aad_layer(f_helmholtz)
        f_helmholtz0 = self.Aad_layer(f_helmholtz0)
        helmholtz = f_helmholtz - f_helmholtz0

        helmholtz = tf.reshape(helmholtz, (-1, ))

        return helmholtz

    @tf.function(input_signature=[tf.TensorSpec(shape=(None,), dtype=type_tf),
                                  tf.TensorSpec(shape=(None,), dtype=type_tf),
                                  tf.TensorSpec(shape=(None,), dtype=type_tf)])
    def dhelmholtz_drho(self, alpha_in, rhoad_in, Tad_in):

        alpha_in = tf.reshape(alpha_in, [-1, 1])
        rhoad_in = tf.reshape(rhoad_in, [-1, 1])
        Tad_in = tf.reshape(Tad_in, [-1, 1])
        rho0 = tf.zeros_like(rhoad_in)

        with tf.GradientTape(persistent=False) as model_tape:
            model_tape.watch(rhoad_in)
            inv_Tad = tf.math.reciprocal(Tad_in)
            f_helmholtz = self.concatenate([alpha_in, rhoad_in, inv_Tad])
            f_helmholtz0 = self.concatenate([alpha_in, rho0, inv_Tad])
            for layer in self.hidden_layers:
                f_helmholtz = layer(f_helmholtz)
                f_helmholtz0 = layer(f_helmholtz0)
            f_helmholtz = self.Aad_layer(f_helmholtz)
            f_helmholtz0 = self.Aad_layer(f_helmholtz0)
            helmholtz = f_helmholtz - f_helmholtz0

        dhelmholtz_drhoad = model_tape.gradient(helmholtz, rhoad_in)

        helmholtz = tf.reshape(helmholtz, (-1, ))
        dhelmholtz_drhoad = tf.reshape(dhelmholtz_drhoad, (-1, ))

        return helmholtz, dhelmholtz_drhoad

    @tf.function(input_signature=[tf.TensorSpec(shape=(None,), dtype=type_tf),
                                  tf.TensorSpec(shape=(None,), dtype=type_tf),
                                  tf.TensorSpec(shape=(None,), dtype=type_tf)])
    def d2helmholtz_drho2(self, alpha_in, rhoad_in, Tad_in):

        alpha_in = tf.reshape(alpha_in, [-1, 1])
        rhoad_in = tf.reshape(rhoad_in, [-1, 1])
        Tad_in = tf.reshape(Tad_in, [-1, 1])
        rho0 = tf.zeros_like(rhoad_in)

        with tf.GradientTape(persistent=False) as model_tape1:
            model_tape1.watch(rhoad_in)
            with tf.GradientTape(persistent=False) as model_tape2:
                model_tape2.watch(rhoad_in)
                inv_Tad = tf.math.reciprocal(Tad_in)
                f_helmholtz = self.concatenate([alpha_in, rhoad_in, inv_Tad])
                f_helmholtz0 = self.concatenate([alpha_in, rho0, inv_Tad])
                for layer in self.hidden_layers:
                    f_helmholtz = layer(f_helmholtz)
                    f_helmholtz0 = layer(f_helmholtz0)
                f_helmholtz = self.Aad_layer(f_helmholtz)
                f_helmholtz0 = self.Aad_layer(f_helmholtz0)
                helmholtz = f_helmholtz - f_helmholtz0
            dhelmholtz_drhoad = model_tape2.gradient(helmholtz, rhoad_in)

        d2helmholtz_drhoad2 = model_tape1.gradient(dhelmholtz_drhoad, rhoad_in)

        helmholtz = tf.reshape(helmholtz, (-1, ))
        dhelmholtz_drhoad = tf.reshape(dhelmholtz_drhoad, (-1, ))
        d2helmholtz_drhoad2 = tf.reshape(d2helmholtz_drhoad2, (-1, ))

        return helmholtz, dhelmholtz_drhoad, d2helmholtz_drhoad2

    @tf.function(input_signature=[tf.TensorSpec(shape=(None,), dtype=type_tf),
                                  tf.TensorSpec(shape=(None,), dtype=type_tf),
                                  tf.TensorSpec(shape=(None,), dtype=type_tf)])
    def dhelmholtz_dT(self, alpha_in, rhoad_in, Tad_in):

        alpha_in = tf.reshape(alpha_in, [-1, 1])
        rhoad_in = tf.reshape(rhoad_in, [-1, 1])
        Tad_in = tf.reshape(Tad_in, [-1, 1])
        rho0 = tf.zeros_like(rhoad_in)

        with tf.GradientTape(persistent=False) as model_tape:
            model_tape.watch(Tad_in)
            inv_Tad = tf.math.reciprocal(Tad_in)
            f_helmholtz = self.concatenate([alpha_in, rhoad_in, inv_Tad])
            f_helmholtz0 = self.concatenate([alpha_in, rho0, inv_Tad])
            for layer in self.hidden_layers:
                f_helmholtz = layer(f_helmholtz)
                f_helmholtz0 = layer(f_helmholtz0)
            f_helmholtz = self.Aad_layer(f_helmholtz)
            f_helmholtz0 = self.Aad_layer(f_helmholtz0)
            helmholtz = f_helmholtz - f_helmholtz0

        dhelmholtz_dTad = model_tape.gradient(helmholtz, Tad_in)

        helmholtz = tf.reshape(helmholtz, (-1, ))
        dhelmholtz_dTad = tf.reshape(dhelmholtz_dTad, (-1, ))

        return helmholtz, dhelmholtz_dTad

    @tf.function(input_signature=[tf.TensorSpec(shape=(None,), dtype=type_tf),
                                  tf.TensorSpec(shape=(None,), dtype=type_tf),
                                  tf.TensorSpec(shape=(None,), dtype=type_tf)])
    def d2helmholtz_dT2(self, alpha_in, rhoad_in, Tad_in):

        alpha_in = tf.reshape(alpha_in, [-1, 1])
        rhoad_in = tf.reshape(rhoad_in, [-1, 1])
        Tad_in = tf.reshape(Tad_in, [-1, 1])
        rho0 = tf.zeros_like(rhoad_in)

        with tf.GradientTape(persistent=False) as model_tape1:
            model_tape1.watch(Tad_in)
            with tf.GradientTape(persistent=False) as model_tape2:
                model_tape2.watch(Tad_in)
                inv_Tad = tf.math.reciprocal(Tad_in)
                f_helmholtz = self.concatenate([alpha_in, rhoad_in, inv_Tad])
                f_helmholtz0 = self.concatenate([alpha_in, rho0, inv_Tad])
                for layer in self.hidden_layers:
                    f_helmholtz = layer(f_helmholtz)
                    f_helmholtz0 = layer(f_helmholtz0)
                f_helmholtz = self.Aad_layer(f_helmholtz)
                f_helmholtz0 = self.Aad_layer(f_helmholtz0)
                helmholtz = f_helmholtz - f_helmholtz0
            dhelmholtz_dTad = model_tape2.gradient(helmholtz, Tad_in)

        d2helmholtz_dTad2 = model_tape1.gradient(dhelmholtz_dTad, Tad_in)

        helmholtz = tf.reshape(helmholtz, (-1, ))
        dhelmholtz_dTad = tf.reshape(dhelmholtz_dTad, (-1, ))
        d2helmholtz_dTad2 = tf.reshape(d2helmholtz_dTad2, (-1, ))

        return helmholtz, dhelmholtz_dTad, d2helmholtz_dTad2

    @tf.function(input_signature=[tf.TensorSpec(shape=(None,), dtype=type_tf),
                                  tf.TensorSpec(shape=(None,), dtype=type_tf),
                                  tf.TensorSpec(shape=(None,), dtype=type_tf)])
    def dhelmholtz(self, alpha_in, rhoad_in, Tad_in):

        alpha_in = tf.reshape(alpha_in, [-1, 1])
        rhoad_in = tf.reshape(rhoad_in, [-1, 1])
        Tad_in = tf.reshape(Tad_in, [-1, 1])
        rho0 = tf.zeros_like(rhoad_in)

        with tf.GradientTape(persistent=False) as model_tape:
            model_tape.watch(rhoad_in)
            model_tape.watch(Tad_in)
            inv_Tad = tf.math.reciprocal(Tad_in)
            f_helmholtz = self.concatenate([alpha_in, rhoad_in, inv_Tad])
            f_helmholtz0 = self.concatenate([alpha_in, rho0, inv_Tad])
            for layer in self.hidden_layers:
                f_helmholtz = layer(f_helmholtz)
                f_helmholtz0 = layer(f_helmholtz0)
            f_helmholtz = self.Aad_layer(f_helmholtz)
            f_helmholtz0 = self.Aad_layer(f_helmholtz0)
            helmholtz = f_helmholtz - f_helmholtz0

        dhelmholtz_drhoad, dhelmholtz_dTad = model_tape.gradient(helmholtz, [rhoad_in, Tad_in])

        helmholtz = tf.reshape(helmholtz, (-1, ))
        dhelmholtz_drhoad = tf.reshape(dhelmholtz_drhoad, (-1, ))
        dhelmholtz_dTad = tf.reshape(dhelmholtz_dTad, (-1, ))

        out = (helmholtz, dhelmholtz_drhoad, dhelmholtz_dTad)
        return out

    @tf.function(input_signature=[tf.TensorSpec(shape=(None,), dtype=type_tf),
                                  tf.TensorSpec(shape=(None,), dtype=type_tf),
                                  tf.TensorSpec(shape=(None,), dtype=type_tf)])
    def d2helmholtz(self, alpha_in, rhoad_in, Tad_in):

        alpha_in = tf.reshape(alpha_in, [-1, 1])
        rhoad_in = tf.reshape(rhoad_in, [-1, 1])
        Tad_in = tf.reshape(Tad_in, [-1, 1])
        rho0 = tf.zeros_like(rhoad_in)

        with tf.GradientTape(persistent=True) as model_tape1:
            model_tape1.watch(rhoad_in)
            model_tape1.watch(Tad_in)
            with tf.GradientTape(persistent=False) as model_tape2:
                model_tape2.watch(rhoad_in)
                model_tape2.watch(Tad_in)
                inv_Tad = tf.math.reciprocal(Tad_in)
                f_helmholtz = self.concatenate([alpha_in, rhoad_in, inv_Tad])
                f_helmholtz0 = self.concatenate([alpha_in, rho0, inv_Tad])
                for layer in self.hidden_layers:
                    f_helmholtz = layer(f_helmholtz)
                    f_helmholtz0 = layer(f_helmholtz0)
                f_helmholtz = self.Aad_layer(f_helmholtz)
                f_helmholtz0 = self.Aad_layer(f_helmholtz0)
                helmholtz = f_helmholtz - f_helmholtz0

            dhelmholtz_drhoad, dhelmholtz_dTad = model_tape2.gradient(helmholtz, [rhoad_in, Tad_in])

        d2helmholtz_drhoad2 = model_tape1.gradient(dhelmholtz_drhoad, rhoad_in)
        d2helmholtz_drhoad_dTad = model_tape1.gradient(dhelmholtz_drhoad, Tad_in)
        d2helmholtz_dTad2 = model_tape1.gradient(dhelmholtz_dTad, Tad_in)
        del model_tape1
        helmholtz = tf.reshape(helmholtz, (-1, ))
        # first order derivatives
        dhelmholtz_drhoad = tf.reshape(dhelmholtz_drhoad, (-1, ))
        dhelmholtz_dTad = tf.reshape(dhelmholtz_dTad, (-1, ))
        # second order derivatives
        d2helmholtz_drhoad2 = tf.reshape(d2helmholtz_drhoad2, (-1, ))
        d2helmholtz_drhoad_dTad = tf.reshape(d2helmholtz_drhoad_dTad, (-1, ))
        d2helmholtz_dTad2 = tf.reshape(d2helmholtz_dTad2, (-1, ))

        out = (helmholtz, dhelmholtz_drhoad, dhelmholtz_dTad,
               d2helmholtz_drhoad2,  d2helmholtz_dTad2,
               d2helmholtz_drhoad_dTad)
        return out

    @tf.function(input_signature=[tf.TensorSpec(shape=(None,), dtype=type_tf),
                                  tf.TensorSpec(shape=(None,), dtype=type_tf),
                                  tf.TensorSpec(shape=(None,), dtype=type_tf)])
    def internal_energy(self, alpha_in, rhoad_in, Tad_in):

        alpha_in = tf.reshape(alpha_in, [-1, 1])
        rhoad_in = tf.reshape(rhoad_in, [-1, 1])
        Tad_in = tf.reshape(Tad_in, [-1, 1])
        rho0 = tf.zeros_like(rhoad_in)

        with tf.GradientTape(persistent=False) as model_tape:
            model_tape.watch(Tad_in)
            inv_Tad = tf.math.reciprocal(Tad_in)
            f_helmholtz = self.concatenate([alpha_in, rhoad_in, inv_Tad])
            f_helmholtz0 = self.concatenate([alpha_in, rho0, inv_Tad])
            for layer in self.hidden_layers:
                f_helmholtz = layer(f_helmholtz)
                f_helmholtz0 = layer(f_helmholtz0)
            f_helmholtz = self.Aad_layer(f_helmholtz)
            f_helmholtz0 = self.Aad_layer(f_helmholtz0)
            helmholtz = f_helmholtz - f_helmholtz0

        dhelmholtz_dTad = model_tape.gradient(helmholtz, Tad_in)

        internal_aux1 = tf.math.multiply(Tad_in, dhelmholtz_dTad)
        internal_aux1 = tf.reshape(internal_aux1, [-1, 1])
        internal = tf.math.subtract(helmholtz, internal_aux1)
        internal = tf.reshape(internal, (-1,))

        return internal

    @tf.function(input_signature=[tf.TensorSpec(shape=(None,), dtype=type_tf),
                                  tf.TensorSpec(shape=(None,), dtype=type_tf),
                                  tf.TensorSpec(shape=(None,), dtype=type_tf)])
    def cv_residual(self, alpha_in, rhoad_in, Tad_in):

        alpha_in = tf.reshape(alpha_in, [-1, 1])
        rhoad_in = tf.reshape(rhoad_in, [-1, 1])
        Tad_in = tf.reshape(Tad_in, [-1, 1])
        rho0 = tf.zeros_like(rhoad_in)

        with tf.GradientTape(persistent=False) as model_tape2:
            model_tape2.watch(Tad_in)
            with tf.GradientTape(persistent=False) as model_tape1:
                model_tape1.watch(Tad_in)
                inv_Tad = tf.math.reciprocal(Tad_in)
                f_helmholtz = self.concatenate([alpha_in, rhoad_in, inv_Tad])
                f_helmholtz0 = self.concatenate([alpha_in, rho0, inv_Tad])
                for layer in self.hidden_layers:
                    f_helmholtz = layer(f_helmholtz)
                    f_helmholtz0 = layer(f_helmholtz0)
                f_helmholtz = self.Aad_layer(f_helmholtz)
                f_helmholtz0 = self.Aad_layer(f_helmholtz0)
                helmholtz = f_helmholtz - f_helmholtz0
            dhelmholtz_dTad = model_tape1.gradient(helmholtz, Tad_in)

        d2helmholtz_dTad2 = model_tape2.gradient(dhelmholtz_dTad, Tad_in)

        Cv = - tf.math.multiply(d2helmholtz_dTad2, Tad_in)
        Cv = tf.reshape(Cv, (-1,))

        return Cv

    @tf.function(input_signature=[tf.TensorSpec(shape=(None,), dtype=type_tf),
                                  tf.TensorSpec(shape=(None,), dtype=type_tf),
                                  tf.TensorSpec(shape=(None,), dtype=type_tf)])
    def enthalpy(self, alpha_in, rhoad_in, Tad_in):

        alpha_in = tf.reshape(alpha_in, [-1, 1])
        rhoad_in = tf.reshape(rhoad_in, [-1, 1])
        Tad_in = tf.reshape(Tad_in, [-1, 1])
        rho0 = tf.zeros_like(rhoad_in)

        with tf.GradientTape(persistent=False) as model_tape:
            model_tape.watch(rhoad_in)
            model_tape.watch(Tad_in)
            inv_Tad = tf.math.reciprocal(Tad_in)
            f_helmholtz = self.concatenate([alpha_in, rhoad_in, inv_Tad])
            f_helmholtz0 = self.concatenate([alpha_in, rho0, inv_Tad])
            for layer in self.hidden_layers:
                f_helmholtz = layer(f_helmholtz)
                f_helmholtz0 = layer(f_helmholtz0)
            f_helmholtz = self.Aad_layer(f_helmholtz)
            f_helmholtz0 = self.Aad_layer(f_helmholtz0)
            helmholtz = f_helmholtz - f_helmholtz0

        dhelmholtz_drhoad, dhelmholtz_dTad = model_tape.gradient(helmholtz, [rhoad_in, Tad_in])

        internal_aux1 = tf.math.multiply(Tad_in, dhelmholtz_dTad)
        internal_aux1 = tf.reshape(internal_aux1, [-1, 1])
        internal = tf.math.subtract(helmholtz, internal_aux1)
        internal = tf.reshape(internal, (-1,))

        PV = tf.math.multiply(rhoad_in, dhelmholtz_drhoad)
        PV = tf.reshape(PV, (-1, ))

        enthalpy_res = internal + PV

        return enthalpy_res

    @tf.function(input_signature=[tf.TensorSpec(shape=(None,), dtype=type_tf),
                                  tf.TensorSpec(shape=(None,), dtype=type_tf),
                                  tf.TensorSpec(shape=(None,), dtype=type_tf)])
    def chemical_potential(self, alpha_in, rhoad_in, Tad_in):

        alpha_in = tf.reshape(alpha_in, [-1, 1])
        rhoad_in = tf.reshape(rhoad_in, [-1, 1])
        Tad_in = tf.reshape(Tad_in, [-1, 1])
        rho0 = tf.zeros_like(rhoad_in)

        with tf.GradientTape(persistent=False) as model_tape:
            model_tape.watch(rhoad_in)
            inv_Tad = tf.math.reciprocal(Tad_in)
            f_helmholtz = self.concatenate([alpha_in, rhoad_in, inv_Tad])
            f_helmholtz0 = self.concatenate([alpha_in, rho0, inv_Tad])
            for layer in self.hidden_layers:
                f_helmholtz = layer(f_helmholtz)
                f_helmholtz0 = layer(f_helmholtz0)
            f_helmholtz = self.Aad_layer(f_helmholtz)
            f_helmholtz0 = self.Aad_layer(f_helmholtz0)
            helmholtz = f_helmholtz - f_helmholtz0

        dhelmholtz_drhoad = model_tape.gradient(helmholtz, rhoad_in)
        chem_pot = tf.multiply(rhoad_in, dhelmholtz_drhoad)
        chem_pot = tf.reshape(chem_pot, [-1, 1])
        chem_pot = tf.math.add(helmholtz, chem_pot)
        chem_pot = tf.reshape(chem_pot, (-1, ))

        return chem_pot

    @tf.function(input_signature=[tf.TensorSpec(shape=(None,), dtype=type_tf),
                                  tf.TensorSpec(shape=(None,), dtype=type_tf),
                                  tf.TensorSpec(shape=(None,), dtype=type_tf)])
    def pressure(self, alpha_in, rhoad_in, Tad_in):

        alpha_in = tf.reshape(alpha_in, [-1, 1])
        rhoad_in = tf.reshape(rhoad_in, [-1, 1])
        Tad_in = tf.reshape(Tad_in, [-1, 1])
        rho0 = tf.zeros_like(rhoad_in)

        with tf.GradientTape(persistent=False) as model_tape:
            model_tape.watch(rhoad_in)
            inv_Tad = tf.math.reciprocal(Tad_in)
            f_helmholtz = self.concatenate([alpha_in, rhoad_in, inv_Tad])
            f_helmholtz0 = self.concatenate([alpha_in, rho0, inv_Tad])
            for layer in self.hidden_layers:
                f_helmholtz = layer(f_helmholtz)
                f_helmholtz0 = layer(f_helmholtz0)
            f_helmholtz = self.Aad_layer(f_helmholtz)
            f_helmholtz0 = self.Aad_layer(f_helmholtz0)
            helmholtz = f_helmholtz - f_helmholtz0

        dhelmholtz_drhoad = model_tape.gradient(helmholtz, rhoad_in)
        pressure = tf.math.multiply(tf.math.square(rhoad_in), dhelmholtz_drhoad)
        # pressure = tf.reshape(pressure, [-1, 1])
        pressure = tf.reshape(pressure, (-1, ))

        return pressure

    @tf.function(input_signature=[tf.TensorSpec(shape=(None,), dtype=type_tf),
                                  tf.TensorSpec(shape=(None,), dtype=type_tf),
                                  tf.TensorSpec(shape=(None,), dtype=type_tf)])
    def dpressure_drho(self, alpha_in, rhoad_in, Tad_in):
        # alpha_in, rhoad_in, Tad_in = inputs

        alpha_in = tf.reshape(alpha_in, [-1, 1])
        rhoad_in = tf.reshape(rhoad_in, [-1, 1])
        Tad_in = tf.reshape(Tad_in, [-1, 1])
        rho0 = tf.zeros_like(rhoad_in)

        with tf.GradientTape(persistent=False) as model_tape1:
            model_tape1.watch(rhoad_in)
            with tf.GradientTape(persistent=False) as model_tape2:
                model_tape2.watch(rhoad_in)
                inv_Tad = tf.math.reciprocal(Tad_in)
                f_helmholtz = self.concatenate([alpha_in, rhoad_in, inv_Tad])
                f_helmholtz0 = self.concatenate([alpha_in, rho0, inv_Tad])
                for layer in self.hidden_layers:
                    f_helmholtz = layer(f_helmholtz)
                    f_helmholtz0 = layer(f_helmholtz0)
                f_helmholtz = self.Aad_layer(f_helmholtz)
                f_helmholtz0 = self.Aad_layer(f_helmholtz0)
                helmholtz = f_helmholtz - f_helmholtz0
            dhelmholtz_drhoad = model_tape2.gradient(helmholtz, rhoad_in)

        d2helmholtz_drhoad2 = model_tape1.gradient(dhelmholtz_drhoad, rhoad_in)
        pressure = tf.math.multiply(tf.math.square(rhoad_in), dhelmholtz_drhoad)
        pressure = tf.reshape(pressure, (-1, ))

        dpressure = 2. * tf.math.multiply(rhoad_in, dhelmholtz_drhoad)
        dpressure += tf.math.multiply(tf.math.square(rhoad_in), d2helmholtz_drhoad2)
        dpressure = tf.reshape(dpressure, (-1, ))

        return pressure, dpressure

    @tf.function(input_signature=[tf.TensorSpec(shape=(None,), dtype=type_tf),
                                  tf.TensorSpec(shape=(None,), dtype=type_tf),
                                  tf.TensorSpec(shape=(None,), dtype=type_tf)])
    def d2pressure_drho2(self, alpha_in, rhoad_in, Tad_in):

        alpha_in = tf.reshape(alpha_in, [-1, 1])
        rhoad_in = tf.reshape(rhoad_in, [-1, 1])
        Tad_in = tf.reshape(Tad_in, [-1, 1])
        rho0 = tf.zeros_like(rhoad_in)

        with tf.GradientTape(persistent=False) as model_tape1:
            model_tape1.watch(rhoad_in)
            with tf.GradientTape(persistent=False) as model_tape2:
                model_tape2.watch(rhoad_in)
                with tf.GradientTape(persistent=False) as model_tape3:
                    model_tape3.watch(rhoad_in)
                    inv_Tad = tf.math.reciprocal(Tad_in)
                    f_helmholtz = self.concatenate([alpha_in, rhoad_in, inv_Tad])
                    f_helmholtz0 = self.concatenate([alpha_in, rho0, inv_Tad])
                    for layer in self.hidden_layers:
                        f_helmholtz = layer(f_helmholtz)
                        f_helmholtz0 = layer(f_helmholtz0)
                    f_helmholtz = self.Aad_layer(f_helmholtz)
                    f_helmholtz0 = self.Aad_layer(f_helmholtz0)
                    helmholtz = f_helmholtz - f_helmholtz0
                dhelmholtz_drhoad = model_tape3.gradient(helmholtz, rhoad_in)
            d2helmholtz_drhoad2 = model_tape2.gradient(dhelmholtz_drhoad, rhoad_in)
        d3helmholtz_drhoad3 = model_tape1.gradient(d2helmholtz_drhoad2, rhoad_in)

        pressure = tf.math.multiply(tf.math.square(rhoad_in), dhelmholtz_drhoad)
        pressure = tf.reshape(pressure, (-1, ))

        dpressure = 2. * tf.math.multiply(rhoad_in, dhelmholtz_drhoad)
        dpressure += tf.math.multiply(tf.math.square(rhoad_in), d2helmholtz_drhoad2)
        dpressure = tf.reshape(dpressure, (-1, ))

        d2pressure = 2. * dhelmholtz_drhoad
        d2pressure += 4. * tf.math.multiply(rhoad_in, d2helmholtz_drhoad2)
        d2pressure += tf.math.multiply(tf.math.square(rhoad_in), d3helmholtz_drhoad3)
        d2pressure = tf.reshape(d2pressure, (-1, ))
        return pressure, dpressure, d2pressure

    @tf.function(input_signature=[tf.TensorSpec(shape=(None,), dtype=type_tf),
                                  tf.TensorSpec(shape=(None,), dtype=type_tf),
                                  tf.TensorSpec(shape=(None,), dtype=type_tf)])
    def dpressure_dT(self, alpha_in, rhoad_in, Tad_in):

        alpha_in = tf.reshape(alpha_in, [-1, 1])
        rhoad_in = tf.reshape(rhoad_in, [-1, 1])
        Tad_in = tf.reshape(Tad_in, [-1, 1])
        rho0 = tf.zeros_like(rhoad_in)

        with tf.GradientTape(persistent=False) as model_tape2:
            model_tape2.watch(Tad_in)
            with tf.GradientTape(persistent=False) as model_tape1:
                model_tape1.watch(rhoad_in)
                model_tape1.watch(Tad_in)
                inv_Tad = tf.math.reciprocal(Tad_in)
                f_helmholtz = self.concatenate([alpha_in, rhoad_in, inv_Tad])
                f_helmholtz0 = self.concatenate([alpha_in, rho0, inv_Tad])
                for layer in self.hidden_layers:
                    f_helmholtz = layer(f_helmholtz)
                    f_helmholtz0 = layer(f_helmholtz0)
                f_helmholtz = self.Aad_layer(f_helmholtz)
                f_helmholtz0 = self.Aad_layer(f_helmholtz0)
                helmholtz = f_helmholtz - f_helmholtz0
            dhelmholtz_drhoad = model_tape1.gradient(helmholtz, rhoad_in)

        d2helmholtz_dTaddrho = model_tape2.gradient(dhelmholtz_drhoad, Tad_in)

        pressure = tf.math.multiply(tf.math.square(rhoad_in), dhelmholtz_drhoad)
        pressure = tf.reshape(pressure, (-1,))

        dpressure_dTad = tf.math.multiply(d2helmholtz_dTaddrho, tf.math.square(rhoad_in))
        dpressure_dTad = tf.reshape(dpressure_dTad, (-1,))
        return pressure, dpressure_dTad
