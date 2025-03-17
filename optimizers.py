import numpy as np

class Optimizer:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    def update(self, parameters, gradients):
        """Update parameters based on gradients"""
        updated_params = parameters.copy()
        for key in parameters:
            if key in gradients:
                updated_params[key] -= self.learning_rate * gradients[key]
        return updated_params

class SGD(Optimizer):
    def update(self, parameters, gradients):
        """Simple Stochastic Gradient Descent"""
        for key in parameters:
            if key.startswith('W') or key.startswith('b'):
                parameters[key] -= self.learning_rate * gradients[key]
        return parameters

class Momentum(Optimizer):
    def __init__(self, learning_rate=0.01, momentum=0.9):
        super().__init__(learning_rate)
        self.momentum = momentum
        self.velocity = {}

    def initialize(self, parameters):
        """Initialize velocity terms"""
        for key in parameters:
            self.velocity[key] = np.zeros_like(parameters[key])

    def update(self, parameters, gradients):
        """Momentum-based gradient descent"""
        if not self.velocity:
            self.initialize(parameters)

        for key in parameters:
            # Update velocity
            self.velocity[key] = (self.momentum * self.velocity[key] 
                                  + (1 - self.momentum) * gradients[key])
            # Update parameters
            parameters[key] -= self.learning_rate * self.velocity[key]
        return parameters

class NAG(Optimizer):
    def __init__(self, learning_rate=0.01, momentum=0.9):
        super().__init__(learning_rate)
        self.momentum = momentum
        self.velocity = {}

    def initialize(self, parameters):
        """Initialize velocity terms"""
        for key in parameters:
            self.velocity[key] = np.zeros_like(parameters[key])

    def update(self, parameters, gradients):
        """Nesterov Accelerated Gradient"""
        if not self.velocity:
            self.initialize(parameters)

        # Store original parameters
        original_params = {}
        for key in parameters:
            original_params[key] = parameters[key].copy()
            # Lookahead update
            parameters[key] -= self.momentum * self.velocity[key]

        # Compute gradients at lookahead position
        # Note: This requires recomputing gradients, which would be done in the training loop

        for key in parameters:
            # Update velocity
            self.velocity[key] = (self.momentum * self.velocity[key] 
                                  + (1 - self.momentum) * gradients[key])
            # Update parameters
            parameters[key] = original_params[key] - self.learning_rate * self.velocity[key]
        return parameters

class RMSprop(Optimizer):
    def __init__(self, learning_rate=0.001, beta=0.9, epsilon=1e-8):
        super().__init__(learning_rate)
        self.beta = beta
        self.epsilon = epsilon
        self.cache = {}

    def initialize(self, parameters):
        """Initialize cache terms"""
        for key in parameters:
            self.cache[key] = np.zeros_like(parameters[key])

    def update(self, parameters, gradients):
        """RMSprop optimizer"""
        if not self.cache:
            self.initialize(parameters)

        for key in parameters:
            # Update cache (moving average of squared gradients)
            self.cache[key] = (self.beta * self.cache[key] 
                               + (1 - self.beta) * np.square(gradients[key]))
            
            # Update parameters
            parameters[key] -= (self.learning_rate * gradients[key] 
                                / (np.sqrt(self.cache[key]) + self.epsilon))
        return parameters

class Adam(Optimizer):
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = {}  # First moment
        self.v = {}  # Second moment
        self.t = 0   # Timestep

    def initialize(self, parameters):
        """Initialize first and second moment terms"""
        for key in parameters:
            self.m[key] = np.zeros_like(parameters[key])
            self.v[key] = np.zeros_like(parameters[key])

    def update(self, parameters, gradients):
        """Adam optimizer"""
        if not self.m:
            self.initialize(parameters)

        self.t += 1

        for key in parameters:
            # Update biased first moment estimate
            self.m[key] = (self.beta1 * self.m[key] 
                           + (1 - self.beta1) * gradients[key])
            
            # Update biased second moment estimate
            self.v[key] = (self.beta2 * self.v[key] 
                           + (1 - self.beta2) * np.square(gradients[key]))
            
            # Bias correction
            m_hat = self.m[key] / (1 - self.beta1**self.t)
            v_hat = self.v[key] / (1 - self.beta2**self.t)
            
            # Update parameters
            parameters[key] -= (self.learning_rate * m_hat 
                                / (np.sqrt(v_hat) + self.epsilon))
        return parameters

class NAdam(Optimizer):
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = {}  # First moment
        self.v = {}  # Second moment
        self.t = 0   # Timestep

    def initialize(self, parameters):
        """Initialize first and second moment terms"""
        for key in parameters:
            self.m[key] = np.zeros_like(parameters[key])
            self.v[key] = np.zeros_like(parameters[key])

    def update(self, parameters, gradients):
        """Nesterov-accelerated Adam optimizer"""
        if not self.m:
            self.initialize(parameters)

        self.t += 1

        for key in parameters:
            # Update biased first moment estimate
            self.m[key] = (self.beta1 * self.m[key] 
                           + (1 - self.beta1) * gradients[key])
            
            # Update biased second moment estimate
            self.v[key] = (self.beta2 * self.v[key] 
                           + (1 - self.beta2) * np.square(gradients[key]))
            
            # Bias correction
            m_hat = self.m[key] / (1 - self.beta1**self.t)
            v_hat = self.v[key] / (1 - self.beta2**self.t)
            
            # Nesterov momentum
            m_bar = (self.beta1 * m_hat) + ((1 - self.beta1) * gradients[key] / (1 - self.beta1**self.t))
            
            # Update parameters
            parameters[key] -= (self.learning_rate * m_bar 
                                / (np.sqrt(v_hat) + self.epsilon))
        return parameters