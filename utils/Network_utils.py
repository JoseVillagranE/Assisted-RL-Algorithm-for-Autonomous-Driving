import numpy as np

def freeze_params(model, params=None, verbose=True):
    for name, child in model.named_children():
        for param in child.parameters():
            if params is None or param in params:
                param.requires_grad = False
            freeze_params(child)


def conv2d_size_out(size, kernels_size, strides, paddings, dilations):
    for kernel_size, stride, padding, dilation in zip(kernels_size, strides, paddings, dilations):
        size = (size + 2*padding - dilation*(kernel_size - 1) - 1)//stride + 1
    return size

class OUNoise(object):

    def __init__(self, action_space, mu=0.0, theta=0.6, max_sigma=0.4, min_sigma=0,
                decay_period=100):

        self.mu = mu
        self.theta = theta
        self.sigma = max_sigma
        self.max_sigma = max_sigma
        self.min_sigma = min_sigma
        self.decay_period = decay_period
        #self.action_dim = action_space.shape[0]
        self.action_dim = action_space
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dim)*self.mu

    def evolve_state(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma*np.random.randn(self.action_dim)
        self.state = x + dx
        return self.state
    
    def update_sigma(self, t):
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma)*min(1.0, t/self.decay_period)

    def get_action(self, action, t=0):
        ou_state = self.evolve_state()
        self.update_sigma(t)
        return  action+ou_state

if __name__ == "__main__":
    
    import matplotlib.pyplot as plt
    
    action_space = 2 
    mu = 0.0
    theta = 0.3
    max_sigma = 0.4
    min_sigma = 0.0
    decay_period = 50
    noise = OUNoise(action_space, mu=mu, theta=theta, max_sigma=max_sigma, min_sigma=min_sigma,
                    decay_period=decay_period)
    
    a = []
    for i in range(100):
        ou_state = noise.evolve_state()
        noise.update_sigma(i)
        a.append(ou_state)
        
    plt.plot(a)