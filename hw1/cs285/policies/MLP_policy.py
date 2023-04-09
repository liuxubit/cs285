import numpy as np
import functools
import inspect
import torch
import torch.nn as nn
import torch.nn.functional as F
# from base_imp import MLP, convert_args_to_tensor

class MLP(nn.Module):

    def __init__(self, input_size, output_size, n_layers,
                size, activation=torch.tanh, output_activation=None):
        super(MLP, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation
        self.size = size
        self.n_layers = n_layers
        self.output_activations = output_activation

        layer_size = [self.input_size] + [[self.size] * self.n_layers] + [self.output_size]
        self.layers = nn.ModuleList(
            [
                nn.Linear(layer_size[i], layer_size[i+1]) for i in range(len(layer_size)-1)
            ]
        )

        for layer in self.layers:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
    
    def forward(self, x):
        out = x
        for i, layer in enumerate(self.layers):
            if i != len(self.layers) - 1:
                out = self.activation(layer(out))
            else:
                out = layer(out)
        if self.output_activations:
            out = self.output_activations(out)
        return out

def multivariate_normal_diag(loc, scale_diag):
    normal = torch.distributions.Normal(loc, scale=scale_diag)
    return torch.distributions.Independent(normal, 1)



def _is_method(func):
    spec = inspect.signature(func)
    return 'self' in spec.parameters

def convert_args_to_tensor(positional_args_list=None, keyword_args_list=None, device='cpu'):
    """A decorator which converts args in positional_args_list to torch.Tensor

    Args:
        positional_args_list ([list]): [arguments to be converted to torch.Tensor. If None, 
        it will convert all positional arguments to Tensor]
        keyword_args_list ([list]): [arguments to be converted to torch.Tensor. If None, 
        it will convert all keyword arguments to Tensor]
        device ([str]): [pytorch will run on this device]
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            
            
            
            _device = device
            _keyword_args_list = keyword_args_list
            _positional_args_list = positional_args_list
            
            if keyword_args_list is None:
                _keyword_args_list = list(kwargs.keys())

            if positional_args_list is None:
                _positional_args_list = list(range(len(args)))
            
                if _is_method(func):
                    _positional_args_list = _positional_args_list[1:]
            
            args = list(args)
            for i, arg in enumerate(args):
                if i in _positional_args_list:
                    if type(arg) == np.ndarray:
                        if arg.dtype == np.double:
                            args[i] = torch.from_numpy(arg).type(torch.float32).to(_device)
                        else:
                            args[i] = torch.from_numpy(arg).to(_device)
                    elif type(arg) == list:
                        args[i] = torch.tensor(arg).to(_device)
                    elif type(arg) == torch.Tensor or type(arg) == int or type(arg) == float or type(arg) == bool:
                        continue
                    else:
                        raise ValueError('Arguments should be Numpy arrays, but argument in position {} is not: {}'.format(str(i), type(arg)))
            
            for key, arg in kwargs.items():
                if key in _keyword_args_list:
                    if type(arg) == np.ndarray:
                        if arg.dtype == np.double:
                            kwargs[key] = torch.from_numpy(arg).type(torch.float32).to(_device)
                        else:
                            kwargs[key] = torch.from_numpy(arg).to(_device) 
                    elif type(arg) == list:
                        kwargs[key] = torch.tensor(arg).to(_device)
                    elif type(arg) == torch.Tensor or type(arg) == int or type(arg) == float or type(arg) == bool:
                        continue
                    else:
                        raise ValueError('Arguments should be Numpy arrays, but argument in position {} is not: {}'.format(str(i), type(arg)))

            return func(*args, **kwargs)

        return wrapper

    return decorator

@convert_args_to_tensor([0], ['labels'])
def torch_one_hot(labels, one_hot_size):
    one_hot = torch.zeros(labels.shape[0], one_hot_size)
    one_hot[torch.arange(labels.shape[0]), labels] = 1
    return one_hot

@convert_args_to_tensor()
def gather_nd(params, indices):
    """params is of "n" dimensions and has size [x1, x2, x3, ..., xn], indices is of 2 dimensions  and has size [num_samples, m] (m <= n)"""
    assert type(indices) == torch.Tensor
    return params[indices.transpose(0,1).long().numpy().tolist()]
class MLPPolicy(nn.Module):

    def __init__(self,
        ac_dim,
        ob_dim,
        n_layers,
        size,
        device,
        lr = 1e-4,
        training=True,
        discrete=False, # unused for now
        nn_baseline=False, # unused for now
        learning_rate = 1e-4,
        **kwargs):
        super().__init__()

        # init vars
        self.device = device
        self.ac_dim = ac_dim
        self.ob_dim = ob_dim
        self.n_layers = n_layers
        self.size = size
        self.learning_rate = learning_rate
        self.training = training
        self.device = device

        # network architecture
        #TODO -build the network architecture
        #HINT -build an nn.Modulelist() using the passed in parameters
        self.build_graph()

        #loss and optimizer
        if self.training:
            # TODO define the loss that will be used to train this policy
            self.loss_func = torch.nn.MSELoss(reduction='mean')
            self.optimizer = torch.optim.Adam(self.parameters(), lr)

        self.to(device)
    
    def build_graph(self):
        self.define_forward_pass_parameters()
        if self.training:
            self.define_train_op()
    
    def define_forward_pass_parameters(self):
        mean = MLP(self.ob_dim, output_size=self.ac_dim, n_layers=self.n_layers, size=self.size).to(self.device)
        logstd = torch.zeros(self.ac_dim, required_grad=True, device=self.device)
        self.parameters = (mean, logstd)


    ##################################
    def _build_action_sampling(self, observation):
        mean, logstd = self.parameters
        probs_out = mean(observation)
        sample_ac = probs_out + torch.exp(logstd) + torch.randn(probs_out.size(), device = self.device)
        return sample_ac

    ##################################

    ##################################

    def forward(self, x):
        for layer in self.mlp:
            x = layer(x)
        return x

    ##################################

    def save(self, filepath):
        torch.save(self.state_dict(), filepath)

    def restore(self, filepath):
        self.load_state_dict(torch.load(filepath))

    ##################################

    # query this policy with observation(s) to get selected action(s)
    @convert_args_to_tensor()
    def get_action(self, obs):
        with torch.no_grad():
            if len(obs.shape)>1:
                observation = obs
            else:
                observation = obs[None]

        # TODO return the action that the policy prescribes
        return self._build_action_sampling(observation).cpu().numpy()

    # update/train this policy
    def update(self, observations, actions):
        raise NotImplementedError

#####################################################
#####################################################

class MLPPolicySL(MLPPolicy):

    """
        This class is a special case of MLPPolicy,
        which is trained using supervised learning.
        The relevant functions to define are included below.
    """

    def update(self, observations, actions):
        assert self.training, 'Policy must be created with training = true in order to perform training updates...'

        # TODO define network update
        # HINT - you need to calculate the prediction loss and then use optimizer.step()
        actions, observations = actions.to(self.device), observations.to(self.device)
        sample_ac = self._build_action_sampling(observations)
        loss = self.loss_func(actions, sample_ac)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
