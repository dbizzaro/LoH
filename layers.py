import abc
import torch
from torch.nn import functional as F
from torch.distributions.gumbel import Gumbel
import itertools
import numpy as np
import random


class PrecomputedNoise:
    """Precomputes noise to improve efficiency.
    Retrive the noise with get_noise_tensor(shape)."""

    def __init__(self, buffer_size, noise_distribution='Gumbel', device=None, **kwargs):
        self.buffer_size = buffer_size
        self.noise_distribution = noise_distribution
        self.device = device
        self.kwargs = kwargs
        self.initialization()

    def initialization(self):
        self.sample_and_store()
        list1 = random.sample(range(self.buffer_size), self.buffer_size)
        list2 = random.sample(range(1, self.buffer_size), self.buffer_size - 1)
        self.indices_generator = itertools.product(list1, list2)

    def sample_and_store(self):
        if self.noise_distribution == 'Gumbel':
            scaling_factor = self.kwargs.get('weight_random', 1.)
            self.noise_vector = Gumbel(0,1).sample((self.buffer_size,)).to(self.device) * scaling_factor
        elif self.noise_distribution == 'Uniform':
            bounds = self.kwargs.get('uniform_noise_bounds', (0.75, 1.25))
            self.noise_vector = bounds[0] + (bounds[1] - bounds[0]) * torch.rand((self.buffer_size,), device=self.device)
        elif self.noise_distribution == 'Bernoulli':
            p = self.kwargs['p']
            self.noise_vector = torch.bernoulli(p * torch.ones((self.buffer_size,), device=self.device)).to(torch.bool)

    def get_noise_slice(self, start_idx, step_size, length):
        indices = (torch.arange(start_idx, start_idx + step_size * length, step_size, device=self.device) % self.buffer_size)
        return self.noise_vector[indices]
    
    def get_noise_tensor(self, shape):
        length = np.prod(shape)
        try:
            start_idx, step_size = next(self.indices_generator)
        except StopIteration:
            self.initialization()
            start_idx, step_size = next(self.indices_generator)
        return self.get_noise_slice(start_idx, step_size, length=length).reshape(shape)


def subtract_mean_of_two_along_dim(tensor, dim):
    """Subtract the mean of the two largest values in the given tensor along the given dimension."""
    top2, _ = torch.topk(tensor, 2, dim=dim)
    mean_top2 = top2.mean(dim=dim, keepdim=True)
    return tensor - mean_top2


class MaxMinTensorMultiply(torch.autograd.Function):
    """"  x -> A, shape (m, n, r)
    weights -> B, shape (n, p, r)
    result  ->    shape (m, p, r)
    Result_ikl = max_j min(A_ijl, B_jkl)"""

    @staticmethod
    def forward(ctx, A, B):
        #A_expanded = A.unsqueeze(-1).unsqueeze(-1) # when A shape just (m, n)
        A_expanded = A.unsqueeze(-2)  # Shape: (m, n, 1, r)
        B_expanded = B.unsqueeze(0)   # Shape: (1, n, p, r)
        result, indices = torch.max(torch.minimum(A_expanded, B_expanded), dim=1)
        ctx.save_for_backward(A, B, indices)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        A, B, indices = ctx.saved_tensors
        #A = A.unsqueeze(-1).expand(-1, -1, B.shape[-1]) # when A shape just (m, n)
        grad_A = torch.zeros_like(A)  # Shape: (m, n, r)
        grad_B = torch.zeros_like(B)  # Shape: (n, p, r)
        C = A.gather(1, indices) < B.gather(0, indices)
        grad_A = grad_A.scatter_add_(1, indices, grad_output.masked_fill(~C, 0))#.sum(dim=-1) # when A shape just (m, n)
        grad_B = grad_B.scatter_add_(0, indices, grad_output.masked_fill(C, 0))       
        return grad_A, grad_B

    
class MinMaxTensorMultiply(torch.autograd.Function):
    """"  x -> A, shape (m, n, r)
    weights -> B, shape (n, p, r)
    result  ->    shape (m, p, r)
    Result_ikl = min_j max(A_ijl, B_jkl)"""

    @staticmethod
    def forward(ctx, A, B):
        #A_expanded = A.unsqueeze(-1).unsqueeze(-1) # when A shape just (m, n)
        A_expanded = A.unsqueeze(-2)  # Shape: (m, n, 1, r)
        B_expanded = B.unsqueeze(0)   # Shape: (1, n, p, r)
        result, indices = torch.min(torch.maximum(A_expanded, B_expanded), dim=1)
        ctx.save_for_backward(A, B, indices)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        A, B, indices = ctx.saved_tensors
        #A = A.unsqueeze(-1).expand(-1, -1, B.shape[-1]) # when A shape just (m, n)
        grad_A = torch.zeros_like(A)  # Shape: (m, n, r)
        grad_B = torch.zeros_like(B)  # Shape: (n, p, r)
        C = A.gather(1, indices) > B.gather(0, indices)
        grad_A = grad_A.scatter_add_(1, indices, grad_output.masked_fill(~C, 0))#.sum(dim=-1) # when A shape just (m, n)
        grad_B = grad_B.scatter_add_(0, indices, grad_output.masked_fill(C, 0))       
        return grad_A, grad_B



class LogicalLayer(torch.nn.Module, abc.ABC):
    def __init__(self, device, buffer_size=1000, no_noise=False, clipped=False):
        super().__init__()
        self.no_noise = no_noise
        self.clipped = clipped
        if not no_noise:
            self.additive_noise = PrecomputedNoise(buffer_size, noise_distribution='Gumbel', device=device)
        self.w = None
        self.device = device
    
    @abc.abstractmethod
    def forward(self, x, **kwargs):
        pass

    def produce_weights(self, dim_subtraction, weight_random=0.0, temp=1.0):
        if self.training and not self.no_noise:
            r = self.additive_noise.get_noise_tensor(self.Z.shape) * weight_random
            w = (self.Z + r) / temp
        else:
            w = self.Z / temp
        if dim_subtraction is not None:
            normalized_w = subtract_mean_of_two_along_dim(w, dim=dim_subtraction)
        else:
            normalized_w = w
        if self.clipped:
            self.w = torch.clip(0.5 + normalized_w, 0, 1)
        else:
            self.w = torch.nn.Sigmoid()(normalized_w)

            
class FFAnd(LogicalLayer):
    """A feed-forward layer that implements the AND and (possibly) NOT operations.
    The meta-rule is: for all j (AND_i [P^l_i, 1]) -> P^{l+1}_j
    or, if negations: for all j (AND_i [P^l_i, not P^l_i, 1]) -> P^{l+1}_j
    """
    def __init__(self, input_size, output_size, negations = False, device=None, dual=False, **kwargs):
        super().__init__(device, **kwargs)
        self.negations = negations
        self.input_size = input_size
        self.output_size = output_size
        self.internal_size = 3 if negations else 2
        self.dual = dual
        self.Z = torch.nn.Parameter(0.5*torch.randn(output_size, input_size, self.internal_size, device=device))
        self.layer_type = 'and'
    
    def forward(self, x, weight_random=0.0, temp=1.0):
        if self.negations:
            extended_x = torch.cat((x.unsqueeze(-1).unsqueeze(-3),
                                1. - x.unsqueeze(-1).unsqueeze(-3),
                                torch.ones_like(x.unsqueeze(-1).unsqueeze(-3))), dim=-1)
        else:
            extended_x = torch.cat((x.unsqueeze(-1).unsqueeze(-3),
                                torch.ones_like(x.unsqueeze(-1).unsqueeze(-3))), dim=-1)
        self.produce_weights(2, weight_random, temp)
        n_dims = extended_x.ndim - self.w.ndim
        for _ in range(n_dims):
            w = self.w.unsqueeze(0)
        if self.dual:
            z = torch.min(torch.maximum(1. - w, extended_x), dim=-1)[0]
        else:
            z = torch.max(torch.minimum(w, extended_x), dim=-1)[0]
        return torch.min(z, dim=-1)[0]
    
    def to_rules(self, input_propositions=None, output_units=None, output_propositions=None, implications=False):
        """Convert the weights to a list of rules (strings)."""
        if input_propositions is None:
            input_propositions = ['i'+str(x) for x in range(self.w.shape[1])]
        if output_units is None:
            output_units = range(self.w.shape[0])
        if output_propositions is None:
            output_propositions = ['o'+str(x) for x in output_units]
        rules = []
        operator = ' & '
        tensor = torch.argmax(self.w, dim=-1)
        for o in output_units:
            rule = ''
            for i in range(self.w.shape[1]):
                if tensor[o, i] == 0:
                    rule += input_propositions[i] + operator
                elif self.negations and tensor[o, i] == 1:
                    rule += '~ ' + input_propositions[i] + operator
            rule = rule[:-len(operator)]
            if rule == '':
                rule = 'True'
            if implications:
                rule = rule + ' >> ' + output_propositions[o]
            rules.append(rule)
        return rules
    
    def to_clauses(self, output_units = None):
        """Convert the weights to a list of clauses (sets of integers)."""
        if output_units is None:
            output_units = range(self.w.shape[0])
        clauses = []
        tensor = torch.argmax(self.w, dim=-1)
        for o in output_units:
            clause = []
            for i in range(tensor.shape[1]):
                if tensor[o, i] == 0:
                    clause.append(i)
                elif self.negations and self.tensor[o, i] == 1:
                    clause.append(-i)
            clauses.append(set(clause))
        return clauses



class FFOr(LogicalLayer):
    """A feed-forward layer that implements the OR and (possibly) NOT operations.
    The meta-rule is: for all j (OR_i [P^l_i, 0]) -> P^{l+1}_j
    or, if negations: for all j (OR_i [P^l_i, not P^l_i, 0]) -> P^{l+1}_j
    """
    def __init__(self, input_size, output_size, negations = False, device=None, **kwargs):
        super().__init__(device, **kwargs)
        self.negations = negations
        self.input_size = input_size
        self.output_size = output_size
        self.internal_size = 3 if negations else 2
        self.negations = negations
        self.Z = torch.nn.Parameter(0.5*torch.randn(output_size, input_size, self.internal_size, device=device))
        self.layer_type = 'or'

    def forward(self, x, weight_random=0.0, temp=1.0):
        if self.negations:
            extended_x = torch.cat((x.unsqueeze(-1).unsqueeze(-3),
                                1. - x.unsqueeze(-1).unsqueeze(-3),
                                torch.zeros_like(x.unsqueeze(-1).unsqueeze(-3))), dim=-1)
        else:
            extended_x = torch.cat((x.unsqueeze(-1).unsqueeze(-3),
                                torch.zeros_like(x.unsqueeze(-1).unsqueeze(-3))), dim=-1)

        self.produce_weights(2, weight_random, temp)
        n_dims = extended_x.ndim - self.w.ndim
        for _ in range(n_dims):
          w = self.w.unsqueeze(0)
        z = torch.max(torch.minimum(w, extended_x), dim=-1)[0]
        return torch.max(z, dim=-1)[0]
    
    def to_rules(self, input_propositions=None, output_units=None, output_propositions=None, implications=False):
        """Convert the weights to a list of rules (strings)."""
        if input_propositions is None:
            input_propositions = ['i'+str(x) for x in range(self.w.shape[1])]
        if output_units is None:
            output_units = range(self.w.shape[0])
        if output_propositions is None:
            output_propositions = ['o'+str(x) for x in output_units]
        rules = []
        operator = ' | '
        tensor = torch.argmax(self.w, dim=-1)
        for o in output_units:
            rule = ''
            for i in range(self.w.shape[1]):
                if tensor[o, i] == 0:
                    rule += input_propositions[i] + operator
                elif self.negations and tensor[o, i] == 1:
                    rule += '~ ' + input_propositions[i] + operator
            rule = rule[:-len(operator)]
            if rule == '':
                rule = 'false'
            if implications:
                rule = rule + ' >> ' + output_propositions[o]
            rules.append(rule)
        return rules
    
    def to_clauses(self, output_units = None):
        """Convert the weights to a list of clauses (sets of integers)."""
        if output_units is None:
            output_units = range(self.w.shape[0])
        clauses = []
        tensor = torch.argmax(self.w, dim=-1)
        for o in output_units:
            clause = []
            for i in range(tensor.shape[1]):
                if tensor[o, i] == 0:
                    clause.append(i)
                elif self.negations and self.tensor[o, i] == 1:
                    clause.append(-i)
            clauses.append(set(clause))
        return clauses



class AndClause(LogicalLayer):
    """A feed-forward layer that implements the AND and (possibly) NOT operations.
    The meta-rule is (for k=2): for all j [P^l] AND [P^l]  -> P^{l+1}_j
    or, if negations = True, k=2: for all j ([P^l U not P^l] AND [P^l, not P^l]) -> P^{l+1}_j
    """
    def __init__(self, input_size, output_size, k, negations=False, device=None, dual=False, **kwargs):
        super().__init__(device, **kwargs)
        self.internal_size = k
        self.input_size = 2*input_size if negations else input_size
        self.output_size = output_size
        self.negations = negations
        self.dual = dual
        if negations:
            self.Z = torch.nn.Parameter(0.5*torch.randn(2*input_size, output_size, k, device=device))
        else:
            self.Z = torch.nn.Parameter(0.5*torch.randn(input_size, output_size, k, device=device))
        self.layer_type = 'and'
        
    def forward(self, x, weight_random=0.0, temp=1.0):
        self.produce_weights(0, weight_random, temp)
        extended_x = torch.cat((x, 1. - x), dim=1) if self.negations else x
        extended_x = extended_x.unsqueeze(-1).expand(-1, -1, self.w.shape[-1])
        if self.dual:
            z = MinMaxTensorMultiply.apply(extended_x, 1. - self.w)
        else:
            z = MaxMinTensorMultiply.apply(extended_x, self.w)
        return torch.min(z, dim=-1)[0]
    
    def to_rules(self, input_propositions=None, output_units=None, output_propositions=None, implications=False):
        """Convert the weights to a list of rules (strings)."""
        n_input_props = self.input_size//2 if self.negations else self.input_size
        if input_propositions is None:
            input_propositions = ['i'+str(x) for x in range(n_input_props)]
        if output_units is None:
            output_units = range(self.w.shape[1])
        if output_propositions is None:
            output_propositions = ['o'+str(x) for x in output_units]
        rules = []
        operator = ' & '
        tensor = torch.argmax(self.w, dim=0)
        for o in output_units:
            rule = ''
            for i in range(self.internal_size):
                index = tensor[o, i]
                if index < n_input_props:
                    rule += input_propositions[index] + operator
                else:
                    rule += '~ ' + input_propositions[index - n_input_props] + operator
            rule = rule[:-len(operator)]
            if implications:
                rule = rule + ' >> ' + output_propositions[o]
            rules.append(rule)
        return rules
      
    def to_clauses(self, output_units = None):
        """Convert the weights to a list of clauses (sets of integers)."""
        n_input_props = self.w.shape[0]//2 if self.negations else self.w.shape[0]
        if output_units is None:
            output_units = range(self.w.shape[1])
        clauses = []
        tensor = torch.argmax(self.w, dim=0)
        for o in output_units:
            clause = []
            for i in range(self.internal_size):
                index = tensor[o, i].item()
                if index < n_input_props:
                    clause.append(index)
                else:
                    clause.append(-index)
            if self.w[:, o, :].max(axis=0)[0].min() > 0.5:
                clauses.append(set(clause))
            else:
                clauses.append(set())
        return clauses
    

class OrClause(LogicalLayer):
    """A feed-forward layer that implements the OR and (possibly) NOT operations.
    The meta-rule is (for k=2): for all j [P^l] OR [P^l]  -> P^{l+1}_j
    or, if negations = True, k=2: for all j ([P^l U not P^l] OR [P^l, not P^l]) -> P^{l+1}_j
    """
    def __init__(self, input_size, output_size, k, negations=False, device=None, **kwargs):
        super().__init__(device, **kwargs)
        self.negations = negations
        self.internal_size = k
        self.input_size = input_size
        self.output_size = output_size
        if negations:
            self.Z = torch.nn.Parameter(0.5*torch.randn(2*input_size, output_size, k, device=device))
        else:
            self.Z = torch.nn.Parameter(0.5*torch.randn(input_size, output_size, k, device=device))
        self.layer_type = 'or'

    def forward(self, x, weight_random=0.0, temp=1.0):
        self.produce_weights(0, weight_random, temp)
        extended_x = torch.cat((x, 1. - x), dim=1) if self.negations else x
        extended_x = extended_x.unsqueeze(-1).expand(-1, -1, self.w.shape[-1])
        z = MaxMinTensorMultiply.apply(extended_x, self.w)
        return torch.max(z, dim=-1)[0]
    
    def to_rules(self, input_propositions=None, output_units=None, output_propositions=None, implications=False):
        """Convert the weights to a list of rules (strings)."""
        n_input_props = self.w.shape[0]//2 if self.negations else self.w.shape[0]
        if input_propositions is None:
            input_propositions = ['i'+str(x) for x in range(n_input_props)]
        if output_units is None:
            output_units = range(self.w.shape[1])
        if output_propositions is None:
            output_propositions = ['o'+str(x) for x in output_units]
        rules = []
        operator = ' | '
        tensor = torch.argmax(self.w, dim=0)
        for o in output_units:
            rule = ''
            for i in range(self.internal_size):
                index = tensor[o, i]
                if index < n_input_props:
                    rule += input_propositions[index] + operator
                else:
                    rule += '~ ' + input_propositions[index - n_input_props] + operator
            rule = rule[:-len(operator)]
            if implications:
                rule = rule + ' >> ' + output_propositions[o]
            rules.append(rule)
        return rules
    
    def to_clauses(self, output_units = None):
        """Convert the weights to a list of clauses (sets of integers)."""
        n_input_props = self.w.shape[0]//2 if self.negations else self.w.shape[0]
        if output_units is None:
            output_units = range(self.w.shape[1])
        clauses = []
        tensor = torch.argmax(self.w, dim=0)
        for o in output_units:
            clause = []
            for i in range(self.internal_size):
                index = tensor[o, i].item()
                if index < n_input_props:
                    clause.append(index)
                else:
                    clause.append(-index)
            clauses.append(set(clause))
        return clauses
    

class MultiplyWithoutGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, v, w):
        return v * w

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


class Classifier(torch.nn.Module):
    def __init__(self, vector_size, device=None, uniform_noise_bounds=[0.75, 1.25]):
        super().__init__()
        self.vector_size = vector_size
        self.device = device
        self.uniform_noise_bounds = uniform_noise_bounds
        self.additive_noise = PrecomputedNoise(1000, noise_distribution='Gumbel', device=device)
        self.multiplicative_noise = PrecomputedNoise(1000, noise_distribution='Uniform', device=device, uniform_noise_bounds=uniform_noise_bounds)

    def forward(self, x, weight_random=0.0, temp=1.0):
        x = torch.logit(x, eps=1e-5)
        if self.training:
            r = self.additive_noise.get_noise_tensor(x.shape) * weight_random
            r2 = self.multiplicative_noise.get_noise_tensor(x.shape)
            x = (x + r) / temp
        else:
            x = x / temp
            r2 = 1.
        normalized_x = subtract_mean_of_two_along_dim(x, dim=1)
        return torch.nn.Sigmoid()(MultiplyWithoutGrad.apply(normalized_x, r2))