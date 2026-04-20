import torch
from layers import *
import re


class AndOrModel(torch.nn.Module):
    """A model that uses AND and OR layers in alternation."""
    def __init__(self, n_units, ks=None, device=None, starting='and', negations=False, dual=True, **kwargs):
        """
        Parameters:
            n_units (List[int]): the number of units in each layer
            ks (List[int]): If None, the model uses FF layers, selecting clauses of any size. 
                            Otherwise, the model uses layers with fixed clause size ks[i].
            device (torch.device): the device to use for the model
            starting (str): 'and' or 'or', the type of the first layer
            negations (bool): whether to use negations in the clauses
            dual (bool): use conjunctive compilation on conjunctive layers (instead of always disjunctive compilation)
        """
        super().__init__()
        self.n_units = n_units
        self.ks = ks
        self.negations = negations
        input_len = n_units[0]
        self.n_classes = n_units[-1]
        self.layers = []
        and_layer = (starting == 'and')
        for i, p in enumerate(n_units[1:]):
            if and_layer:
                if ks is None:
                    self.layers.append(FFAnd(input_len, p, device=device, negations=negations, dual=dual, **kwargs))
                else:
                    self.layers.append(AndClause(input_len, p, ks[i], device=device, negations=negations, dual=dual, **kwargs))
                and_layer = False
            else:
                if ks is None:
                    self.layers.append(FFOr(input_len, p, device=device, negations=negations, **kwargs))
                else:
                    self.layers.append(OrClause(input_len, p, ks[i], negations=negations, device=device, **kwargs))
                and_layer = True
            input_len = p
        if self.n_classes > 1:
            self.classifier=Classifier(self.n_classes, device=device)
        self.module_list = torch.nn.ModuleList(self.layers)
        
    def forward(self, x, weight_random=0.0, temp=1.0, weight_random_classifier=0.0):
        for layer in self.layers:
            x = layer(x, weight_random, temp)
        if layer.output_size == 1:
            return x
        return self.classifier(x, weight_random_classifier, temp)
    

    def to_formula(self, dead_nodes=[[], []], features_names=[], verbose=False):
        output_layer = self.layers[-1]
        formulas = output_layer.to_rules(['0i'+str(x) for x in range(output_layer.input_size)])
        if verbose:
            print(f'layer {-1}: {formulas}')
        for f in range(len(formulas)):
            formula = formulas[f]
            for l, layer in enumerate(reversed(self.layers[:-1])):
                if l < len(self.layers)-2:
                    layer_rules = layer.to_rules([str(l+1)+'i'+str(x) for x in range(layer.input_size)])
                else:
                    layer_rules = layer.to_rules(features_names)
                if f == 0 and verbose:
                    print(f'layer {-2-l}: {layer_rules}')
                for i in range(layer.output_size):
                    substring = str(l)+'i'+str(i)
                    if l < len(self.layers)-2:
                        formula = re.sub(r'\b'+substring+r'\b', '('+layer_rules[i]+')', formula)
                    else:
                        if i in dead_nodes[0]:
                            formula = re.sub(r'\b'+substring+r'\b', 'false', formula)
                        elif i in dead_nodes[1]:
                            formula = re.sub(r'\b'+substring+r'\b', 'true', formula)
                        else:
                            formula = re.sub(r'\b'+substring+r'\b', '('+layer_rules[i]+')', formula)
            substring = str(l+1)+'i'
            #formulas[f] = re.sub(r'(?<!\w|\()'+substring+r'(?!\w|\))', 'i', formula)
            formulas[f] = formula.replace(substring, 'i')
        return formulas
    
    def parameter_count(self, statistics=False):
        count = 0
        for layer in self.layers:
            count += layer.input_size * layer.output_size * layer.internal_size
            if statistics:
                print(f'Layer {layer.__class__.__name__}: {layer.input_size} inputs, {layer.output_size} outputs, {layer.internal_size} internal size')
                hist = torch.histc(layer.w, bins=10, min=0, max=1)
                print(f"\t Histogram: {hist.cpu().numpy()}")
        return count