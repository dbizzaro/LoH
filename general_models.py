import pyparsing as pp
import torch
from torch.distributions.gumbel import Gumbel

from layers import subtract_mean_of_two_along_dim


class Prop(torch.nn.Module):
    """Proposition (leaf node in the computational graph)
    Each proposition is a column in the input tensor."""
    def __init__(self, index, name=None):
        super().__init__()
        self.index = index
        if name is not None:
            self.name = name
        else:
            self.name = f'P_{index + 1}'
    def forward(self, x):
        return x[:, self.index]
    def __str__(self):
        return self.name
    def extract_formula(self):
        return str(self)
    
def proposition_list(names):
    """Creates a list of Prop instances from a list of names."""
    return [Prop(index, name) for index, name in enumerate(names)]

class Top(torch.nn.Module):
    """True"""
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return torch.ones_like(x[:,0])
    def __str__(self):
        return 'True'
    def extract_formula(self):
        return str(self)

class Bot(torch.nn.Module):
    """False"""
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return torch.zeros_like(x[:,0])
    def __str__(self):
        return 'False'
    def extract_formula(self):
        return str(self)

class Not(torch.nn.Module):
    """Negation"""
    def __init__(self, operand):
        super().__init__()
        self.operand = operand
        self.operand_values = None
        self.result = None
    def forward(self, x):
        self.result = 1. - self.operand(x)
        return self.result
    def __str__(self):
        return f'~ {self.operand}'
    def extract_formula(self):
        return f'~ {self.operand.extract_formula()}'

class Or(torch.nn.Module):
    """Disjunction"""
    def __init__(self, *operands):
        super().__init__()
        self.operands = torch.nn.ModuleList(operands)
        self.operand_values = None
        self.result = None
    def forward(self, x):
        self.operand_values = torch.stack([operand(x) for operand in self.operands], dim=1)
        self.result = self.operand_values.max(dim=1)[0]
        return self.result
    def __str__(self):
        return '(' + ' | '.join([str(operand) for operand in self.operands]) + ')'
    def extract_formula(self):
        return '(' + ' | '.join([operand.extract_formula() for operand in self.operands]) + ')'

class And(torch.nn.Module):
    """Conjunction"""
    def __init__(self, *operands):
        super().__init__()
        self.operands = torch.nn.ModuleList(operands)
    def forward(self, x):
        results = torch.stack([operand(x) for operand in self.operands], dim=1)
        return results.min(dim=1)[0]
    def __str__(self):
        return '(' + ' & '.join([str(operand) for operand in self.operands]) + ')'
    def extract_formula(self):
        return '(' + ' & '.join([operand.extract_formula() for operand in self.operands]) + ')'

class Choose(torch.nn.Module):
    """Disjunctive compilation of a Choice operator"""
    def __init__(self, *operands):
        super().__init__()
        self.operands = torch.nn.ModuleList(operands)
        self.Z = torch.nn.Parameter(torch.zeros(len(operands))) #weights initialization
        self.activation = torch.sigmoid # alternative activation: torch.nn.Softmax(dim=0), or lambda x: x
        self.additive_noise_distribution = Gumbel(0,1)
        #self.multiplicative_noise_distribution = torch.distributions.Uniform(0.75,1.5)
        self.weight_random = 1.
        self.temp = 1.
        self.w = None
        self.operand_values = None
        self.weighted_values = None
        self.result = None

    def produce_weights(self, eval=False):
        if self.training and not eval:
            r_additive = self.additive_noise_distribution.sample(self.Z.shape).to(self.Z.device) * self.weight_random
            #r_multiplicative = self.multiplicative_noise_distribution.sample(self.Z.shape).to(self.Z.device)
            w = (self.Z + r_additive) / self.temp
        else:
            w = self.Z / self.temp
            #r_multiplicative = 1.
        normalized_w = subtract_mean_of_two_along_dim(w, dim=0)
        #normalized_w = MultiplyWithoutGrad.apply(normalized_w, r_multiplicative)
        self.w = self.activation(normalized_w)
    
    def get_weights(self, temp=1.0):
        w = self.Z.detach() / temp
        normalized_w = subtract_mean_of_two_along_dim(w, dim=0)
        return self.activation(normalized_w)

    def forward(self, x):
        self.produce_weights()
        self.operand_values = torch.stack([operand(x) for operand in self.operands], dim=1)
        self.weighted_values = torch.minimum(self.operand_values, self.w[None, :])
        self.result = torch.max(self.weighted_values, dim=1)[0]
        return self.result    

    def __str__(self):
        return '[' + ', '.join([str(operand) for operand in self.operands]) + ']'
    
    def extract_formula(self):
        weights = self.get_weights()
        choice = torch.argmax(weights).item()
        return self.operands[choice].extract_formula()
    


class ChooseDual(torch.nn.Module):
    """Conjunctive compilation of a Choice operator"""
    def __init__(self, *operands):
        super().__init__()
        self.operands = torch.nn.ModuleList(operands)
        self.Z = torch.nn.Parameter(torch.zeros(len(operands))) #weights initialization
        self.activation = torch.sigmoid # alternative activation: torch.nn.Softmax(dim=0), or lambda x: x
        self.additive_noise_distribution = Gumbel(0,1)
        #self.multiplicative_noise_distribution = torch.distributions.Uniform(0.75,1.5)
        self.weight_random = 1.
        self.temp = 1.
        self.w = None
        self.operand_values = None
        self.weighted_values = None
        self.result = None

    def produce_weights(self, eval=False):
        if self.training and not eval:
            r_additive = self.additive_noise_distribution.sample(self.Z.shape).to(self.Z.device) * self.weight_random
            #r_multiplicative = self.multiplicative_noise_distribution.sample(self.Z.shape).to(self.Z.device)
            w = (self.Z + r_additive) / self.temp
        else:
            w = self.Z / self.temp
            #r_multiplicative = 1.
        normalized_w = subtract_mean_of_two_along_dim(w, dim=0)
        #normalized_w = MultiplyWithoutGrad.apply(normalized_w, r_multiplicative)
        self.w = self.activation(normalized_w)
    
    def get_weights(self, temp=1.0):
        w = self.Z.detach() / temp
        normalized_w = subtract_mean_of_two_along_dim(w, dim=0)
        return self.activation(normalized_w)

    def forward(self, x):
        self.produce_weights()
        self.operand_values = torch.stack([operand(x) for operand in self.operands], dim=1)
        self.weighted_values = torch.maximum(self.operand_values, 1. - self.w[None, :])
        self.result = torch.min(self.weighted_values, dim=1)[0]
        return self.result

    def __str__(self):
        return '[' + ', '.join([str(operand) for operand in self.operands]) + ']_'
    
    def extract_formula(self):
        weights = self.get_weights()
        choice = torch.argmax(weights).item()
        return self.operands[choice].extract_formula()
    


class LogicalParser:
    """
    A parser for logical formulas with ~, &, |, (), a 'choose' operator [...], and a 'choose dual' operator [...]_
    Instantiate the class and call the `parse` method to produce the architecture
    """
    def __init__(self, prop_names=None):
        """Initializes the parser, building the grammar if not already built"""
        self.prop_names = prop_names
        if prop_names is None:
            self.counter = 0
            self.prop_names = []
        else:
            self.counter = len(prop_names)
        self._parser_expr = self._build_grammar()
        
    def _init_prop(self, name):
        """Initializes a proposition with a given name"""
        if name in self.prop_names:
            return Prop(self.prop_names.index(name), name)
        self.counter += 1
        self.prop_names.append(name)
        return Prop(self.counter-1, name)

    def _build_grammar(self):
        """Builds the pyparsing grammar"""

        def make_nary(cls):
            def parse_action(tokens):
                """flatten nested structures for associative operators"""
                items = tokens[0].asList()
                operands = [items[i] for i in range(0, len(items), 2)]
                return cls(*operands)
            return parse_action
        
        # Define the grammar for the logical expressions
        formula = pp.Forward()

        # Suppress punctuation and operators
        LPAREN, RPAREN, LBRACK, RBRACK, COMMA, UNDERSCORE = map(pp.Suppress, "()[],_")
        TILDE, AMPERSAND, PIPE = map(pp.Literal, "~&|") # Operators

        # Define constants True and False
        true_const = pp.Keyword("True").setParseAction(lambda _: Top())
        false_const = pp.Keyword("False").setParseAction(lambda _: Bot())

        # Proposition: alphanumeric + underscore creates Prop object
        prop_name = pp.Word(pp.alphas, pp.alphanums + "_")
        prop = prop_name.copy().setParseAction(lambda t: self._init_prop(t[0]))

        # Choose dual operator: [ formula, ... ]_ creates Choose object
        # Need pp.Group around formula for delimitedList when formula itself might be complex
        choosedual_expr = (LBRACK + pp.Optional(pp.delimitedList(pp.Group(formula))) + RBRACK + UNDERSCORE)
        choosedual_expr.setParseAction(lambda t: ChooseDual(*[el[0] for el in t]))

        # Choose operator: [ formula, ... ] creates Choose object
        # Need pp.Group around formula for delimitedList when formula itself might be complex
        choose_expr = (LBRACK + pp.Optional(pp.delimitedList(pp.Group(formula))) + RBRACK)
        choose_expr.setParseAction(lambda t: Choose(*[el[0] for el in t]))

        # Base elements for the grammar
        atom = true_const | false_const | prop | choosedual_expr | choose_expr | (LPAREN + formula + RPAREN)

        # Define operator precedence and actions using infixNotation. Order: ~, &, |
        formula <<= pp.infixNotation(atom, [
            (TILDE, 1, pp.opAssoc.RIGHT, lambda t: Not(t[0][1])),
            (AMPERSAND, 2, pp.opAssoc.LEFT, make_nary(And)),
            (PIPE, 2, pp.opAssoc.LEFT, make_nary(Or)),
        ])
        return formula

    def parse(self, text):
        result = self._parser_expr.parseString(text, parseAll=True)
        return result[0]
    


def select_one_rule_architecture(rules_strs, proposition_names, choosedual=False):
    """
    Create a LoH model choosing a subformula.
    Parameters:
        rules_strs (List[str]):  e.g. ['P_1', '~P_2', 'P_3 | ~P_4']
        proposition_names (List[str]): e.g. ['P_1', 'P_2', 'P_3', 'P_4']
        choosedual (bool): use conjunctive compilation instead of disjunctive
    Returns:
        torch.nn.Module: The compilation of the LoH model (e.g. [P_1, ~P_2, P_3 | ~P_4])
    """
    parser = LogicalParser(proposition_names)
    architecture = "[" + ", ".join(rule for rule in rules_strs) + "]"
    if choosedual:
        architecture += "_"
    return parser.parse(architecture)

def select_one_rule_per_list_architecture(rules_strs_list, proposition_names, conjunction=True, choosedual=False):
    """
    Create a LoH model from a list of lists of rules, selecting one for each list.
    Parameters:
        rules_strs_list (List[List[str]]): e.g. [['P_1', '~P_2', 'P_3 | ~P_4'], ['~P_1 & P_2', '~P_3 | P_4']]
        proposition_names (List[str]): e.g. ['P_1', 'P_2', 'P_3', 'P_4']
        conjunction (bool): put the rules in a conjunction (otherwise disjunction)
        choosedual (bool): use conjunctive compilation instead of disjunctive
    Returns:
        torch.nn.Module: The compilation of the LoH model (e.g. [P_1, ~P_2, P_3 | ~P_4] & ['~P_1 & P_2', '~P_3 | P_4'])
    """
    subparts = [select_one_rule_architecture(rules_strs, proposition_names, choosedual) for rules_strs in rules_strs_list]
    if conjunction:
        return And(*subparts)
    else:
        return Or(*subparts)

def select_rules_architecture(rules_strs, proposition_names, conjunction=True, choosedual=False):
    """
    Create a LoH model from a list of rules, selecting a subset.
    Parameters:
        rules_strs (List[str]): e.g. ['P_1', '~P_2', 'P_3 | ~P_4']
        proposition_names (List[str]): e.g. ['P_1', 'P_2', 'P_3', 'P_4']
        conjunction (bool): put the rules in a conjunction (otherwise disjunction)
        choosedual (bool): use conjunctive compilation instead of disjunctive
    Returns:
        torch.nn.Module: The compilation of the LoH model (e.g. [P_1, True] & [~P_2, True] & [P_3 | ~P_4, True])
    """
    parser = LogicalParser(proposition_names)
    if conjunction and choosedual:
        architecture = " & ".join([f"[{rule}, True]_" for rule in rules_strs])
    elif conjunction and not choosedual:
        architecture = " & ".join([f"[{rule}, True]" for rule in rules_strs])
    elif choosedual and not conjunction:
        architecture = " | ".join([f"[{rule}, False]_" for rule in rules_strs])
    elif not choosedual and not conjunction:
        architecture = " | ".join([f"[{rule}, False]" for rule in rules_strs])
    return parser.parse(architecture)