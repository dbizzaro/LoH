'''Optimized combinatorial class
'''
import logging

from src.ilp import Rule_Manger
from src.core import Atom, Term, Clause

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class Optimized_Combinatorial_Generator(Rule_Manger):

    @staticmethod
    def __is_valid_clause(clause, head, body1_terms, body2_terms, target_variables, allow_intensional, intensional_predicates):
        # All variables in head should be in the body
        if not set(target_variables).issubset([v.name for v in body1_terms] + [v.name for v in body2_terms]):
            return False
        if head == clause.body[0] or head == clause.body[1]:  # No Circular
            return False
        # NOTE: Based on appendix requires to have a intensional predicate
        if allow_intensional and not (
            clause.body[0].predicate in intensional_predicates or clause.body[1].predicate in intensional_predicates
        ):
            return False
        return True

    def generate_clauses(self):
        '''Generate all clauses with some level of optimization
        '''
        rule_matrix = []
        for rule in self.rules:
            # logger.info('Generating clauses')
            if rule == None:
                rule_matrix.append([None])
                continue
            clauses = []
            if(rule.allow_intensional):
                p = list(set(self.p_e + self.p_i + [self.target]))
                p_i = list(set(self.p_i))
                intensional_predicates = [atom.predicate for atom in p_i]
            else:
                p = list(set(self.p_e))
                intensional_predicates = []
            variables = ['X_%d' %
                         i for i in range(0, self.target.arity + rule.v)]
            target_variables = ['X_%d' %
                                i for i in range(0, self.target.arity)]

            # Generate the body list
            body_list = []
            head = Atom(
                [Term(True, var) for var in target_variables], self.target.predicate)
            for var1 in variables:
                for var2 in variables:
                    term1 = Term(True, var1)
                    term2 = Term(True, var2)
                    body_list.append([term1, term2])

            # If explicit clause whitelist is supplied, avoid combinatorial enumeration.
            if hasattr(rule, 'has_clause_whitelist') and rule.has_clause_whitelist():
                added_pred = {}
                for clause in rule.clause_whitelist:
                    if clause is None:
                        continue
                    if clause in added_pred:
                        continue
                    body1_terms = clause.body[0].terms
                    body2_terms = clause.body[1].terms
                    if not Optimized_Combinatorial_Generator.__is_valid_clause(
                        clause,
                        head,
                        body1_terms,
                        body2_terms,
                        target_variables,
                        rule.allow_intensional,
                        intensional_predicates,
                    ):
                        continue
                    if not rule.keep(clause):
                        continue
                    added_pred[clause] = 1
                    clauses.append(clause)
                rule_matrix.append(clauses)
                continue

            # Generate the list
            added_pred = {}
            for ind1 in range(0, len(p)):
                pred1 = p[ind1]
                for b1 in body_list:
                    for ind2 in range(ind1, len(p)):
                        pred2 = p[ind2]
                        for b2 in body_list:
                            body1 = Atom([b1[index]
                                          for index in range(0, pred1.arity)], pred1.predicate)
                            body2 = Atom([b2[index]
                                          for index in range(0, pred2.arity)], pred2.predicate)

                            clause = Clause(head, [body1, body2])
                            if not Optimized_Combinatorial_Generator.__is_valid_clause(
                                clause,
                                head,
                                b1,
                                b2,
                                target_variables,
                                rule.allow_intensional,
                                intensional_predicates,
                            ):
                                continue
                            if not rule.keep(clause):
                                continue
                            if clause in added_pred:
                                continue
                            added_pred[clause] = 1
                            clauses.append(clause)
            rule_matrix.append(clauses)
            # logger.info('Clauses Generated')
        return rule_matrix
