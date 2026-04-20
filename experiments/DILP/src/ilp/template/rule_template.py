'''Defines the rule template
'''


class Rule_Template():

    def __init__(self, v: int, allow_intensional: bool, clause_whitelist=None, keep_clause=None):
        '''

        Arguments:
            v {int} -- numberof existentially quantified variable allowed in the clause
            allow_intensional {bool} -- True is intensional predicates are allowed, False if only extensional predicates
            clause_whitelist {iterable|None} -- explicit list/set of admissible clauses
            keep_clause {callable|None} -- filter function keep_clause(clause)->bool
        '''

        if clause_whitelist is not None and keep_clause is not None:
            raise ValueError('Specify at most one of clause_whitelist or keep_clause')
        if keep_clause is not None and not callable(keep_clause):
            raise ValueError('keep_clause must be callable')

        self._v = v
        self._allow_intensional = allow_intensional
        self._clause_whitelist = None if clause_whitelist is None else tuple(clause_whitelist)
        self._clause_whitelist_lookup = None if clause_whitelist is None else set(clause_whitelist)
        self._keep_clause = keep_clause

    @property
    def v(self):
        return self._v

    @property
    def allow_intensional(self):
        return self._allow_intensional

    @property
    def clause_whitelist(self):
        return self._clause_whitelist

    @property
    def keep_clause(self):
        return self._keep_clause

    def has_clause_whitelist(self):
        return self._clause_whitelist is not None

    def keep(self, clause):
        if self._clause_whitelist_lookup is not None:
            return clause in self._clause_whitelist_lookup
        if self._keep_clause is not None:
            return bool(self._keep_clause(clause))
        return True
