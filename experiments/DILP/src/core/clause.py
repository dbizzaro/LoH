'''Defines the clauses
'''
from src.core.atom import Atom
from functools import reduce


class Clause():

    def __init__(self, head: Atom, body: list):
        '''
        Arguments:
            head {Atom} -- Head atom of clause
            body {list} -- List of body atoms
        '''
        self._head = head
        self._body = body
        self._variable = sorted(list(head.variables.union(
            set(reduce(set.union, [atom.variables for atom in body])))))

    def __str__(self):
        return '%s <- %s' % (str(self._head), ','.join(str(atom) for atom in self._body))

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        return other is not None and all([(atom in other.body) for atom in self._body]) and all([(atom in self._body) for atom in other.body]) and self._head == other.head

    def __hash__(self):
        # __eq__ treats body atoms as an unordered set, so hash must do the same.
        body_key = frozenset(str(atom) for atom in self._body)
        return hash((str(self._head), body_key, tuple(self._variable)))

    @property
    def body(self):
        return self._body

    @body.setter
    def body(self, value):
        self._body = value

    @property
    def head(self):
        return self._head

    @head.setter
    def head(self, value):
        self._head = value

    @property
    def variables(self):
        '''Return all the variable used in the clause
        '''

        return self._variable
