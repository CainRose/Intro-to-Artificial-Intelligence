'''
All models need to return a CSP object, and a list of lists of Variable objects 
representing the board. The returned list of lists is used to access the 
solution. 

For example, after these three lines of code

    csp, var_array = kenken_csp_model(board)
    solver = BT(csp)
    solver.bt_search(prop_FC, var_ord)

var_array[0][0].get_assigned_value() should be the correct value in the top left
cell of the KenKen puzzle.

The grid-only models do not need to encode the cage constraints.

1. binary_ne_grid (worth 10/100 marks)
    - A model of a KenKen grid (without cage constraints) built using only 
      binary not-equal constraints for both the row and column constraints.

2. nary_ad_grid (worth 10/100 marks)
    - A model of a KenKen grid (without cage constraints) built using only n-ary 
      all-different constraints for both the row and column constraints. 

3. kenken_csp_model (worth 20/100 marks) 
    - A model built using your choice of (1) binary binary not-equal, or (2) 
      n-ary all-different constraints for the grid.
    - Together with KenKen cage constraints.

'''

from cspbase import Variable, Constraint, CSP
from itertools import permutations
import operator


def binary_ne_grid(kenken_grid):
    # Get the size of the grid and buid the domain of each variable
    size = kenken_grid[0][0]
    domain = list(range(1, size + 1))
    # Instantiate all variables in size x size list and instantiate csp
    vars = []
    for i in domain:
        vars.append([Variable(str(i) + str(j), domain) for j in domain])
    csp = CSP('kenken', [var for row in vars for var in row])
    # Create binary constraints
    valid = list(permutations(domain, 2))
    for i in range(size):
        for j in range(size):
            for k in range(j + 1, size):
                cons_x = Constraint('{0}{1}-{0}{2}'.format(i, j, k),
                                    [vars[i][j], vars[i][k]])
                cons_y = Constraint('{1}{0}-{2}{0}'.format(i, j, k),
                                    [vars[j][i], vars[k][i]])
                cons_x.add_satisfying_tuples(valid)
                cons_y.add_satisfying_tuples(valid)
                csp.add_constraint(cons_x)
                csp.add_constraint(cons_y)

    return csp, vars


def nary_ad_grid(kenken_grid):
    # Get the size of the grid and buid the domain of each variable
    size = kenken_grid[0][0]
    domain = list(range(1, size + 1))
    # Instantiate all variables in size x size list and instantiate csp
    vars = []
    for i in domain:
        vars.append([Variable(str(i) + str(j), domain) for j in domain])
    csp = CSP('kenken', [var for row in vars for var in row])
    # Create n-ary constraints
    valid = list(permutations(domain, size))
    for i in range(size):
        cons_x = Constraint('r' + str(i), vars[i])
        cons_y = Constraint('c' + str(i), [v[i] for v in vars])
        cons_x.add_satisfying_tuples(valid)
        cons_y.add_satisfying_tuples(valid)
        csp.add_constraint(cons_x)
        csp.add_constraint(cons_y)

    return csp, vars


def get_var(vars, coord):
    c = tuple([int(i) - 1 for i in str(coord)])
    return vars[c[0]][c[1]]


def augment_valid(valid, valid_i, val):
    for v in valid_i:
        if v:
            v.append(val)
    valid.extend(valid_i)


def valid_given_op(n, targ, dom, op, ordered=False):
    if ordered:
        dom = dom[::-1]
    comp = operator.gt if not ordered else operator.lt
    valid = []
    for i, val in enumerate(dom):
        valid_i = __valid_given_op(dom[i], n - 1, targ, dom[i:], op, comp)
        augment_valid(valid, valid_i, val)
    valid = [tuple(v) for p in valid for v in permutations(p)]
    return list(set(valid))


def __valid_given_op(cur, n, targ, dom, op, comp):
    # print(cur, n, targ, dom)
    if comp(cur, targ):
        return []
    if n == 1:
        for val in dom:
            if op(cur, val) == targ:
                return [[val]]
        return []

    valid = []
    for i, val in enumerate(dom):
        new_cur = op(cur, val)
        valid_i = __valid_given_op(new_cur, n - 1, targ, dom[i:], op, comp)
        augment_valid(valid, valid_i, val)
    return valid


ops = ['+', '-', '/', '*']
def kenken_csp_model(kenken_grid):
    size = kenken_grid[0][0]
    domain = list(range(1, size + 1))
    csp, vars = nary_ad_grid(kenken_grid)
    for i, cage in enumerate(kenken_grid[1:]):
        if len(cage) == 2:
            # Forced value constraint
            cons = Constraint('g{} f'.format(i), [get_var(vars, cage[0])])
            cons.add_satisfying_tuples([(cage[1],)])
        else:
            # Extract relevant data
            n_vars = len(cage) - 2
            target = cage[-2]
            operation = cage[-1]
            # Determine scope of the constraint and initialize
            vars_i = [get_var(vars, c) for c in cage[:-2]]
            cons = Constraint('g{} {}{}'.format(i, target, ops[operation]),
                              vars_i)
            # Determine satisfying tuples
            valid = []
            if operation == 0:  # Addition
                valid = valid_given_op(n_vars, target, domain,
                                       operator.add)
            elif operation == 1:  # Subtraction
                valid = valid_given_op(n_vars, target, domain,
                                       operator.sub, True)
            elif operation == 2:  # Division
                valid = valid_given_op(n_vars, target, domain,
                                       operator.floordiv, True)
            elif operation == 3:  # Multiplication
                valid = valid_given_op(n_vars, target, domain,
                                       operator.mul)
            cons.add_satisfying_tuples(valid)

        csp.add_constraint(cons)

    return csp, vars