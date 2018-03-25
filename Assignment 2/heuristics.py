'''
This file will contain different variable ordering heuristics to be used within
bt_search.

1. ord_dh(csp)
    - Takes in a CSP object (csp).
    - Returns the next Variable to be assigned as per the DH heuristic.
2. ord_mrv(csp)
    - Takes in a CSP object (csp).
    - Returns the next Variable to be assigned as per the MRV heuristic.
3. val_lcv(csp, var)
    - Takes in a CSP object (csp), and a Variable object (var)
    - Returns a list of all of var's potential values, ordered from best value 
      choice to worst value choice according to the LCV heuristic.

The heuristics can use the csp argument (CSP object) to get access to the 
variables and constraints of the problem. The assigned variables and values can 
be accessed via methods.
'''

import random
from copy import deepcopy
from propagators import prop_FC

def ord_dh(csp):
    var_dh, max_dh = None, -9999999
    for var in csp.get_all_unasgn_vars():
        cons = csp.get_cons_with_var(var)
        cur_val = sum([len(c.get_scope()) - 1 for c in cons]) - len(cons)
        if cur_val >= max_dh:
            max_dh = cur_val
            var_dh = var
    return var_dh


def ord_mrv(csp):
    var_mrv, min_mrv = None, 9999999
    for var in csp.get_all_unasgn_vars():
        dom_size = var.cur_domain_size()
        if dom_size < min_mrv:
            min_mrv = dom_size
            var_mrv = var
    return var_mrv

def val_lcv(csp, var):
    val_lcv = []
    for val in var.cur_domain():
        var.assign(val)
        status, pruned = prop_FC(csp, var)
        var.unassign()
        for var_p, val_p in pruned:
            var_p.unprune_value(val_p)
        if status:
            val_lcv.append((val, len(pruned)))
    val_lcv.sort(key=lambda v: v[1])
    return [v[0] for v in val_lcv]