from tests import *
from cspbase import *
from heuristics import *
from kenken_csp import *


def helper_prop(board, prop=prop_FC, var_ord=ord_mrv):
    csp, var_array = kenken_csp_model(board)
    solver = BT(csp)
    solver.quiet()
    solver.bt_search(prop, var_ord)

if __name__ == '__main__':
    board = BOARDS[2]
    start_time = time.clock()
    helper_prop(board, prop_FC, None)
    end_time = time.clock()
    print(end_time - start_time)