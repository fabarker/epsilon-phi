from collections import OrderedDict
import numpy as np
import logging
from fractions import Fraction

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Constraints:
    def __init__(self, constraints: str):
        self.constraints = constraints


class ConstraintsParser:

    def __init__(self, constraints: str):
        self.constraints = constraints


    def break_down_constraint_single(s, asset_list):
        error = 0
        temp = ''
        elements = []
        br = 0

        for i in range(len(s)):
            idx = '+-'.find(s[i])
            if idx != -1 and br == 0:
                elements.append(temp)
                temp = ''
            else:
                temp = temp + s[i]
                if s[i] == '(':
                    br += 1
                elif s[i] == ')':
                    br -= 1
        elements.append(temp)
        mat = np.zeros(len(asset_list))

        for ss in elements:

            if ss[0] == "(":
                k = 1
                while ss[k] != ")":
                    k += 1
                str = ss[1:k]
                if '/' in str:
                    mult = float(Fraction(str))
                else:
                    mult = float(str)
                ss = ss[k+2:]
            elif ss.find('*') != -1:
                k = ss.find("*")
                str = ss[0:k]
                if '/' in str:
                    mult = float(Fraction(str))
                else:
                    mult = float(str)
                ss = ss[k+1:]
            else:
                mult = 1

            if not ss in asset_list:
                logger.error('Asset: {} from constraint: {} not present in optimization list'.format(ss, s))
                error = 17
            else:
                h = asset_list.index(ss)
                mat[h] = mult
        return mat, error



    def format_result(self, const_result, asset_list):

        const_list = []
        for i in range(const_result.mat.shape[0]):

            const = OrderedDict()
            for j in range(const_result.mat.shape[0]):
                const[asset_list[j]] = const_result.mat[i, j]

        if not const_result.LB is None:
            LB_list = list(const_result.LB)
        else:
            LB_list = None

        UB_list = list(const_result.UB)

        j = 0
        for c in const_list:

            for i in c:
                if c[i] != 0:
                    logger.debug(('Constraint {} ---> Constraint {} {}'.format(j, i, c[i])))
            j += 1

        if not LB_list is None:
            i = 0
            for i in LB_list:
                i += 1

        if not UB_list is None:
            i = 0
            for i in UB_list:
                i += 1

        return const_list, LB_list, UB_list

    def generate_constraints_UB_LB(
            self,
            constraints: str,
            asset_list: list
    ):
        pass


    def generate_constraints_UB(
            self,
            constraints: str,
            asset_list: list
    ):

        error = 0
        result = Constraints(constraints)
        result.constraints = constraints

        # operate on tmp_str
        tmp_str = constraints

        # split by delimeter ;
        splt_str = tmp_str.split(':')
        splt_str = sorted([
            x for x in splt_str if x
        ])

        # make sure the other constrain is not binding
        num_const = len(splt_str)
        LB = np.ones(num_const) * 100 * (-len(asset_list))
        UB = np.ones(num_const) * 100 * (len(asset_list))
        mat = np.zeros((num_const, len(asset_list)))

        eq_constraint_idx = []
        for i in range(num_const):

            # strip the % off the string
            s = splt_str[i].replace('%', "")

            lt_idx = s.find('<')
            gt_idx = s.find('>')
            eq_idx = s.find('=')

            if lt_idx != -1:
               # matrix equality
               mat[i,:], error = self.break_down_constraint_single(
                   s[0:lt_idx],
                   asset_list
               )
               # inequality
               UB[i] = float(s[lt_idx+1:])

            if gt_idx != -1:
                mat[i, :], error = self.break_down_constraint_single(
                    s[0:gt_idx],
                    asset_list
                )
                UB[i] = float(s[gt_idx + 1:]) / (1-0) * (-1)

            if eq_idx != -1:
                mat[i, :], error = self.break_down_constraint_single(
                    s[0:eq_idx],
                    asset_list
                )
                UB[i] = float(s[eq_idx + 1:]) / (1 - 0)
                eq_constraint_idx.append(i)

                if error != 0:
                    logger.error("Error in constraints parser... error code = {}".format(error))

        final_result = Constraints(constraints)
        mat_prime = np.zeros((num_const + len(eq_constraint_idx), len(asset_list)))
        UB_prime = np.zeros(num_const + len(eq_constraint_idx))

        for i in range(num_const):
            mat_prime[i,:] = mat[i,:]
            UB_prime[i,:] = UB[i,:]

        i = num_const
        for j in range(len(eq_constraint_idx)):
            mat_prime[i+j] = mat[eq_constraint_idx[j]] * (-1)
            UB_prime[i+j] = UB[eq_constraint_idx[j]]

        const, LB, UB = self.format_result(
            final_result,
            asset_list,
        )

        return result, const, LB, UB, error


