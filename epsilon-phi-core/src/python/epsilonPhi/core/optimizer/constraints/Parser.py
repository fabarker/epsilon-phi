from collections import OrderedDict
import numpy as np
from typing import Optional
import logging
from fractions import Fraction

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Constraints:
    def __init__(self, constraints: Optional[str] = None):
        self.constraints = constraints
        self.UB = None
        self.LB = None
        self.mat = None
        self.UB_box = None
        self.LB_box = None

    def get_LBs(self):
        return self.LB if self.LB is not None else -np.inf * np.ones(self.mat.shape[0])

    def get_UBs(self):
        return self.UB if self.UB is not None else -np.inf * np.ones(self.mat.shape[0])



class ConstraintsParser:

    def __init__(self, constraints: str):
        self.constraints = constraints


    def break_down_constraint_single(self, s: str, asset_list: list):
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



    def format_result(self, const_result, asset_list: list):

        const_list = []
        for i in range(const_result.mat.shape[0]):

            const = OrderedDict()
            for j in range(const_result.mat.shape[1]):
                const[asset_list[j]] = const_result.mat[i, j]
            const_list.append(const)

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
            for l in LB_list:
                i += 1

        if not UB_list is None:
            i = 0
            for l in UB_list:
                i += 1

        return const_list, LB_list, UB_list

    def generate_constraints_UB_LB(
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
        splt_str = tmp_str.split(';')
        splt_str = sorted([
            x for x in splt_str if x
        ])

        # make sure the other constrain is not binding
        num_const = len(splt_str)
        result.LB = np.ones(num_const) * 100 * (-len(asset_list))
        result.UB = np.ones(num_const) * 100 * (len(asset_list))
        result.mat = np.zeros((num_const, len(asset_list)))

        for i in range(num_const):

            # strip the % off the string
            s = splt_str[i].replace('%', "")

            lt_idx = s.find('<')
            gt_idx = s.find('>')
            eq_idx = s.find('=')

            if lt_idx != -1:
                # matrix equality
                result.mat[i, :], error = self.break_down_constraint_single(
                    s[0:lt_idx],
                    asset_list
                )
                # inequality
                result.UB[i] = float(s[lt_idx + 1:]) / (1 - 0)

            if gt_idx != -1:
                result.mat[i, :], error = self.break_down_constraint_single(
                    s[0:gt_idx],
                    asset_list
                )
                result.LB[i] = float(s[gt_idx + 1:]) / (1 - 0)
                result.mat[i, :] = result.mat[i, :]

            if eq_idx != -1:
                result.mat[i, :], error = self.break_down_constraint_single(
                    s[0:eq_idx],
                    asset_list
                )
                result.UB[i] = float(s[eq_idx + 1:]) / (1 - 0)
                result.LB[i] = float(s[eq_idx + 1:]) / (1 - 0)

                if error != 0:
                    logger.error("Error in constraints parser... error code = {}".format(error))


        const, LB, UB = self.format_result(
            result,
            asset_list,
        )

        return result, const, LB, UB, error


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
        splt_str = tmp_str.split(';')
        splt_str = sorted([
            x for x in splt_str if x
        ])

        # make sure the other constrain is not binding
        num_const = len(splt_str)
        result.LB = np.ones(num_const) * 100 * (-len(asset_list))
        result.UB = np.ones(num_const) * 100 * (len(asset_list))
        result.mat = np.zeros((num_const, len(asset_list)))

        eq_constraint_idx = []
        for i in range(num_const):

            # strip the % off the string
            s = splt_str[i].replace('%', "")

            lt_idx = s.find('<')
            gt_idx = s.find('>')
            eq_idx = s.find('=')

            if lt_idx != -1:
               # matrix equality
               result.mat[i,:], error = self.break_down_constraint_single(
                   s[0:lt_idx],
                   asset_list
               )
               # inequality
               result.UB[i] = float(s[lt_idx+1:])

            if gt_idx != -1:
                result.mat[i, :], error = self.break_down_constraint_single(
                    s[0:gt_idx],
                    asset_list
                )
                result.UB[i] = float(s[gt_idx + 1:]) / (1-0) * (-1)

            if eq_idx != -1:
                result.mat[i, :], error = self.break_down_constraint_single(
                    s[0:eq_idx],
                    asset_list
                )
                result.UB[i] = float(s[eq_idx + 1:]) / (1 - 0)
                eq_constraint_idx.append(i)

                if error != 0:
                    logger.error("Error in constraints parser... error code = {}".format(error))

        final_result = Constraints(constraints)
        final_result.mat = np.zeros((num_const + len(eq_constraint_idx), len(asset_list)))
        final_result.UB = np.zeros(num_const + len(eq_constraint_idx))

        for i in range(num_const):
            final_result.mat[i,:] = result.mat[i,:]
            final_result.UB[i] = result.UB[i]

        i = num_const
        for j in range(len(eq_constraint_idx)):
            final_result.mat[i+j] = result.mat[eq_constraint_idx[j]] * (-1)
            final_result.UB[i+j] = result.UB[eq_constraint_idx[j]]


        const, LB, UB = self.format_result(
            final_result,
            asset_list,
        )

        return result, const, LB, UB, error

if __name__ == '__main__':

    constr_str = '(-45)*RUS1G+40*RUS1V=0;(-45)*RUS2G+5*RUS1V=0;(-45)*RUS2V_GE10+10*RUS1V=0;(-1)*ACEMLPIT+MSWDIF=0;(-1)*SPGLREITH+MSWDIF=0;MSEM+(-1)*ACEMLPIT=0;'
    assetList = [
        'RUS1G',
        'RUS2G',
        'RUS1V',
        'RUS2V',
        'RUS2V_GE10',
        'LHUSMIT10',
        'IUSHHY_GE25',
        'ACEMLPIT',
        'MSEM',
        'ACEMLPIT',
        'MSWDIF',
        'SPGLREITH',
        'CSFBED_GE25',
        'CSFBSLE',
        'CSFBMA',
        'PEALLQ',
        'PEDIS',
        'PEEN',
        'PEMEZ',
        'PEVEN',
        'PE_adj_MSWDIF',
        'RETBI']

    p = ConstraintsParser(constr_str)
    p.generate_constraints_UB(constr_str, assetList)



