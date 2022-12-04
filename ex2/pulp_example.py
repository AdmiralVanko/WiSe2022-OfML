#!/usr/bin/env python3
#
# Author: Stefan Haller <stefan.haller@iwr.uni-heidelberg.de>

#
# Install the Python LP library via anaconda or pip.
#
# e.g. pip3 install --user pulp
#
import pulp

#
# The documentation for the pulp library is available at
#   <https://pythonhosted.org/PuLP/pulp.html>.
#

# Create problem + variables.
lp = pulp.LpProblem('Test Problem')
v0 = pulp.LpVariable('v0', lowBound=0, upBound=1)
v1 = pulp.LpVariable('v1', lowBound=0, upBound=1)

# Create objective function (<x, c>)
# LpAffineExpression takes list of tuple (LpVariable, cofficent)
obj = pulp.LpAffineExpression([(v0, 10), (v1, 5)])
lp += obj # or lp.setObjective(obj)

# Create a single simplex constraint (sum of variables = 1)
lhs = pulp.LpAffineExpression([(v0, 1), (v1, 1)])
constr = pulp.LpConstraint(lhs, pulp.LpConstraintEQ, 'simplex constraint', 1)
lp += constr

lp.writeLP('/tmp/test.lp')
lp.solve()

print('v0 = {}\nv1 = {}\nobj = {}'.format(v0.value(), v1.value(), lp.objective.value()))
# Obviously, the result is to set v1 to 1, as we only pay a cost of 5 instead
# of 10. Due to the simplex constraint the sum of both variables must be one,
# so the other variable is set to 0.
#
# Do you recognize that even though we did not solve the problem as an integer
# linear problem (ILP) the solution only has integer values?
# You can read in the script/book about the tightness of the LP relaxations!

# To programmatically create a sum of LpVariables (for constraints, etc.):
my_list = [v0, v1]
affine_expression = pulp.lpSum(my_list)
print(type(affine_expression))
print(affine_expression)

# You don't have to go through the trouble of creating all the
# LpAffineExpression/LpConstraints and so...
print(type(v0 + v1)) # -> affine expression of the sum of the variables (coefficients are 1)
print(type(v0 + v1 == 1)) # -> constraint (sum of variables must be equal to 1)
