from casadi import SX, Function, CodeGenerator, vertcat, jtimes, gradient
from sys import argv, path
from os.path import join, dirname

path.insert(0, join(dirname(__file__), '..', '..', '..', 'src', 'alpaqa'))
import casadi_generator

if len(argv) < 2:
    print(f"Usage:    {argv[0]} <name>")
    exit(0)

x = SX.sym("x")
y = SX.sym("y")
z = SX.sym("z")
unknwns = vertcat(x, y, z)

p = SX.sym("p")

# Formulate the NLP
f = x**2 + p * z**2
g = z + (1 - x)**2 - y

cg, _, _, _ = casadi_generator.generate_casadi_problem(
    Function("f", [unknwns, p], [f]),
    Function("g", [unknwns, p], [g]),
    name=argv[1])
cg.generate()
