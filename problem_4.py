from problem_4.problem_4a import problem_4a
from problem_4.problem_4b import problem_4b

import sys
import numpy as np
np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

problem_4a()
problem_4b()