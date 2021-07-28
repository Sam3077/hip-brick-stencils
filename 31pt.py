from st.expr import Index, ConstRef
from st.grid import Grid

radius = 5

# Declare indices
i = Index(0)
j = Index(1)
k = Index(2)

param = [
       ConstRef("dev_coeff[0]"),
       ConstRef("dev_coeff[1]"),
       ConstRef("dev_coeff[2]"),
       ConstRef("dev_coeff[3]"),
       ConstRef("dev_coeff[4]"),
       ConstRef("dev_coeff[5]"),
       ConstRef("dev_coeff[6]"),
       ConstRef("dev_coeff[7]"),
       ConstRef("dev_coeff[8]")
]

# Declare grid
input = Grid("bIn", 3)
output = Grid("bOut", 3)

calc = param[0] * input(i, j, k)
for a in range(1, radius + 1):
    calc = calc + param[a] * ( \
            input(i, j, k + a) + input(i, j + a, k) + input(i + a, j, k) + \
            input(i, j, k - a) + input(i, j - a, k) + input(i - a, j, k))
output(i, j, k).assign(calc)
STENCIL = [output]
