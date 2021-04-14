import pyspbla as sp

#
# Basic setup example
#

a = sp.Matrix.empty(shape=(2, 3))
a[0, 0] = True
a[1, 2] = True

b = sp.Matrix.empty(shape=(3, 4))
b[0, 1] = True
b[0, 2] = True
b[1, 3] = True
b[2, 1] = True

print(a, b, a.mxm(b), sep="\n")

#
# Transitive closure example
#

a = sp.Matrix.empty(shape=(4, 4))
a[0, 1] = True
a[1, 2] = True
a[2, 0] = True
a[2, 3] = True
a[3, 2] = True

t = a.dup()                             # Duplicate matrix where to store result
total = 0                               # Current number of values

while total != t.nvals:
    total = t.nvals
    t.mxm(t, out=t, accumulate=True)    # t += t * t

print(a, t, sep="\n")

#
# Export matrices set to graph viz graph
#

name = "Test"                           # Displayed graph name
shape = (4, 4)                          # Adjacency matrices shape
colors = {"a": "red", "b": "green"}     # Colors per label

a = sp.Matrix.empty(shape=shape)        # Edges labeled as 'a'
a[0, 1] = True
a[1, 2] = True
a[2, 0] = True

b = sp.Matrix.empty(shape=shape)        # Edges labeled as 'b'
b[2, 3] = True
b[3, 2] = True

print(sp.matrices_to_gviz(matrices={"a": a, "b": b}, graph_name=name, edge_colors=colors))



