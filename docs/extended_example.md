## Example library usage

The following C code snipped demonstrates, how library functions and primitives can be used for the transitive closure
evaluation of the directed graph, represented as an adjacency matrix with boolean values. The transitive closure
provides info about reachable vertices in the graph:

```c++
/**
 * Performs transitive closure for directed graph
 *
 * @param A Adjacency matrix of the graph
 * @param T Reference to the handle where to allocate and store result
 *
 * @return Status on this operation
 */
spbla_Status TransitiveClosure(spbla_Matrix A, spbla_Matrix* T) {
    spbla_Matrix_Duplicate(A, T);                       /* Duplicate A to result T */
    
    spbla_Index total = 0;
    spbla_Index current;
    
    spbla_Matrix_Nvals(*T, &current);                   /* Query current nvals value */
    
    while (current != total) {                          /* Iterate, while new values are added */
        total = current;
        spbla_MxM(*T, *T, *T, SPBLA_HINT_ACCUMULATE);   /* T += T x T */
        spbla_Matrix_Nvals(*T, &current);
    }
    
    return SPBLA_STATUS_SUCCESS;
}
```

The following Python code snippet demonstrates, how the library python wrapper can be used to compute the same
transitive closure problem for the directed graph within python environment:

```python
import pyspbla as sp


def transitive_closure(a: sp.Matrix):
    """
    Evaluates transitive closure for the provided
    adjacency matrix of the graph.

    :param a: Adjacency matrix of the graph
    :return: The transitive closure adjacency matrix
    """

    t = a.dup()  # Duplicate matrix where to store result
    total = 0  # Current number of values

    while total != t.nvals:
        total = t.nvals
        t.mxm(t, out=t, accumulate=True)  # t += t * t

    return t
```