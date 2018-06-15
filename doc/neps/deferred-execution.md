# Deferred Execution
- Author: Matt Harrigan <harrigan.matthew@gmail.com>
- Created: 03-April-2017

## Abstract
This NEP describes a proposal to add deferred execution of operations on arrays.  This will allow various performance
optimizations, most notably reducing memory bandwidth by processing cache sized blocks through multiple operations in a
single pass.  This NEP is largely a variation on this [NEP](https://docs.scipy.org/doc/numpy-dev/neps/deferred-ufunc-evaluation.html)
The key difference is leveraging the `__array_ufunc__` method for a cleaner API.

## Proposed API
The functionality would be defined in a new class `DeferredExecution`.  Two key methods contain the core of the proposed
functionality:
- execute(): returns the result of all operations as an ndarray, or a namedtuple of arrays for the result and each output call.
- output(name=None): can be called after any operation so the subsequent execute call additionally returns the result at that stage
  of the operator graph.  Name is an optional argument which defines the variable name in the output namedtuple, default is
  'output_{}''.format(index)

Other required methods to complete the basic functionality would be:
- the constructor would take an ndarray and wrap it
- methods for debug printing and introspection
- many of the special methods for basic arithmetic, logical operations, assignment, and indexing

## Example Usage
```
def foo(arr):
    # function to convert from degree to radians and find the sum of squares
    # very little modification to typical numpy code is required for deferred execution

    # create deferred execution array
    deferred = DeferredExecutionArray(arr)

    # created the first operation
    # no calculations are performed
    # the returned object has references to the inputs and operations, the beginning of the operation graph
    scaled = deferred * np.pi / 180

    # add the next operations to the deferred array
    # regular ufuncs can be used
    sum_square = np.add.reduce(np.square(scaled)

    # execute, returning the result as an ndarray
    result = sum_square.execute()

    return result


def bar(arr):
    plus1 = DeferredExecutionArray(arr) + 1
    plus1.output()
    plus2 = plus1 + 1

    # returns a namedtuple of DeferredExecutionResult(output_0=arr+1, result=arr+2)
    # arguments are returned in order of persist calls, last being the final result
    result = plus2.execute()
```

## Rationale
### output
A user would often want to keep intermediate results in a long chain of operations.  Therefore output() was added.  It
allows for those arrays to be returned, but also allows for optimization over the largest possible number of operations,
since execute would typically only be called once.  The optimization algorithm should know what needs to be returned,
since if an array is only a temporary intermediate, more optimizations may be possible such as operation reordering and
further reductions in memory bandwidth.  It also would allow for reduction in total memory required.  Therefore the
proposed API requires the user to explicitly indicate any intermediate results that must be returned.

### execute
The prior NEP had implicit execution.  For example, a print statement may trigger an execution.  Many other possible
functions could trigger an execute.  That adds quite a bit of complexity.  It also limits the optimization opportunities.
Therefore in the proposed API all executions are explicitly triggered by the user invoking the execute method.  Calling
print on an instance of DeferredExecutionArray would print (hopefully useful) debug information about the inputs and
operations it refers to, but would not trigger and execution.

## Implementation
The key to allowing for this functionality is `__array_ufunc__`.  It allows for arbitrary objects to be returned when
normal ufuncs are used.  In this case the returned object is a proxy which keeps track of inputs and operations as a
graph.  They are executed efficiently after performing any number of optimizations.

DeferredExecution is not a subclass of ndarray.  The main reason is that it would violate the substitution principle.
DeferredExecution objects are fundamentally quite different in many respects.

If any input or output argument if of type DeferredExecution, the result is a DeferredExecution object.  This applys
to ufuncs, assignment, and selection.  Passing a regular ndarry as out when other arguments are DeferredExecution objects
is a TypeError.

## Error Handling
Error checking that does not require the operation to be performed should be done when that operation is defined.  A
common example would be checking for compatible shapes.  Errors during execution such as divide by zero are propagated
up when execute is called.

## Roadmap
Initially a subset of numpy functionality could be supported.  Supporting regular ufuncs (no core dimensions) might
be a good initial start.  Subsequently other generalized ufuncs or non-ufunc functions could be added.  Also the
number and complexity of optimizations could increase over time.
