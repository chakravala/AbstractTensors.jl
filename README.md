# AbstractTensors.jl

*TensorAlgebra abstract type system interoperability with VectorSpace parameter*

[![Build Status](https://travis-ci.org/chakravala/AbstractTensors.jl.svg?branch=master)](https://travis-ci.org/chakravala/AbstractTensors.jl)
[![Build status](https://ci.appveyor.com/api/projects/status/yey8huk505h4b81u?svg=true)](https://ci.appveyor.com/project/chakravala/abstracttensors-jl)
[![Coverage Status](https://coveralls.io/repos/chakravala/AbstractTensors.jl/badge.svg?branch=master&service=github)](https://coveralls.io/github/chakravala/AbstractTensors.jl?branch=master)
[![codecov.io](http://codecov.io/github/chakravala/AbstractTensors.jl/coverage.svg?branch=master)](http://codecov.io/github/chakravala/AbstractTensors.jl?branch=master)

This package is intended for universal interoperability of the abstract `TensorAlgebra` type system.
All `TensorAlgebra{V}` subtypes contain `V` in their type parameters, used to store a `VectorSpace` value obtained from the [DirectSum.jl](https://github.com/chakravala/DirectSum.jl) package.

For example, this is mainly used in [Grassmann.jl](https://github.com/chakravala/Grassmann.jl) to define various `SubAlgebra`, `TensorTerm` and `TensorMixed` types, each with subtypes. Externalizing the abstract type helps extend the dispatch to other packages.
```Julia
julia> Grassmann.TensorTerm{V,G} <: AbstractTensors.TensorAlgebra{V}
true
```
By itself, this package does not impose any structure or specifications on the `TensorAlgebra{V}` subtypes and elements, aside from requiring `V` to be a `VectorSpace`.
This means that different packages can create special types of tensors with shared method names and a common underlying `VectorSpace` structure.

## Interoperability

Since `VectorSpace` choices are fundamental to `TensorAlgebra` operations, the universal interoperability between `TensorAlgebra{V}` elements with different associated `VectorSpace` choices is naturally realized by applying the `union` morphism to operations.

```Julia
function op(::TensorAlgebra{V},::TensorAlgebra{V}) where V
    # well defined operations if V is shared
end # but what if V ≠ W in the input types?

function op(a::TensorAlgebra{V},b::TensorAlgebra{W}) where {V,W}
    VW = V ∪ W        # VectorSpace type union
    op(VW(a),VW(b))   # makes call well-defined
end # this option is automatic with interop(a,b)

# alternatively for evaluation of forms, VW(a)(VW(b))
```
Some of the method names like `+,-,*,⊗` for `TensorAlgebra` elements are shared across different packages, some of the interoperability is taken care of in this package.
Additionally, a universal unit volume element can be specified in terms of `LinearAlgebra.UniformScaling`, which is independent of `V` and has its interpretation only instantiated by the context of the `TensorAlgebra{V}` element being operated on.

### Example with a new subtype

Suppose we are dealing with a new subtype in another project, such as
```Julia
using AbstractTensors, DirectSum
struct SpecialTensor{V} <: TensorAlgebra{V} end
a = SpecialTensor{ℝ}()
b = SpecialTensor{ℝ'}()
```
To define additional specialized interoperability for further methods, it is necessary to define dispatch that catches well-defined operations for equal `VectorSpace` choices and a fallback method for interoperability, along with a `VectorSpace` morphism:
```Julia
(W::VectorSpace)(s::SpecialTensor{V}) where V = SpecialTensor{W}() # conversions
op(a::SpecialTensor{V},b::SpecialTensor{V}) where V = a # do some kind of operation
op(a::TensorAlgebra{V},b::TensorAlgebra{W}) where {V,W} = interop(op,a,b) # compat
```
which should satisfy (using the `∪` operation as defined in `DirectSum`)
```Julia
julia> op(a,b) |> vectorspace == vectorspace(a) ∪ vectorspace(b)
true
```
Thus, interoperability is simply a matter of defining one additional fallback method for the operation and also a new form `VectorSpace` compatibility morphism.

#### UniformScaling pseudoscalar

The universal interoperability of `LinearAlgebra.UniformScaling` as a pseudoscalar element which takes on the `VectorSpace` form of any other `TensorAlgebra` element is handled globally by defining the dispatch:
```Julia
(W::VectorSpace)(s::UniformScaling) = ones(ndims(W)) # interpret a unit pseudoscalar
op(a::TensorAlgebra{V},b::UniformScaling) where V = op(a,V(b)) # right pseudoscalar
op(a::UniformScaling,b::TensorAlgebra{V}) where V = op(V(a),b) # left pseudoscalar
```
This enables the usage of `I` from `LinearAlgebra` as a universal pseudoscalar element.

##### Tensor evaluation

To support a generalized interface for `TensorAlgebra` element evaluation, a similar compatibility interface is constructible.

```Julia
(a::SpecialTensor{V})(b::SpecialTensor{V}) where V = a # conversion of some form
(a::SpecialTensor{W})(b::SpecialTensor{V}) where {V,W} = interform(a,b) # compat
```
which should satisfy (using the `∪` operation as defined in `DirectSum`)
```Julia
julia> b(a) |> vectorspace == vectorspace(a) ∪ vectorspace(b)
true
```
The purpose of the `interop` and `interform` methods is to help unify the interoperability of `TensorAlgebra` elements.

### Deployed applications

By importing the `AbstractTensors` module, the [Reduce.jl](https://github.com/chakravala/Reduce.jl) is able to correctly bypass operations on `TensorAlgebra` elements to the correct methods within the scope of the `Reduce.Algebra` module.
This requires no additional overhead for the `Grassmann` or `Reduce` packages, because the `AbstractTensors` interoperability interface enables separate precompilation of both.
Additionally, the `VectorSpace` interoperability also enables more arbitrary inputs.
