# AbstractTensors.jl

*Tensor algebra abstract type interoperability with vector bundle parameter*

[![DOI](https://zenodo.org/badge/169811826.svg)](https://zenodo.org/badge/latestdoi/169811826)
[![Docs Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://grassmann.crucialflow.com/stable)
[![Docs Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://grassmann.crucialflow.com/dev)
[![Gitter](https://badges.gitter.im/Grassmann-jl/community.svg)](https://gitter.im/Grassmann-jl/community?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge)
[![Build Status](https://travis-ci.org/chakravala/AbstractTensors.jl.svg?branch=master)](https://travis-ci.org/chakravala/AbstractTensors.jl)
[![Build status](https://ci.appveyor.com/api/projects/status/yey8huk505h4b81u?svg=true)](https://ci.appveyor.com/project/chakravala/abstracttensors-jl)
[![Coverage Status](https://coveralls.io/repos/chakravala/AbstractTensors.jl/badge.svg?branch=master&service=github)](https://coveralls.io/github/chakravala/AbstractTensors.jl?branch=master)
[![codecov.io](https://codecov.io/github/chakravala/AbstractTensors.jl/coverage.svg?branch=master)](https://codecov.io/github/chakravala/AbstractTensors.jl?branch=master)

The `AbstractTensors` package is intended for universal interoperability of the abstract `TensorAlgebra` type system.
All `TensorAlgebra{V}` subtypes have type parameter `V`, used to store a `TensorBundle` value obtained from [DirectSum.jl](https://github.com/chakravala/DirectSum.jl).

For example, this is mainly used in [Grassmann.jl](https://github.com/chakravala/Grassmann.jl) to define various `SubAlgebra`, `TensorGraded` and `TensorMixed` types, each with subtypes. Externalizing the abstract type helps extend the dispatch to other packages.
By itself, this package does not impose any specifications or structure on the `TensorAlgebra{V}` subtypes and elements, aside from requiring `V` to be a `Manifold`.
This means that different packages can create tensor types having a common underlying `TensorBundle` structure.

## Interoperability

Since `TensorBundle` choices are fundamental to `TensorAlgebra` operations, the universal interoperability between `TensorAlgebra{V}` elements with different associated `TensorBundle` choices is naturally realized by applying the `union` morphism to operations.

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
Some of operations like `+,-,*,⊗,⊛,⊙,⊠,⨼,⨽,⋆` and postfix operators `⁻¹,ǂ,₊,₋,ˣ` for `TensorAlgebra` elements are shared across different packages, some of the interoperability is taken care of in this package.
Additionally, a universal unit volume element can be specified in terms of `LinearAlgebra.UniformScaling`, which is independent of `V` and has its interpretation only instantiated by the context of the `TensorAlgebra{V}` element being operated on.

Utility methods such as `scalar, involute, norm, norm2, unit, even, odd` are also defined.

### Example with a new subtype

Suppose we are dealing with a new subtype in another project, such as
```Julia
using AbstractTensors, DirectSum
struct SpecialTensor{V} <: TensorAlgebra{V} end
a = SpecialTensor{ℝ}()
b = SpecialTensor{ℝ'}()
```
To define additional specialized interoperability for further methods, it is necessary to define dispatch that catches well-defined operations for equal `TensorBundle` choices and a fallback method for interoperability, along with a `Manifold` morphism:
```Julia
(W::Signature)(s::SpecialTensor{V}) where V = SpecialTensor{W}() # conversions
op(a::SpecialTensor{V},b::SpecialTensor{V}) where V = a # do some kind of operation
op(a::TensorAlgebra{V},b::TensorAlgebra{W}) where {V,W} = interop(op,a,b) # compat
```
which should satisfy (using the `∪` operation as defined in `DirectSum`)
```Julia
julia> op(a,b) |> Manifold == Manifold(a) ∪ Manifold(b)
true
```
Thus, interoperability is simply a matter of defining one additional fallback method for the operation and also a new form `TensorBundle` compatibility morphism.

#### UniformScaling pseudoscalar

The universal interoperability of `LinearAlgebra.UniformScaling` as a pseudoscalar element which takes on the `TensorBundle` form of any other `TensorAlgebra` element is handled globally by defining the dispatch:
```Julia
(W::Signature)(s::UniformScaling) = ones(ndims(W)) # interpret a unit pseudoscalar
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
julia> b(a) |> Manifold == Manifold(a) ∪ Manifold(b)
true
```
The purpose of the `interop` and `interform` methods is to help unify the interoperability of `TensorAlgebra` elements.

### Deployed applications

The key to making the whole interoperability work is that each `TensorAlgebra` subtype shares a `TensorBundle` parameter (with all `isbitstype` parameters), which contains all the info needed at compile time to make decisions about conversions. So other packages need only use the vector space information to decide on how to convert based on the implementation of a type. If external methods are needed, they can be loaded by `Requires` when making a separate package with `TensorAlgebra` interoperability.
