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

Additionally, `TupleVector` is provided as a light weight alternative to [StaticArrays.jl](https://github.com/JuliaArrays/StaticArrays.jl).

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

## TupleVector

*Statically sized tuple vectors for Julia*

**TupleVector** provides a framework for implementing statically sized tuple vectors
in Julia, using the abstract type `TupleVector{N,T} <: AbstractVector{T}`.
Subtypes of `TupleVector` will provide fast implementations of common array and
linear algebra operations. Note that here "statically sized" means that the
size can be determined from the *type*, and "static" does **not** necessarily
imply `immutable`.

The package also provides some concrete static vector types: `Values` which may be used as-is (or else embedded in your own type).
Mutable versions `Variables` are also exported, as well
as `FixedVector` for annotating standard `Vector`s with static size information.

### Quick start

Add *AbstractTensors* from the [Pkg REPL](https://docs.julialang.org/en/latest/stdlib/Pkg/#Getting-Started-1), i.e., `pkg> add AbstractTensors`. Then:
```julia
using AbstractTensors

# Create Values using various forms, using constructors, functions or macros
v1 = Values(1, 2, 3)
v1.v === (1, 2, 3) # Values uses a tuple for internal storage
v2 = Values{3,Float64}(1, 2, 3) # length 3, eltype Float64
v5 = zeros(Values{3}) # defaults to Float64
v7 = Values{3}([1, 2, 3]) # Array conversions must specify size

# Can get size() from instance or type
size(v1) == (3,)
size(typeof(v1)) == (3,)

# Supports all the common operations of AbstractVector
v7 = v1 + v2
v8 = sin.(v2)

# Indexing can also be done using static vectors of integers
v1[1] === 1
v1[:] === v1
typeof(v1[[1,2,3]]) <: Vector # Can't determine size from the type of [1,2,3]
```

### Approach

The package provides a range of different useful built-in `TupleVector` types,
which include mutable and immutable vectors based upon tuples, vectors based upon
structs, and wrappers of `Vector`. There is a relatively simple interface for
creating your own, custom `TupleVector` types, too.

This package also provides methods for a wide range of `AbstractVector` functions,
specialized for (potentially immutable) `TupleVector`s. Many of Julia's
built-in method definitions inherently assume mutability, and further
performance optimizations may be made when the size of the vector is known to the
compiler. One example of this is by loop unrolling, which has a substantial
effect on small arrays and tends to automatically trigger LLVM's SIMD
optimizations. In combination with intelligent fallbacks to
the methods in Base, we seek to provide a comprehensive support for statically
sized vectors, large or small, that hopefully "just works".

`TupleVector` is directly inspired from [StaticArrays.jl](https://github.com/JuliaArrays/StaticArrays.jl).
