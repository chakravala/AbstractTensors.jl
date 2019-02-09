# AbstractTensors.jl

*TensorAlgebra abstract type system interface with VectorSpace parameter*

This package is intended for universal interopability of the abstract `TensorAlgebra` root-type system.
All `TensorAlgebra{V}` elements and subtypes contain `V` in their type parameters, which is used to store a `VectorSpace` value obtrained from the [DirectSum.jl](https://github.com/chakravala/DirectSum.jl) package.

For example, this is mainly used in the [Grassmann.jl](https://github.com/chakravala/Grassmann.jl) package to define various `SubAlgebra`, `TensorTerm` and `TensorMixed` types, each with subtypes.
```Julia
julia> Grassmann.TensorTerm{V,G} <: AbstractTensors.TensorAlgebra{V}
true
```

## Interopability

Since `VectorSpace` choices are fundamental to `TensorAlgebra` operations, the universal interopability between `TensorAlgebra` elements with different associated `VectorSpace` choices is naturally realized by applying the `union` morphism to operations.

```Julia
function op(::TensorAlgebra{V},::TensorAlgebra{V}) where V
    # well defined operations
end

function op(a::TensorAlgebra{V},b::TensorAlgebra{W}) where {V,W}
    VW = V ∪ W        # VectorSpace type union
    op(VW(a),VW(b))   # makes call well-defined
end
```
Some of the method names like `+,-,*,⊗` for `TensorAlgebra` elements are shared across different packages, some of the interopability is taken care of in this package.

To define additional specialized interopability for further methods, it is necessary to define dispatch that catches well-defined operations for equal `VectorSpace` choices and a fallback method for interopability, along with a `VectorSpace` morphism:

```Julia
struct SpecialTensor{V} <: TensorAlgebra{V} end
op(s::SpecialTensor{V},::SpecialTensor{V}) where V = s
op(a::TensorAlgebra{V},b::TensorAlgebra{W}) where {V,W} = interop(op,a,b)
(W::VectorSpace)(s::SpecialTensor{V}) where V = SpecialTensor{W}()
```
which should satisfy
```Julia
juila> vectorspace(op(SpecialTensor{ℝ}(),SpecialTensor{ℝ'}())) == ℝ⊕ℝ'
true
```
Thus, interopability simply a matter of defining one additional fallback method for the operation and also a `VectorSpace` compatiblity morphism.

## Tensor evaluation

To support a generalized interface for `TensorAlgebra` element evaluation, a similar compatibility interface is constructible.


