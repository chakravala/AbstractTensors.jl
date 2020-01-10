using AbstractTensors
using Test, DirectSum, LinearAlgebra

# example data
struct SpecialTensor{V} <: TensorAlgebra{V} end
a,b = SpecialTensor{ℝ}(), SpecialTensor{ℝ'}()
@test ndims(+(a)) == ndims(b)

## tensor operation (trivial test)
op(s::SpecialTensor{V},::SpecialTensor{V}) where V = s
op(a::TensorAlgebra{V},b::TensorAlgebra{W}) where {V,W} = interop(op,a,b)
(W::Signature)(s::SpecialTensor{V}) where V = SpecialTensor{W}()
@test Manifold(op(a,b)) == ℝ⊕ℝ'
@test Manifold(interop(op,a,b)) == ℝ⊕ℝ'
@test Manifold(op(a,a)) == ℝ
@test Manifold(interop(op,a,a)) == ℝ
Base.:+(s::SpecialTensor{V},::SpecialTensor{V}) where V = s
@test Manifold(+(a,b)) == ℝ⊕ℝ'
@test Manifold(interop(+,a,b)) == ℝ⊕ℝ'
@test Manifold(+(a,a)) == ℝ
@test Manifold(interop(+,a,a)) == ℝ

## tensor pseudoscalar (trivial test)
op(a::TensorAlgebra{V},b::UniformScaling) where V = op(a,V(b))
op(a::UniformScaling,b::TensorAlgebra{V}) where V = op(V(a),b)
(W::Signature)(s::UniformScaling) where V = SpecialTensor{W}()
@test Manifold(op(a,LinearAlgebra.I)) == ℝ
@test Manifold(op(LinearAlgebra.I,a)) == ℝ
@test Manifold(+(a,LinearAlgebra.I)) == ℝ
@test Manifold(+(LinearAlgebra.I,a)) == ℝ

## tensor evaluation (trivial test)
(a::SpecialTensor{V})(b::SpecialTensor{V}) where V = a
(a::SpecialTensor{W})(b::SpecialTensor{V}) where {V,W} = interform(a,b)
@test Manifold(b(a)) == ℝ⊕ℝ'
@test Manifold(interform(b,a)) == ℝ⊕ℝ'
@test Manifold(a(a)) == ℝ
@test Manifold(interform(a,a)) == ℝ

## algebraic tests
@test !I == |(I)
@test (2)⁻¹ == 1/2
@test (im)ǂ == -im
@test (sqrt(2))ˣ == sqrt(2)
@test (sqrt(2))₊ == sqrt(2)
@test (sqrt(2))₋ == 0
