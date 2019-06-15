using AbstractTensors
using Test, DirectSum, LinearAlgebra
import DirectSum: VectorSpace

# example data
struct SpecialTensor{V} <: TensorAlgebra{V} end

## tensor operation (trivial test)
op(s::SpecialTensor{V},::SpecialTensor{V}) where V = s
op(a::TensorAlgebra{V},b::TensorAlgebra{W}) where {V,W} = interop(op,a,b)
(W::Signature)(s::SpecialTensor{V}) where V = SpecialTensor{W}()
@test vectorspace(op(SpecialTensor{ℝ}(),SpecialTensor{ℝ'}())) == ℝ⊕ℝ'
@test vectorspace(interop(op,SpecialTensor{ℝ}(),SpecialTensor{ℝ'}())) == ℝ⊕ℝ'
Base.:+(s::SpecialTensor{V},::SpecialTensor{V}) where V = s
@test vectorspace(+(SpecialTensor{ℝ}(),SpecialTensor{ℝ'}())) == ℝ⊕ℝ'
@test vectorspace(interop(+,SpecialTensor{ℝ}(),SpecialTensor{ℝ'}())) == ℝ⊕ℝ'


## tensor pseudoscalar (trivial test)
op(a::TensorAlgebra{V},b::UniformScaling) where V = op(a,V(b))
op(a::UniformScaling,b::TensorAlgebra{V}) where V = op(V(a),b)
(W::Signature)(s::UniformScaling) where V = SpecialTensor{W}()
@test vectorspace(op(SpecialTensor{ℝ}(),LinearAlgebra.I)) == ℝ
@test vectorspace(op(LinearAlgebra.I,SpecialTensor{ℝ}())) == ℝ
@test vectorspace(+(SpecialTensor{ℝ}(),LinearAlgebra.I)) == ℝ
@test vectorspace(+(LinearAlgebra.I,SpecialTensor{ℝ}())) == ℝ

## tensor evaluation (trivial test)
(a::SpecialTensor{V})(b::SpecialTensor{V}) where V = a
(a::SpecialTensor{W})(b::SpecialTensor{V}) where {V,W} = interform(a,b)
@test vectorspace(SpecialTensor{ℝ'}()(SpecialTensor{ℝ}())) == ℝ⊕ℝ'
@test vectorspace(interform(SpecialTensor{ℝ'}(),SpecialTensor{ℝ}())) == ℝ⊕ℝ'

## algebraic tests
@test !I == |(I)
@test (2)⁻¹ == 1/2
@test (im)ǂ == -im
@test (1)ˣ == 1
@test (1)₊ == 1
@test (1)₋ == 0
