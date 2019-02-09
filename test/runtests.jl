using AbstractTensors
using Test, DirectSum
import DirectSum: VectorSpace

# example data
struct SpecialTensor{V} <: TensorAlgebra{V} end

## tensor operation (trivial test)
op(s::SpecialTensor{V},::SpecialTensor{V}) where V = s
op(a::TensorAlgebra{V},b::TensorAlgebra{W}) where {V,W} = interop(op,a,b)
(W::VectorSpace)(s::SpecialTensor{V}) where V = SpecialTensor{W}()
@test vectorspace(op(SpecialTensor{ℝ}(),SpecialTensor{ℝ'}())) == ℝ⊕ℝ'
@test vectorspace(interop(op,SpecialTensor{ℝ}(),SpecialTensor{ℝ'}())) == ℝ⊕ℝ'

## tensor evaluation (trivial test)
(a::SpecialTensor{V})(b::SpecialTensor{V}) where V = a
(a::SpecialTensor{W})(b::SpecialTensor{V}) where {V,W} = interform(a,b)
@test vectorspace(SpecialTensor{ℝ'}()(SpecialTensor{ℝ}())) == ℝ⊕ℝ'
@test vectorspace(interform(SpecialTensor{ℝ'}(),SpecialTensor{ℝ}())) == ℝ⊕ℝ'
