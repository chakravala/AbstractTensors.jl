
#   This file is part of AbstractTensors.jl. It is licensed under the GPL license
#   Grassmann Copyright (C) 2019 Michael Reed

module AbstractTensors

# universal root Tensor type

abstract type TensorAlgebra{V} end

# V, VectorSpace produced by DirectSum

import DirectSum: vectorspace, value
import LinearAlgebra: dot, cross, UniformScaling

# parameters accessible from anywhere

Base.@pure vectorspace(::T) where T<:TensorAlgebra{V} where V = V

# universal vector space interopability

@inline interop(op::Function,a::A,b::B) where {A<:TensorAlgebra{V},B<:TensorAlgebra{V}} where V = op(a,b)

# ^^ identity ^^ | vv union vv #

@inline function interop(op::Function,a::A,b::B) where {A<:TensorAlgebra{V},B<:TensorAlgebra{W}} where {V,W}
    VW = V ∪ W
    return op(VW(a),VW(b))
end

# abstract tensor form evaluation

@inline interform(a::A,b::B) where {A<:TensorAlgebra{V},B<:TensorAlgebra{V}} where V = a(b)
@inline function interform(a::A,b::B) where {A<:TensorAlgebra{V},B<:TensorAlgebra{W}} where {V,W}
    VW = V ∪ W
    return VW(a)(VW(b))
end

# extended compatibility interface

export interop, TensorAlgebra, interform, ⊗, ⊛, ⊙, ⊠, ⨼, ⨽, ⋆

# some shared presets

for op ∈ (:(Base.:+),:(Base.:-),:(Base.:*),:⊗,:⊛,:⊙,:⊠,:⨼,:⨽,:dot,:cross,:(Base.:|),:(Base.:(==)),:(Base.:<),:(Base.:>),:(Base.:<<),:(Base.:>>),:(Base.:>>>),:(Base.div),:(Base.rem),:(Base.:&),:(Base.:^))
    @eval begin
        @inline $op(a::A,b::B) where {A<:TensorAlgebra,B<:TensorAlgebra} = interop($op,a,b)
        @inline $op(a::A,b::UniformScaling) where A<:TensorAlgebra{V} where V = $op(a,V(b))
        @inline $op(a::UniformScaling,b::B) where B<:TensorAlgebra{V} where V = $op(V(a),b)
    end
end

for op ∈ (:(Base.:+),:(Base.:*))
    @eval $op(t::T) where T<:TensorAlgebra = t
end

# postfix ? (:⁻¹,:ǂ,:₊,:₋,:ˣ)

⋆(t::UniformScaling{T}) where T = T<:Bool ? (t.λ ? 1 : -1) : t.λ
⨽(a::TensorAlgebra{V},b::TensorAlgebra{V}) where V = dot(a,b)
Base.:|(a::TensorAlgebra{V},b::TensorAlgebra{V}) where V = dot(a,b)
for op ∈ (:|,:!), T ∈ (TensorAlgebra,UniformScaling)
    @eval Base.$op(t::$T) = ⋆(t)
end

# absolute value norm

@inline Base.abs(t::T) where T<:TensorAlgebra = sqrt(abs(value(dot(t,t))))
@inline Base.abs2(t::T) where T<:TensorAlgebra = value(dot(t,t))
@inline unit(t::T) where T<:TensorAlgebra = t/abs(t)

end # module
