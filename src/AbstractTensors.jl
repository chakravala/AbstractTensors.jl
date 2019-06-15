
#   This file is part of AbstractTensors.jl. It is licensed under the GPL license
#   Grassmann Copyright (C) 2019 Michael Reed

module AbstractTensors

# universal root Tensor type

abstract type TensorAlgebra{V} end

# V, VectorSpace produced by DirectSum

import DirectSum: vectorspace, value, dual
import LinearAlgebra: dot, cross, norm, UniformScaling

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

export TensorAlgebra, interop, interform, scalar, involute, norm, norm2, unit, even, odd
export ⊗, ⊛, ⊙, ⊠, ⨼, ⨽, ⋆, ⁻¹, ǂ, ₊, ₋, ˣ

# some shared presets

for op ∈ (:(Base.:+),:(Base.:-),:(Base.:*),:⊗,:⊛,:⨼,:⨽,:dot,:cross,:(Base.:|),:(Base.:(==)),:(Base.:<),:(Base.:>),:(Base.:<<),:(Base.:>>),:(Base.:>>>),:(Base.div),:(Base.rem),:(Base.:&),:(Base.:^))
    @eval begin
        @inline $op(a::A,b::B) where {A<:TensorAlgebra,B<:TensorAlgebra} = interop($op,a,b)
        @inline $op(a::A,b::UniformScaling) where A<:TensorAlgebra{V} where V = $op(a,V(b))
        @inline $op(a::UniformScaling,b::B) where B<:TensorAlgebra{V} where V = $op(V(a),b)
    end
end

for op ∈ (:(Base.:+),:(Base.:*))
    @eval $op(t::T) where T<:TensorAlgebra = t
end

⋆(t::UniformScaling{T}) where T = T<:Bool ? (t.λ ? 1 : -1) : t.λ
⨽(a::TensorAlgebra{V},b::TensorAlgebra{V}) where V = dot(a,b)
Base.:|(a::TensorAlgebra{V},b::TensorAlgebra{V}) where V = dot(a,b)
for op ∈ (:|,:!), T ∈ (TensorAlgebra,UniformScaling)
    @eval Base.$op(t::$T) = ⋆(t)
end
for op ∈ (:⊙,:⊠)
    @eval function $op end
end
for op ∈ (:scalar,:involute,:even)
    @eval $op(x::T) where T<:Number = x
end
odd(x::T) where T<:Number = 0

# absolute value norm

@inline Base.abs(t::T) where T<:TensorAlgebra = sqrt(abs(value(dot(t,t))))
@inline Base.abs2(t::T) where T<:TensorAlgebra = value(dot(t,t))
@inline norm(t::T) where T<:TensorAlgebra = sqrt(abs(value(dot(t,~t))))
@inline norm2(t::T) where T<:TensorAlgebra = t*~t
@inline unit(t::T) where T<:TensorAlgebra = t/abs(t)

# postfix operators

struct Postfix{Op} end
@inline Base.:*(t,op::Postfix) = op(t)
for op ∈ (:⁻¹,:ǂ,:₊,:₋,:ˣ)
    @eval const $op = $(Postfix{op}())
end
@inline (::Postfix{:⁻¹})(t) = inv(t)
@inline (::Postfix{:ǂ})(t) = conj(t)
@inline (::Postfix{:₊})(t) = even(t)
@inline (::Postfix{:₋})(t) = odd(t)
@inline (::Postfix{:ˣ})(t) = involute(t)

end # module
