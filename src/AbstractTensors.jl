
#   This file is part of AbstractTensors.jl. It is licensed under the GPL license
#   Grassmann Copyright (C) 2019 Michael Reed

module AbstractTensors

# universal root Tensor type

abstract type TensorAlgebra{V} <: Number end

# V, VectorSpace produced by DirectSum

import DirectSum
import DirectSum: vectorspace, value, dual
import LinearAlgebra: norm, dot, cross, UniformScaling, I
import AbstractLattices: ∨

# parameters accessible from anywhere

Base.@pure vectorspace(::T) where T<:TensorAlgebra{V} where V = V
Base.@pure Base.ndims(::T) where T<:TensorAlgebra{V} where V = ndims(V)

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

export TensorAlgebra, interop, interform, scalar, involute, norm, unit, even, odd
export ⊖, ⊗, ⊛, ⊙, ⊠, ⨼, ⨽, ⋆, ∗, ⁻¹, ǂ, ₊, ₋, ˣ

# some shared presets

for op ∈ (:(Base.:+),:(Base.:-),:(Base.:*),:⊗,:⊛,:∗,:⨼,:⨽,:dot,:cross,:contraction,:(Base.:|),:(Base.:(==)),:(Base.:<),:(Base.:>),:(Base.:<<),:(Base.:>>),:(Base.:>>>),:(Base.div),:(Base.rem),:(Base.:&))
    @eval begin
        @inline $op(a::A,b::B) where {A<:TensorAlgebra,B<:TensorAlgebra} = interop($op,a,b)
        @inline $op(a::A,b::UniformScaling) where A<:TensorAlgebra{V} where V = $op(a,V(b))
        @inline $op(a::UniformScaling,b::B) where B<:TensorAlgebra{V} where V = $op(V(a),b)
    end
end

const ⊖ = *
@inline ⋆(t::UniformScaling{T}) where T = T<:Bool ? (t.λ ? 1 : -1) : t.λ
@inline ⊛(a::A,b::B) where {A<:TensorAlgebra{V},B<:TensorAlgebra{V}} where V = scalar(contraction(a,b))
@inline ⨽(a::A,b::B) where {A<:TensorAlgebra{V},B<:TensorAlgebra{V}} where V = contraction(a,b)
@inline ⨼(a::A,b::B) where {A<:TensorAlgebra{V},B<:TensorAlgebra{V}} where V = contraction(b,a)
@inline Base.:<(a::A,b::B) where {A<:TensorAlgebra{V},B<:TensorAlgebra{V}} where V = contraction(b,a)
@inline Base.:>(a::A,b::B) where {A<:TensorAlgebra{V},B<:TensorAlgebra{V}} where V = contraction(a,b)
@inline Base.:|(a::A,b::B) where {A<:TensorAlgebra{V},B<:TensorAlgebra{V}} where V = contraction(a,b)
@inline Base.:\(a::A,b::B) where {A<:TensorAlgebra,B<:TensorAlgebra} = inv(a)*b

for op ∈ (:(Base.:+),:(Base.:*))
    @eval $op(t::T) where T<:TensorAlgebra = t
end
for op ∈ (:|,:!), T ∈ (TensorAlgebra,UniformScaling)
    @eval Base.$op(t::$T) = ⋆(t)
end
for op ∈ (:⊙,:⊠,:¬)
    @eval function $op end
end
for op ∈ (:scalar,:involute,:even)
    @eval $op(t::T) where T<:Real = t
end
odd(::T) where T<:Real = 0

@inline Base.exp(t::T) where T<:TensorAlgebra = 1+expm1(t)
@inline Base.log(b,t::T) where T<:TensorAlgebra = log(t)/log(b)
@inline Base.:^(b::S,t::T) where {S<:Number,T<:TensorAlgebra} = exp(t*log(b))
@inline Base.:^(a::A,b::UniformScaling) where A<:TensorAlgebra{V} where V = ^(a,V(b))
@inline Base.:^(a::UniformScaling,b::B) where B<:TensorAlgebra{V} where V = ^(V(a),b)

for base ∈ (2,10)
    fl,fe = (Symbol(:log,base),Symbol(:exp,base))
    @eval Base.$fl(t::T) where T<:TensorAlgebra = $fl(ℯ)*log(t)
    @eval Base.$fe(t::T) where T<:TensorAlgebra = exp(log($base)*t)
end

@inline Base.cos(t::T) where T<:TensorAlgebra{V} where V = cosh(V(I)*t)
@inline Base.sin(t::T) where T<:TensorAlgebra{V} where V = (i=V(I);sinh(i*t)/i)
@inline Base.tan(t::T) where T<:TensorAlgebra = sin(t)/cos(t)
@inline Base.cot(t::T) where T<:TensorAlgebra = cos(t)/sin(t)
@inline Base.sec(t::T) where T<:TensorAlgebra = inv(cos(t))
@inline Base.csc(t::T) where T<:TensorAlgebra = inv(sin(t))
@inline Base.asec(t::T) where T<:TensorAlgebra = acos(inv(t))
@inline Base.acsc(t::T) where T<:TensorAlgebra = asin(inv(t))
@inline Base.sech(t::T) where T<:TensorAlgebra = inv(cosh(t))
@inline Base.csch(t::T) where T<:TensorAlgebra = inv(sinh(t))
@inline Base.asech(t::T) where T<:TensorAlgebra = acosh(inv(t))
@inline Base.acsch(t::T) where T<:TensorAlgebra = asinh(inv(t))
@inline Base.tanh(t::T) where T<:TensorAlgebra = sinh(t)/cosh(t)
@inline Base.coth(t::T) where T<:TensorAlgebra = cosh(t)/sinh(t)
@inline Base.asinh(t::T) where T<:TensorAlgebra = log(t+sqrt(1+t^2))
@inline Base.acosh(t::T) where T<:TensorAlgebra = log(t+sqrt(t^2-1))
@inline Base.atanh(t::T) where T<:TensorAlgebra = (log(1+t)-log(1-t))/2
@inline Base.acoth(t::T) where T<:TensorAlgebra = (log(t+1)-log(t-1))/2
Base.asin(t::T) where T<:TensorAlgebra{V} where V = (i=V(I);-i*log(i*t+sqrt(1-t^2)))
Base.acos(t::T) where T<:TensorAlgebra{V} where V = (i=V(I);-i*log(t+i*sqrt(1-t^2)))
Base.atan(t::T) where T<:TensorAlgebra{V} where V = (i=V(I);(-i/2)*(log(1+i*t)-log(1-i*t)))
Base.acot(t::T) where T<:TensorAlgebra{V} where V = (i=V(I);(-i/2)*(log(t-i)-log(t+i)))
Base.sinc(t::T) where T<:TensorAlgebra{V} where V =iszero(t) ? one(V) : (x=(1π)*t;sin(x)/x)
Base.cosc(t::T) where T<:TensorAlgebra{V} where V = iszero(t) ? zero(V) : (x=(1π)*t; cos(x)/t - sin(x)/(x*t))

# absolute value norm

@inline Base.abs(t::T) where T<:TensorAlgebra{V} where V = sqrt(abs2(t))
@inline Base.abs2(t::T) where T<:TensorAlgebra = t∗t
@inline norm(t::T) where T<:TensorAlgebra = DirectSum.norm(value(t))
@inline unit(t::T) where T<:TensorAlgebra = t/abs(t)
@inline Base.iszero(t::T) where T<:TensorAlgebra = norm(t) ≈ 0
@inline Base.isone(t::T) where T<:TensorAlgebra = norm(t) ≈ value(scalar(t)) ≈ 1

# identity elements

for id ∈ (:zero,:one)
    @eval begin
        @inline Base.$id(::T) where T<:TensorAlgebra{V} where V = zero(V)
        @inline Base.$id(::Type{T}) where T<:TensorAlgebra{V} where V = zero(V)
    end
end

# postfix operators

struct Postfix{Op} end
@inline Base.:*(t,op::P) where P<:Postfix = op(t)
for op ∈ (:⁻¹,:ǂ,:₊,:₋,:ˣ)
    @eval const $op = $(Postfix{op}())
end
@inline (::Postfix{:⁻¹})(t) = inv(t)
@inline (::Postfix{:ǂ})(t) = conj(t)
@inline (::Postfix{:₊})(t) = even(t)
@inline (::Postfix{:₋})(t) = odd(t)
@inline (::Postfix{:ˣ})(t) = involute(t)

end # module
