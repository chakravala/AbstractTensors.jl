
#   This file is part of AbstractTensors.jl. It is licensed under the AGPL license
#   Grassmann Copyright (C) 2019 Michael Reed

module AbstractTensors

# universal root Tensor type

"""
    TensorAlgebra{V} <: Number

Universal root tensor type with `DirectSum.Manifold` instance parameter.
"""
abstract type TensorAlgebra{V} <: Number end

## Manifold{n}

"""
    Manifold{n}

Basis parametrization locally homeomorphic to `ℝ^n` product topology.
"""
abstract type Manifold{n} <: TensorAlgebra{n} end

# V, VectorSpace produced by DirectSum

value(x::T) where T<:Number = x
value(x::T) where T<:Manifold = 1
const vectorspace = Manifold
import LinearAlgebra
import LinearAlgebra: dot, cross, UniformScaling, I
import AbstractLattices: ∨

# parameters accessible from anywhere

Base.@pure Manifold(::T) where T<:TensorAlgebra{V} where V = V
Base.@pure Manifold(V::T) where T<:Manifold = V
Base.@pure Base.ndims(::T) where T<:TensorAlgebra{V} where V = ndims(V)
Base.@pure Base.ndims(::T) where T<:Manifold{n} where n = n
Base.@pure LinearAlgebra.rank(::T) where T<:Manifold{n} where n = n
function valuetype end
function isapprox(a::S,b::T) where {S<:TensorAlgebra,T<:TensorAlgebra}
    rtol = Base.rtoldefault(valuetype(a), valuetype(b), 0)
    LinearAlgebra.norm(Base.:-(a,b))≤rtol*max(LinearAlgebra.norm(a),LinearAlgebra.norm(b))
end

# universal vector space interopability

@inline interop(op::Function,a::A,b::B) where {A<:TensorAlgebra{V},B<:TensorAlgebra{V}} where V = op(a,b)

# ^^ identity ^^ | vv union vv #

for T ∈ (TensorAlgebra,Manifold)
    @eval begin
        @inline function interop(op::Function,a::A,b::B) where {A<:$T,B<:$T}
            M = Manifold(a) ∪ Manifold(b)
            return op(M(a),M(b))
        end
        @inline function interform(a::A,b::B) where {A<:$T,B<:$T}
            M = Manifold(a) ∪ Manifold(b)
            return M(a)(M(b))
        end
    end
end

# abstract tensor form evaluation

@inline interform(a::A,b::B) where {A<:TensorAlgebra{V},B<:TensorAlgebra{V}} where V = a(b)

# extended compatibility interface

export TensorAlgebra, interop, interform, scalar, involute, unit, even, odd
export ⊖, ⊗, ⊛, ⊙, ⊠, ⨼, ⨽, ⋆, ∗, ⁻¹, ǂ, ₊, ₋, ˣ

# some shared presets

for op ∈ (:(Base.:+),:(Base.:-),:(Base.:*),:⊗,:⊛,:∗,:⨼,:⨽,:dot,:cross,:contraction,:(Base.:|),:(Base.:(==)),:(Base.:<),:(Base.:>),:(Base.:<<),:(Base.:>>),:(Base.:>>>),:(Base.div),:(Base.rem),:(Base.:&))
    @eval begin
        @inline $op(a::A,b::B) where {A<:TensorAlgebra,B<:TensorAlgebra} = interop($op,a,b)
        @inline $op(a::A,b::UniformScaling) where A<:TensorAlgebra = $op(a,Manifold(a)(b))
        @inline $op(a::UniformScaling,b::B) where B<:TensorAlgebra = $op(Manifold(b)(a),b)
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
@inline Base.:/(a::A,b::B) where {A<:TensorAlgebra,B<:TensorAlgebra} = a*Base.inv(b)
@inline Base.:/(a::UniformScaling,b::B) where B<:TensorAlgebra = Manifold(b)(a)*Base.inv(b)
@inline Base.:/(a::A,b::UniformScaling) where A<:TensorAlgebra = a*Base.inv(Manifold(a)(b))
@inline Base.:\(a::A,b::B) where {A<:TensorAlgebra,B<:TensorAlgebra} = Base.inv(a)*b
@inline Base.:\(a::UniformScaling,b::B) where B<:TensorAlgebra = Base.inv(Manifold(b)(a))*b
@inline Base.:\(a::A,b::UniformScaling) where A<:TensorAlgebra = Base.inv(a)*Manifold(a)(b)

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

@inline Base.exp(t::T) where T<:TensorAlgebra = 1+Base.expm1(t)
@inline Base.log(b,t::T) where T<:TensorAlgebra = Base.log(t)/Base.log(b)
@inline Base.:^(b::S,t::T) where {S<:Number,T<:TensorAlgebra} = Base.exp(t*Base.log(b))
@inline Base.:^(a::A,b::UniformScaling) where A<:TensorAlgebra = Base.:^(a,Manifold(a)(b))
@inline Base.:^(a::UniformScaling,b::B) where B<:TensorAlgebra = Base.:^(Manifold(b)(a),b)

for base ∈ (2,10)
    fl,fe = (Symbol(:log,base),Symbol(:exp,base))
    @eval Base.$fl(t::T) where T<:TensorAlgebra = Base.$fl(ℯ)*Base.log(t)
    @eval Base.$fe(t::T) where T<:TensorAlgebra = Base.exp(Base.log($base)*t)
end

@inline Base.cos(t::T) where T<:TensorAlgebra = Base.cosh(Manifold(t)(I)*t)
@inline Base.sin(t::T) where T<:TensorAlgebra = (i=Manifold(t)(I);Base.:/(Base.sinh(i*t),i))
@inline Base.tan(t::T) where T<:TensorAlgebra = Base.:/(Base.sin(t),Base.cos(t))
@inline Base.cot(t::T) where T<:TensorAlgebra = Base.:/(Base.cos(t),Base.sin(t))
@inline Base.sec(t::T) where T<:TensorAlgebra = Base.inv(Base.cos(t))
@inline Base.csc(t::T) where T<:TensorAlgebra = Base.inv(Base.sin(t))
@inline Base.asec(t::T) where T<:TensorAlgebra = Base.acos(Base.inv(t))
@inline Base.acsc(t::T) where T<:TensorAlgebra = Base.asin(Base.inv(t))
@inline Base.sech(t::T) where T<:TensorAlgebra = Base.inv(Base.cosh(t))
@inline Base.csch(t::T) where T<:TensorAlgebra = Base.inv(Base.sinh(t))
@inline Base.asech(t::T) where T<:TensorAlgebra = Base.acosh(Base.inv(t))
@inline Base.acsch(t::T) where T<:TensorAlgebra = Base.asinh(Base.inv(t))
@inline Base.tanh(t::T) where T<:TensorAlgebra = Base.:/(Base.sinh(t),Base.cosh(t))
@inline Base.coth(t::T) where T<:TensorAlgebra = Base.:/(Base.cosh(t),Base.sinh(t))
@inline Base.asinh(t::T) where T<:TensorAlgebra = Base.log(t+Base.sqrt(1+Base.:^(t,2)))
@inline Base.acosh(t::T) where T<:TensorAlgebra = Base.log(t+Base.sqrt(Base.:-(Base.:^(t,2),1)))
@inline Base.atanh(t::T) where T<:TensorAlgebra = Base.:/(Base.:-(Base.log(1+t),Base.log(Base.:-(1,t))),2)
@inline Base.acoth(t::T) where T<:TensorAlgebra = Base.:/(Base.:-(Base.log(t+1),Base.log(Base.:-(t,1))),2)
Base.asin(t::T) where T<:TensorAlgebra = (i=Manifold(t)(I);Base.:-(i)*Base.log(i*t+Base.sqrt(Base.:-(1,Base.:^(t,2)))))
Base.acos(t::T) where T<:TensorAlgebra = (i=Manifold(t)(I);Base.:-(i)*Base.log(t+i*Base.sqrt(Base.:-(1,Base.:^(t,2)))))
Base.atan(t::T) where T<:TensorAlgebra = (i=Manifold(t)(I);Base.:/(Base.:-(i),2)*Base.:-(Base.log(1+i*t),Base.log(Base.:-(1,i*t))))
Base.acot(t::T) where T<:TensorAlgebra = (i=Manifold(t)(I);Base.:/(Base.:-(i),2)*Base.:-(Base.log(Base.:-(t,i)),Base.log(t+i)))
Base.sinc(t::T) where T<:TensorAlgebra = iszero(t) ? one(Manifold(t)) : (x=(1π)*t;Base.:/(Base.sin(x),x))
Base.cosc(t::T) where T<:TensorAlgebra = iszero(t) ? zero(Manifold(t)) : (x=(1π)*t; Base.:-(Base.:/(Base.cos(x),t), Base.:/(sin(x),(x*t))))

# absolute value norm

@inline Base.abs(t::T) where T<:TensorAlgebra = Base.sqrt(Base.abs2(t))
@inline Base.abs2(t::T) where T<:TensorAlgebra = t∗t
@inline norm(z) = LinearAlgebra.norm(z)
@inline LinearAlgebra.norm(t::T) where T<:TensorAlgebra = norm(value(t))
@inline unit(t::T) where T<:TensorAlgebra = Base.:/(t,Base.abs(t))
@inline Base.iszero(t::T) where T<:TensorAlgebra = LinearAlgebra.norm(t) ≈ 0
@inline Base.isone(t::T) where T<:TensorAlgebra = LinearAlgebra.norm(t)≈value(scalar(t))≈1

# identity elements

for id ∈ (:zero,:one)
    @eval begin
        @inline Base.$id(t::T) where T<:TensorAlgebra = zero(Manifold(t))
        @inline Base.$id(::Type{T}) where T<:TensorAlgebra{V} where V = zero(V)
        @inline Base.$id(t::Type{T}) where T<:Manifold = zero(t())
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

# dispatch

signbit(x::Symbol) = false
signbit(x::Expr) = x.head == :call && x.args[1] == :-
-(x) = Base.:-(x)
-(x::Symbol) = :(-$x)

for op ∈ (:conj,:inv,:sqrt,:abs,:exp,:expm1,:log,:log1p,:sin,:cos,:sinh,:cosh,:signbit)
    @eval @inline $op(z) = Base.$op(z)
end

for op ∈ (:/,:-,:^,:≈)
    @eval @inline $op(a,b) = Base.$op(a,b)
end

for (OP,op) ∈ ((:∏,:*),(:∑,:+))
    @eval begin
        @inline $OP(x...) = Base.$op(x...)
        @inline $OP(x::AbstractVector{T}) where T<:Any = $op(x...)
    end
end

const PROD,SUM,SUB = ∏,∑,-

@inline norm(z::Expr) = abs(z)
@inline norm(z::Symbol) = z

for T ∈ (Expr,Symbol)
    @eval begin
        ≈(a::$T,b::$T) = a == b
        ≈(a::$T,b) = false
        ≈(a,b::$T) = false
    end
end

end # module
