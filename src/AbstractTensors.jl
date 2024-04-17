
#   This file is part of AbstractTensors.jl
#   It is licensed under the MIT license
#   AbstractTensors Copyright (C) 2019 Michael Reed
#       _           _                         _
#      | |         | |                       | |
#   ___| |__   __ _| | ___ __ __ ___   ____ _| | __ _
#  / __| '_ \ / _` | |/ / '__/ _` \ \ / / _` | |/ _` |
# | (__| | | | (_| |   <| | | (_| |\ V / (_| | | (_| |
#  \___|_| |_|\__,_|_|\_\_|  \__,_| \_/ \__,_|_|\__,_|
#
#   https://github.com/chakravala
#   https://crucialflow.com
#  _______  __            __                       __
# |   _   ||  |--..-----.|  |_ .----..---.-..----.|  |_
# |       ||  _  ||__ --||   _||   _||  _  ||  __||   _|
# |___|___||_____||_____||____||__|  |___._||____||____|
#   _______
#  |_     _|.-----..-----..-----..-----..----..-----.
#    |   |  |  -__||     ||__ --||  _  ||   _||__ --|
#    |___|  |_____||__|__||_____||_____||__|  |_____|

module AbstractTensors

# universal root Tensor type, Manifold

"""
    TensorAlgebra{V} <: Number

Universal root tensor type with `Manifold` instance parameter `V`.
"""
abstract type TensorAlgebra{V} <: Number end
TensorAlgebra{V}(t::TensorAlgebra{V}) where V = t
TensorAlgebra{V}(t::TensorAlgebra{W}) where {V,W} = (V∪W)(t)
Base.@pure istensor(t::T) where T<:TensorAlgebra = true
Base.@pure istensor(t) = false

"""
    Manifold{V} <: TensorAlgebra{V}

Basis parametrization locally homeomorphic to `ℝ^n` product topology.
"""
abstract type Manifold{V} <: TensorAlgebra{V} end
Base.@pure ismanifold(t::T) where T<:Manifold = true
Base.@pure ismanifold(t) = false

"""
    TensorGraded{V,G} <: Manifold{V} <: TensorAlgebra

Graded elements of a `TensorAlgebra` in a `Manifold` topology.
"""
abstract type TensorGraded{V,G} <: Manifold{V} end
const TAG = (:TensorAlgebra,:TensorGraded)
Base.@pure isgraded(t::T) where T<:TensorGraded = true
Base.@pure isgraded(t) = false

"""
    Scalar{V} <: TensorGraded{V,0}

Graded scalar elements of a `TensorAlgebra` in a `Manifold` topology.
"""
const Scalar{V} = TensorGraded{V,0}

"""
    GradedVector{V} <: TensorGraded{V,1}

Graded vector elements of a `TensorAlgebra` in a `Manifold` topology.
"""
const GradedVector{V} = TensorGraded{V,1}

"""
    Bivector{V} <: TensorGraded{V,2}

Graded bivector elements of a `TensorAlgebra` in a `Manifold` topology.
"""
const Bivector{V} = TensorGraded{V,2}

"""
    Trivector{V} <: TensorGraded{V,3}

Graded trivector elements of a `TensorAlgebra` in a `Manifold` topology.
"""
const Trivector{V} = TensorGraded{V,3}

"""
    TensorTerm{V,G} <: TensorGraded{V,G}

Terms of a `TensorAlgebra` having a single coefficient.
"""
abstract type TensorTerm{V,G} <: TensorGraded{V,G} end
Base.@pure isterm(t::T) where T<:TensorTerm = true
Base.@pure isterm(t) = false
Base.isfinite(b::T) where T<:TensorTerm = isfinite(value(b))

"""
    TensorMixed{V} <: TensorAlgebra{V}

Elements of `TensorAlgebra` having non-homogenous grade.
"""
abstract type TensorMixed{V} <: TensorAlgebra{V} end
Base.@pure ismixed(t::T) where T<:TensorMixed = true
Base.@pure ismixed(t) = false

# parameters accessible from anywhere

for T ∈ (:T,:(Type{T}))
    @eval begin
        Base.@pure Manifold(::$T) where T<:TensorAlgebra{V} where V = V
        Base.@pure Manifold(::$T) where T<:TensorGraded{V} where V = V
        Base.@pure Manifold(V::$T) where T<:Manifold = V
        Base.@pure Base.parent(V::$T) where T<:TensorAlgebra = Manifold(V)
    end
end

import LinearAlgebra
import LinearAlgebra: UniformScaling, I, rank

"""
    rank(::Manifold{n})

Dimensionality `n` of the `Manifold` subspace representation.
"""
Base.@pure LinearAlgebra.rank(::T) where T<:TensorGraded{V,G} where {V,G} = G
Base.@pure LinearAlgebra.rank(::Type{M}) where M<:Manifold = mdims(M)

"""
    mdims(t::TensorAlgebra{V})

Dimensionality of the pseudoscalar `V` of that `TensorAlgebra`.
"""
Base.@pure mdims(M::T) where T<:TensorAlgebra = mdims(Manifold(M))
Base.@pure mdims(M::Type{T}) where T<:TensorAlgebra = mdims(Manifold(M))
Base.@pure mdims(M::Int) = M

for (part,G) ∈ ((:scalar,0),(:vector,1),(:bivector,2),(:trivector,3))
    ispart = Symbol(:is,part)
    str = """
    $part(::TensorAlgebra)

Return the $part (rank $G) part of any `TensorAlgebra` element.
    """
    @eval begin
        @doc $str $part
        @inline $part(t::T) where T<:TensorGraded{V} where V = zero(V)
        @inline $part(t::T) where T<:TensorGraded{V,$G} where V = t
        @inline $ispart(t::T) where T<:TensorGraded = rank(t) == $G || iszero(t)
    end
end

"""
    pseudoscalar(::TensorAlgebra), volume(::TensorAlgebra)

Return the pseudoscalar (full rank) part of any `TensorAlgebra` element.
"""
@inline pseudoscalar(t::T) where T<:Manifold = t
const volume = pseudoscalar
@inline volume(t::T) where T<:TensorGraded{V,G} where {V,G} = G == mdims(V) ? t : zero(V)
@inline isvolume(t::T) where T<:TensorGraded = rank(t) == mdims(t) || iszero(t)

"""
    values(::TensorAlgebra)

Returns the internal `Values` representation of a `TensorAlgebra` element.
"""
values(t::T) where T<:Number = t
values(t::T) where T<:AbstractArray = t
const value = values

"""
    valuetype(t::TensorAlgebra)

Returns type of a `TensorAlgebra` element value's internal representation.
"""
Base.@pure valuetype(::T) where T<:Number = T
#Base.@pure valuetype(::T) where T<:TensorAlgebra{V,𝕂} where V where 𝕂 = 𝕂

Base.real(::Type{T}) where T<:TensorAlgebra = real(valuetype(T))
Base.rtoldefault(::Type{T}) where T<:TensorAlgebra = Base.rtoldefault(valuetype(T))
function Base.isapprox(a::S,b::T;atol::Real=0,rtol::Real=Base.rtoldefault(a,b,atol),nans::Bool=false,norm::Function=LinearAlgebra.norm) where {S<:TensorAlgebra,T<:TensorAlgebra}
    x,y = norm(a),norm(b)
    (isfinite(x) && isfinite(y) && norm(Base.:-(a,b))≤max(atol,rtol*max(x,y))) || (nans && isnan(x) && isnan(y))
end
function Base.isapprox(a::S,b::T;atol::Real=0,rtol::Real=Base.rtoldefault(a,b,atol),nans::Bool=false,norm::Function=LinearAlgebra.norm) where {S<:TensorGraded,T<:TensorGraded}
    Manifold(a)==Manifold(b) && if rank(a)==rank(b)
        x,y = norm(a),norm(b)
        (isfinite(x) && isfinite(y) && norm(Base.:-(a,b))≤max(atol,rtol*max(x,y))) || (nans && isnan(x) && isnan(y))
    else
        isnull(a) && isnull(b)
    end
end

# universal vector space interopability, abstract tensor form evaluation, contraction

for X ∈ TAG, Y ∈ TAG
    @eval begin
        @inline interop(op::Function,a::A,b::B) where {A<:$X{V},B<:$Y{V}} where V = op(a,b)
        @inline interform(a::A,b::B) where {A<:$X{V},B<:$Y{V}} where V = a(b)
        # ^^ identity ^^ | vv union vv #
        @inline function interop(op::Function,a::A,b::B) where {A<:$X,B<:$Y}
            M = Manifold(a) ∪ Manifold(b)
            return op(M(a),M(b))
        end
        @inline function interform(a::A,b::B) where {A<:$X,B<:$Y}
            M = Manifold(a) ∪ Manifold(b)
            return M(a)(M(b))
        end
        @inline ∗(a::A,b::B) where {A<:$X{V},B<:$Y{V}} where V = (~a)⟑b
        @inline ⊛(a::A,b::B) where {A<:$X{V},B<:$Y{V}} where V = scalar(contraction(a,b))
        @inline ⨼(a::A,b::B) where {A<:$X{V},B<:$Y{V}} where V = contraction(b,a)
        @inline Base.:<<(a::A,b::B) where {A<:$X{V},B<:$Y{V}} where V = contraction(b,~a)
        @inline Base.:>>(a::A,b::B) where {A<:$X{V},B<:$Y{V}} where V = contraction(~a,b)
        @inline Base.:<(a::A,b::B) where {A<:$X{V},B<:$Y{V}} where V = contraction(b,a)
    end
    for op ∈ (:⨽,:(Base.:>),:(Base.:|),:(LinearAlgebra.dot))
        @eval @inline $op(a::A,b::B) where {A<:$X{V},B<:$Y{V}} where V = contraction(a,b)
    end
end

# lattice defaults

import AbstractLattices: ∧, ∨, wedge, vee

@inline ∧() = 1
@inline ∨() = I

# extended compatibility interface

export TensorAlgebra, Manifold, TensorGraded, Distribution, expansion, metric, pseudometric
export Scalar, GradedVector, Bivector, Trivector, contraction, wedgedot, veedot, @pseudo
export istensor, ismanifold, isterm, isgraded, ismixed, rank, mdims, values, sandwich
export scalar, isscalar, vector, isvector, bivector, isbivector, volume, isvolume,hodge
export value, valuetype, interop, interform, involute, unit, unitize, unitnorm, even, odd
export ⟑, ⊘, ⊖, ⊗, ⊛, ⊙, ⊠, ×, ⨼, ⨽, ⋆, ∗, ⁻¹, ǂ, ₊, ₋, ˣ, antiabs, antiabs2, geomabs

# some shared presets

for op ∈ (:(Base.:+),:(Base.:-),:(Base.:*),:sandwich,:⊛,:∗,:⨼,:⨽,:contraction,:expansion,:veedot,:(LinearAlgebra.dot),:(Base.:|),:(Base.:(==)),:(Base.:<),:(Base.:>),:(Base.:<<),:(Base.:>>),:(Base.:>>>),:(Base.div),:(Base.rem),:(Base.:&))
    @eval begin
        @inline $op(a::A,b::UniformScaling) where A<:TensorAlgebra = $op(a,Manifold(a)(b))
        @inline $op(a::UniformScaling,b::B) where B<:TensorAlgebra = $op(Manifold(b)(a),b)
    end
end
#const plus,minus,times = Base.:+,Base.:-,Base.:*
Base.:+(a::A,b::B) where {A<:TensorAlgebra,B<:TensorAlgebra} = plus(a,b)
Base.:-(a::A,b::B) where {A<:TensorAlgebra,B<:TensorAlgebra} = minus(a,b)
Base.:*(a::A,b::B) where {A<:TensorAlgebra,B<:TensorAlgebra} = times(a,b)
LinearAlgebra.dot(a::A,b::B) where {A<:TensorAlgebra,B<:TensorAlgebra} = contraction(a,b)
Base.:(==)(a::A,b::B) where {A<:TensorAlgebra,B<:TensorAlgebra} = equal(a,b)
for op ∈ (:plus,:minus,:wedgedot,:contraction,:equal,:sandwich,:⊛,:∗,:(Base.:|),:(Base.:<),:(Base.:>),:(Base.:<<),:(Base.:>>),:(Base.:>>>),:(Base.div),:(Base.rem),:(Base.:&))
    @eval @inline $op(a::A,b::B) where {A<:TensorAlgebra,B<:TensorAlgebra} = interop($op,a,b)
end

for op ∈ (:(Base.:!),:complementrighthodge)
    for T ∈ (:Real,:Complex)
        @eval @inline $op(t::T) where T<:$T = UniformScaling(t)
    end
end

const complement = !
const complementright = !
const ⋆ = complementrighthodge
const hodge = complementrighthodge
const ⊘ = sandwich
const ⊖,⟑,times,antidot,pseudodot = wedgedot,wedgedot,wedgedot,expansion,expansion
@inline Base.:|(t::T) where T<:TensorAlgebra = hodge(t)
@inline Base.:!(t::UniformScaling{T}) where T = T<:Bool ? (t.λ ? 1 : 0) : t.λ
@inline Base.:/(a::A,b::B) where {A<:TensorAlgebra,B<:TensorAlgebra} = a⟑Base.inv(b)
@inline Base.:/(a::UniformScaling,b::B) where B<:TensorAlgebra = Manifold(b)(a)⟑Base.inv(b)
@inline Base.:/(a::A,b::UniformScaling) where A<:TensorAlgebra = a⟑Base.inv(Manifold(a)(b))
@inline Base.:\(a::A,b::B) where {A<:TensorAlgebra,B<:TensorAlgebra} = Base.inv(a)⟑b
@inline Base.:\(a::UniformScaling,b::B) where B<:TensorAlgebra = Base.inv(Manifold(b)(a))⟑b
@inline Base.:\(a::A,b::UniformScaling) where A<:TensorAlgebra = Base.inv(a)⟑Manifold(a)(b)
@inline ⊗(a::A,b::B) where {A<:TensorAlgebra,B<:Real} = a*b
@inline ⊗(a::A,b::B) where {A<:TensorAlgebra,B<:Complex} = a*b
@inline ⊗(a::A,b::B) where {A<:Real,B<:TensorAlgebra} = a*b
@inline ⊗(a::A,b::B) where {A<:Complex,B<:TensorAlgebra} = a*b
Base.:∘(a::A,b::B) where {A<:TensorAlgebra,B<:TensorAlgebra} = expansion(a,b)

for op ∈ (:(Base.:+),:(Base.:*),:wedgedot)
    @eval $op(t::T) where T<:TensorAlgebra = t
end
for op ∈ (:⊙,:⊠,:¬,:⋆,:clifford,:basis,:complementleft,:complementlefthodge,:complementleftanti,:complementrightanti,:metric,:antimetric,:veedot)
    @eval function $op end
end
for op ∈ (:scalar,:involute,:even)
    @eval $op(t::T) where T<:Real = t
end
odd(::T) where T<:Real = 0
LinearAlgebra.cross(a::A,b::B) where {A<:TensorAlgebra,B<:TensorAlgebra} = hodge(∧(a,b))

@inline Base.exp(t::T) where T<:TensorAlgebra{V} where V = one(V)+Base.expm1(t)
@inline Base.log(b,t::T) where T<:TensorAlgebra = Base.log(t)/Base.log(b)
@inline Base.:^(b::S,t::T) where {S<:Number,T<:TensorAlgebra} = Base.exp(t*Base.log(b))
@inline Base.:^(a::A,b::UniformScaling) where A<:TensorAlgebra = Base.:^(a,Manifold(a)(b))
@inline Base.:^(a::UniformScaling,b::B) where B<:TensorAlgebra = Base.:^(Manifold(b)(a),b)

for base ∈ (2,10)
    fl,fe = (Symbol(:log,base),Symbol(:exp,base))
    @eval Base.$fl(t::T) where T<:TensorAlgebra = Base.$fl(ℯ)*Base.log(t)
    @eval Base.$fe(t::T) where T<:TensorAlgebra = Base.exp(Base.log($base)*t)
end

@inline Base.cos(t::T) where T<:TensorAlgebra{V} where V = Base.cosh(V(I)⟑t)
@inline Base.sin(t::T) where T<:TensorAlgebra{V} where V = (i=V(I);Base.:/(Base.sinh(i⟑t),i))
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
@inline Base.asinh(t::T) where T<:TensorAlgebra{V} where V = Base.log(t+Base.sqrt(one(V)+(t⟑t)))
@inline Base.acosh(t::T) where T<:TensorAlgebra{V} where V = Base.log(t+Base.sqrt(Base.:-(t⟑t,one(V))))
@inline Base.atanh(t::T) where T<:TensorAlgebra{V} where V = Base.:/(Base.:-(Base.log(one(V)+t),Base.log(Base.:-(one(V),t))),2)
@inline Base.acoth(t::T) where T<:TensorAlgebra{V} where V = Base.:/(Base.:-(Base.log(t+one(V)),Base.log(Base.:-(t,one(V)))),2)
Base.asin(t::T) where T<:TensorAlgebra{V} where V = (i=V(I);Base.:-(i)*Base.log(i*t+Base.sqrt(Base.:-(one(V),t⟑t))))
Base.acos(t::T) where T<:TensorAlgebra{V} where V = (i=V(I);Base.:-(i)*Base.log(t+i*Base.sqrt(Base.:-(one(V),t⟑t))))
Base.atan(t::T) where T<:TensorAlgebra{V} where V = (i=V(I);it=i⟑t;Base.:/(Base.:-(i),2)*Base.:-(Base.log(one(V)+it),Base.log(Base.:-(one(V),it))))
Base.acot(t::T) where T<:TensorAlgebra{V} where V = (i=V(I);Base.:/(Base.:-(i),2)*Base.:-(Base.log(Base.:-(t,i)),Base.log(t+i)))
Base.sinc(t::T) where T<:TensorAlgebra{V} where V = iszero(t) ? one(V) : (x=(1π)*t;Base.:/(Base.sin(x),x))
Base.cosc(t::T) where T<:TensorAlgebra{V} where V = iszero(t) ? zero(V) : (x=(1π)*t; Base.:-(Base.:/(Base.cos(x),t), Base.:/(sin(x),(x⟑t))))

# absolute value norm

@inline Base.abs(t::T) where T<:TensorAlgebra = Base.sqrt(Base.abs2(t))
@inline Base.abs2(t::T) where T<:TensorAlgebra = (a=(~t)⟑t; isscalar(a) ? scalar(a) : a)
@inline Base.abs2(t::T) where T<:TensorGraded = contraction(t,t)
@inline geomabs(t::T) where T<:TensorAlgebra = Base.abs(t)+pseudoabs(t)
@inline norm(z) = LinearAlgebra.norm(z)
@inline LinearAlgebra.norm(t::T) where T<:TensorAlgebra = norm(value(t))
@inline unit(t::T) where T<:Number = Base.:/(t,Base.abs(t))
@inline unitize(t::T) where T<:Number = Base.:/(t,value(pseudoabs(t)))
@inline unitnorm(t::T) where T<:Number = Base.:/(t,norm(geomabs(t)))
@inline Base.iszero(t::T) where T<:TensorAlgebra = LinearAlgebra.norm(t) ≈ 0
@inline Base.isone(t::T) where T<:TensorAlgebra = LinearAlgebra.norm(t)≈value(scalar(t))≈1
@inline LinearAlgebra.dot(a::A,b::B) where {A<:TensorGraded,B<:TensorGraded} = contraction(a,b)

macro pseudo(fun)
    ant = Symbol(:pseudo,fun.args[1])
    com = [:(complementright($(esc(fun.args[t])))) for t ∈ 2:length(fun.args)]
    return Expr(:function,Expr(:call,esc(ant),esc.(fun.args[2:end])...),
        Expr(:block,:(complementleft($(esc(fun.args[1]))($(com...))))))
end

for fun ∈ (:abs,:abs2,:sqrt,:cbrt,:exp,:log,:inv,:sin,:cos,:tan,:sinh,:cosh,:tanh)
    ant = Symbol(:pseudo,fun)
    @eval begin
        export $ant
        @inline $ant(t::T) where T<:TensorAlgebra = complementleft(Base.$fun(complementright(t)))
    end
end
const antiabs,antiabs2,pseudometric = pseudoabs,pseudoabs2,antimetric
pseudosandwich(a,b) = complementleft(sandwich(complementright(a),complementright(b)))

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

import StaticVectors: inv, ∏, ∑, -, /

@inline norm(z::Expr) = abs(z)
@inline norm(z::Symbol) = z
Base.@pure isnull(::Expr) = false
Base.@pure isnull(::Symbol) = false
isnull(n) = iszero(n)
signbit(x::Symbol) = false
signbit(x::Expr) = x.head == :call && x.args[1] == :-

@inline dot(x,y) = LinearAlgebra.dot(x,y)
@inline exp(z) = Base.exp(z)

@inline inv(z::Z) where Z<:TensorAlgebra = Base.inv(z)
for op ∈ (:conj,:sqrt,:abs,:expm1,:log,:log1p,:sin,:cos,:sinh,:cosh,:signbit)
    @eval begin
        @inline $op(z) = Base.$op(z)
        @inline $op(z::Z) where Z<:TensorAlgebra = Base.$op(z)
    end
end

for op ∈ (:^,:≈)
    @eval @inline $op(a,b) = Base.$op(a,b)
end
for op ∈ (:-,:/,:^,:≈)
    @eval @inline $op(a::A,b::B) where {A<:TensorAlgebra,B<:TensorAlgebra} = Base.$op(a,b)
end

for T ∈ (Expr,Symbol)
    @eval begin
        ≈(a::$T,b::$T) = a == b
        ≈(a::$T,b) = false
        ≈(a,b::$T) = false
    end
end

for (OP,op) ∈ ((:∏,:*),(:∑,:+))
    @eval @inline $OP(x::AbstractVector{T}) where T<:Any = $op(x...)
end

const PROD,SUM,SUB,√ = ∏,∑,-,sqrt

if VERSION >= v"1.10.0"; @eval begin
    const $(Symbol("⟇")) = veedot
    export $(Symbol("⟇"))
end end

export FloatVector, FloatMatrix, FloatArray
const FloatVector{T<:AbstractFloat} = AbstractVector{T}
const FloatMatrix{T<:AbstractFloat} = AbstractMatrix{T}
const FloatArray{N,T<:AbstractFloat} = AbstractArray{T,N}

export RealVector, RealMatrix, RealArray
const RealVector{T<:Real} = AbstractVector{T}
const RealMatrix{T<:Real} = AbstractMatrix{T}
const RealArray{N,T<:Real} = AbstractArray{T,N}

export TupleVector, Values, Variables, FixedVector

if haskey(ENV,"STATICJL")
    import StaticArrays: SVector, MVector, SizedVector, StaticVector, _diff
    const Values,Variables,FixedVector,TupleVector = SVector,MVector,SizedVector,StaticVector
    Base.@pure countvalues(a::Int,b::Int) = Values{max(0,b-a+1),Int}(a:b...)
    Base.@pure evenvalues(a::Int,b::Int) = Values{((b-a)÷2)+1,Int}(a:2:b...)
else
    import StaticVectors: Values, Variables, FixedVector, TupleVector, _diff
    import StaticVectors: SVector, MVector, SizedVector, countvalues, evenvalues
end

end # module
