
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
    TensorAlgebra{V,T} <: Number

Universal root tensor type with `Manifold` instance `V` with scalar field `T`.
"""
abstract type TensorAlgebra{V,T} <: Number end
TensorAlgebra{V}(t::TensorAlgebra{V}) where V = t
TensorAlgebra{V}(t::TensorAlgebra{W}) where {V,W} = (V∪W)(t)

"""
    istensor(t) -> Bool

Test whether `t` is some subtype of `TensorAlgebra`.
"""
Base.@pure istensor(t::T) where T<:TensorAlgebra = true
Base.@pure istensor(t) = false

"""
    Manifold{V,T} <: TensorAlgebra{V,T}

Basis parameter locally homeomorphic to `V::Submanifold{M}` T-module product topology.
"""
abstract type Manifold{V,T} <: TensorAlgebra{V,T} end

"""
    ismanifold(t) -> Bool

Test whether `t` is some subtype of `Manifold`.
"""
Base.@pure ismanifold(t::T) where T<:Manifold = true
Base.@pure ismanifold(t) = false

"""
    TensorGraded{V,G,T} <: Manifold{V,T} <: TensorAlgebra

Grade `G` elements of a `Manifold` instance `V` with scalar field `T`.
"""
abstract type TensorGraded{V,G,T} <: Manifold{V,T} end
const TAG = (:TensorAlgebra,:TensorGraded)

"""
    isgraded(t) -> Bool

Test whether `t` is some subtype of `TensorGraded`.
"""
Base.@pure isgraded(t::T) where T<:TensorGraded = true
Base.@pure isgraded(t) = false

"""
    Scalar{V,T} <: TensorGraded{V,0,T}

Graded `scalar` elements of a `Manifold` instance `V` with scalar field `T`.
"""
const Scalar{V,T} = TensorGraded{V,0,T}

"""
    GradedVector{V,T} <: TensorGraded{V,1,T}

Graded `vector` elements of a `Manifold` instance `V` with scalar field `T`.
"""
const GradedVector{V,T} = TensorGraded{V,1,T}

"""
    Bivector{V,T} <: TensorGraded{V,2,T}

Graded `bivector` elements of a `Manifold` instance `V` with scalar field `T`.
"""
const Bivector{V,T} = TensorGraded{V,2,T}

"""
    Trivector{V,T} <: TensorGraded{V,3,T}

Graded `trivector` elements of a `Manifold` instance `V` with scalar field `T`.
"""
const Trivector{V,T} = TensorGraded{V,3,T}

"""
    TensorTerm{V,G,T} <: TensorGraded{V,G,T}

Single coefficient for grade `G` of a `Manifold` instance `V` with scalar field `T`.
"""
abstract type TensorTerm{V,G,T} <: TensorGraded{V,G,T} end

"""
    isterm(t) -> Bool

Test whether `t` is some subtype of `TensorTerm`.
"""
Base.@pure isterm(t::T) where T<:TensorTerm = true
Base.@pure isterm(t) = false
Base.isfinite(b::T) where T<:TensorTerm = isfinite(value(b))

"""
    TensorMixed{V,T} <: TensorAlgebra{V,T}

Elements of `Manifold` instance `V` having non-homogenous grade with scalar field `T`.
"""
abstract type TensorMixed{V,T} <: TensorAlgebra{V,T} end

"""
    ismixed(t) -> Bool

Test whether `t` is some subtype of `TensorMixed`.
"""
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

"""
    tdims(t::TensorAlgebra{V})

Dimensionality of the superalgebra of `V` for that `TensorAlgebra`.
"""
Base.@pure tdims(M::T) where T<:TensorAlgebra = 1<<mdims(M)
Base.@pure tdims(M::Type{T}) where T<:TensorAlgebra = 1<<mdims(M)
Base.@pure tdims(M::Int) = 1<<M

"""
    gdims(t::TensorGraded{V,G})

Dimensionality of the grade `G` of `V` for that `TensorAlgebra`.
"""
Base.@pure gdims(t::TensorGraded{V,G}) where {V,G} = gdims(mdims(t),G)
Base.@pure gdims(t::Type{T}) where {V,G,T<:TensorGraded{V,G}} = gdims(mdims(t),G)
Base.@pure gdims(N,G) = Base.binomial(N,G)

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
@inline volume(t::T) where T<:TensorGraded{V,G} where {V,G} = G == mdims(t) ? t : zero(V)
@inline isvolume(t::T) where T<:TensorGraded = rank(t) == mdims(t) || iszero(t)

"""
    value(::TensorAlgebra)

Returns the internal `Values` representation of a `TensorAlgebra` element.
"""
value(t::T) where T<:Number = t
value(t::T) where T<:AbstractArray = t

"""
    valuetype(t::TensorAlgebra{V,T}) where {V,T} = T

Returns type of a `TensorAlgebra` element value's internal representation.
"""
Base.@pure valuetype(::T) where T<:TensorAlgebra{V,K} where V where K = K
Base.@pure valuetype(::Type{<:TensorAlgebra{V,T} where V}) where T = T
Base.@pure valuetype(::T) where T<:Number = T
Base.@pure valuetype(::Type{T}) where T<:Number = T
const valtype = valuetype; export valtype

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

export TensorAlgebra, Manifold, TensorGraded, Scalar, GradedVector, Bivector, Trivector
export wedgedot, veedot, contraction, expansion, metric, pseudometric, cometric, @pseudo
export istensor, ismanifold, isterm, isgraded, ismixed, rank, mdims, tdims, gdims, sandwich
export scalar, isscalar, vector, isvector, bivector, isbivector, volume, isvolume, hodge
export value, valuetype, interop, interform, involute, unit, unitize, unitnorm, even, odd
export ⟑, ⊘, ⊖, ⊗, ⊛, ⊙, ⊠, ×, ⨼, ⨽, ⋆, ∗, ⁻¹, ǂ, ₊, ₋, ˣ, antiabs, antiabs2, geomabs, @co

# some shared presets

for op ∈ (:(Base.:+),:(Base.:-),:(Base.:*),:sandwich,:⊛,:∗,:⨼,:⨽,:contraction,:contraction_metric,:expansion,:veedot,:wedgedot,:wedgedot_metric,:(LinearAlgebra.dot),:(Base.:|),:(Base.:(==)),:(Base.:<),:(Base.:>),:(Base.:<<),:(Base.:>>),:(Base.:>>>),:(Base.div),:(Base.rem),:(Base.:&))
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
for op ∈ (:plus,:minus,:wedgedot,:wedgedot_metric,:contraction,:contraction_metric,:equal,:sandwich,:⊛,:∗,:(Base.:|),:(Base.:<),:(Base.:>),:(Base.:<<),:(Base.:>>),:(Base.:>>>),:(Base.div),:(Base.rem),:(Base.:&))
    @eval @inline $op(a::A,b::B) where {A<:TensorAlgebra,B<:TensorAlgebra} = interop($op,a,b)
end

for op ∈ (:(Base.:!),:complementrighthodge)
    for T ∈ (:Real,:Complex)
        @eval @inline $op(t::T,g=nothing) where T<:$T = UniformScaling(t)
    end
end

const complement = !
const complementright = !
const ⋆ = complementrighthodge
const hodge = complementrighthodge
const ⊘ = sandwich
const ⊖,⟑,times,antidot,pseudodot,codot = wedgedot,wedgedot,wedgedot,expansion,expansion,expansion
@inline Base.:|(t::T) where T<:TensorAlgebra = hodge(t)
@inline Base.:!(t::UniformScaling{T}) where T = T<:Bool ? (t.λ ? 1 : 0) : t.λ

for (op,logm,field) ∈ ((:⟑,:(Base.log),false),(:wedgedot_metric,:log_metric,true)); args = field ? (:g,) : ()
    @eval begin
        @inline Base.:/(a::A,b::B,$(args...)) where {A<:TensorAlgebra,B<:TensorAlgebra} = $op(a,Base.inv(b,$(args...)),$(args...))
        @inline Base.:/(a::UniformScaling,b::B,$(args...)) where B<:TensorAlgebra = $op(Manifold(b)(a),Base.inv(b,$(args...)),$(args...))
        @inline Base.:/(a::A,b::UniformScaling,$(args...)) where A<:TensorAlgebra = $op(a,Base.inv(Manifold(a)(b),$(args...)),$(args...))
        @inline Base.:\(a::A,b::B,$(args...)) where {A<:TensorAlgebra,B<:TensorAlgebra} = $op(Base.inv(a,$(args...)),b,$(args...))
        @inline Base.:\(a::UniformScaling,b::B,$(args...)) where B<:TensorAlgebra = $op(Base.inv(Manifold(b)(a),$(args...)),b,$(args...))
        @inline Base.:\(a::A,b::UniformScaling,$(args...)) where A<:TensorAlgebra = $op(Base.inv(a,$(args...)),Manifold(a)(b),$(args...))
        @inline Base.:^(b::S,t::T,$(args...)) where {S<:Number,T<:TensorAlgebra} = Base.exp($op(t,$logm(b,$(args...)),$(args...)),$(args...))
        @inline Base.:^(a::A,b::UniformScaling,$(args...)) where A<:TensorAlgebra = Base.:^(a,Manifold(a)(b),$(args...))
        @inline Base.:^(a::UniformScaling,b::B,$(args...)) where B<:TensorAlgebra = Base.:^(Manifold(b)(a),b,$(args...))
        @inline Base.exp(t::T,$(args...)) where T<:TensorAlgebra{V} where V = one(V)+Base.expm1(t,$(args...))
        @inline $logm(b,t::T,$(args...)) where T<:TensorAlgebra = $logm(t,$(args...))/$logm(b,$(args...))
    end
end
@inline ⊗(a::A,b::B) where {A<:TensorAlgebra,B<:Real} = a*b
@inline ⊗(a::A,b::B) where {A<:TensorAlgebra,B<:Complex} = a*b
@inline ⊗(a::A,b::B) where {A<:Real,B<:TensorAlgebra} = a*b
@inline ⊗(a::A,b::B) where {A<:Complex,B<:TensorAlgebra} = a*b
Base.:∘(a::A,b::B) where {A<:TensorAlgebra,B<:TensorAlgebra} = expansion(a,b)

for op ∈ (:(Base.:+),:(Base.:*),:wedgedot)
    @eval $op(t::T) where T<:TensorAlgebra = t
end
for op ∈ (:⊙,:⊠,:¬,:⋆,:clifford,:basis,:complementleft,:complementlefthodge,:complementleftanti,:complementrightanti)
    @eval function $op end
end
for op ∈ (:scalar,:involute,:even)
    @eval $op(t::T) where T<:Real = t
end
odd(::T) where T<:Real = 0
LinearAlgebra.cross(a::A,b::B) where {A<:TensorAlgebra,B<:TensorAlgebra} = hodge(∧(a,b))
wedgedot(a,b) = Base.:*(a,b)
contraction(a,b) = LinearAlgebra.dot(a,b)
wedgedot_metric(a::Real,b,g) = Base.:*(a,b)
wedgedot_metric(a,b::Real,g) = Base.:*(a,b)
wedgedot_metric(a::Real,b::Real,g) = Base.:*(a,b)
wedgedot_metric(a::Complex,b::Real,g) = Base.:*(a,b)
wedgedot_metric(a::Real,b::Complex,g) = Base.:*(a,b)
wedgedot_metric(a::Complex,b,g) = Base.:*(a,b)
wedgedot_metric(a,b::Complex,g) = Base.:*(a,b)
wedgedot_metric(a::Complex,b::Complex,g) = Base.:*(a,b)
contraction_metric(a::Real,b::Real,g) = contraction(a,b)
contraction_metric(a::Complex,b::Complex,g) = contraction(a,b)
contraction_metric(a::Real,b::TensorAlgebra,g) = contraction(a,b)
contraction_metric(a::Complex,b::TensorAlgebra,g) = contraction(a,b)
contraction_metric(a::TensorAlgebra,b::Real,g) = contraction(a,b)
contraction_metric(a::TensorAlgebra,b::Complex,g) = contraction(a,b)
LinearAlgebra.norm(a::TensorMixed,b::TensorMixed) = norm(a-b)
LinearAlgebra.norm(a::TensorGraded,b::TensorGraded) = norm(a-b)
for (op,fun) ∈ ((:metric,:abs),(:cometric,:pseudoabs))
    @eval begin
        $op(a::TensorMixed,b::TensorMixed) = $fun(a-b)
        $op(a::TensorGraded,b::TensorGraded) = $fun(a-b)
        $op(a::TensorMixed,b::TensorGraded) = $fun(a-b)
        $op(a::TensorGraded,b::TensorMixed) = $fun(a-b)
        $op(a::Real,b::Real) = $fun(a-b)
        $op(a::Complex,b::Complex) = $fun(a-b)
        $op(a::Real,b::Complex) = $fun(a-b)
        $op(a::Complex,b::Real) = $fun(a-b)
    end
end

for base ∈ (2,10)
    fl,fe = (Symbol(:log,base),Symbol(:exp,base))
    @eval Base.$fl(t::T) where T<:TensorAlgebra = Base.$fl(ℯ)*Base.log(t)
    @eval Base.$fe(t::T) where T<:TensorAlgebra = Base.exp(Base.log($base)*t)
end

for op ∈ (:/,:^)
    @eval begin
        @inline Base.$op(a::Real,b::Real,g) = Base.$op(a,b)
        @inline Base.$op(a::Real,b::Complex,g) = Base.$op(a,b)
        @inline Base.$op(a::Complex,b::Real,g) = Base.$op(a,b)
        @inline Base.$op(a::Complex,b::Complex,g) = Base.$op(a,b)
    end
end
for op ∈ (:abs,:abs2,:cos,:sin,:tan,:cot,:sec,:csc,:asec,:acsc,:sech,:csch,:asech,:acsch,:tanh,:coth,:asinh,:acosh,:atanh,:acoth,:asin,:acos,:atan,:acot,:sinc,:cosc,:cis,:sqrt,:cbrt,:exp,:exp2,:exp10,:log2,:log10)
    @eval begin
        @inline Base.$op(t::Real,g) = Base.$op(t)
        @inline Base.$op(t::Complex,g) = Base.$op(t)
    end
end
@inline Base.log(t::Real,g::TensorAlgebra) = Base.log(t)
@inline Base.log(t::Complex,g::TensorAlgebra) = Base.log(t)
for (op,logm,field) ∈ ((:⟑,:(Base.log),false),(:wedgedot_metric,:log_metric,true)); args = field ? (:g,) : ()
    @eval begin
@inline Base.cos(t::T,$(args...)) where T<:TensorAlgebra{V} where V = Base.cosh($op(V(I),t,$(args...)),$(args...))
@inline Base.sin(t::T,$(args...)) where T<:TensorAlgebra{V} where V = (i=V(I);Base.:/(Base.sinh($op(i,t,$(args...)),$(args...)),i,$(args...)))
@inline Base.tan(t::T,$(args...)) where T<:TensorAlgebra = Base.:/(Base.sin(t,$(args...)),Base.cos(t,$(args...)),$(args...))
@inline Base.cot(t::T,$(args...)) where T<:TensorAlgebra = Base.:/(Base.cos(t,$(args...)),Base.sin(t,$(args...)),$(args...))
@inline Base.sec(t::T,$(args...)) where T<:TensorAlgebra = Base.inv(Base.cos(t,$(args...)),$(args...))
@inline Base.csc(t::T,$(args...)) where T<:TensorAlgebra = Base.inv(Base.sin(t,$(args...)),$(args...))
@inline Base.asec(t::T,$(args...)) where T<:TensorAlgebra = Base.acos(Base.inv(t,$(args...)),$(args...))
@inline Base.acsc(t::T,$(args...)) where T<:TensorAlgebra = Base.asin(Base.inv(t,$(args...)),$(args...))
@inline Base.sech(t::T,$(args...)) where T<:TensorAlgebra = Base.inv(Base.cosh(t,$(args...)),$(args...))
@inline Base.csch(t::T,$(args...)) where T<:TensorAlgebra = Base.inv(Base.sinh(t,$(args...)),$(args...))
@inline Base.asech(t::T,$(args...)) where T<:TensorAlgebra = Base.acosh(Base.inv(t,$(args...)),$(args...))
@inline Base.acsch(t::T,$(args...)) where T<:TensorAlgebra = Base.asinh(Base.inv(t,$(args...)),$(args...))
@inline Base.tanh(t::T,$(args...)) where T<:TensorAlgebra = Base.:/(Base.sinh(t,$(args...)),Base.cosh(t,$(args...)),$(args...))
@inline Base.coth(t::T,$(args...)) where T<:TensorAlgebra = Base.:/(Base.cosh(t,$(args...)),Base.sinh(t,$(args...)),$(args...))
@inline Base.asinh(t::T,$(args...)) where T<:TensorAlgebra{V} where V = $logm(t+Base.sqrt(one(V)+$op(t,t,$(args...)),$(args...)),$(args...))
@inline Base.acosh(t::T,$(args...)) where T<:TensorAlgebra{V} where V = $logm(t+Base.sqrt(Base.:-($op(t,t,$(args...)),one(V)),$(args...)),$(args...))
@inline Base.atanh(t::T,$(args...)) where T<:TensorAlgebra{V} where V = Base.:/(Base.:-($logm(one(V)+t,$(args...)),$logm(Base.:-(one(V),t),$(args...))),2)
@inline Base.acoth(t::T,$(args...)) where T<:TensorAlgebra{V} where V = Base.:/(Base.:-($logm(t+one(V),$(args...)),$logm(Base.:-(t,one(V)),$(args...))),2)
Base.asin(t::T,$(args...)) where T<:TensorAlgebra{V} where V = (i=V(I);$op(Base.:-(i),$logm($op(i,t,$(args...))+Base.sqrt(Base.:-(one(V),$op(t,t,$(args...))),$(args...)),$(args...)),$(args...)))
Base.acos(t::T,$(args...)) where T<:TensorAlgebra{V} where V = (i=V(I);$op(Base.:-(i),$logm(t+$op(i,Base.sqrt(Base.:-(one(V),$op(t,t,$(args...))),$(args...)),$(args...)),$(args...)),$(args...)))
Base.atan(t::T,$(args...)) where T<:TensorAlgebra{V} where V = (i=V(I);it=$op(i,t,$(args...));$op(Base.:/(Base.:-(i),2),Base.:-($logm(one(V)+it,$(args...)),$logm(Base.:-(one(V),it),$(args...))),$(args...)))
Base.acot(t::T,$(args...)) where T<:TensorAlgebra{V} where V = (i=V(I);$op(Base.:/(Base.:-(i),2),Base.:-($logm(Base.:-(t,i),$(args...)),$logm(t+i,$(args...))),$(args...)))
Base.sinc(t::T,$(args...)) where T<:TensorAlgebra{V} where V = iszero(t) ? one(V) : (x=(1π)*t;Base.:/(Base.sin(x,$(args...)),x,$(args...)))
Base.cosc(t::T,$(args...)) where T<:TensorAlgebra{V} where V = iszero(t) ? zero(V) : (x=(1π)*t; Base.:-(Base.:/(Base.cos(x,$(args...)),t,$(args...)), Base.:/(sin(x,$(args...)),$op(x,t,$(args...)),$(args...))))
end end

# absolute value norm

@inline Base.abs(t::T) where T<:TensorAlgebra = Base.sqrt(Base.abs2(t))
@inline Base.abs(t::T,g) where T<:TensorAlgebra = Base.sqrt(Base.abs2(t,g),g)
@inline Base.abs2(t::T) where T<:TensorAlgebra = (a=(~t)⟑t; isscalar(a) ? scalar(a) : a)
@inline Base.abs2(t::T,g) where T<:TensorAlgebra = (a=wedgedot_metric(~t,t,g); isscalar(a) ? scalar(a) : a)
@inline Base.abs2(t::T) where T<:TensorGraded = contraction(t,t)
@inline Base.abs2(t::T,g) where T<:TensorGraded = contraction_metric(t,t,g)
@inline norm(z) = LinearAlgebra.norm(z)
@inline LinearAlgebra.norm(t::T) where T<:TensorAlgebra = norm(value(t))
@inline Base.iszero(t::T) where T<:TensorAlgebra = LinearAlgebra.norm(t) ≈ 0
@inline Base.isone(t::T) where T<:TensorAlgebra = LinearAlgebra.norm(t)≈value(scalar(t))≈1
@inline LinearAlgebra.dot(a::A,b::B) where {A<:TensorGraded,B<:TensorGraded} = contraction(a,b)

"""
    geomabs(t::TensorAlgebra)

Geometric norm defined as `geomabs(t) = abs(t) + coabs(t)`.
"""
@inline geomabs(t::T) where T<:TensorAlgebra = Base.abs(t)+coabs(t)
@inline geomabs(t::T,g) where T<:TensorAlgebra = Base.abs(t,g)+coabs(t,g)

"""
    unit(t::Number)

Normalization defined as `unit(t) = t/abs(t)`.
"""
@inline unit(t::T) where T<:Number = Base.:/(t,Base.abs(t))
@inline unit(t::T,g) where T<:Number = Base.:/(t,Base.abs(t,g),g)

"""
    counit(t::TensorAlgebra)

Pseudo-normalization defined as `unitize(t) = t/value(coabs(t))`.
"""
@inline counit(t::T) where T<:Number = Base.:/(t,value(coabs(t)))
@inline counit(t::T,g) where T<:Number = Base.:/(t,value(coabs(t,g)),g)
const unitize = counit

"""
    unitnorm(t::TensorAlgebra)

Geometric normalization defined as `unitnorm(t) = t/norm(geomabs(t))`.
"""
@inline unitnorm(t::T) where T<:Number = Base.:/(t,norm(geomabs(t)))
@inline unitnorm(t::T,g) where T<:Number = Base.:/(t,norm(geomabs(t,g)),g)

"""
    @co fun(args...)

Use the macro `@co` to make a pseudoscalar `complement` variant of any functions:

```Julia
julia> @co myfun(x)
comyfun (generic function with 1 method)
```

Now `comyfun(x) = complementleft(myfun(complementright(x)))` is defined.

```Julia
julia> @co myproduct(a,b)
comyproduct (generic function with 1 method)
```
Now `comyproduct(a,b) = complementleft(myproduct(!a,!b))` is defined.
"""
macro co(fun)
    ant = Symbol(:co,fun.args[1])
    com = [:(complementright($(esc(fun.args[t])))) for t ∈ 2:length(fun.args)]
    return Expr(:function,Expr(:call,esc(ant),esc.(fun.args[2:end])...),
        Expr(:block,:(complementleft($(esc(fun.args[1]))($(com...))))))
end

"""
    @pseudo fun(args...)

Use the macro `@pseudo` to make a pseudoscalar `complement` variant of any functions:

```Julia
julia> @pseudo myfun(x)
pseudomyfun (generic function with 1 method)
```

Now `pseudomyfun(x) = complementleft(myfun(complementright(x)))` is defined.

```Julia
julia> @pseudo myproduct(a,b)
pseudomyproduct (generic function with 1 method)
```
Now `pseudomyproduct(a,b) = complementleft(myproduct(!a,!b))` is defined.
"""
macro pseudo(fun)
    ant = Symbol(:pseudo,fun.args[1])
    com = [:(complementright($(esc(fun.args[t])))) for t ∈ 2:length(fun.args)]
    return Expr(:function,Expr(:call,esc(ant),esc.(fun.args[2:end])...),
        Expr(:block,:(complementleft($(esc(fun.args[1]))($(com...))))))
end

for fun ∈ (:abs,:abs2,:sqrt,:cbrt,:exp,:log,:inv,:sin,:cos,:tan,:sinh,:cosh,:tanh)
    for ant ∈ (Symbol(:pseudo,fun),Symbol(:co,fun))
        str = """
    $ant(t::TensorAlgebra)

Complemented `$fun` defined as `complementleft($fun(complementright(t)))`.
        """
        @eval begin
            export $ant
            @doc $str $ant
            @inline $ant(t::T) where T<:TensorAlgebra = complementleft(Base.$fun(complementright(t)))
        end
        fun ≠ :log && @eval begin
            @inline $ant(t::T,g) where T<:TensorAlgebra = complementleft(Base.$fun(complementright(t),g))
        end
    end
end
@inline colog_metric(t::T,g) where T<:TensorAlgebra = complementleft(log_metric(complementright(t),g))
@inline pseudolog_metric(t::T,g) where T<:TensorAlgebra = complementleft(log_metric(complementright(t),g))
const antiabs,antiabs2,antimetric,pseudometric = coabs,coabs2,cometric,cometric
export cosandwich, pseudosandwich, antisandwich, antimetric

"""
    cosandwich(x::TensorAlgebra,R::TensorAlgebra)

Defined as `complementleft(sandwich(complementright(x),complementright(R)))`.
"""
cosandwich(x,R) = complementleft(sandwich(complementright(x),complementright(R)))
cosandwich(x,R,g) = complementleft(sandwich(complementright(x),complementright(R),g))
const pseudosandwich = cosandwich

"""
    antisandwich(x::TensorAlgebra,R::TensorAlgebra)

Defined as `complementleft(complementright(R)>>>complementright(x))`.
"""
antisandwich(R,x) = complementleft(complementright(R)>>>complementright(x))
antisandwich(R,x,g) = complementleft(>>>(complementright(R),complementright(x),g))

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

import StaticVectors: Values, Variables, FixedVector, TupleVector, evens, _diff
import StaticVectors: SVector, MVector, SizedVector, countvalues, evenvalues

end # module
