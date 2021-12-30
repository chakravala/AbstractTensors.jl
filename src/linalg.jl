
# This file is adapted from JuliaArrays/StaticArrays.jl License is MIT:
# https://github.com/JuliaArrays/StaticArrays.jl/blob/master/LICENSE.md

#--------------------------------------------------
# Vector space algebra

# Unary ops
@inline Base.:-(a::TupleVector) = map(-, a)
@inline Base.:-(a::TupleVector{n,<:Number}) where n = map(Base.:-, a)

# Binary ops
# Between vectors
@inline Base.:+(a::TupleVector, b::TupleVector) = map(∑, a, b)
@inline Base.:+(a::AbstractVector, b::TupleVector) = map(∑, a, b)
@inline Base.:+(a::TupleVector, b::AbstractVector) = map(∑, a, b)

@inline Base.:+(a::TupleVector{n,<:Number}, b::TupleVector{n,<:Number}) where n = map(Base.:+, a, b)
@inline Base.:+(a::AbstractVector{<:Number}, b::TupleVector{n,<:Number}) where n = map(Base.:+, a, b)
@inline Base.:+(a::TupleVector{n,<:Number}, b::AbstractVector{<:Number}) where n = map(Base.:+, a, b)

@inline Base.:-(a::TupleVector, b::TupleVector) = map(-, a, b)
@inline Base.:-(a::AbstractVector, b::TupleVector) = map(-, a, b)
@inline Base.:-(a::TupleVector, b::AbstractVector) = map(-, a, b)

@inline Base.:-(a::TupleVector{n,<:Number}, b::TupleVector{n,<:Number}) where n = map(Base.:-, a, b)
@inline Base.:-(a::AbstractVector{<:Number}, b::TupleVector{n,<:Number}) where n = map(Base.:-, a, b)
@inline Base.:-(a::TupleVector{n,<:Number}, b::AbstractVector{<:Number}) where n = map(Base.:-, a, b)

# Scalar-vector
@inline Base.:*(a::Number, b::TupleVector{n,<:Number}) where n = map(c->Base.:*(a,c), b)
@inline Base.:*(a::TupleVector{n,<:Number}, b::Number) where n = map(c->Base.:*(c,b), a)

@inline Base.:*(a, b::TupleVector) = broadcast(∏, a, b)
@inline Base.:*(a::TupleVector, b) = broadcast(∏, a, b)

@inline Base.:*(a::Expr, b::TupleVector) = broadcast(∏, Ref(a), b)
@inline Base.:*(a::TupleVector, b::Expr) = broadcast(∏, a, Ref(b))

@inline Base.:*(a::Symbol, b::TupleVector) = broadcast(∏, Ref(a), b)
@inline Base.:*(a::TupleVector, b::Symbol) = broadcast(∏, a, Ref(b))

@inline Base.:/(a::TupleVector{n,<:Number}, b::Number) where n = broadcast(Base.:/, a, b)
@inline Base.:\(a::Number, b::TupleVector{n,<:Number}) where n = broadcast(Base.:\, a, b)

@inline Base.:/(a::TupleVector, b) = broadcast(/, a, b)
@inline Base.:\(a, b::TupleVector) = broadcast(\, a, b)

# Ternary ops
@inline Base.muladd(scalar::Number, a::TupleVector, b::TupleVector) = map((ai, bi) -> muladd(scalar, ai, bi), a, b)
@inline Base.muladd(a::TupleVector, scalar::Number, b::TupleVector) = map((ai, bi) -> muladd(ai, scalar, bi), a, b)

#--------------------------------------------------
# Vector products
@inline LinearAlgebra.dot(a::TupleVector, b::TupleVector) = _vecdot(a, b, LinearAlgebra.dot)
@inline bilinear_vecdot(a::TupleVector, b::TupleVector) = _vecdot(a, b, Base.:*)

@inline function _vecdot(a::TupleVector{S}, b::TupleVector{S}, product) where S
    if S == 0
        za, zb = zero(eltype(a)), zero(eltype(b))
    else
        # Use an actual element if there is one, to support e.g. Vector{<:Number}
        # element types for which runtime size information is required to construct
        # a zero element.
        za, zb = zero(a[1]), zero(b[1])
    end
    ret = product(za, zb) + product(za, zb)
    @inbounds @simd for j = 1:S
        ret += product(a[j], b[j])
    end
    return ret
end

#--------------------------------------------------
# Norms
_inner_eltype(v::AbstractVector) = isempty(v) ? eltype(v) : _inner_eltype(first(v))
_inner_eltype(x::Number) = typeof(x)
@inline _init_zero(v::TupleVector) = float(norm(zero(_inner_eltype(v))))

@inline function LinearAlgebra.norm_sqr(v::TupleVector)
    return mapreduce(LinearAlgebra.norm_sqr, Base.:+, v; init=_init_zero(v))
end

@inline LinearAlgebra.norm(a::TupleVector) = _norm(a)
@generated function _norm(a::TupleVector{S}) where S
    if S == 0
        return :(_init_zero(a))
    end
    expr = :(LinearAlgebra.norm_sqr(a[1]))
    for j = 2:S
        expr = :($expr + LinearAlgebra.norm_sqr(a[$j]))
    end
    return quote
        $(Expr(:meta, :inline))
        @inbounds return Base.sqrt($expr)
    end
end

function _norm_p0(x)
    T = _inner_eltype(x)
    return float(norm(iszero(x) ? zero(T) : one(T)))
end

@inline LinearAlgebra.norm(a::TupleVector, p::Real) = _norm(a, p)
@generated function _norm(a::TupleVector{S,T}, p::Real) where {S,T}
    if S == 0
        return :(_init_zero(a))
    end
    fun = T<:Number ? :(LinearAlgebra.norm) : :norm
    expr = :($fun(a[1])^p)
    for j = 2:S
        expr = :($expr + $fun(a[$j])^p)
    end
    expr_p1 = :($fun(a[1]))
    for j = 2:S
        expr_p1 = :($expr_p1 + $fun(a[$j]))
    end
    return quote
        $(Expr(:meta, :inline))
        if p == Inf
            return mapreduce($fun, max, a)
        elseif p == 1
            @inbounds return $expr_p1
        elseif p == 2
            return LinearAlgebra.norm(a)
        elseif p == 0
            return mapreduce(_norm_p0, $(T<:Number ? :(Base.:+) : :∑), a)
        else
            @inbounds return $(T<:Number ? :(Base.:^) : :^)($expr,$(T<:Number ? :(Base.inv) : :inv)(p))
        end
    end
end

@inline LinearAlgebra.normalize(a::TupleVector) = ∏(inv(LinearAlgebra.norm(a)),a)
@inline LinearAlgebra.normalize(a::TupleVector, p::Real) = ∏(inv(LinearAlgebra.norm(a, p)),a)

@inline LinearAlgebra.normalize!(a::TupleVector) = (a .*= inv(LinearAlgebra.norm(a)); return a)
@inline LinearAlgebra.normalize!(a::TupleVector, p::Real) = (a .*= inv(LinearAlgebra.norm(a, p)); return a)

@inline LinearAlgebra.normalize(a::TupleVector{n,<:Number}) where n = Base.:*(Base.inv(LinearAlgebra.norm(a)),a)
@inline LinearAlgebra.normalize(a::TupleVector{n,<:Number}, p::Real) where n = Base.:*(Base.inv(LinearAlgebra.norm(a, p)),a)

@inline LinearAlgebra.normalize!(a::TupleVector{n,<:Number}) where n = (a .*= Base.inv(LinearAlgebra.norm(a)); return a)
@inline LinearAlgebra.normalize!(a::TupleVector{n,<:Number}, p::Real) where n = (a .*= Base.inv(LinearAlgebra.norm(a, p)); return a)
