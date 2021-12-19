
# This file is adapted from JuliaArrays/StaticArrays.jl License is MIT:
# https://github.com/JuliaArrays/StaticArrays.jl/blob/master/LICENSE.md

@inline Base.zero(a::SA) where {SA<:TupleVector} = zeros(SA)
@inline Base.zero(a::Type{SA}) where {SA<:TupleVector} = zeros(SA)

@inline Base.zeros(::Type{SA}) where {SA<:TupleVector{N}} where N = _zeros(Val(N), SA)
@generated function _zeros(::Val{N}, ::Type{SA}) where SA<:TupleVector{N} where N
    T = eltype(SA)
    if T == Any
        T = Float64
    end
    v = [:(zero($T)) for i = 1:N]
    return quote
        @_inline_meta
        $SA(tuple($(v...)))
    end
end

@inline Base.ones(::Type{SA}) where {SA<:TupleVector{N}} where N = _ones(Val(N), SA)
@generated function _ones(::Val{N}, ::Type{SA}) where SA<:TupleVector{N} where N
    T = eltype(SA)
    if T == Any
        T = Float64
    end
    v = [:(one($T)) for i = 1:N]
    return quote
        @_inline_meta
        $SA(tuple($(v...)))
    end
end

@inline Base.fill(val,::SA) where {SA<:TupleVector{N}} where N = _fill(val, Val(N), SA)
@inline Base.fill(val,::Type{SA}) where {SA<:TupleVector{N}} where N = _fill(val, Val(N), SA)
@generated function _fill(val,::Val{N},::Type{SA}) where SA<:TupleVector{N} where N
    v = [:val for i = 1:N]
    return quote
        @_inline_meta
        $SA(tuple($(v...)))
    end
end

# Also consider randcycle, randperm? Also faster rand!(staticarray, collection)

using Random
import Random: SamplerType, AbstractRNG
@inline Base.rand(rng::AbstractRNG,::Type{SA},dims::Dims) where SA<:TupleVector = rand!(rng, Array{SA}(undef, dims), SA)
@inline Base.rand(rng::AbstractRNG,::SamplerType{SA}) where SA<:TupleVector{N} where N = _rand(rng, Val(N), SA)

@generated function _rand(rng::AbstractRNG,::Val{N},::Type{SA}) where {SA <: TupleVector{N}} where N
    T = eltype(SA)
    if T == Any
        T = Float64
    end
    v = [:(rand(rng, $T)) for i = 1:N]
    return quote
        @_inline_meta
        $SA(tuple($(v...)))
    end
end

@inline Base.rand(rng::AbstractRNG,range::AbstractArray, ::Type{SA}) where {SA<:TupleVector{N}} where N = _rand(rng, range, Val(N), SA)
@inline Base.rand(range::AbstractArray,::Type{SA}) where {SA<:TupleVector{N}} where N = _rand(Random.GLOBAL_RNG, range, Val(N), SA)
@generated function _rand(rng::AbstractRNG, range::AbstractArray, ::Val{N}, ::Type{SA}) where SA<:TupleVector{N} where N
    v = [:(rand(rng, range)) for i = 1:N]
    return quote
        @_inline_meta
        $SA(tuple($(v...)))
    end
end

#@inline Base.rand(rng::MersenneTwister, range::AbstractArray, ::Type{SA}) where {SA <: TupleVector} = _rand(rng, range, Val(SA), SA)

@inline Base.randn(rng::AbstractRNG,::Type{SA}) where SA<:TupleVector{N} where N = _randn(rng, Val(N), SA)
@generated function _randn(rng::AbstractRNG,::Val{N},::Type{SA}) where SA<:TupleVector{N} where N
    T = eltype(SA)
    if T == Any
        T = Float64
    end
    v = [:(randn(rng, $T)) for i = 1:N]
    return quote
        @_inline_meta
        $SA(tuple($(v...)))
    end
end

@inline Random.randexp(rng::AbstractRNG,::Type{SA}) where SA<:TupleVector{N} where N = _randexp(rng, Val(N), SA)
@generated function _randexp(rng::AbstractRNG,::Val{N},::Type{SA}) where SA <: TupleVector{N} where N
    T = eltype(SA)
    if T == Any
        T = Float64
    end
    v = [:(randexp(rng, $T)) for i = 1:N]
    return quote
        @_inline_meta
        $SA(tuple($(v...)))
    end
end

# Mutable versions

# Why don't these two exist in Base?
# @generated function Base.zeros!{SA <: TupleVector}(a::SA)
# @generated function Base.ones!{SA <: TupleVector}(a::SA)

@inline Base.fill!(a::SA,val) where SA<:TupleVector{N} where N = _fill!(Val(N), a, val)
@generated function _fill!(::Val{N},a::SA,val) where SA<:TupleVector{N} where N
    exprs = [:(a[$i] = valT) for i = 1:N]
    return quote
        @_inline_meta
        valT = convert(eltype(SA), val)
        @inbounds $(Expr(:block, exprs...))
        return a
    end
end

@inline Random.rand!(rng::AbstractRNG,a::SA) where SA<:TupleVector{N} where N = _rand!(rng, Val(N), a)
@generated function _rand!(rng::AbstractRNG,::Val{N},a::SA) where SA <: TupleVector{N} where N
    exprs = [:(a[$i] = rand(rng, eltype(SA))) for i = 1:N]
    return quote
        @_inline_meta
        @inbounds $(Expr(:block, exprs...))
        return a
    end
end

@inline Random.rand!(rng::MersenneTwister,a::SA) where SA<:TupleVector{N,Float64} where N = _rand!(rng, Val(N), a)

@inline Random.randn!(rng::AbstractRNG,a::SA) where SA<:TupleVector{N} where N = _randn!(rng, Val(N), a)
@generated function _randn!(rng::AbstractRNG,::Val{N},a::SA) where SA<:TupleVector{N} where N
    exprs = [:(a[$i] = randn(rng, eltype(SA))) for i = 1:N]
    return quote
        @_inline_meta
        @inbounds $(Expr(:block, exprs...))
        return a
    end
end

@inline Random.randexp!(rng::AbstractRNG,a::SA) where SA<:TupleVector{N} where N = _randexp!(rng, Val(N), a)
@generated function _randexp!(rng::AbstractRNG,::Val{N},a::SA) where SA<:TupleVector{N} where N
    exprs = [:(a[$i] = randexp(rng, eltype(SA))) for i = 1:N]
    return quote
        @_inline_meta
        @inbounds $(Expr(:block, exprs...))
        return a
    end
end
