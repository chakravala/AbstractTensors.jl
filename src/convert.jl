
# This file is adapted from JuliaArrays/StaticArrays.jl License is MIT:
# https://github.com/JuliaArrays/StaticArrays.jl/blob/master/LICENSE.md

(::Type{SA})(x::Tuple{Tuple{Tuple{<:Tuple}}}) where {SA <: TupleVector} =
    throw(DimensionMismatch("No precise constructor for $SA found. Val of input was $(length(x[1][1][1]))."))

@inline (::Type{SA})(x...) where {SA <: TupleVector} = SA(x)
@inline (::Type{SA})(a::TupleVector) where {SA<:TupleVector} = SA(Tuple(a))
@propagate_inbounds (::Type{SA})(a::AbstractVector) where {SA <: TupleVector} = convert(SA, a)

# this covers most conversions and "statically-sized reshapes"
#@inline Base.convert(::Type{SA}, sa::TupleVector) where {SA<:TupleVector} = SA(Tuple(sa))
#@inline Base.convert(::Type{SA}, sa::SA) where {SA<:TupleVector} = sa
@inline Base.convert(::Type{SA}, x::Tuple) where {SA<:TupleVector} = SA(x) # convert -> constructor. Hopefully no loops...

# support conversion to AbstractVector
Base.AbstractVector{T}(sa::TupleVector{N,T}) where {N,T} = sa
Base.AbstractVector{T}(sa::TupleVector{N,U}) where {N,T,U} = similar_type(typeof(sa),T,Val(N))(sa)

# Constructing a Tuple from a TupleVector
@inline Base.Tuple(a::TupleVector{N}) where N = unroll_tuple(a, Val(N))

@noinline function dimension_mismatch_fail(SA::Type, a::AbstractVector)
    throw(DimensionMismatch("expected input vector of length $(length(SA)), got length $(length(a))"))
end

@propagate_inbounds function Base.convert(::Type{SA}, a::AbstractVector) where {SA<:TupleVector{N}} where N
    @boundscheck if length(a) != length(SA)
        dimension_mismatch_fail(SA, a)
    end

    return _convert(SA, a, Val(N))
end

@inline _convert(SA, a, l::Val) = SA(unroll_tuple(a, l))
@inline _convert(SA::Type{<:TupleVector{N,T}}, a, ::Val{0}) where {N,T} = similar_type(SA, T)(())
@inline _convert(SA, a, ::Val{0}) = similar_type(SA, eltype(a))(())

@pure length_val(n::Val{S}) where S = n
@pure length_val(n::TupleVector{S}) where S = Val(S)
@pure length_val(n::Type{<:TupleVector{S}}) where S = Val(S)
@pure length_val(a::T) where {T <: TupleMatrixLike{S}} where S = Val(S)
@pure length_val(a::Type{T}) where {T<:TupleMatrixLike{S}} where S = Val(S)

@generated function unroll_tuple(a::AbstractVector, ::Val{N}) where N
    exprs = [:(a[$j]) for j = 1:N]
    quote
        @_inline_meta
        @inbounds return $(Expr(:tuple, exprs...))
    end
end

# `float` and `real` of TupleVector types, analogously to application to scalars (issue 935)
Base.float(::Type{TV}) where TV<:TupleVector{T,_N} where {T,_N} = similar_type(TV, float(T))
Base.real(::Type{TV}) where TV<:TupleVector{T,_N} where {T,_N} = similar_type(TV, real(T))
