
# this file is inspired from StaticArrays.jl
# https://github.com/JuliaArrays/StaticArrays.jl

import Base: @propagate_inbounds, @_inline_meta, @pure

abstract type TupleVector{N,T} <: AbstractVector{T} end

# Being a member of TupleMatrixLike or TupleVectorLike implies that Val(A)
# returns a static Val instance (none of the dimensions are Dynamic). The converse may not be true.
# These are akin to aliases like StridedArray and in similarly bad taste, but the current approach
# in Base necessitates their existence.
const TupleMatrixLike{n,T} = Union{
    LinearAlgebra.Transpose{T, <:TupleVector{n,T}},
    LinearAlgebra.Adjoint{T, <:TupleVector{n,T}},
    LinearAlgebra.Diagonal{T, <:TupleVector{n,T}},
}
const TupleVectorLike{n,T} = Union{TupleVector{n,T}, TupleMatrixLike{n,T}}

@pure Base.length(::T) where T<:TupleVector{N} where N = N
@pure Base.length(::Type{<:TupleVector{N}}) where N = N
@pure Base.lastindex(::T) where T<:TupleVector{N} where N = N
@pure Base.size(::T) where T<:TupleVector{N} where N = (N,)
@pure Base.size(::Type{<:TupleVector{N}}) where N = (N,)
@pure @inline Base.size(t::T,d::Int) where T<:TupleVector{N} where N = d > 1 ? 1 : length(t)
@pure @inline Base.size(t::Type{<:TupleVector},d::Int) = d > 1 ? 1 : length(t)

include("SOneTo.jl")
include("util.jl")
include("traits.jl")
include("Values.jl")
include("Variables.jl")
include("FixedVector.jl")
include("initializers.jl")
include("convert.jl")
include("abstractvector.jl")
include("indexing.jl")
include("broadcast.jl")
include("mapreduce.jl")
include("arraymath.jl")
include("linalg.jl")
