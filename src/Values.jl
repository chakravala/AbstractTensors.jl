
# This file is adapted from JuliaArrays/StaticArrays.jl License is MIT:
# https://github.com/JuliaArrays/StaticArrays.jl/blob/master/LICENSE.md

# SArray.jl

struct Values{N,T} <: TupleVector{N,T}
    v::NTuple{N,T}
    Values{N,T}(x::NTuple{N,T}) where {N,T} = new{N,T}(x)
    Values{N,T}(x::NTuple{N,Any}) where {N,T} = new{N,T}(convert_ntuple(T, x))
end

@pure @generated function (::Type{Values{N,T}})(x::Tuple) where {T, N}
    return quote
        @_inline_meta
        Values{N,T}(x)
    end
end

@noinline function generator_too_short_error(inds::CartesianIndices, i::CartesianIndex)
    error("Generator produced too few elements: Expected exactly $(shape_string(inds)) elements, but generator stopped at $(shape_string(i))")
end
@noinline function generator_too_long_error(inds::CartesianIndices)
    error("Generator produced too many elements: Expected exactly $(shape_string(inds)) elements, but generator yields more")
end

shape_string(inds::CartesianIndices) = join(length.(inds.indices), '×')
shape_string(inds::CartesianIndex) = join(Tuple(inds), '×')

@inline throw_if_nothing(x, inds, i) =
    (x === nothing && generator_too_short_error(inds, i); x)

@generated function tvcollect(::Type{TV}, gen) where {TV <: TupleVector{N}} where N
    stmts = [:(Base.@_inline_meta)]
    args = []
    iter = :(iterate(gen))
    inds = CartesianIndices(Tuple(N))
    for i in inds
        el = Symbol(:el, i)
        push!(stmts, :(($el,st) = throw_if_nothing($iter, $inds, $i)))
        push!(args, el)
        iter = :(iterate(gen,st))
    end
    push!(stmts, :($iter === nothing || generator_too_long_error($inds)))
    push!(stmts, :(TV($(args...))))
    Expr(:block, stmts...)
end
"""
   tvcollect(TV, gen)

Construct a statically-sized vector of type `TV`.from a generator
`gen`. `TV` needs to have a size parameter since the length of `vec`
is unknown to the compiler. `TV` can optionally specify the element
type as well.

Example:

    tvcollect(Values{3, Int}, 2i+1 for i in 1:3)

This creates the same statically-sized vector as if the generator were
collected in an array, but is more efficient since no array is
allocated.

Equivalent:

    Values{3, Int}([2i+1 for i in 1:3])
"""
tvcollect

@inline (::Type{TV})(gen::Base.Generator) where {TV <: TupleVector} =
    tvcollect(TV, gen)

@inline Values(a::TupleVector{N}) where N = Values{N}(Tuple(a))
@propagate_inbounds Base.getindex(v::Values, i::Int) = v.v[i]
@inline Tuple(v::Values) = v.v
Base.dataids(::Values) = ()

# See #53
Base.cconvert(::Type{Ptr{T}}, a::Values) where {T} = Base.RefValue(a)
Base.unsafe_convert(::Type{Ptr{T}}, a::Base.RefValue{TV}) where {N,T,TV<:Values{N,T}} = Ptr{T}(Base.unsafe_convert(Ptr{Values{N,T}}, a))

# SVector.jl

@inline Values(x::NTuple{N,Any}) where N = Values{N}(x)
@inline Values{N}(x::NTuple{N,T}) where {N,T} = Values{N,T}(x)
@inline Values{N}(x::T) where {N,T<:Tuple} = Values{N,promote_tuple_eltype(T)}(x)

@inline Values{N, T}(gen::Base.Generator) where {N, T} = tvcollect(Values{N, T}, gen)
@inline Values{N}(gen::Base.Generator) where {N} = tvcollect(Values{N}, gen)

# Some more advanced constructor-like functions
@pure @inline Base.zeros(::Type{Values{N}}) where N = zeros(Values{N,Float64})
@pure @inline Base.ones(::Type{Values{N}}) where N = ones(Values{N,Float64})

# Converting a CartesianIndex to an SVector
Base.convert(::Type{Values}, I::CartesianIndex) = Values(I.I)
Base.convert(::Type{Values{N}}, I::CartesianIndex{N}) where {N} = Values{N}(I.I)
Base.convert(::Type{Values{N,T}}, I::CartesianIndex{N}) where {N,T} = Values{N,T}(I.I)

@pure Base.promote_rule(::Type{Values{N,T}}, ::Type{CartesianIndex{N}}) where {N,T} = Values{N,promote_type(T,Int)}

