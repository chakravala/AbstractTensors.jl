
# this file is inspired from TupleVectors.jl
# https://github.com/JuliaArrays/TupleVectors.jl

import Base: @propagate_inbounds, @_inline_meta, @pure

abstract type TupleVector{N,T} <: AbstractVector{T} end

@pure Base.length(::T) where T<:TupleVector{N} where N = N
@pure Base.length(::Type{<:TupleVector{N}}) where N = N
@pure Base.lastindex(::T) where T<:TupleVector{N} where N = N
@pure Base.size(::T) where T<:TupleVector{N} where N = (N,)
@pure Base.size(::Type{<:TupleVector{N}}) where N = (N,)
@pure @inline Base.size(t::T,d::Int) where T<:TupleVector{N} where N = d > 1 ? 1 : length(t)
@pure @inline Base.size(t::Type{<:TupleVector},d::Int) = d > 1 ? 1 : length(t)

Base.IndexStyle(::Type{T}) where {T<:TupleVector} = IndexLinear()


################################
## Non-scalar linear indexing ##
################################

@inline function Base.getindex(a::TupleVector{N}, ::Colon) where N
    _getindex(a::TupleVector, Val(N), :)
end

@generated function _getindex(a::TupleVector, s::Val{N}, ::Colon) where N
    exprs = [:(a[$i]) for i = 1:N]
    return quote
        @_inline_meta
        @inbounds return similar_type(a,s)(tuple($(exprs...)))
    end
end

@propagate_inbounds function Base.getindex(a::TupleVector, inds::TupleVector{N,Int}) where N
    _getindex(a, Val(N), inds)
end

@generated function _getindex(a::TupleVector, s::Val{N}, inds::TupleVector{N, Int}) where N
    exprs = [:(a[inds[$i]]) for i = 1:N]
    return quote
        Base.@_propagate_inbounds_meta
        similar_type(a, s)(tuple($(exprs...)))
    end
end

@inline function Base.setindex!(a::TupleVector{N}, v, ::Colon) where N
    _setindex!(a::TupleVector, v, Val(N), :)
    return v
end

@generated function _setindex!(a::TupleVector, v, ::Val{L}, ::Colon) where {L}
    exprs = [:(a[$i] = v) for i = 1:L]
    return quote
        @_inline_meta
        @inbounds $(Expr(:block, exprs...))
    end
end

@generated function _setindex!(a::TupleVector, v::AbstractVector, ::Val{L}, ::Colon) where {L}
    exprs = [:(a[$i] = v[$i]) for i = 1:L]
    return quote
        Base.@_propagate_inbounds_meta
        @boundscheck if length(v) != L
            throw(DimensionMismatch("tried to assign $(length(v))-element array to length-$L destination"))
        end
        @inbounds $(Expr(:block, exprs...))
    end
end

@generated function _setindex!(a::TupleVector, v::TupleVector{M}, ::Val{L}, ::Colon) where {M,L}
    exprs = [:(a[$i] = v[$i]) for i = 1:L]
    return quote
        Base.@_propagate_inbounds_meta
        @boundscheck if M != L
            throw(DimensionMismatch("tried to assign $M-element array to length-$L destination"))
        end
        $(Expr(:block, exprs...))
    end
end

@propagate_inbounds function Base.setindex!(a::TupleVector, v, inds::TupleVector{N,Int}) where N
    _setindex!(a, v, Val(N), inds)
    return v
end

@generated function _setindex!(a::TupleVector, v, ::Val{N}, inds::TupleVector{N,Int}) where N
    exprs = [:(a[inds[$i]] = v) for i = 1:N]
    return quote
        Base.@_propagate_inbounds_meta
        similar_type(a, s)(tuple($(exprs...)))
    end
end

@generated function _setindex!(a::TupleVector, v::AbstractVector, ::Val{N}, inds::TupleVector{N,Int}) where N
    exprs = [:(a[inds[$i]] = v[$i]) for i = 1:N]
    return quote
        Base.@_propagate_inbounds_meta
        @boundscheck if length(v) != $N
            throw(DimensionMismatch("tried to assign $(length(v))-element array to length-$N destination"))
        end
        $(Expr(:block, exprs...))
    end
end

@generated function _setindex!(a::TupleVector, v::TupleVector{M}, ::Val{N}, inds::TupleVector{N,Int}) where {N,M}
    exprs = [:(a[inds[$i]] = v[$i]) for i = 1:N]
    return quote
        Base.@_propagate_inbounds_meta
        @boundscheck if M != N
            throw(DimensionMismatch("tried to assign $M-element array to length-$N destination"))
        end
        $(Expr(:block, exprs...))
    end
end

# generators

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

# conversion

(::Type{SA})(x::Tuple{Tuple{Tuple{<:Tuple}}}) where {SA <: TupleVector} =
    throw(DimensionMismatch("No precise constructor for $SA found. Val of input was $(length(x[1][1][1]))."))

@inline (::Type{SA})(x...) where {SA <: TupleVector} = SA(x)
@inline (::Type{SA})(a::TupleVector) where {SA<:TupleVector} = SA(Tuple(a))
@propagate_inbounds (::Type{SA})(a::AbstractArray) where {SA <: TupleVector} = convert(SA, a)

# this covers most conversions and "statically-sized reshapes"
#@inline Base.convert(::Type{SA}, sa::TupleVector) where {SA<:TupleVector} = SA(Tuple(sa))
#@inline Base.convert(::Type{SA}, sa::SA) where {SA<:TupleVector} = sa
@inline Base.convert(::Type{SA}, x::Tuple) where {SA<:TupleVector} = SA(x) # convert -> constructor. Hopefully no loops...

# support conversion to AbstractArray
Base.AbstractArray{T}(sa::TupleVector{N,T}) where {N,T} = sa
Base.AbstractArray{T,1}(sa::TupleVector{N,T}) where {N,T} = sa
Base.AbstractArray{T}(sa::TupleVector{N,U}) where {N,T,U} = similar_type(typeof(sa),T,Val(N))(sa)
Base.AbstractArray{T,1}(sa::TupleVector{N,U}) where {N,T,U} = similar_type(typeof(sa),T,Val(N))(sa)

# Constructing a Tuple from a TupleVector
@inline Base.Tuple(a::TupleVector{N}) where N = unroll_tuple(a, Val(N))

@noinline function dimension_mismatch_fail(SA::Type, a::AbstractArray)
    throw(DimensionMismatch("expected input array of length $(length(SA)), got length $(length(a))"))
end

@propagate_inbounds function Base.convert(::Type{SA}, a::AbstractArray) where {SA<:TupleVector{N}} where N
    @boundscheck if length(a) != length(SA)
        dimension_mismatch_fail(SA, a)
    end

    return _convert(SA, a, Val(N))
end

@inline _convert(SA, a, l::Val) = SA(unroll_tuple(a, l))
@inline _convert(SA::Type{<:TupleVector{N,T}}, a, ::Val{0}) where {N,T} = similar_type(SA, T)(())
@inline _convert(SA, a, ::Val{0}) = similar_type(SA, eltype(a))(())

@generated function unroll_tuple(a::AbstractArray, ::Val{N}) where N
    exprs = [:(a[$j]) for j = 1:N]
    quote
        @_inline_meta
        @inbounds return $(Expr(:tuple, exprs...))
    end
end

# Cast any Tuple to an TupleN{T}
@inline convert_ntuple(::Type{T},d::T) where {T} = T # For zero-dimensional arrays
@inline convert_ntuple(::Type{T},d::NTuple{N,T}) where {N,T} = d
@generated function convert_ntuple(::Type{T}, d::NTuple{N,Any}) where {N,T}
    exprs = ntuple(i -> :(convert(T, d[$i])), Val(N))
    return quote
        @_inline_meta
        $(Expr(:tuple, exprs...))
    end
end

# Base gives up on tuples for promote_eltype... (TODO can we improve Base?)
@generated function promote_tuple_eltype(::Union{T,Type{T}}) where T <: Tuple
    t = Union{}
    for i = 1:length(T.parameters)
        tmp = T.parameters[i]
        if tmp <: Vararg
            tmp = tmp.parameters[1]
        end
        t = :(promote_type($t, $tmp))
    end
    return quote
        @_inline_meta
        $t
    end
end

# Diff is slightly different
@inline LinearAlgebra.diff(a::TupleVector{N}; dims=Val(1)) where N = _diff(Val(N),a,dims)

@inline function _diff(sz::Val, a::TupleVector, D::Int)
    _diff(sz,a,Val(D))
end
@generated function _diff(::Val{N}, a::TupleVector, ::Val{1}) where N
    Snew = N-1
    exprs = Array{Expr}(undef, Snew)
    for i1 = Base.product(1:Snew)
        i2 = copy([i1...])
        i2[1] = i1[1] + 1
        exprs[i1...] = :(a[$(i2...)] - a[$(i1...)])
    end
    return quote
        @_inline_meta
        elements = tuple($(exprs...))
        @inbounds return similar_type(a, eltype(elements), Val($Snew))(elements)
    end
end

# Values

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

@inline Values(a::TupleVector{N}) where N = Values{N}(Tuple(a))
@propagate_inbounds Base.getindex(v::Values, i::Int) = v.v[i]
@inline Tuple(v::Values) = v.v
Base.dataids(::Values) = ()

# See #53
Base.cconvert(::Type{Ptr{T}}, a::Values) where {T} = Base.RefValue(a)
Base.unsafe_convert(::Type{Ptr{T}}, a::Base.RefValue{SA}) where {N,T,SA<:Values{N,T}} = Ptr{T}(Base.unsafe_convert(Ptr{Values{N,T}}, a))

@inline Values(x::NTuple{N,Any}) where N = Values{N}(x)
@inline Values{N}(x::NTuple{N,T}) where {N,T} = Values{N,T}(x)
@inline Values{N}(x::T) where {N,T<:Tuple} = Values{N,promote_tuple_eltype(T)}(x)

# Some more advanced constructor-like functions
@pure @inline Base.zeros(::Type{Values{N}}) where N = zeros(Values{N,Float64})
@pure @inline Base.ones(::Type{Values{N}}) where N = ones(Values{N,Float64})

# Converting a CartesianIndex to an SVector
Base.convert(::Type{Values}, I::CartesianIndex) = Values(I.I)
Base.convert(::Type{Values{N}}, I::CartesianIndex{N}) where {N} = Values{N}(I.I)
Base.convert(::Type{Values{N,T}}, I::CartesianIndex{N}) where {N,T} = Values{N,T}(I.I)

@pure Base.promote_rule(::Type{Values{N,T}}, ::Type{CartesianIndex{N}}) where {N,T} = Values{N,promote_type(T,Int)}

# Variables

mutable struct Variables{N,T} <: TupleVector{N,T}
    v::NTuple{N,T}
    Variables{N,T}(x::NTuple{N,T}) where {N,T} = new{N,T}(x)
    Variables{N,T}(x::NTuple{N,Any}) where {N,T} = new{N,T}(convert_ntuple(T, x))
    Variables{N,T}(::UndefInitializer) where {N,T} = new{N,T}()
end

@inline Variables(a::TupleVector{N}) where N = Variables{N}(Tuple(a))
@generated function (::Type{Variables{N,T}})(x::Tuple) where {N,T}
    return quote
        $(Expr(:meta, :inline))
        Variables{N,T}(x)
    end
end
@generated function (::Type{Variables{N}})(x::T) where {N,T<:Tuple}
    return quote
        $(Expr(:meta, :inline))
        Variables{N,promote_tuple_eltype(T)}(x)
    end
end

@propagate_inbounds function Base.getindex(v::Variables, i::Int)
    @boundscheck checkbounds(v,i)
    T = eltype(v)
    if isbitstype(T)
        return GC.@preserve v unsafe_load(Base.unsafe_convert(Ptr{T}, pointer_from_objref(v)), i)
    end
    v.v[i]
end
@propagate_inbounds function Base.setindex!(v::Variables, val, i::Int)
    @boundscheck checkbounds(v,i)
    T = eltype(v)
    if isbitstype(T)
        GC.@preserve v unsafe_store!(Base.unsafe_convert(Ptr{T}, pointer_from_objref(v)), convert(T, val), i)
    else
        # This one is unsafe (#27)
        # unsafe_store!(Base.unsafe_convert(Ptr{Ptr{Nothing}}, pointer_from_objref(v.data)), pointer_from_objref(val), i)
        error("setindex!() with non-isbitstype eltype is not supported by TupleVectors. Consider using FixedVector.")
    end
    return val
end

@inline Base.Tuple(v::Variables) = v.v
Base.dataids(ma::Variables) = (UInt(pointer(ma)),)

@inline function Base.unsafe_convert(::Type{Ptr{T}}, a::Variables{N,T}) where {N,T}
    Base.unsafe_convert(Ptr{T}, pointer_from_objref(a))
end

function Base.promote_rule(::Type{<:Variables{N,T}}, ::Type{<:Variables{N,U}}) where {N,T,U}
    Variables{N,promote_type(T,U)}
end

@inline Variables(x::NTuple{N,Any}) where N = Variables{N}(x)
@inline Variables{N}(x::NTuple{N,T}) where {N,T} = Variables{N,T}(x)
@inline Variables{N}(x::NTuple{N,Any}) where N = Variables{N, promote_tuple_eltype(typeof(x))}(x)

# Some more advanced constructor-like functions
@inline Base.zeros(::Type{Variables{N}}) where N = zeros(Variables{N,Float64})
@inline Base.ones(::Type{Variables{N}}) where N = ones(Variables{N,Float64})

# FixedVector

struct FixedVector{N,T} <: TupleVector{N,T}
    v::Vector{T}
    function FixedVector{N,T}(a::Vector) where {N,T}
        if length(a) != N
            throw(DimensionMismatch("Dimensions $(size(a)) don't match static size $S"))
        end
        new{N,T}(a)
    end
    function FixedVector{N,T}(::UndefInitializer) where {N,T}
        new{N,T}(Vector{T}(undef,N))
    end
end

@inline FixedVector{N}(a::Vector{T}) where {N,T} = FixedVector{N,T}(a)

@generated function FixedVector{N,T}(x::NTuple{N,Any}) where {N,T}
    exprs = [:(a[$i] = x[$i]) for i = 1:N]
    return quote
        $(Expr(:meta, :inline))
        a = FixedVector{N,T}(undef)
        @inbounds $(Expr(:block, exprs...))
        return a
    end
end

@inline FixedVector{N,T}(x::Tuple) where {N,T} = FixedVector{N,T}(x)
@inline FixedVector{N}(x::NTuple{N,T}) where {N,T} = FixedVector{N,T}(x)

# Overide some problematic default behaviour
@inline Base.convert(::Type{SA}, sa::FixedVector) where {SA<:FixedVector} = SA(sa.v)
@inline Base.convert(::Type{SA}, sa::SA) where {SA<:FixedVector} = sa

# Back to Array (unfortunately need both convert and construct to overide other methods)
@inline Base.Array(sa::FixedVector) = Vector(sa.v)
@inline Base.Array{T}(sa::FixedVector{N,T}) where {N,T} = Vector{T}(sa.v)
@inline Base.Array{T,1}(sa::FixedVector{N,T}) where {N,T} = Vector{T}(sa.v)

@inline Base.convert(::Type{Array}, sa::FixedVector) = sa.v
@inline Base.convert(::Type{Array{T}}, sa::FixedVector{N,T}) where {N,T} = sa.v
@inline Base.convert(::Type{Array{T,1}}, sa::FixedVector{N,T}) where {N,T} = sa.v

@propagate_inbounds Base.getindex(a::FixedVector, i::Int) = getindex(a.v, i)
@propagate_inbounds Base.setindex!(a::FixedVector, v, i::Int) = setindex!(a.v, v, i)

Base.dataids(sa::FixedVector) = Base.dataids(sa.v)

function Base.promote_rule(::Type{<:FixedVector{N,T}}, ::Type{<:FixedVector{N,U}}) where {N,T,U}
    FixedVector{N,promote_type(T,U)}
end

# SOneTo

struct SOneTo{n} <: AbstractUnitRange{Int} end

@pure SOneTo(n::Int) = SOneTo{n}()
function SOneTo{n}(r::AbstractUnitRange) where n
    ((first(r) == 1) & (last(r) == n)) && return SOneTo{n}()

    errmsg(r) = throw(DimensionMismatch("$r is inconsistent with SOneTo{$n}")) # avoid GC frame
    errmsg(r)
end
Base.Tuple(::SOneTo{N}) where N = ntuple(identity, Val(N))

@pure Base.axes(s::SOneTo) = (s,)
@pure Base.size(s::SOneTo{n}) where n = (n,)
@pure Base.length(s::SOneTo{n}) where n = n

# The axes of a Slice'd SOneTo use the SOneTo itself
Base.axes(S::Base.Slice{<:SOneTo}) = (S.indices,)
Base.unsafe_indices(S::Base.Slice{<:SOneTo}) = (S.indices,)
Base.axes1(S::Base.Slice{<:SOneTo}) = S.indices

@propagate_inbounds function Base.getindex(s::SOneTo, i::Int)
    @boundscheck checkbounds(s, i)
    return i
end
@propagate_inbounds function Base.getindex(s::SOneTo, s2::SOneTo)
    @boundscheck checkbounds(s, s2)
    return s2
end

@pure Base.first(::SOneTo) = 1
@pure Base.last(::SOneTo{n}) where n = n::Int
@pure Base.iterate(::SOneTo{n}) where n = n::Int < 1 ? nothing : (1, 1)
@pure function Base.iterate(::SOneTo{n}, s::Int) where {n}
    if s < n::Int
        s2 = s + 1
        return (s2, s2)
    else
        return nothing
    end
end

function Base.getproperty(::SOneTo{n}, s::Symbol) where {n}
    if s === :start
        return 1
    elseif s === :stop
        return n::Int
    else
        error("type SOneTo has no property $s")
    end
end

Base.show(io::IO, ::SOneTo{n}) where {n} = print(io, "SOneTo(", n::Int, ")")
Base.@pure function Base.checkindex(::Type{Bool}, ::SOneTo{n1}, ::SOneTo{n2}) where {n1, n2}
    return n1::Int >= n2::Int
end

Base.promote_rule(a::Type{Base.OneTo{T}}, ::Type{SOneTo{n}}) where {T,n} =
    Base.OneTo{promote_type(T, Int)}

# Broadcast

import Base.Broadcast: BroadcastStyle, AbstractArrayStyle, Broadcasted, DefaultArrayStyle, materialize!
import Base.Broadcast: _bcs1  # for SOneTo axis information
using Base.Broadcast: _bcsm
# Add a new BroadcastStyle for TupleVectors, derived from AbstractArrayStyle
# A constructor that changes the style parameter N (array dimension) is also required
struct TupleVectorStyle{N} <: AbstractArrayStyle{N} end
TupleVectorStyle{M}(::Val{N}) where {M,N} = TupleVectorStyle{N}()
BroadcastStyle(::Type{<:TupleVector{N,<:Any}}) where N = TupleVectorStyle{N}()
BroadcastStyle(::Type{<:LinearAlgebra.Transpose{<:Any,<:TupleVector{N,<:Any}}}) where N = TupleVectorStyle{N}()
BroadcastStyle(::Type{<:LinearAlgebra.Adjoint{<:Any,<:TupleVector{N,<:Any}}}) where N = TupleVectorStyle{N}()
# Precedence rules
BroadcastStyle(::TupleVectorStyle{M}, ::DefaultArrayStyle{N}) where {M,N} =
    DefaultArrayStyle(Val(max(M, N)))
BroadcastStyle(::TupleVectorStyle{M}, ::DefaultArrayStyle{0}) where {M} =
    TupleVectorStyle{M}()
# copy overload
@inline function Base.copy(B::Broadcasted{TupleVectorStyle{M}}) where M
    flat = Broadcast.flatten(B); as = flat.args; f = flat.f
    argsizes = broadcast_sizes(as...)
    destsize = combine_sizes(argsizes)
    _broadcast(f, destsize, argsizes, as...)
end
# copyto! overloads
@inline Base.copyto!(dest, B::Broadcasted{<:TupleVectorStyle}) = _copyto!(dest, B)
@inline Base.copyto!(dest::AbstractArray, B::Broadcasted{<:TupleVectorStyle}) = _copyto!(dest, B)
@inline function _copyto!(dest, B::Broadcasted{TupleVectorStyle{M}}) where M
    flat = Broadcast.flatten(B); as = flat.args; f = flat.f
    argsizes = broadcast_sizes(as...)
    destsize = combine_sizes((Val(M), argsizes...))
    #=if Val(destsize) === Val{Dynamic()}()
        # destination dimension cannot be determined statically; fall back to generic broadcast!
        return copyto!(dest, convert(Broadcasted{DefaultArrayStyle{M}}, B))
    end=#
    _broadcast!(f, destsize, dest, argsizes, as...)
end

# Resolving priority between dynamic and static axes
_bcs1(a::SOneTo, b::SOneTo) = _bcsm(b, a) ? b : (_bcsm(a, b) ? a : throw(DimensionMismatch("arrays could not be broadcast to a common size")))
_bcs1(a::SOneTo, b::Base.OneTo) = _bcs1(Base.OneTo(a), b)
_bcs1(a::Base.OneTo, b::SOneTo) = _bcs1(a, Base.OneTo(b))

###################################################
## Internal broadcast machinery for TupleVectors ##
###################################################

broadcast_indices(A::TupleVector) = indices(A)

# TODO: just use map(broadcast_size, as)?
@inline broadcast_sizes(a, as...) = (broadcast_size(a), broadcast_sizes(as...)...)
@inline broadcast_sizes() = ()
@inline broadcast_size(a) = Val(0)
@inline broadcast_size(a::AbstractVector) = Val(length(a))
@inline broadcast_size(a::NTuple{N}) where N = Val(N)

function broadcasted_index(oldsize, newindex)
    index = ones(Int, length(oldsize))
    for i = 1:length(oldsize)
        if oldsize[i] != 1
            index[i] = newindex[i]
        end
    end
    return LinearIndices((oldsize,))[index...]
end

# similar to Base.Broadcast.combine_indices:
@generated function combine_sizes(s::Tuple{Vararg{Val}})
    sizes = [sz.parameters[1] for sz ∈ s.parameters]
    ndims = 0
    for i = 1:length(sizes)
        ndims = max(ndims,sizes[i])
    end 
    quote
        @_inline_meta
        Val($ndims)
    end
end

scalar_getindex(x) = x
scalar_getindex(x::Ref) = x[]

@generated function _broadcast(f, ::Val{newsize}, s::Tuple{Vararg{Val}}, a...) where newsize
    first_staticarray = a[findfirst(ai -> ai <: Union{TupleVector, LinearAlgebra.Transpose{<:Any, <:TupleVector}, LinearAlgebra.Adjoint{<:Any, <:TupleVector}}, a)]
    if newsize == 0
        # Use inference to get eltype in empty case (see also comments in _map)
        eltys = [:(eltype(a[$i])) for i ∈ 1:length(a)]
        return quote
            @_inline_meta
            T = Core.Compiler.return_type(f, Tuple{$(eltys...)})
            @inbounds return similar_type($first_staticarray, T, Val(newsize))()
        end
    end
    sizes = [sz.parameters[1] for sz ∈ s.parameters]
    indices = CartesianIndices((newsize,))
    exprs = similar(indices, Expr)
    for (j, current_ind) ∈ enumerate(indices)
        exprs_vals = [
            (!(a[i] <: AbstractArray || a[i] <: Tuple) ? :(scalar_getindex(a[$i])) : :(a[$i][$(broadcasted_index(sizes[i], current_ind))]))
            for i = 1:length(sizes)
        ]
        exprs[j] = :(f($(exprs_vals...)))
    end
    return quote
        @_inline_meta
        @inbounds elements = tuple($(exprs...))
        @inbounds return similar_type($first_staticarray, eltype(elements), Val(newsize))(elements)
    end
end

####################################################
## Internal broadcast! machinery for TupleVectors ##
####################################################

@generated function _broadcast!(f, ::Val{newsize}, dest::AbstractArray, s::Tuple{Vararg{Val}}, as...) where {newsize}
    sizes = [sz.parameters[1] for sz ∈ s.parameters]
    sizes = tuple(sizes...)
    indices = CartesianIndices((newsize,))
    exprs = similar(indices, Expr)
    for (j, current_ind) ∈ enumerate(indices)
        exprs_vals = [
            (!(as[i] <: AbstractArray || as[i] <: Tuple) ? :(as[$i][]) : :(as[$i][$(broadcasted_index(sizes[i], current_ind))]))
            for i = 1:length(sizes)
        ]
        exprs[j] = :(dest[$j] = f($(exprs_vals...)))
    end
    return quote
        Base.@_propagate_inbounds_meta
        @inbounds $(Expr(:block, exprs...))
        return dest
    end
end

# Other

Base.axes(::TupleVector{N}) where N = _axes(Val(N))
@pure function _axes(::Val{sizes}) where {sizes}
    map(SOneTo, (sizes,))
end
Base.axes(rv::LinearAlgebra.Adjoint{<:Any,<:Values})   = (SOneTo(1), axes(rv.parent)...)
Base.axes(rv::LinearAlgebra.Transpose{<:Any,<:Values}) = (SOneTo(1), axes(rv.parent)...)

# Base.strides is intentionally not defined for SArray, see PR #658 for discussion
Base.strides(a::Variables) = Base.size_to_strides(1, size(a)...)
Base.strides(a::FixedVector) = strides(a.v)

similar_type(::SA) where {SA<:TupleVector} = similar_type(SA,eltype(SA))
similar_type(::Type{SA}) where {SA<:TupleVector} = similar_type(SA,eltype(SA))

similar_type(::SA,::Type{T}) where {SA<:TupleVector{N},T} where N = similar_type(SA,T,Val(N))
similar_type(::Type{SA},::Type{T}) where {SA<:TupleVector{N},T} where N = similar_type(SA,T,Val(N))

similar_type(::A,n::Val) where {A<:AbstractArray} = similar_type(A,eltype(A),n)
similar_type(::Type{A},n::Val) where {A<:AbstractArray} = similar_type(A,eltype(A),n)

similar_type(::A,::Type{T},n::Val) where {A<:AbstractArray,T} = similar_type(A,T,n)

# We should be able to deal with SOneTo axes
similar_type(s::SOneTo) = similar_type(typeof(s))
similar_type(::Type{SOneTo{n}}) where n = similar_type(SOneTo{n}, Int, Val(n))

similar_type(::A, shape::Tuple{SOneTo, Vararg{SOneTo}}) where {A<:AbstractArray} = similar_type(A, eltype(A), shape)
similar_type(::Type{A}, shape::Tuple{SOneTo, Vararg{SOneTo}}) where {A<:AbstractArray} = similar_type(A, eltype(A), shape)

similar_type(::A,::Type{T}, shape::Tuple{SOneTo, Vararg{SOneTo}}) where {A<:AbstractArray,T} = similar_type(A, T, Val(length(last.(shape))))
similar_type(::Type{A},::Type{T}, shape::Tuple{SOneTo, Vararg{SOneTo}}) where {A<:AbstractArray,T} = similar_type(A, T, Val(length(last.(shape))))

# Default types
# Generally, use SArray
similar_type(::Type{A},::Type{T},n::Val) where {A<:AbstractArray,T} = default_similar_type(T,n)
default_similar_type(::Type{T},::Val{N}) where {N,T} = Values{N,T}

similar_type(::Type{SA},::Type{T},n::Val) where {SA<:Variables,T} = mutable_similar_type(T,n)

mutable_similar_type(::Type{T},::Val{N}) where {N,T} = Variables{N,T}

similar_type(::Type{<:FixedVector},::Type{T},n::Val) where T = sizedarray_similar_type(T,n)
# Should FixedVector also be used for normal Array?
#similar_type(::Type{<:Array},::Type{T},n::Val) where T = sizedarray_similar_type(T,n)

sizedarray_similar_type(::Type{T},::Val{N}) where {N,T} = FixedVector{N,T}

Base.similar(::SA) where {SA<:TupleVector} = similar(SA,eltype(SA))
Base.similar(::Type{SA}) where {SA<:TupleVector} = similar(SA,eltype(SA))

Base.similar(::SA,::Type{T}) where {SA<:TupleVector{N},T} where N = similar(SA,T,Val(N))
Base.similar(::Type{SA},::Type{T}) where {SA<:TupleVector{N},T} where N = similar(SA,T,Val(N))

# Cases where a Val is given as the dimensions
Base.similar(::A,n::Val) where A<:AbstractArray = similar(A,eltype(A),n)
Base.similar(::Type{A},n::Val) where A<:AbstractArray = similar(A,eltype(A),n)

Base.similar(::A,::Type{T},n::Val) where {A<:AbstractArray,T} = similar(A,T,n)

# defaults to built-in mutable types
Base.similar(::Type{A},::Type{T},n::Val) where {A<:AbstractArray,T} = mutable_similar_type(T,n)(undef)

# both FixedVector and Array return FixedVector
Base.similar(::Type{SA},::Type{T},n::Val) where {SA<:FixedVector,T} = sizedarray_similar_type(T,n)(undef)
Base.similar(::Type{A},::Type{T},n::Val) where {A<:Array,T} = sizedarray_similar_type(T,n)(undef)

# Support tuples of mixtures of `SOneTo`s alongside the normal `Integer` and `OneTo` options
# by simply converting them to either a tuple of Ints or a Val, re-dispatching to either one
# of the above methods (in the case of Val) or a base fallback (in the case of Ints).
const HeterogeneousShape = Union{Integer, Base.OneTo, SOneTo}

Base.similar(A::AbstractArray, ::Type{T}, shape::Tuple{HeterogeneousShape, Vararg{HeterogeneousShape}}) where {T} = similar(A, T, homogenize_shape(shape))
Base.similar(::Type{A}, shape::Tuple{HeterogeneousShape, Vararg{HeterogeneousShape}}) where {A<:AbstractArray} = similar(A, homogenize_shape(shape))
# Use an Array for TupleVectors if we don't have a statically-known size
Base.similar(::Type{A}, shape::Tuple{Int, Vararg{Int}}) where {A<:TupleVector} = Array{eltype(A)}(undef, shape)

homogenize_shape(::Tuple{}) = ()
homogenize_shape(shape::Tuple{Vararg{SOneTo}}) = Val(prod(map(last, shape)))
homogenize_shape(shape::Tuple{Vararg{HeterogeneousShape}}) = map(last, shape)


@inline Base.copy(a::TupleVector) = typeof(a)(Tuple(a))
@inline Base.copy(a::FixedVector) = typeof(a)(copy(a.v))

@inline Base.reverse(v::Values) = typeof(v)(_reverse(v))

@generated function _reverse(v::Values{N,T}) where {N,T}
    return Expr(:tuple, (:(v[$i]) for i = N:(-1):1)...)
end

#--------------------------------------------------
# Vector space algebra

# Unary ops
@inline Base.:-(a::TupleVector) = map(-, a)
@inline Base.:-(a::TupleVector{n,<:Number}) where n = map(Base.:-, a)

# Binary ops
# Between arrays
@inline Base.:+(a::TupleVector, b::TupleVector) = map(∑, a, b)
@inline Base.:+(a::AbstractArray, b::TupleVector) = map(∑, a, b)
@inline Base.:+(a::TupleVector, b::AbstractArray) = map(∑, a, b)

@inline Base.:+(a::TupleVector{n,<:Number}, b::TupleVector{n,<:Number}) where n = map(Base.:+, a, b)
@inline Base.:+(a::AbstractArray{<:Number}, b::TupleVector{n,<:Number}) where n = map(Base.:+, a, b)
@inline Base.:+(a::TupleVector{n,<:Number}, b::AbstractArray{<:Number}) where n = map(Base.:+, a, b)

@inline Base.:-(a::TupleVector, b::TupleVector) = map(-, a, b)
@inline Base.:-(a::AbstractArray, b::TupleVector) = map(-, a, b)
@inline Base.:-(a::TupleVector, b::AbstractArray) = map(-, a, b)

@inline Base.:-(a::TupleVector{n,<:Number}, b::TupleVector{n,<:Number}) where n = map(Base.:-, a, b)
@inline Base.:-(a::AbstractArray{<:Number}, b::TupleVector{n,<:Number}) where n = map(Base.:-, a, b)
@inline Base.:-(a::TupleVector{n,<:Number}, b::AbstractArray{<:Number}) where n = map(Base.:-, a, b)

# Scalar-array
@inline Base.:*(a::Number, b::TupleVector{n,<:Number}) where n = broadcast(Base.:*, a, b)
@inline Base.:*(a::TupleVector{n,<:Number}, b::Number) where n = broadcast(Base.:*, a, b)

@inline Base.:*(a, b::TupleVector) = broadcast(∏, a, b)
@inline Base.:*(a::TupleVector, b) = broadcast(∏, a, b)

@inline Base.:/(a::TupleVector{n,<:Number}, b::Number) where n = broadcast(Base.:/, a, b)
@inline Base.:\(a::Number, b::TupleVector{n,<:Number}) where n = broadcast(Base.:\, a, b)

@inline Base.:/(a::TupleVector, b) = broadcast(/, a, b)
@inline Base.:\(a, b::TupleVector) = broadcast(\, a, b)

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
@inline LinearAlgebra.norm_sqr(v::TupleVector) = mapreduce(Base.abs2, Base.:+, v; init=zero(real(eltype(v))))

@inline LinearAlgebra.norm(a::TupleVector) = _norm(a)
@generated function _norm(a::TupleVector{S}) where S
    if S == 0
        return :(zero(real(eltype(a))))
    end
    expr = :(Base.abs2(a[1]))
    for j = 2:S
        expr = :($expr + Base.abs2(a[$j]))
    end
    return quote
        $(Expr(:meta, :inline))
        @inbounds return Base.sqrt($expr)
    end
end

_norm_p0(x) = x == 0 ? zero(x) : one(x)

@inline LinearAlgebra.norm(a::TupleVector, p::Real) = _norm(a, p)
@generated function _norm(a::TupleVector{S,T}, p::Real) where {S,T}
    if S == 0
        return :(zero(real(eltype(a))))
    end
    fun = T<:Number ? :(Base.abs) : :abs
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
            return mapreduce(Base.abs, max, a)
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
