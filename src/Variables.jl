
# This file is adapted from JuliaArrays/StaticArrays.jl License is MIT:
# https://github.com/JuliaArrays/StaticArrays.jl/blob/master/LICENSE.md

# MArray.jl

mutable struct Variables{N,T} <: TupleVector{N,T}
    v::NTuple{N,T}
    Variables{N,T}(x::NTuple{N,T}) where {N,T} = new{N,T}(x)
    Variables{N,T}(x::NTuple{N,Any}) where {N,T} = new{N,T}(convert_ntuple(T, x))
    Variables{N,T}(::UndefInitializer) where {N,T} = new{N,T}()
end

@inline Variables(a::TupleVector{N,T}) where {N,T} = Variables{N,T}(Tuple(a))
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
    return v
end

@inline Base.Tuple(v::Variables) = v.v
Base.dataids(ma::Variables) = (UInt(pointer(ma)),)

@inline function Base.unsafe_convert(::Type{Ptr{T}}, a::Variables{N,T}) where {N,T}
    Base.unsafe_convert(Ptr{T}, pointer_from_objref(a))
end

function Base.promote_rule(::Type{<:Variables{N,T}}, ::Type{<:Variables{N,U}}) where {N,T,U}
    Variables{N,promote_type(T,U)}
end

function Base.view(
    a::Variables{N},
    index::Union{Integer, Colon, TupleVector, Base.Slice, SOneTo},
) where N
    view_from_invoke = invoke(view, Tuple{AbstractVector, typeof(index)}, a, index)
    return FixedVector{new_out_size(Val(N), index)}(view_from_invoke)
end

Base.elsize(::Type{<:Variables{S,T}}) where {S,T} = sizeof(T)

# MVector.jl

@inline Variables(x::Values{N,T}) where {N,T} = Variables{N,T}(x)
@inline Variables(x::NTuple{N,Any}) where N = Variables{N}(x)
@inline Variables{N}(x::NTuple{N,T}) where {N,T} = Variables{N,T}(x)
@inline Variables{N}(x::NTuple{N,Any}) where N = Variables{N, promote_tuple_eltype(typeof(x))}(x)

# Some more advanced constructor-like functions
@inline Base.zeros(::Type{Variables{N}}) where N = zeros(Variables{N,Float64})
@inline Base.ones(::Type{Variables{N}}) where N = ones(Variables{N,Float64})
