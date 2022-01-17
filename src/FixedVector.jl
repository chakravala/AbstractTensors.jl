
# This file is adapted from JuliaArrays/StaticArrays.jl License is MIT:
# https://github.com/JuliaArrays/StaticArrays.jl/blob/master/LICENSE.md

require_one_based_indexing(A...) = !Base.has_offset_axes(A...) ||
    throw(ArgumentError("offset arrays are not supported but got an array with index other than 1"))

struct FixedVector{N,T,TData<:AbstractVector{T}} <: TupleVector{N,T}
    v::TData
    function FixedVector{N,T,TData}(a::TData) where {N,T,TData<:AbstractVector{T}}
        require_one_based_indexing(a)
        if length(a) != N
            throw(DimensionMismatch("Dimensions $(length(a)) don't match static size $N"))
        end
        new{N,T,TData}(a)
    end
    function FixedVector{N,T,TData}(::UndefInitializer) where {N,T,TData<:AbstractVector{T}}
        new{N,T,TData}(TData(undef,N))
    end
end

@inline function FixedVector{N,T}(a::TData) where {N,T,TData<:AbstractVector{T}}
    return FixedVector{N,T,TData}(a)
end
@inline function FixedVector{N}(a::TData) where {N,T,TData<:AbstractVector{T}}
    return FixedVector{N,T,TData}(a)
end
@inline FixedVector{N,T}(::UndefInitializer) where {N,T} = FixedVector{N,T,Vector{T}}(undef)
@generated function (::Type{FixedVector{N,T,TData}})(x::NTuple{N,Any}) where {N,T,TData<:AbstractVector{T}}
    exprs = [:(a[$i] = x[$i]) for i = 1:N]
    return quote
        $(Expr(:meta, :inline))
        a = FixedVector{N,T}(undef)
        @inbounds $(Expr(:block, exprs...))
        return a
    end
end

@inline FixedVector{N,T}(x::Tuple) where {N,T} = FixedVector{N,T,Vector{T}}(x)
@inline FixedVector{N}(x::NTuple{N,T}) where {N,T} = FixedVector{N,T}(x)

# Overide some problematic default behaviour
@inline Base.convert(::Type{TV}, tv::FixedVector) where {TV<:FixedVector} = TV(tv.v)
@inline Base.convert(::Type{TV}, tv::TV) where {TV<:FixedVector} = tv

# Back to Vector (unfortunately need both convert and construct to overide other methods)
@inline Base.Vector(tv::FixedVector) = Vector(tv.v)
@inline Base.Vector{T}(tv::FixedVector{N,T}) where {N,T} = Vector{T}(tv.v)

@inline function Base.convert(::Type{Vector}, tv::FixedVector{N}) where N
    return Vector(tv.v)
end
@inline function Base.convert(::Type{Vector}, tv::FixedVector{N,T,Vector{T}}) where {N,T}
    return tv.v
end
@inline function Base.convert(::Type{Vector{T}}, tv::FixedVector{N,T}) where {N,T}
    return Vector(tv.v)
end
@inline function Base.convert(::Type{Vector{T}}, tv::FixedVector{N,T,Vector{T}}) where {N,T}
    return tv.v
end

@propagate_inbounds Base.getindex(a::FixedVector, i::Int) = getindex(a.v, i)
@propagate_inbounds Base.setindex!(a::FixedVector, v, i::Int) = setindex!(a.v, v, i)

Base.parent(tv::FixedVector) = tv.v

Base.pointer(tv::FixedVector) = pointer(tv.v)

Base.unsafe_convert(::Type{Ptr{T}}, tv::FixedVector) where T = Base.unsafe_convert(Ptr{T}, tv.v)
Base.elsize(::Type{FixedVector{S,T}}) where {S,T} = Base.elsize(A)

FixedVector(a::TupleVector{N,T}) where {N,T} = FixedVector{N,T}(a)

Base.dataids(sa::FixedVector) = Base.dataids(sa.v)

function Base.promote_rule(
    ::Type{FixedVector{N,T,TDataA}},
    ::Type{FixedVector{N,U,TDataB}},
) where {N,T,U,TDataA,TDataB}
    return FixedVector{N, promote_type(T, U), promote_type(TDataA, TDataB)}
end

function promote_rule(::Type{FixedVector{N,T}},::Type{FixedVector{N,U}}) where {N,T,U}
    return FixedVector{N, promote_type(T, U)}
end

### Code that makes views of statically sized arrays also statically sized (where possible)

# Note, _get_tuple_vector_length is used in a generated function so it's strictly internal and can't be extended
_get_tuple_vector_length(::Type{<:TupleVector{N}}) where N = N

@generated function new_out_size(::Type{Val{N}}, ind) where N
    if ind <: Integer
        Tuple{1} # dimension is fixed
    elseif ind <: TupleVector
        Tuple{_get_tuple_vector_length(ind)}
    elseif ind == Colon || ind <: Base.Slice
        Tuple{N}
    elseif ind <: SOneTo
        Tuple{ind.parameters[1]}
    else
        error("Unknown index type: $ind")
    end
end

@generated new_out_size(::Type{Val{N}}, ::Colon) where N = Tuple{N}

function Base.view(
    a::FixedVector{N},
    index::Union{Integer, Colon, TupleVector, Base.Slice, SOneTo},
) where N
    return FixedVector{new_out_size(Val(N), index)}(view(a.v, index))
end
