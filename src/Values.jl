
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

@inline Values(a::TupleVector{N}) where N = Values{N}(Tuple(a))
@propagate_inbounds Base.getindex(v::Values, i::Int) = v.v[i]
@inline Tuple(v::Values) = v.v
Base.dataids(::Values) = ()

# See #53
Base.cconvert(::Type{Ptr{T}}, a::Values) where {T} = Base.RefValue(a)
Base.unsafe_convert(::Type{Ptr{T}}, a::Base.RefValue{SA}) where {N,T,SA<:Values{N,T}} = Ptr{T}(Base.unsafe_convert(Ptr{Values{N,T}}, a))

# SVector.jl

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

