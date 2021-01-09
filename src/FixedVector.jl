
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
