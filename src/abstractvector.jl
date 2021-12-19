
# This file is adapted from JuliaArrays/StaticArrays.jl License is MIT:
# https://github.com/JuliaArrays/StaticArrays.jl/blob/master/LICENSE.md

Base.axes(::TupleVector{N}) where N = _axes(Val(N))
@pure function _axes(::Val{sizes}) where {sizes}
    map(SOneTo, (sizes,))
end
Base.axes(rv::LinearAlgebra.Adjoint{<:Any,<:Values})   = (SOneTo(1), axes(rv.parent)...)
Base.axes(rv::LinearAlgebra.Transpose{<:Any,<:Values}) = (SOneTo(1), axes(rv.parent)...)

# Base.strides is intentionally not defined for SArray, see PR #658 for discussion
Base.strides(a::Variables) = Base.size_to_strides(1, size(a)...)
Base.strides(a::FixedVector) = strides(a.v)

Base.IndexStyle(::Type{T}) where {T<:TupleVector} = IndexLinear()

similar_type(::SA) where {SA<:TupleVector} = similar_type(SA,eltype(SA))
similar_type(::Type{SA}) where {SA<:TupleVector} = similar_type(SA,eltype(SA))

similar_type(::SA,::Type{T}) where {SA<:TupleVector{N},T} where N = similar_type(SA,T,Val(N))
similar_type(::Type{SA},::Type{T}) where {SA<:TupleVector{N},T} where N = similar_type(SA,T,Val(N))

similar_type(::A,n::Val) where {A<:AbstractArray} = similar_type(A,eltype(A),n)
similar_type(::Type{A},n::Val) where {A<:AbstractArray} = similar_type(A,eltype(A),n)

similar_type(::A,::Type{T},n::Val) where {A<:AbstractArray,T} = similar_type(A,T,n)

# We should be able to deal with SOneTo axes
@pure similar_type(s::SOneTo) = similar_type(typeof(s))
@pure similar_type(::Type{SOneTo{n}}) where n = similar_type(SOneTo{n}, Int, Val(n))

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
# Concatenation
@inline Base.vcat(a::TupleVectorLike) = a
@inline Base.vcat(a::TupleVectorLike{N}, b::TupleVectorLike{M}) where {N,M} = _vcat(Val(N), Val(M), a, b)
@inline Base.vcat(a::TupleVectorLike, b::TupleVectorLike, c::TupleVectorLike...) = vcat(vcat(a,b), vcat(c...))

@generated function _vcat(::Val{Sa}, ::Val{Sb}, a::TupleVectorLike, b::TupleVectorLike) where {Sa, Sb}

    # TODO cleanup?
    Snew = Sa + Sb
    exprs = vcat([:(a[$i]) for i = 1:Sa],
                 [:(b[$i]) for i = 1:Sb])
    return quote
        @_inline_meta
        @inbounds return similar_type(a, promote_type(eltype(a), eltype(b)), Val($Snew))(tuple($(exprs...)))
    end
end
