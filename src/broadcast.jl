
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
