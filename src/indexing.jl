
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
