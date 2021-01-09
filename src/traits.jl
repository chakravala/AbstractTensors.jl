
"""
Return either the statically known Val() or runtime length()
"""
@inline _size(a) = Val(length(a))
@inline _size(a::TupleVector{n}) where n = Val(n)

# Return static array from a set of arrays
@inline _first_static(a1::TupleVector, as...) = a1
@inline _first_static(a1, as...) = _first_static(as...)
@inline _first_static() = throw(ArgumentError("No TupleVector found in argument list"))

"""
Returns the common Val of the inputs (or else throws a DimensionMismatch)
"""
@inline function same_size(as...)
    s = Val(length(_first_static(as...)))
    _sizes_match(s, as...) || _throw_size_mismatch(as...)
    s
end
@inline _sizes_match(s::Val, a1, as...) = ((s == _size(a1)) ? _sizes_match(s, as...) : false)
@inline _sizes_match(s::Val) = true
@noinline function _throw_size_mismatch(as...)
    throw(DimensionMismatch("Sizes $(map(_size, as)) of input arrays do not match"))
end
