
#   This file is part of AbstractTensors.jl. It is licensed under the GPL license
#   Grassmann Copyright (C) 2019 Michael Reed

module AbstractTensors

# universal root Tensor type

abstract type TensorAlgebra{V} end

# V, VectorSpace produced by DirectSum

import DirectSum: vectorspace
import LinearAlgebra: dot, cross, UniformScaling

# parameters accessible from anywhere

Base.@pure vectorspace(::T) where T<:TensorAlgebra{V} where V = V

# universal vector space interopability

@inline interop(op::Function,a::A,b::B) where {A<:TensorAlgebra{V},B<:TensorAlgebra{V}} where V = op(a,b)

# ^^ identity ^^ | vv union vv #

@inline function interop(op::Function,a::A,b::B) where {A<:TensorAlgebra{V},B<:TensorAlgebra{W}} where {V,W}
    VW = V ∪ W
    return op(VW(a),VW(b))
end

# abstract tensor form evaluation

@inline interform(a::A,b::B) where {A<:TensorAlgebra{V},B<:TensorAlgebra{V}} where V = a(b)
@inline function interform(a::A,b::B) where {A<:TensorAlgebra{V},B<:TensorAlgebra{W}} where {V,W}
    VW = V ∪ W
    return VW(a)(VW(b))
end

# extended compatibility interface

export interop, TensorAlgebra, interform, ⊗

# some shared presets

for op ∈ (:(Base.:+),:(Base.:-),:(Base.:*),:⊗,:dot,:cross,:(Base.:(==)))
    @eval begin
        @inline $op(a::A,b::B) where {A<:TensorAlgebra,B<:TensorAlgebra} = interop($op,a,b)
        @inline $op(a::A,b::UniformScaling) where A<:TensorAlgebra{V} where V = $op(a,V(b))
        @inline $op(a::UniformScaling,b::B) where B<:TensorAlgebra{V} where V = $op(V(a),b)
    end
end

end # module
