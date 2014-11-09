module DeepReshapes

export structure
export deep_length
export deep_reshape
export flatten
export pack

Scalar = Union(Number, String, Nothing)

structure{T <: Scalar}(x::T; typed::Bool = true) = typed ? T : ()

structure(x::(); typed::Bool = true) = nothing

structure(x::(Any, Any...); typed::Bool = true) =
  tuple([structure(e, typed = typed) for e in x]...)

structure{T <: Scalar}(x::Array{T}; typed::Bool = true) =
  typed ? tuple(T, Base.size(x)...) : Base.size(x)

structure(x::Array; typed::Bool = true) =
  map!(e -> structure(e, typed = typed), similar(x, Any), x)

deep_length(x) = structure_length(structure(x, typed = false))

structure_length(_::Nothing) = 0
structure_length(_::()) = 1
structure_length(x::(Integer...)) = prod(x)
structure_length(x) = sum(structure_length, x)

deep_reshape(x, structure) =
  consume_stucture(@task(produce_scalars(x)), structure)

deep_reshape(x, structure...) = deep_reshape(x, structure)

produce_scalars(x::Scalar) = produce(x)
produce_scalars(x) = for e in x; produce_scalars(e) end
  
function consume_stucture(
  producer::Task,
  MemberType::DataType,
  dimensions::Integer...
)
  if dimensions == ()
    convert(MemberType, consume(producer))
  else
    result = Array(MemberType, dimensions...)
    copy_object(i...) = result[i...] = consume(producer)
    cartesianmap(copy_object, dimensions)
  
    result
  end
end

consume_stucture(producer::Task, structure::(Integer...)) =
  consume_stucture(producer, Any, structure...)

consume_stucture(producer::Task, structure::(DataType, Integer...)) =
  consume_stucture(producer, structure...)

consume_stucture(producer::Task, structure::Tuple) =
  tuple([consume_stucture(producer::Task, element) for element in structure]...)

function consume_stucture(producer::Task, structure::Array)
  result = Array(Any, size(structure)...)
  
  copy_into_result(i...) = result[i...] =
    consume_stucture(producer, structure[i...])
  cartesianmap(copy_into_result, size(structure))
  
  result
end

consume_stucture(producer::Task, _::Nothing) = ()

flatten(x, MemberType::DataType) = deep_reshape(x, MemberType, deep_length(x))
flatten(x) = flatten(x, Any)

function pack(MemberType::DataType, data...)
  s = structure(data, typed = false)
  l = structure_length(s)
  (deep_reshape(data, MemberType, l), s)
end

end # module
