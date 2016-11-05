

typealias ExperienceLibrary Dict{Strategy, Vector{Experience}}

type ExperienceStore
    library::ExperienceLibrary
    previous_epochs::Vector{ExperienceLibrary}
    batch_size::Int
    max_batches::Int
    epochs_to_keep::Int
    epoch::Int
end

function ExperienceStore()
    library = ExperienceLibrary()
    prev = Vector{ExperienceLibrary}()
    return ExperienceStore(library, prev, 500, 100, 4, 0)
end

const es = ExperienceStore()

function experience_list(store::ExperienceStore, strategy::Strategy)
    if !haskey(store.library, strategy)
        list = Vector{Experience}()
        store.library[strategy] = list
    else
        list = store.library[strategy]
    end
    return list
end

function add_experience!(store::ExperienceStore, strategy::Strategy, experience::Experience)
    list = experience_list(store, strategy)
    push!(list, experience)
end

function add_experience!(store::ExperienceStore, strategy::Strategy, experiences::Vector{Experience})
    list = experience_list(store, strategy)
    append!(list, experiences)
end

add_experience!(strategy::Strategy, experience::Experience) = add_experience!(es, strategy, experience)
add_experience!(strategy::Strategy, targets::OutputVector, inputs::InputVector, action::Any) = add_experience!(strategy, GenericExperience(targets, inputs, action))
add_experience!(strategy::Strategy, experiences::Vector{Experience}) = add_experience!(es, strategy, experiences)

function last_experience(store::ExperienceStore, strategy::Strategy)
    if !has_key(store.library, strategy) || length(store.library[strategy]) == 0
        return nothing
    end
    return store.library[strategy][end]
end

strategies(store::ExperienceStore) = keys(store.library)

function backpropagate!(nn::NN, experience::Experience)
    o = outputs(nn, experience)
    backpropagate!(nn, deltas(nn, experience, o[end]), o)
end

function training_inputs(strategy::Strategy, experience_list::Vector{Experience}, input_size::Int, batch_size::Int)
    batch_inputs = Matrix{Float}(input_size, batch_size)
    batch_experiences = Vector{Experience}()
    for i in 1:batch_size
        experience = pickOne(experience_list)
        batch_inputs[:, i] = inputs(experience)
        push!(batch_experiences, experience)
    end
    return batch_inputs, batch_experiences
end

function training_deltas(strategy::Strategy, batch_experiences::Vector{Experience}, batch_outputs::OutputMatrixList)
    batch_deltas = zeros(batch_outputs[end])
    rows, columns = size(batch_deltas)
    for i in 1:rows
        exp = batch_experiences[i]
        out_vec = vec(batch_outputs[end][i, :])
        batch_deltas[i, :] = deltas(strategy.nn, exp, out_vec)
    end
    return batch_deltas
end

function all_experiences(strategy::Strategy, store::ExperienceStore)
    experience_list = copy(store.library[strategy])
    for previous_epoch in store.previous_epochs
        if haskey(previous_epoch, strategy)
            append!(experience_list, previous_epoch[strategy])
        end
    end
    return experience_list
end

function train_strategy!(strategy::Strategy, store::ExperienceStore)
    println("[learning $(typeof(strategy))]")
    experience_list = all_experiences(strategy, store)
    nn = strategy.nn
    for batch in 1:store.max_batches
        if (batch % 5) == 0 
            println("Processing batch $batch")
        end
        batch_inputs, batch_experiences = training_inputs(strategy, experience_list, n_inputs(nn), store.batch_size)
        batch_outputs = feedforward(nn, batch_inputs)
        batch_deltas  = training_deltas(strategy, batch_experiences, batch_outputs)
        backpropagate!(nn, batch_deltas, batch_outputs)
        update!(nn)
    end
end

function learn_from_experience!(store::ExperienceStore)
    store.epoch += 1
    heading("EPOCH $(store.epoch)")
    for strategy in strategies(store)
      train_strategy!(strategy, store)
    end
    push!(store.previous_epochs, store.library)
    if(length(store.previous_epochs) > store.epochs_to_keep)
        shift!(store.previous_epochs)
    end
    store.library = ExperienceLibrary()

end

learn_from_experience!() = learn_from_experience!(es)


