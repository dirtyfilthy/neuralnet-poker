abstract Strategy


@enum ACTIONTYPE fold=0 call=1 raise=2

type Action
    action::ACTIONTYPE
    amount::Int
end


function new_game(s::Strategy)

end

function new_hand(s::Strategy)

end

function hand_finished(s::Strategy, result::Int)

end


function game_finished(s::Strategy)

end

function is_log_space(s::Strategy)
    return false
end