function [ out ] = qfunc(Q, old_state, state, action, alpha, gamma, reward)

    old_value = Q(old_state.pos(1), old_state.pos(2), action);
    
    out = old_value ... 
        + alpha * (reward + gamma * max(Q(state.pos(1), state.pos(2), :)) - (old_value));


end