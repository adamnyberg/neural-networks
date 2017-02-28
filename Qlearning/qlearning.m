world = 1;
actions = [1 2 3 4];

% Q-Learning parameters
alpha = 1; % learning rate. lower if randomness.
gamma = 0.8; % discount factor. determines the importance of future rewards.
epsilon = 0.5; % exploration factor. 

gwinit(world);

state = gwstate();

Q = zeros(state.xsize, state.ysize, length(actions)); % zero init
Q(1, :, 2) = -Inf; % top border
Q(:, state.ysize, 3) = -Inf; % right 
Q(state.xsize, :, 1) = -Inf; % bottom 
Q(:, 1, 4) = -Inf; % left


goal_count = 0;
goal_limit = 100000;
total_iterations = 0;

%%
while goal_count < goal_limit
    x = state.pos(1);
    y = state.pos(2);
    action = choose_action(Q, state.pos(1), state.pos(2), ...
        actions, [0.25 0.25 0.25 0.25], [1-epsilon epsilon]);
    
    new_state = gwaction(action);
    
    if (new_state.isvalid == 1)
        new_value = qfunc(Q, state, new_state, action, alpha, gamma, new_state.feedback);
        Q(state.pos(1), state.pos(2), action) = new_value; 
        
        state = new_state;
    else
        %new_state
        %new_state.pos
        
    end
    
    if (new_state.isterminal == 1)
        Q(new_state.pos(1), new_state.pos(2), :) = 0;
        gwinit(world);
        current_iteration = current_iteration + 1;
    end
    goal_count = goal_count + 1;
end



%%

gwdraw();

figure(2);
subplot(2,2,1)
imagesc(Q(:, :, 1));
subplot(2,2,2)
imagesc(Q(:, :, 2));
subplot(2,2,3)
imagesc(Q(:, :, 3));
subplot(2,2,4)
imagesc(Q(:, :, 4));

figure(3)
gwdraw();
for x = 1:state.xsize
   for y = 1:state.ysize
       [~, I] = max(Q(x,y,:));
       gwplotarrow([x y], I);
   end
end

