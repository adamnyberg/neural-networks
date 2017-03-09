world = 1;
actions = [1 2 3 4];

% Q-Learning parameters
alpha = 0.8; % learning rate. lower if randomness.
gamma = 0.4; % discount factor. determines the importance of future rewards.
epsilon = 0.05; % exploration factor.
epsilon_end = 0.5;

gwinit(world);

state = gwstate();

Q = zeros(state.xsize, state.ysize, length(actions)); % zero init
Q(1, :, 2) = -Inf; % top border
Q(:, state.ysize, 3) = -Inf; % right 
Q(state.xsize, :, 1) = -Inf; % bottom 
Q(:, 1, 4) = -Inf; % left


goal_count = 0;
goal_limit = 1000;
total_iterations = 0;

epsilon_step = abs(epsilon_end-epsilon)/goal_limit;

%%
tic
percent = 0;
while goal_count < goal_limit
    action = choose_action(Q, state.pos(1), state.pos(2), ...
        actions, [0.25 0.25 0.25 0.25], [1-epsilon epsilon]);
    
    new_state = gwaction(action);
    reward = new_state.feedback;
    
    % protect from unwanted actions
    true_pos = state.pos + [(action==1) - (action==2),(action==3) - (action==4)]';
    if (true_pos ~= new_state.pos)
        state = new_state;
        continue;
    end
    
    if (new_state.isvalid == 1)
        new_value = qfunc(Q, state, new_state, action, alpha, gamma, reward);
        Q(state.pos(1), state.pos(2), action) = new_value; 
        
        state = new_state;  
    end
    
    if (new_state.isterminal == 1)
        Q(new_state.pos(1), new_state.pos(2), :) = 0;
        gwinit(world);
        goal_count = goal_count + 1;
        
        epsilon = epsilon + epsilon_step;
        
        if ((goal_count/goal_limit) > percent)
            display(num2str(percent*100))
            percent = percent + 0.01;
        end
        
    end
    total_iterations = total_iterations + 1;
end

trainingTime = toc;
display(['Time spent training: ' num2str(trainingTime) ' sec'])

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

