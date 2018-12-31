function nn = nnapplygrads(nn)
% NNAPPLYGRADS updates weights and biases with calculated gradients
    
    for i = 1 : (nn.n - 1)
        
        if length(nn.weightPenaltyL2) == 1
            pl2 = nn.weightPenaltyL2(1);
            pl1 = nn.weightPenaltyL1(1);
        else
            pl2 = nn.weightPenaltyL2(i);
            pl1 = nn.weightPenaltyL1(i);
        end
        
        if pl2 > 0
            dW = nn.dW{i} + pl2 * [zeros(size(nn.W{i}, 1),1) nn.W{i}(:,2:end)];
        else
            dW = nn.dW{i};
        end
        
        
        if pl1 > 0
            dW = (1-pl1) * dW + pl1 * [zeros(size(nn.W{i}, 1),1) sign(nn.W{i}(:,2:end))];
        end    
        
        % muliplied with learning rate
        dW = nn.learningRate * dW;
        
        % momentum
        if(nn.momentum>0)
            nn.vW{i} = nn.momentum*nn.vW{i} + dW;
            dW = nn.vW{i};
        end
        
        % gradient descent
        nn.W{i} = nn.W{i} - dW;
    end
end
