function nn = nnapplygrads(nn)
% NNAPPLYGRADS updates weights and biases with calculated gradients
% nn = nnapplygrads(nn) returns an neural network structure with updated
% weights and biases
    
    dv = nn.dv;    
    dv = dv - 0.001 * norm(nn.v);
%     dv = dv - 0.1 * sign(nn.v);
    dv = nn.learningRate * dv;
    nn.v = nn.v + dv;
%     nn.v = nn.v / norm(nn.v);
end
