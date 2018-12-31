function nn = nnbp(nn,x, y)
%NNBP performs backpropagation
% nn = nnbp(nn) returns an neural network structure with updated weights 
    
    n = nn.n;           % # of layers
    m = size(x, 1);     % m : # of samples
    sparsityError = 0;  % sparsity error
    
    % the class label index
    index = find(y==1);
    
    switch nn.output
        case 'sigm'                 % sigmoid : f'(z) = f(z)*(1-f(z))
%             d{n} = (nn.a{n}(index) .* (1 - nn.a{n}(index)));   
            d{n} = 1;   
        case {'softmax','linear'}   % linear, softmax : f'(z) = 1 (softmax is with 'exp' and 'log')
            d{n} = 1;          
        case 'tanh'
            d{n} = 1;
    end
    
    % Derivative of the activation function
    for i = (n - 1) : -1 : 1
        switch nn.activation_function 
            case 'sigm'             % sigmoid : f'(z) = f(z)*(1-f(z))         
                d_act = nn.a{i} .* (1 - nn.a{i});   
            case 'tanh_opt'         % tanh optimized
                d_act = 1.7159 * 2/3 * (1 - 1/(1.7159)^2 * nn.a{i}.^2); 
            case 'tanh'             % tanh
                d_act = (1 - nn.a{i}.^2); 
        end
        
        if i+1 == n
            d{i} = nn.W{i}(index,:) .* d_act;
        elseif i == 1
            d{i} = nn.W{i};
        else
            d{i} = nn.W{i} .* repmat(d_act,size(nn.W{i},1),1);
        end
        
    end
    
    td = d{n};
    for i = (n-1) : -1 : 1
        td = td * d{i}(:,2:end);
    end    
    nn.dv(1,:) = td;

end
