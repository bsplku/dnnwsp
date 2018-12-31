function nn = nnbp(nn)
%NNBP performs backpropagation
% nn = nnbp(nn) returns an neural network structure with updated weights 
    
    n = nn.n;           % # of layers
    sparsityError = 0;  % sparsity error
    
    % Error term. cf) UFLDL. the highest layer output derivative. for all weight/bias update term.
    % d{n} : error term. -(y-a)*f'(z).
    switch nn.output
        case 'sigm'                 % sigmoid : f'(z) = f(z)*(1-f(z))
            d{n} = - nn.e .* (nn.a{n} .* (1 - nn.a{n}));   
        case {'softmax','linear'}   % linear, softmax : f'(z) = 1 (softmax is with 'exp' and 'log')
            d{n} = - nn.e;          
    end
    
    % Derivative of the activation function
    for i = (n - 1) : -1 : 2
        switch nn.activation_function 
            case 'sigm'             % sigmoid : f'(z) = f(z)*(1-f(z))         
                d_act = nn.a{i} .* (1 - nn.a{i});   
%                   d_act = dSigmoid(nn.a{i});   
            case 'tanh_opt'         % tanh optimized
                d_act = 1.7159 * 2/3 * (1 - 1/(1.7159)^2 * nn.a{i}.^2); 
            case 'tanh'             % tanh
                d_act = (1 - nn.a{i}.^2); 
%                     d_act = dTanh(nn.a{i}); 
            case 'relu'
%                 nn.a{i} = gather(nn.a{i});
                d_act = double(nn.a{i}>0);
%                 d_act = gpuArray(d_act);
%                 d_act = dRelu(nn.a{i});   
        end
        
        % sparsity
        if(nn.nonSparsityPenalty>0)
            pi = repmat(nn.p{i}, size(nn.a{i}, 1), 1);  % pi : average hidden activation of vectors
            sparsityError = [zeros(size(nn.a{i},1),1) nn.nonSparsityPenalty * (-nn.sparsityTarget ./ pi + (1 - nn.sparsityTarget) ./ (1 - pi))]; % add bias (zeros)
        end
        
        % Error term update. UFLDL.
        if i+1==n % in this case in d{n} there is not the bias term to be removed.          
            d{i} = (d{i + 1} * nn.W{i} + sparsityError) .* d_act; % Bishop (5.56)
        else % in this case in d{i} the bias term has to be removed.
            d{i} = (d{i + 1}(:,2:end) * nn.W{i} + sparsityError) .* d_act;
        end
        
        % dropout
        if(nn.dropoutFraction>0)
            d{i} = d{i} .* [ones(size(d{i},1),1) nn.dropOutMask{i}];
        end

    end

    % dW for all layers with error term. dW(i) = error term(i+1) * activation(i). 
    for i = 1 : (n - 1)
        if i+1==n
            nn.dW{i} = (d{i + 1}' * nn.a{i}) / size(d{i + 1}, 1);   % divided numbatches?
        else
            nn.dW{i} = (d{i + 1}(:,2:end)' * nn.a{i}) / size(d{i + 1}, 1);      
        end
    end
    
end
