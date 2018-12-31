function nn = nnsetup(architecture)
% nn = nnsetup(architecture) returns an neural network structure with n=numel(architecture)
% layers, architecture being a n x 1 vector of layer sizes e.g. [784 100 10]
    
    % nn struct
    nn.size   = architecture;           % nn.size. [784 100 10]
    nn.n      = numel(nn.size);         % nn.n : # of layers. 3
    
    nn.activation_function              = 'tanh_opt';   %  Activation functions of hidden layers: 'sigm' (sigmoid) or 'tanh_opt' (optimal tanh).
    nn.learningRate                     = 0.1;          %  Learning rate Note: typically needs to be lower when using 'sigm' activation function and non-normalized inputs.
    nn.momentum                         = 0.1;          %  Momentum
    nn.scaling_learningRate             = 1;            %  Scaling factor for the learning rate (each epoch)
    nn.weightPenaltyL2                  = [0 0 0 0 0 0 0 0 0 0];	%  L2 regularization
    nn.nonSparsityPenalty               = 0;            %  Non sparsity penalty
    nn.sparsityTarget                   = 0;            %  Sparsity target
    nn.inputZeroMaskedFraction          = 0;            %  Used for Denoising AutoEncoders
    nn.dropoutFraction                  = 0;            %  Dropout level (http://www.cs.toronto.edu/~hinton/absps/dropout.pdf)
    nn.testing                          = 0;            %  Internal variable. nntest sets this to one.
    nn.output                           = 'sigm';       %  output unit 'sigm' (=logistic), 'linear' and 'softmax'
    nn.loss                             = [];
    
    % weight sparsity 2015.02.06
    nn.weightPenaltyL1                  = 0;            % beta value
    nn.nzr                              = [0 0 0 0 0 0 0 0 0 0];    % target non-zero ratio for "each" layer
    nn.beginAnneal                      = 0;            % epoch index for starting fine-tuning 
    nn.mNZR                             = cell(nn.n-1,1);	% mean non-zero ratio
    nn.max_beta = [];                                  
    nn.beta = [];
    nn.lr = [];	
    nn.rho = [];                         
    nn.er = [];      
    nn.kurtosis = [];
    nn.gini = [];
    nn.hoyer = [];

    for i = 2 : nn.n   
        nn.W{i - 1} = (rand(nn.size(i), nn.size(i - 1)+1) - 0.5) * 4 * sqrt(6 / (nn.size(i) + nn.size(i - 1))); % fc
%         nn.W{i - 1} = 0.01 * randn(nn.size(i), nn.size(i - 1)+1) ; 
        nn.vW{i - 1} = zeros(size(nn.W{i - 1}));
                
        % average activations (hidden sparsity)
        nn.p{i}     = zeros(1, nn.size(i));   
    end    
end
