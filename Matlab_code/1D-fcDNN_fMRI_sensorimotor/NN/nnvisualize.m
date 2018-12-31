function [nn, L]  = nnvisualize(nn, class, opts)

% assert(isfloat(test_x), 'train_x must be a float');                            % why input float?
% assert(nargin == 4,'number ofinput arguments must be 4 or 6')    % 'nargin' is # of parameters.

loss.train.e               = [];    % training loss function
loss.train.e_frac          = [];    % training error rate
loss.val.e                 = [];    % validation loss function
loss.val.e_frac            = [];    % validation error rate
opts.validation = 0;

numepochs = opts.numepochs; % # of epochs
n = 1;                              % for all batches for all samples

%% randomly initialize input space
% nn.v = 0.01 * (rand(1, nn.size(1)) - 0.5);
% nn.v = 0.001 * (randn(1, nn.size(1)));
nn.v = zeros(1, nn.size(1));
nn.dv = zeros(size(nn.v));
nn.dvnrm = [];
nn.score = [];

for i = 1 : numepochs
    tic;       
    
    x = nn.v(1,:);
    y = zeros(1,nn.size(end)); y(1,class) = 1;

    nn = nnff(nn, x, y);    % Feed-forward
    nn = nnvis_bp(nn, x, y);% Back-propagation
    nn = nnvis_apply(nn);   % Gradient-descent update

    L(n) = nn.L;        % loss function with all batches     
    n = n + 1;          % all batches for all samples
    t = toc;
    
    scores = (nn.a{end - 1} * nn.W{end}'); 
        
	nn.lr = [nn.lr nn.learningRate];
    nn.rho = [nn.rho mean(nn.p{2}(:))];
    nn.dvnrm = [nn.dvnrm norm(nn.dv)];
    nn.score = [nn.score scores(class)];
       
	disp(['epoch ' num2str(i) '/' num2str(opts.numepochs) '. Took ' num2str(t) ' seconds' '. Scores = ' num2str(scores)]);
% 	disp(['epoch ' num2str(i) '/' num2str(opts.numepochs) '. Took ' num2str(t) ' seconds' '. Mini-batch mean squared error on training set is ' num2str(mean(L((n-numbatches):(n-1)))) str_perf]);
      
end

end

