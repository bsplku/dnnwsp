function [nn, L]  = nntrain(nn, train_x, train_y, opts, val_x, val_y)

% assert(X, A) == if(~X) error(A)
assert(isfloat(train_x), 'train_x must be a float');                            % why input float?
assert(nargin == 4 || nargin == 6,'number ofinput arguments must be 4 or 6')    % 'nargin' is # of parameters.

loss.train.e               = [];    % training loss function
loss.train.e_frac          = [];    % training error rate
loss.train.abser           = [];   
loss.val.e                 = [];    % validation loss function
loss.val.e_frac            = [];    % validation error rate
loss.val.abser             = [];
opts.validation = 0;

if nargin == 6
    opts.validation = 1;
end

% plot figure handle
fhandle = [];
if isfield(opts,'plot') && opts.plot == 1
    fhandle = figure();
end

m = size(train_x, 1);   % m = # of training samples

batchsize = opts.batchsize; % batch size
numepochs = opts.numepochs; % # of epochs
numbatches = m / batchsize; % # of batches

% weight sparsity
epsilon = 0.0001;          % zero
betarate = 0.001;          % beta rate 0.00001.

assert(rem(numbatches, 1) == 0, 'numbatches must be a integer');

L = zeros(numepochs*numbatches,1);  % L = sum of squared error "in minibatches". For stochastic gradient descent?
n = 1;                              % for all batches for all samples
  nn.weightPenaltyL1 = {};
for i = 1 : numepochs
    tic;       
    kk = randperm(m);   	% kk : random training index
    
    %% weight sparsity one-by-one
   
    for j = 1 : (nn.n-1)
        nn.weightPenaltyL1{j} = zeros(1,nn.size(j+1));  
    end
    
    for l = 1 : numbatches
        batch_x = train_x(kk((l - 1) * batchsize + 1 : l * batchsize), :);  % train_x to batch_x
        batch_y = train_y(kk((l - 1) * batchsize + 1 : l * batchsize), :);	% train_y to batch_y
        
        %Add noise to input (for use in denoising autoencoder) 
        if(nn.inputZeroMaskedFraction ~= 0)
            batch_x = batch_x.*(rand(size(batch_x))>nn.inputZeroMaskedFraction);
        end
        
        
        nn = nnff(nn, batch_x, batch_y);    % Feed-forward
        nn = nnbp(nn);                      % Back-propagation
        nn = nnapplygrads(nn);              % Gradient-1descent update
     
       % weight sparsity 
       for j = 1 : (nn.n-1)               
            if length(nn.weightPenaltyL1) == 1
%                 pl1 = nn.weightPenaltyL1(1);
                pl1 = nn.weightPenaltyL1{1};
            else
%                 pl1 = nn.weightPenaltyL1(j);
                pl1 = nn.weightPenaltyL1{j};
            end               
            %% non-zero ratio
            if nn.nzr(j) ~= 0                 
                mNZR = length( find( abs(nn.W{j}(:)) > epsilon ) ) / numel(nn.W{j});
                pl1 = pl1 + betarate*sign(mNZR - nn.nzr(j));
                if pl1 > nn.max_beta(j)
                        pl1 = nn.max_beta(j);
                elseif pl1 < 0
                    pl1 = 0;
                end
                nn.weightPenaltyL1{j} = pl1;
            end    
            %% hoyer
%             if nn.hoyer(j) ~= 0                 
%                 mHoyer = hoyer(nn.W{j}(:));
%                 pl1 = pl1 + betarate*sign(nn.hoyer(j) - mHoyer);
%                 if pl1 > nn.max_beta(j)
%                         pl1 = nn.max_beta(j);
%                 elseif pl1 < 0
%                     pl1 = 0;
%                 end
%                 nn.weightPenaltyL1(j) = gather(pl1);
%             end    
            %% hoyer one-by-one
%             if nn.hoyer(j) ~= 0                     
%                 mHoyer = gather(hoyer(nn.W{j}))';
%                 for k = 1:size(nn.W{j},1)          
%                     pl1(k) = pl1(k) + betarate*sign(nn.hoyer(j) - mHoyer(k));
%                     if pl1(k) > nn.max_beta(j)
%                             pl1(k) = nn.max_beta(j);
%                     elseif pl1(k) < 0
%                         pl1(k) = 0;
%                     end
%                 end    
%                 nn.weightPenaltyL1{j} = gather(pl1);
%             end    
        end
     
        L(n) =  gather(nn.L);          
%         L(n) = nn.L;        % loss function with all batches     
        n = n + 1;          % all batches for all samples
    end    
    t = toc;
    
    if opts.validation == 1
        loss = nneval(nn, loss, train_x, train_y, val_x, val_y);
        str_perf = sprintf('; Full-batch train mse = %f, val mse = %f', loss.train.e(end), loss.val.e(end));
    else
        loss = nneval(nn, loss, train_x, train_y);
        str_perf = sprintf('; Full-batch train err = %f', loss.train.e(end));
    end
    if ishandle(fhandle)
        nnupdatefigures(nn, fhandle, loss, opts, i);
    end
    
    % weight sparsity for a test..
    nn.loss = loss;
    for j = 1 : (nn.n-1)
        mNZR = length( find( abs(nn.W{j}(:)) > epsilon ) ) / numel(nn.W{j});
        nn.mNZR{j} = [nn.mNZR{j} mNZR];
%         mHoyer = hoyer(nn.W{j}(:));
%         nn.mHoyer{j} = [nn.mHoyer{j} mHoyer];
        mHoyer = hoyer(nn.W{j})';
        nn.mHoyer{j} = [nn.mHoyer{j}; mHoyer];
%         nn.beta{j} =[nn.beta{j}; nn.weightPenaltyL1(j)];
        nn.beta{j} =[nn.beta{j}; nn.weightPenaltyL1{j}];
    end
    nn.lr = [nn.lr nn.learningRate];
    nn.rho = [nn.rho mean(nn.p{2}(:))];
	nn.er = [nn.er loss.train.e(end)];
       
%     disp([num2str(nn.mNZR{1}(end)) ' ' num2str(nn.mNZR{2}(end)) ' ' num2str(nn.mNZR{3}(end))]);
%     disp([num2str(nn.weightPenaltyL1(1)) ' ' num2str(nn.weightPenaltyL1(2)) ' ' num2str(nn.weightPenaltyL1(3))]);
	disp(['epoch ' num2str(i) '/' num2str(opts.numepochs) '. Took ' num2str(t) ' seconds' '. Mini-batch mean squared error on training set is ' num2str(mean(L((n-numbatches):(n-1)))) str_perf]);
%     disp([num2str(mean(nn.p{2}(:))) ' epoch ' num2str(i) '/' num2str(opts.numepochs) '. Took ' num2str(t) ' seconds' '. Mini-batch mean squared error on training set is ' num2str(mean(L((n-numbatches):(n-1)))) str_perf]);
   
    % Annealing
    if nn.beginAnneal == 0
        nn.learningRate = nn.learningRate * nn.scaling_learningRate;
    elseif i > nn.beginAnneal
        decayrate = -0.0001;    
        nn.learningRate = max( 1e-6, ( decayrate*i+(1-decayrate*nn.beginAnneal) ) * nn.learningRate ); 
    end
    
%     figure(1); hist(nn.W{2}(:),300);
%     drawnow;
    
end
end

