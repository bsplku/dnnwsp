clear;

addpath ../DBN  ../NN

%% Sensorimotor data pre-training
load sensorimotor_1D_sample.mat;
gpuDevice();
x0 = lhrhadvs;    y0 = labels;
x0 = gpuArray(x0);  y0 = gpuArray(y0); 

%% Leave-one-subject-out cross-validation
% for j = 1:12
for j = 1
    test_idx = (1:120) + (j-1)*120;
    test_x = x0(test_idx,:);    test_y = y0(test_idx,:);
    train_x = x0;               train_y = y0;
    train_x(test_idx,:) = [];   train_y(test_idx,:) = [];

   %% Save pre-trained files
%     if ~exist(strcat('sensorimotor/sbj', num2str(j)), 'dir')
%         mkdir(strcat('sensorimotor/sbj', num2str(j)));
%     end
    
    %% DBN training
    rng('shuffle');
    dbn.sizes       = [100 100 100];
    opts.activation_function = 'relu';
    opts.numepochs	= 300;
    opts.batchsize	= 10;
    opts.alpha      = [0.00001 0.01 0.01];  % 0.00001/0.01/0.01 
    opts.momentum	= 0.5;
    opts.gbrbm      = 1;    
    opts.max_beta   = [0.5 0.01 0.01];        % 0.5/0.01/0.01
    opts.hsparsityTarget    = 0;    % sparsity 
    opts.wsparsityTarget    = [0 0 0 0];	
    opts.weightPenaltyL1	= [0 0 0 0]; 
%     opts.weightPenaltyL1	= [0.1 0.001 0.001];   
    opts.weightPenaltyL2	= 0;    
    opts.dropoutFraction	= 0;

    dbn = dbnsetup(dbn, train_x, opts); 
    dbn = dbntrain(dbn, train_x, opts);   

%     save(strcat('sensorimotor/sbj', num2str(j),'/dbn_pretrain.mat'), 'dbn');
% end

%% DNN training
%  load(strcat('sensorimotor/sbj', num2str(j),'/dbn_pretrain.mat'));
%   unfold dbn to nn. 
    nn = dbnunfoldtonn(dbn, 4);   
%   nn = nnsetup([68218 100 100 100 4]);
    nn.activation_function = 'relu';
    nn.output = 'softmax';
    nn.learningRate = 0.001;
    nn.weightPenaltyL1 = [0 0 0 0 0];
    % non-zero weight sparsity level
    nn.nzr = [0.001 0 0 0];	
    nn.beginAnneal = 100;
    nn.max_beta = [0.01 0.01 0.01 0];
%     nn.inputZeroMaskedFraction = 0.3;

    % train nn
    opts.numepochs = 500;
    opts.batchsize = 10;
    nn = nntrain(nn, train_x, train_y, opts, test_x, test_y);
    er = nn.loss.val.e_frac(end);

%     save(strcat('sensorimotor/sbj',num2str(j) ,'/dnn_relu_ws_001.mat'), 'er', 'nn');
%     tt_er(j) = er;

end