function rbm = rbmtrain(rbm, x, opts)
    
    h = size(rbm.c, 1);                 % hidden size
    n = size(x, 2);                     % input size
    m = size(x, 1);                     % # of samples
    numbatches = m / opts.batchsize;    % # of batches
            
    epsilon = 0.001;                    % weight sparsity    
    betarate = 0.01;         
    %% all
%     p = 0;
    %% one-by-one
    p = zeros(h,1);
    
    if rbm.gbrbm == 1                   % gaussian rbm    
        sigma = 1;
        x = x - repmat(mean(x),size(x,1),1);
        x = x ./ repmat(std(x),size(x,1),1) * sigma;
    else
        sigma = 1;
    end
    
    assert(isfloat(x), 'x must be a float');   
    assert(rem(numbatches, 1) == 0, 'numbatches not integer');  
        
    for i = 1 : opts.numepochs
        kk = randperm(m);	% random sample index
        err = 0;            % errors in a epoch
        batch_rho = [];     % sparsity
        
        for l = 1 : numbatches
            batch = x(kk((l - 1) * opts.batchsize + 1 : l * opts.batchsize), :);    % x to batch
            
            % dropout
            if rbm.dropoutFraction > 0  
                dropout_mask = (rand(opts.batchsize, h) > rbm.dropoutFraction);
            end

            switch rbm.activation_function
                case 'sigm'
                    v1 = batch;     
                    h1p = sigm(repmat(rbm.c', opts.batchsize, 1) + v1 * rbm.W' / sigma^2); 
                        if rbm.hsparsityTarget > 0,  rho = mean(h1p, 1); end
                        if rbm.dropoutFraction > 0, h1p = h1p .* dropout_mask; end
                    h1 = double(h1p) > rand(size(h1p));
                    if rbm.gbrbm == 1
                        v2 = repmat(rbm.b', opts.batchsize, 1) + double(h1) * rbm.W; % mean-field
%                         v2 = repmat(rbm.b', opts.batchsize, 1) + h1 * rbm.W + sigma^2 * randn(opts.batchsize, n);
                    else
                        v2 = sigm(repmat(rbm.b', opts.batchsize, 1) + double(h1) * rbm.W); 
                    end
                    h2 = sigm(repmat(rbm.c', opts.batchsize, 1) + v2 * rbm.W' / sigma^2);    
                        if rbm.dropoutFraction > 0, h2 = h2 .* dropout_mask; end 
                case 'tanh'
                    v1 = batch;     
                    h1p = tanh(repmat(rbm.c', opts.batchsize, 1) + v1 * rbm.W' / sigma^2);  
                        if rbm.hsparsityTarget > 0,  rho = mean(h1p, 1); end
                        if rbm.dropoutFraction > 0, h1p = h1p .* dropout_mask; end
                        h1 = h1p;
                    if rbm.gbrbm == 1
                        v2 = repmat(rbm.b', opts.batchsize, 1) + double(h1) * rbm.W;
%                         v2 = repmat(rbm.b', opts.batchsize, 1) + double(h1) * rbm.W + sigma^2 * randn(opts.batchsize, n);
                    else
                        v2 = tanh(repmat(rbm.b', opts.batchsize, 1) + double(h1) * rbm.W);  
                    end
                    h2 = tanh(repmat(rbm.c', opts.batchsize, 1) + v2 * rbm.W' / sigma^2); 
                        if rbm.dropoutFraction > 0, h2 = h2 .* dropout_mask; end     
                case 'relu'
                    v1 = batch;  
                        s1 = repmat(rbm.c', opts.batchsize, 1) + v1 * rbm.W' / sigma^2; 
                    h1p = max(0, s1 + sqrt(sigm(s1)) .* randn(size(s1)));
                        if rbm.hsparsityTarget > 0,  rho = mean(h1p, 1); end
                        if rbm.dropoutFraction > 0, h1p = h1p .* dropout_mask; end
                        h1 = h1p;
                    v2 = repmat(rbm.b', opts.batchsize, 1) + h1 * rbm.W;
%                     v2 = repmat(rbm.b', opts.batchsize, 1) + h1 * rbm.W + sigma^2 * randn(opts.batchsize, n);
                        s2 = repmat(rbm.c', opts.batchsize, 1) + v2 * rbm.W' / sigma^2;       
                    h2 = max(0, s2 + sqrt(sigm(s2)) .* randn(size(s2)));    
                        if rbm.dropoutFraction > 0, h2 = h2 .* dropout_mask; end    
            end
            
            % update term       
            c1 = h1p' * v1;  % h1p*v1 for the positive phase update. h1 or h1p?
            c2 = h2' * v2;  % h2*v2 for the negative phase update
            vW = rbm.alpha * (c1 - c2)      /sigma^2    / opts.batchsize;    % vW
            vb = rbm.alpha * sum(v1 - v2)'	/sigma^2    / opts.batchsize;    % vb
            vc = rbm.alpha * sum(h1p - h2)'             / opts.batchsize;    % vc
            
            % hidden sparsity
            if rbm.hsparsityTarget > 0	
                batch_rho = [batch_rho mean(rho)];
            end
            
            % weight penalty L1
            if rbm.weightPenaltyL1 > 0
                dW = rbm.alpha * rbm.weightPenaltyL1 * sign(rbm.W);
                vW = vW - dW;
            end       
            
            % weight penalty L2
            if rbm.weightPenaltyL2 > 0
                dW = rbm.alpha * rbm.weightPenaltyL2 * rbm.W;
                vW = vW - dW;
            end   
             
            % weight sparsity
            if rbm.wsparsityTarget > 0 || rbm.hoyerTarget > 0
                %% all
%                 mNZR = length( find( abs(rbm.W(:)) > epsilon ) ) / numel(rbm.W);
%                 mHoyer = hoyer(rbm.W(:));
                
                %% all - nzr
%                 p = p + betarate*sign(mNZR - rbm.wsparsityTarget);
%                 if p > rbm.max_beta
%                         p = rbm.max_beta;
%                 elseif p < 0
%                     p = 0;
%                 end
%                 dW = rbm.alpha * p * sign(rbm.W);
%                 vW = vW - dW;
                
                %% all - hoyer
%                 p = p + betarate*sign(rbm.hoyerTarget - mHoyer);
%                 if p > rbm.max_beta
%                         p = rbm.max_beta;
%                 elseif p < 0
%                     p = 0;
%                 end
%                 dW = rbm.alpha * p * sign(rbm.W);
%                 vW = vW - dW;
                    
                %% one-by-one
                for k=1:size(rbm.W,1)
%                     mNZR(k) = length( find( abs(rbm.W(k,:)) > epsilon ) ) / numel(rbm.W(k,:));
                    mHoyer(k) = gather(hoyer(rbm.W(k,:)));
                    p(k) = p(k) + betarate*sign(rbm.hoyerTarget - mHoyer(k));
                    if p(k) > rbm.max_beta
                            p(k) = rbm.max_beta;
                    elseif p(k) < 0
                        p(k) = 0;
                    end
%                     dW(k,:) = rbm.alpha * p(k) * sign(rbm.W(k,:));
                    dW(k,:) = gather(rbm.alpha * p(k) * sign(rbm.W(k,:)));
                end
                vW = vW - dW;


            end
                        
            rbm.vW = rbm.momentum * rbm.vW + vW;   
            rbm.vb = rbm.momentum * rbm.vb + vb;
            rbm.vc = rbm.momentum * rbm.vc + vc;
            rbm.W = rbm.W + rbm.vW;
            rbm.b = rbm.b + rbm.vb;
            rbm.c = rbm.c + rbm.vc; 
                        
            err = err + sum(sum((v1 - v2) .^ 2)) / opts.batchsize;  % error. plz consider batch size.
        end
        
        % hidden sparsity
        if rbm.hsparsityTarget > 0	
            epoch_rho = mean(batch_rho);
            rbm.rho = [rbm.rho epoch_rho];
            rbm.c = rbm.c - rbm.hsparsityParam * (epoch_rho-rbm.hsparsityTarget)';
            disp(['sparsity ' num2str(epoch_rho)]);
        end
        
        % weight sparsity progress
        if rbm.wsparsityTarget > 0 || rbm.hoyerTarget > 0
            %% all
%             rbm.mNZR = [rbm.mNZR mNZR]; % nzr
%             rbm.mHoyer = [rbm.mHoyer mHoyer]; % hoyer
%             rbm.beta = [rbm.beta p];
            %% one-by-one
            rbm.mHoyer = [rbm.mHoyer; mHoyer];
            rbm.beta = [rbm.beta; p];      
            
%             disp(['non-zero ratio ' num2str(mNZR)]);
%             disp(['hoyer ' num2str(mHoyer)]);
            disp(['non-zero ratio ' num2str(mean(mHoyer))]);
        end
        
        % Annealing
        if i > rbm.beginAnneal && rbm.beginAnneal ~= 0
            decayrate = -0.0001;    
            rbm.alpha = max( 1e-6, ( decayrate*i+(1-decayrate*rbm.beginAnneal) ) * rbm.alpha ); 
%             rbm.alpha = max( 1e-10, rbm.lr(1)*rbm.beginAnneal/max(i,rbm.beginAnneal));
        end
        rbm.lr = [rbm.lr rbm.alpha];
        
%         disp([num2str(rbm.alpha) ' ' num2str(p) ' epoch ' num2str(i) '/' num2str(opts.numepochs)  '. Reconstruction error is: ' num2str(err / numbatches)]); 
        disp([num2str(rbm.alpha) ' ' num2str(mean(p)) ' epoch ' num2str(i) '/' num2str(opts.numepochs)  '. Reconstruction error is: ' num2str(err / numbatches)]); 
        rbm.error = [rbm.error (err/numbatches)];
                
%         figure(1); hist(rbm.W(1,:),100);          
    end
end
