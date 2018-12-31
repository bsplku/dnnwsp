function rbm = rbmtrain(rbm, x, opts)
    
    assert(isfloat(x), 'x must be a float');   
    h = size(rbm.c, 2);                 % hidden size
    n = size(x, 2);                     % input size
    m = size(x, 1);                     % # of samples
    numbatches = m / opts.batchsize;    % # of batches
        
    assert(rem(numbatches, 1) == 0, 'numbatches not integer');  
 
    for i = 1 : opts.numepochs
        kk = randperm(m);	% random sample index
        err = 0;            % errors in a epoch
        epoch_rho = [];
        
        for l = 1 : numbatches
            batch = x(kk((l - 1) * opts.batchsize + 1 : l * opts.batchsize), :);    % x to batch

           %% v1
            v1 = batch;     
            if rbm.sparsityParam > 0  
                h1p = sigm(repmat(rbm.c', opts.batchsize, 1) + v1 * rbm.W' ); 
                rho = mean(h1p, 1);
            end
            	figure(1); hist(h1p(:),100); xlim([0 1]); ylim([0 200]);     
            	drawnow;
           %% h1
            h1 = sigmrnd(repmat(rbm.c', opts.batchsize, 1) + v1 * rbm.W');  
%             h1 = tanh(repmat(rbm.c', opts.batchsize, 1) + v1 * rbm.W');  
           %% v2
            if rbm.gbrbm == 1           % Gaussian-Bernoulli RBM
                v2 = repmat(rbm.b', opts.batchsize, 1) + h1 * rbm.W;    % mean-field
            else
            	v2 = sigmrnd(repmat(rbm.b', opts.batchsize, 1) + h1 * rbm.W);   % sigmrnd( h1*W + b ).
%             	v2 = tanh(repmat(rbm.b', opts.batchsize, 1) + h1 * rbm.W);   % sigmrnd( h1*W + b ).
            end
           %% h2
            h2 = sigm(repmat(rbm.c', opts.batchsize, 1) + v2 * rbm.W');     % sigm (v2*W + c ). because last.
%             h2 = tanh(repmat(rbm.c', opts.batchsize, 1) + v2 * rbm.W');     % sigm (v2*W + c ). because last.
            
            c1 = h1' * v1;  % h1*v1 for the positive phase update 
            c2 = h2' * v2;  % h2*v2 for the negative phase update

            % update
            rbm.vW = rbm.momentum * rbm.vW + rbm.alpha * (c1 - c2)     / opts.batchsize;    % vW
            rbm.vb = rbm.momentum * rbm.vb + rbm.alpha * sum(v1 - v2)' / opts.batchsize;    % vb
            rbm.vc = rbm.momentum * rbm.vc + rbm.alpha * sum(h1 - h2)' / opts.batchsize;    % vc
            rbm.W = rbm.W + rbm.vW;
            rbm.b = rbm.b + rbm.vb;
            rbm.c = rbm.c + rbm.vc; 
            
            % sparsity
            if rbm.sparsityParam > 0	
                % SSE
                vsW = rbm.alpha * rbm.sparsityParam * bsxfun(@times, (rbm.sparsityTarget-rho)', (h1p.*(1-h1p))'*v1);
                vsc = rbm.alpha * rbm.sparsityParam * ( (rbm.sparsityTarget-rho) .* sum(h1p.*(1-h1p)) )';
                % KL-divergence
%                 derkl = bsxfun(@times, (h1p.*(1-h1p)), (log(rho./(1-rho))-log(rbm.sparsityTarget/(1-rbm.sparsityTarget)))) ...
%                     - repmat(rho-rbm.sparsityTarget,opts.batchsize,1);
%                 vsW = rbm.alpha * rbm.sparsityParam * derkl'*v1;
%                 vsc = rbm.alpha * rbm.sparsityParam * sum(derkl)';
                rbm.W = rbm.W + vsW;
                rbm.c = rbm.c + vsc;
                epoch_rho = [epoch_rho; rho];
            elseif rbm.wsparsityParam > 0
                rbm.W = rbm.W - rbm.alpha * rbm.wsparsityParam * sign(rbm.W);
            end             
                                    
            err = err + sum(sum((v1 - v2) .^ 2)) / opts.batchsize;  % error. plz consider batch size.
        end
        
        if rbm.sparsityParam > 0	
            avg_rho = mean(epoch_rho(:));
            rbm.rho = [rbm.rho avg_rho];
        end
        
        disp([num2str(avg_rho) ' epoch ' num2str(i) '/' num2str(opts.numepochs)  '. Average reconstruction error is: ' num2str(err / numbatches)]);
%         disp(['epoch ' num2str(i) '/' num2str(opts.numepochs)  '. Average reconstruction error is: ' num2str(err / numbatches)]);
        rbm.error = [rbm.error (err/numbatches)];
        
%         figure(1); hist(rbm.W(1,:),100);          
%         drawnow;
    end
end
