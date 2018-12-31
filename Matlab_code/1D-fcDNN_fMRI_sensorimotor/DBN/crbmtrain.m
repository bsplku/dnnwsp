function rbm = crbmtrain(rbm, x, opts)
    
    assert(isfloat(x), 'x must be a float');   
   
    x = trim_image(x, rbm.filtersize, rbm.stride);	% trim the image
    stride = rbm.stride;
    filtersize = rbm.filtersize;
    nfilter = rbm.nfilter;
    ninput = rbm.ninput;
    m = size(x, 1);	% # of samples
    batchsize = opts.batchsize;
    numbatches = m / batchsize;     
    vRows = size(x, 2);               
    vCols = size(x, 3);               
    hRows = vRows - filtersize + 1;
    hCols = vCols - filtersize + 1;
            
    for i = 1 : opts.numepochs
        kk = randperm(m);	% random sample index
        err = 0;            % errors in a epoch
        batch_rho = [];     % sparsity
        
        for l = 1 : numbatches
            % variable priority > nfilter/ninput/batch
            v1 = x( kk((l - 1) * opts.batchsize + 1 : l * opts.batchsize) , :, :, :);    
            
            for f = 1:nfilter
                for n = 1:ninput
                    hidI(:,:,f,n,:) = convn(permute(v1(:,:,:,n), [2,3,1]), ff(rbm.W(:,:,f,n)), 'valid') + rbm.c(f);
                end
                hidI_sum(:,:,f,:) = sum(hidI(:,:,f,:,:),4);
            end
            h1p = exp(hidI_sum) ./ ( 1 + pool(exp(hidI_sum), stride));
            h1p = permute(h1p, [4,1,2,3]);
            if rbm.hsparsityTarget > 0,	rho = mean(h1p(:)); end
%             if rbm.hsparsityTarget > 0,	if i==1, rho=mean(h1p(:)); else rho = mean(h1p(:))*0.01 + rho*0.99; end,	end
            h1 = double( h1p > rand(size(h1p)) );
                        
            for n = 1:ninput
                for f = 1:nfilter
                    visI(:,:,f,n,:) = convn(permute(h1(:,:,:,f), [2,3,1]), rbm.W(:,:,f,n), 'full');
                end
                visI_sum(:,:,n,:) = sum(visI(:,:,:,n,:),3);
            end
            I = visI_sum + repmat(permute(repmat(rbm.b, 1, batchsize), [3,4,1,2]), vRows, vCols);
            if rbm.gbrbm == 1
                v2 = I;
            else
                v2 = sigm(I); 
            end      
            v2 = permute(v2, [4,1,2,3]);
                        
            for f = 1:nfilter
                for n = 1:ninput
                    hidI(:,:,f,n,:) = convn(permute(v2(:,:,:,n), [2,3,1]), ff(rbm.W(:,:,f,n)), 'valid') + rbm.c(f);
                end
                hidI_sum(:,:,f,:) = sum(hidI(:,:,f,:,:),4);
            end
            h2 = exp(hidI_sum) ./ ( 1 + pool(exp(hidI_sum), stride));
            h2 = permute(h2, [4,1,2,3]);
%             h2 = double( h2 > rand(size(h2)) ); % confused.
            
            % updates
            for f = 1:nfilter
                for n = 1:ninput
                    for b = 1:batchsize
                        c1(:,:,b) = conv2(squeeze(v1(b,:,:,n)), ff(squeeze(h1(b,:,:,f))), 'valid');  
                        c2(:,:,b) = conv2(squeeze(v2(b,:,:,n)), ff(squeeze(h2(b,:,:,f))), 'valid'); 
                    end
                    c1_sum = sum(c1, 3);
                    c2_sum = sum(c2, 3);
                    vW(:,:,f,n) = rbm.alpha * (c1_sum - c2_sum) / batchsize;
                end
            end                      
            vb = rbm.alpha * squeeze(sum(sum(sum(v1 - v2)))) / (vRows*vCols) / batchsize;
            vc = rbm.alpha * squeeze(sum(sum(sum(h1 - h2)))) / (hRows*hCols) / batchsize;  
            
            % sparsity
            if rbm.hsparsityTarget > 0	
                batch_rho = [batch_rho rho];
            end
            
            rbm.vW = rbm.momentum * rbm.vW + vW;   
            rbm.vb = rbm.momentum * rbm.vb + vb;
            rbm.vc = rbm.momentum * rbm.vc + vc;
            rbm.W = rbm.W + rbm.vW;
            rbm.b = rbm.b + rbm.vb;
            rbm.c = rbm.c + rbm.vc; 
                        
            err = err + sum(sum(sum(sum( (v1 - v2).^2 )))) / batchsize;  
        end
        
        % hidden sparsity visualization
        if rbm.hsparsityTarget > 0	
            epoch_rho = mean(batch_rho);
            rbm.rho = [rbm.rho epoch_rho];
            rbm.c = rbm.c - rbm.hsparsityParam * (epoch_rho-rbm.hsparsityTarget)';
            disp(['sparsity ' num2str(epoch_rho)]);
        end
        
        disp(['epoch ' num2str(i) '/' num2str(opts.numepochs)  '. Average reconstruction error is: ' num2str(err/numbatches)]);
        rbm.error = [rbm.error err/numbatches];
    end
end

function out = trim_image(in, filtersize, stride)

    if mod(size(in,2)-filtersize+1, stride) ~= 0
        in(:,end,:,:) = [];
    end
    if mod(size(in,3)-filtersize+1, stride) ~= 0
        in(:,:,end,:) = [];
    end
    out = in;
end

function out = ff(in)
    out = in(end:-1:1, end:-1:1);
end

function blocks = pool(in, stride)
    
    nRows = size(in, 1);
    nCols = size(in, 2);
    blocks = zeros(size(in));
    
    for i = 1:floor(nRows/stride)
        rows = (i-1)* stride+1:i*stride;
        for j = 1:floor(nCols/stride)
            cols = (j-1)*stride+1:j*stride;
            
            blockVal = squeeze(sum(sum(in(rows,cols,:,:),1),2));
            blocks(rows,cols,:,:) = repmat(permute(blockVal, [3,4,1,2]), numel(rows), numel(cols));
        end
    end

end
