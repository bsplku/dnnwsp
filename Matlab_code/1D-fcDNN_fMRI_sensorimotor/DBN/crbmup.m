function x = rbmup(rbm, x, opts)
% Convolutional RBM
        
    x = trim_image(x, rbm.filtersize, rbm.stride);

    for i = 1:size(x,1)
        for f = 1:rbm.nfilter
            for n = 1:rbm.ninput
                hidI(i,:,:,f,n) = conv2(squeeze(x(i,:,:,n)), ff(rbm.W(:,:,f,n)), 'valid') + rbm.c(f);
            end
            hidI_sum(i,:,:,f) = sum(hidI(i,:,:,f,:),5);
        end
    end
    % pooling
    p = downsample(hidI_sum, rbm.stride);
    x = p;
            
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
    out = in(end:-1:1, end:-1:1, :);
end

function p = downsample(in, stride)

    in = exp(in);
    nSamples = size(in, 1);
    nRows = size(in, 2);
    nCols = size(in, 3);
    nfilter = size(in, 4);
    
    for s = 1:nSamples
        p = zeros(nSamples, nRows/stride, nCols/stride, nfilter);

        for i = 1:floor(nRows/stride)
            rows = (i-1)* stride+1:i*stride;
            for j = 1:floor(nCols/stride)
                cols = (j-1)*stride+1:j*stride;
                p(s,i,j,:) = sum( sum(in(s,rows,cols,:),2), 3) ./ ( 1+sum( sum(in(s,rows,cols,:),2), 3) );
            end
        end
        
    end

end



