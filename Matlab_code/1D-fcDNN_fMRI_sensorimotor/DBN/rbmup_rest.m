function x = rbmup(rbm, x, opts)

    if rbm.gbrbm == 1 
        sigma = 1;
        x = x - repmat(mean(x),size(x,1),1);
        x = x ./ repmat(std(x), size(x,1),1) * sigma;
%         x = sigm(repmat(rbm.c', size(x, 1), 1) + x * rbm.W' / sigma^2);
        x = tanh(repmat(rbm.c', size(x, 1), 1) + x * rbm.W' / sigma^2);
        
%         s1 = repmat(rbm.c', size(x,1), 1) + x * rbm.W' / sigma^2; 
%         x = max(0, s1 + sqrt(sigm(s1)) .* randn(size(s1)));
%         x(x>=0.5)=1; x(x<0.5)=0;
        
    else
%         x = sigm(repmat(rbm.c', size(x, 1), 1) + x * rbm.W');
        x = tanh(repmat(rbm.c', size(x, 1), 1) + x * rbm.W');
        
%         s1 = repmat(rbm.c', size(x,1), 1) + x * rbm.W'; 
%         x = max(0, s1 + sqrt(sigm(s1)) .* randn(size(s1)));
    end
    
end



