function nn = dbnunfoldtonn(dbn, outputsize)
% DBNUNFOLDTONN Unfolds a DBN to a NN
% dbnunfoldtonn(dbn, outputsize ) returns the unfolded dbn with a final layer of size outputsize added.

% what is unfold? it is pre-training!
% dbn to nn. [784 100 30] => [784 100 30 10].
% W, c trained by dbn is used for W of nn.

    if(exist('outputsize','var'))
        size = [dbn.sizes outputsize];
    else
        size = [dbn.sizes];
    end
    
    nn = nnsetup(size);
    for i = 1 : numel(dbn.rbm)
        nn.W{i} = [dbn.rbm{i}.c dbn.rbm{i}.W];  % combine W,c
    end
end

