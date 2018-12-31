function dbn = cdbntrain(dbn, x, opts)
% Convolutional RBM

    n = dbn.nHLayers;
    dbn.rbm{1} = crbmtrain(dbn.rbm{1}, x, opts);

    for i = 2 : n
        x = crbmup(dbn.rbm{i - 1}, x, opts);
        dbn.rbm{i} = crbmtrain(dbn.rbm{i}, x, opts);
    end
        
end
