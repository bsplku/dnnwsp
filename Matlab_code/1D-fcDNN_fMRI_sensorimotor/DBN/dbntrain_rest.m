function dbn = dbntrain(dbn, x, opts)

    n = numel(dbn.rbm);
    dbn.rbm{1} = rbmtrain_rest(dbn.rbm{1}, x, opts); 

    for i = 2 : n
        x = rbmup_rest(dbn.rbm{i - 1}, x, opts);
        dbn.rbm{i} = rbmtrain_rest(dbn.rbm{i}, x, opts);
    end

end
