function dbn = cdbnsetup(dbn, x, opts)
% Convolutional RBM

    rows = size(x, 2);
    cols = size(x, 3);
    ninput = 1;

    for u = 1 : dbn.nHLayers

        dbn.rbm{u}.alpha    = opts.alpha(u);         
        dbn.rbm{u}.momentum = opts.momentum;        
        dbn.rbm{u}.activation_function = 'sigm';
        dbn.rbm{u}.hsparsityTarget = opts.hsparsityTarget;
        dbn.rbm{u}.hsparsityParam = 1;   

        % convolution parameters
        dbn.rbm{u}.stride = opts.stride(u);
        dbn.rbm{u}.filtersize = opts.filtersize(u);
        dbn.rbm{u}.nfilter = opts.nfilter(u);
        dbn.rbm{u}.ninput = ninput;

        % progress
        dbn.rbm{u}.error = [];
        dbn.rbm{u}.rho = [];

        % Gaussian-Bernoulii RBM
        if u == 1 && opts.gbrbm == 1
            dbn.rbm{u}.gbrbm = 1;
        else
            dbn.rbm{u}.gbrbm = 0;
        end

        % initialization
        fanin = rows * cols; fanout = fanin * dbn.rbm{u}.nfilter;
        dbn.rbm{u}.W  = 2 * sqrt(6/(fanin + fanout)) * (rand(opts.filtersize(u), opts.filtersize(u), opts.nfilter(u), ninput) - 0.5); 
%             dbn.rbm{u}.W  = 0.001 * randn(opts.filtersize(u), opts.filtersize(u), opts.nfilter(u), ninput); 
        dbn.rbm{u}.vW = zeros(size(dbn.rbm{u}.W));              
        dbn.rbm{u}.b  = zeros(ninput, 1);
        dbn.rbm{u}.vb = zeros(size(dbn.rbm{u}.b));
        dbn.rbm{u}.c  = zeros(opts.nfilter(u), 1); 
        dbn.rbm{u}.vc = zeros(size(dbn.rbm{u}.c));       

        rows = rows - opts.filtersize(u) + 1;
        cols = cols - opts.filtersize(u) + 1;
        ninput = opts.nfilter(u);

    end        
end
