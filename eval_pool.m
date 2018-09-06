function outputs = eval_pool(net, imdb)

if strcmp(net.device, 'cpu')
    net.move('gpu');
end

net.conserveMemory = 0;
nsamp = 1;
outputs = [];
while nsamp <= size(imdb.images.data, 4)
    step = min(256, size(imdb.images.data, 4) - nsamp+1);
    images = gpuArray(imdb.images.data(:, :, :, nsamp:nsamp+step-1));
    inputs = {'image', images};
    net.eval(inputs) ;
    nsamp = nsamp + step;
    
    % Gather results.
    index = strfind({net.layers.name}, 'pool_final');
    index = find(not(cellfun('isempty', index)));
    
    % Concat results.
    x = squeeze(gather(net.vars(net.layers(index(1)).outputIndexes(1)).value));
    outputs = cat(2, outputs, x);
end

if strcmp(net.device, 'gpu')
    net.move('cpu');
end

