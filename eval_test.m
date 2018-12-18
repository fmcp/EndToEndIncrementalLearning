function outputs = eval_test(net, imdb)

if strcmp(net.device, 'cpu')
    net.move('gpu');
end

net.conserveMemory = 0;
nsamp = 1;
outputs = {};
while nsamp <= size(imdb.images.data, 4)
    step = min(128, size(imdb.images.data, 4) - nsamp+1);
    images = gpuArray(imdb.images.data(:, :, :, nsamp:nsamp+step-1));
    inputs = {'image', images};
    net.eval(inputs) ;
    nsamp = nsamp + step;
    
    % Gather results.
    index = strfind({net.layers.name}, 'softmax_global'); %softmax
    index = find(not(cellfun('isempty', index)));
    if isempty(index)
	index = strfind({net.layers.name}, 'softmax'); %softmax
        index = find(not(cellfun('isempty', index)));
    end
    npos = length(index);

    if isempty(outputs)
        outputs = cell(1, npos);
    end
    
    for lix = 1:npos
        x = squeeze(gather(net.vars(net.layers(index(lix)).outputIndexes(1)).value));
        outputs{lix} = cat(2, outputs{lix}, x);
    end % lix
end

if strcmp(net.device, 'gpu')
    net.move('cpu');
end

