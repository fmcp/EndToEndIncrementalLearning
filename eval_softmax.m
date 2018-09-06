function outputs = eval_softmax(net, imdb)

if strcmp(net.device, 'cpu')
    net.move('gpu');
end

net.conserveMemory = 0;
nsamp = 1;
outputs = {};
while nsamp <= size(imdb.images.data, 4)
    step = min(256, size(imdb.images.data, 4) - nsamp+1);
    images = gpuArray(single(imdb.images.data(:, :, :, nsamp:nsamp+step-1)));
    inputs = {'image', images};
    net.eval(inputs) ;
    nsamp = nsamp + step;
    
    % Gather results.
	index = strfind({net.layers.name}, 'softmax'); %softmax
    index = find(not(cellfun('isempty', index)));
    npos = length(index);
    
    index2 = strfind({net.layers.name}, 'fc'); %softmax
    index2 = find(not(cellfun('isempty', index2)));
    nposFC = length(index2);

    if isempty(outputs)
        outputs = cell(1, nposFC);
    end
    
    lastPos = 1;
    for lix = 1:npos
        if ~strcmp(net.layers(index(lix)).name, 'softmax_global') && ~strcmp(net.layers(index(lix)).name, 'softmax_old')
            x = squeeze(gather(net.vars(net.layers(index(lix)).outputIndexes(1)).value));
            outputs{lastPos} = cat(2, outputs{lastPos}, x);
            lastPos = lastPos + 1;
        end
    end % lix
end

if strcmp(net.device, 'gpu')
    net.move('cpu');
end

