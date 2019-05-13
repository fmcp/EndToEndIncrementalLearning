function outputs = eval_pool_imagenet(net, imdb)

if strcmp(net.device, 'cpu')
    net.move('gpu');
end

net.conserveMemory = 0;
nsamp = 1;

%% Get output size.
inputs = {'image', gpuArray(single(imresize(imdb.images.data(:, :, :, 1), [224 224])))};
net.eval(inputs) ;

% Gather results.
index = strfind({net.layers.name}, 'pool_final');
index = find(not(cellfun('isempty', index)));
x = squeeze(gather(net.vars(net.layers(index(1)).outputIndexes(1)).value));

% Rerserve memory.
sz = size(x);
sz(end) = size(imdb.images.data, 4);
outputs = zeros(sz, 'single');
while nsamp < size(imdb.images.data, 4)
    step = min(256, size(imdb.images.data, 4) - nsamp+1);
    images = gpuArray(single(imresize(imdb.images.data(:, :, :, nsamp:nsamp+step-1), [224 224])));
    images(:,:,1,:) = images(:,:,1,:) - imdb.meta.dataMean(1);
    images(:,:,2,:) = images(:,:,2,:) - imdb.meta.dataMean(2);
    images(:,:,3,:) = images(:,:,3,:) - imdb.meta.dataMean(3);
    inputs = {'image', images};
    net.eval(inputs) ;
    nsamp = nsamp + step;
    
    % Gather results.
    index = strfind({net.layers.name}, 'pool_final');
    index = find(not(cellfun('isempty', index)));
    
    % Concat results.
    x = squeeze(gather(net.vars(net.layers(index(1)).outputIndexes(1)).value));
    outputs(:, nsamp:nsamp+step-1) = x;
end

if strcmp(net.device, 'gpu')
    net.move('cpu');
end
