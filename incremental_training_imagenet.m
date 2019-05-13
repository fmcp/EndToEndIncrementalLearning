function [net, info, meta, exemplars] = incremental_training_imagenet(net, imdb_or, lastExemplars, opts)

if ~isfield(opts, 'freezeWeights')
    opts.freezeWeights = false;
end

% Add new layers.
[net, derOutputs] = fork_resnet_imagenet(net, ...
    'newtaskdim', opts.newtaskdim, ...
    'distillation_temp', opts.distillation_temp);

imdb = imdb_or;

if ~isfield(imdb.images, 'labels')
    imdb.images.labels = imdb.images.classes;
end


imdb_or.images.labels = imdb_or.images.classes;

opts.train.derOutputs = derOutputs;

net.meta.classes.name = cat(2, net.meta.classes.name, unique(imdb.images.classes));
% Parse labels to fit number of classes
if ~isfield(net.meta, 'eqlabs')
    net.meta.eqlabs = net.meta.classes.name;
end

net.meta.eqlabs = cat(2, net.meta.eqlabs, unique(imdb.images.classes)); % Add previous classes.

net40 = fullfile(opts.train.expDir, 'net-epoch-40.mat');
if ~exist(net40, 'file')
    
    % Build imdb with new imdb + exemplars.
    imdb.meta.exemplars = cat(2, ones(1, length(lastExemplars.images.labels)), zeros(1, length(imdb.images.labels)));
    imdb.meta.classes = cat(2, lastExemplars.meta.classes, imdb.meta.classes);
    ss = size(lastExemplars.images.data);
    aux_ = zeros(ss(1), ss(2), ss(3), ss(4) + size(imdb.images.data, 4), class(imdb.images.data));
    aux_(:,:,:,1:ss(4)) = lastExemplars.images.data;
    aux_(:,:,:,ss(4)+1:end) = imdb.images.data;
    imdb.images.data = int16(aux_);
    clear('aux_');
    %imdb.images.data = cat(4, lastExemplars.images.data, imdb.images.data);
    imdb.images.labels = cat(2, lastExemplars.images.labels, imdb.images.labels);
    imdb.images.classes = cat(2, lastExemplars.images.classes, imdb.images.classes);
    imdb.images.set = cat(2, lastExemplars.images.set, imdb.images.set);
    
    % Data augmentation.
    exemplars_ = imdb;
    sz = size(imdb.images.data);
    posTraining = find(imdb.images.set == 1);
    posTest = find(imdb.images.set == 3);
    newSize = (length(posTraining) * 2) + length(posTest);
    sz(end) = newSize;
    exemplarsFinal = imdb;
    exemplarsFinal.images.data = zeros(224, 224, sz(3), sz(4), class(imdb.images.data));
    exemplarsFinal.images.labels = zeros(1, newSize, class(imdb.images.labels));
    exemplarsFinal.images.classes = zeros(1, newSize, class(imdb.images.classes));
    exemplarsFinal.images.set = zeros(1, newSize, class(imdb.images.set));
    
    % Cat data + mirror.
    exemplarsFinal.images.labels(1:length(posTraining)) = exemplars_.images.labels(posTraining);
    exemplarsFinal.images.labels(length(posTraining)+1:2*length(posTraining)) = exemplars_.images.labels(posTraining);
    
    exemplarsFinal.images.classes(1:length(posTraining)) = exemplars_.images.classes(posTraining);
    exemplarsFinal.images.classes(length(posTraining)+1:2*length(posTraining)) = exemplars_.images.classes(posTraining);
    
    exemplarsFinal.images.set(1:length(posTraining)) = exemplars_.images.set(posTraining);
    exemplarsFinal.images.set(length(posTraining)+1:2*length(posTraining)) = exemplars_.images.set(posTraining);
    
    % Training data.
    for i=1:length(posTraining)
        image = exemplars_.images.data(:,:,:,posTraining(i));
        image2 = fliplr(image);
        
        % Brightness.
        if rand() > 0.5
            brightness = unifrnd(-63, 63);
            image = image + brightness;
            image(image > 255) = 255;
            image(image < 0) = 0;
        end
        
        if rand() > 0.5
            brightness = unifrnd(-63, 63);
            image2 = image2 + brightness;
            image2(image2 > 255) = 255;
            image2(image2 < 0) = 0;
        end
        
        % Contrast.
        if rand() > 0.5
            contrast = unifrnd(0.2, 1.8);
            m1 = mean(mean(image(:,:,1)));
            m2 = mean(mean(image(:,:,2)));
            m3 = mean(mean(image(:,:,3)));
            image(:,:,1) = (image(:,:,1) - m1) * contrast + m1;
            image(:,:,2) = (image(:,:,2) - m2) * contrast + m2;
            image(:,:,3) = (image(:,:,3) - m3) * contrast + m3;
            image(image > 255) = 255;
            image(image < 0) = 0;
        end
        
        if rand() > 0.5
            contrast = unifrnd(0.2, 1.8);
            m1 = mean(mean(image2(:,:,1)));
            m2 = mean(mean(image2(:,:,2)));
            m3 = mean(mean(image2(:,:,3)));
            image2(:,:,1) = (image2(:,:,1) - m1) * contrast + m1;
            image2(:,:,2) = (image2(:,:,2) - m2) * contrast + m2;
            image2(:,:,3) = (image2(:,:,3) - m3) * contrast + m3;
            image2(image2 > 255) = 255;
            image2(image2 < 0) = 0;
        end
        
        % Crop.
        if rand() > 0.5
            cropsx = randi(32);
            cropsy = randi(32);
            inx = cropsx;
            enx = inx + 224 - 1;
            iny = cropsy;
            eny = iny + 224 - 1;
            image = image(inx:enx, iny:eny, :);
        else
            image = imresize(image, [224 224]);
        end
        
        if rand() > 0.5
            cropsx = randi(32);
            cropsy = randi(32);
            inx = cropsx;
            enx = inx + 224 - 1;
            iny = cropsy;
            eny = iny + 224 - 1;
            image2 = image2(inx:enx, iny:eny, :);
        else
            image2 = imresize(image2, [224 224]);
        end
        
        exemplarsFinal.images.data(:,:,:,i) = image;
        exemplarsFinal.images.data(:,:,:,i+length(posTraining)) = image2;
    end
    
    % Test data.
    pos = (2*length(posTraining)) + 1;
    exemplarsFinal.images.data(:,:,:,pos:end) = imresize(exemplars_.images.data(:,:,:,posTest), [224 224]);
    exemplarsFinal.images.labels(pos:end) = exemplars_.images.labels(posTest);
    exemplarsFinal.images.classes(pos:end) = exemplars_.images.classes(posTest);
    exemplarsFinal.images.set(pos:end) = exemplars_.images.set(posTest);
    
    exemplarsFinal.images.data(:,:,1,:) = exemplarsFinal.images.data(:,:,1,:) - exemplarsFinal.meta.dataMean(1);
    exemplarsFinal.images.data(:,:,2,:) = exemplarsFinal.images.data(:,:,2,:) - exemplarsFinal.meta.dataMean(2);
    exemplarsFinal.images.data(:,:,3,:) = exemplarsFinal.images.data(:,:,3,:) - exemplarsFinal.meta.dataMean(3);
    clear('exemplars_');
    imdb = exemplarsFinal;
    clear('exemplarsFinal');
    
    if isvector(imdb.images.labels)
        ulabs = net.meta.eqlabs;
        % Set new ones
        newlabs = zeros(size(imdb.images.labels), class(imdb.images.labels));
        for i = 1:length(ulabs)
            idx = imdb.images.labels == ulabs(i);
            newlabs(idx) = i;
        end
        imdb.images.labels = newlabs;
        fprintf('INFO: reorganized new labels labels!\n');
    end
    
    % Get FC exemplars outputs.
    outputs = eval_softmax(net, imdb);
    
    % Build combined imdb.
    imdb.images.distillationLabels = outputs;
    imdb.meta.inputs = net.getInputs();
    pos = -1;
    for i=1:length(imdb.meta.inputs)
        if strcmp(imdb.meta.inputs{i}, 'global_label')
            pos = i;
        end
    end
    
    if pos > 0
        imdb.meta.inputs(pos) = [];
        imdb.meta.inputs{end+1} = 'global_label';
    end
    imdb.opts = opts.train;
end
% Train!
fprintf('INFO: training!\n');
[net, info] = cnn_train_dag_exemplars(net, imdb, @getIncBatch, 'val', find(imdb.images.set == 3), opts.train);
clear('imdb');
opts.net = net;
aux = unique(lastExemplars.images.labels);
nn = sum(lastExemplars.images.set == 1 & lastExemplars.images.labels == aux(1));
aux2 = opts.maxExemplars;
opts.maxExemplars = opts.newtaskdim * nn;
opts.totalClasses = opts.newtaskdim;
exemplars = build_exemplars_set_imagenet([], imdb_or, opts);
opts.totalClasses = length(unique(lastExemplars.images.labels)) + opts.newtaskdim;
opts.maxExemplars = aux2;
opts.derOutputs = derOutputs;
lastExemplars.images.data = cat(4, lastExemplars.images.data, exemplars.images.data);
lastExemplars.images.labels = cat(2, lastExemplars.images.labels, exemplars.images.labels);
lastExemplars.images.classes = cat(2, lastExemplars.images.classes, exemplars.images.classes);
lastExemplars.images.set = cat(2, lastExemplars.images.set, exemplars.images.set);

%% Distillation
imdb = lastExemplars;
imdb.images.data = imresize(imdb.images.data, [224 224]);
if isvector(imdb.images.labels)
    ulabs = net.meta.eqlabs;
    % Set new ones
    newlabs = zeros(size(imdb.images.labels), class(imdb.images.labels));
    for i = 1:length(ulabs)
        idx = imdb.images.labels == ulabs(i);
        newlabs(idx) = i;
    end
    imdb.images.labels = newlabs;
    fprintf('INFO: reorganized new labels labels!\n');
end
imdb.opts = opts.train;
[net, derOutputs] = fork_resnet_distillation(net, ...
    'newtaskdim', opts.newtaskdim, ...
    'distillation_temp', opts.distillation_temp, ...
    'derOutputs', derOutputs);

% Build combined imdb.
% Get FC exemplars outputs.
outputs = eval_softmax(net, imdb);
imdb.images.distillationLabels = outputs;
imdb.meta.inputs = net.getInputs();
pos = -1;
for i=1:length(imdb.meta.inputs)
    if strcmp(imdb.meta.inputs{i}, 'global_label')
        pos = i;
    end
end

if pos > 0
    imdb.meta.inputs(pos) = [];
    imdb.meta.inputs{end+1} = 'global_label';
end

% Train!
fprintf('INFO: training distillation!\n');
opts.train.derOutputs = derOutputs;
opts.train.learningRate = cat(2, opts.train.learningRate, [0.01*ones(1,10) 0.001*ones(1,10) 0.0001*ones(1,10)]) ;
opts.train.numEpochs = length(opts.train.learningRate);
[net, info] = cnn_train_dag_exemplars(net, imdb, @getIncBatchDist, 'val', find(imdb.images.set == 3), opts.train, 'distillation', true);

opts.derOutputs = derOutputs;
[net, derOutputs] = fork_resnet_remove_distillation(net, derOutputs);
opts.train.derOutputs = derOutputs;

opts.net = net;
%exemplars = fc_buildExemplarsSetImagenet(lastExemplars, imdb_or, opts);
ulabs = unique(lastExemplars.images.labels);
nExemplars = floor(opts.maxExemplars / opts.totalClasses);
exemplars = struct();
exemplars.meta = lastExemplars.meta;
exemplars.images.data = [];
exemplars.images.labels = [];
exemplars.images.classes = [];
if isfield(imdb.images, 'coarseLabels');
    exemplars.images.coarseLabels = [];
end
exemplars.images.set = [];
positionsGlobal = [];
for i = 1:length(ulabs)
    opts.n = nExemplars;
    positions = find(lastExemplars.images.labels == ulabs(i) & lastExemplars.images.set == 1);
    nExemplars_ = min(length(positions), nExemplars);
    positionsGlobal = cat(2, positionsGlobal, positions(1:nExemplars_));
    exemplars.images.labels = cat(2, exemplars.images.labels, lastExemplars.images.labels(positions(1:nExemplars_)));
    exemplars.images.classes = cat(2, exemplars.images.classes, lastExemplars.images.classes(positions(1:nExemplars_)));
    if isfield(exemplars.images, 'coarseLabels');
        exemplars.images.coarseLabels = cat(2, exemplars.images.coarseLabels, lastExemplars.images.coarseLabels(positions(1:nExemplars_)));
    end
    exemplars.images.set = cat(2, exemplars.images.set, lastExemplars.images.set(positions(1:nExemplars_)));
end

% Keep all test exemplars.
positions = find(lastExemplars.images.set == 3);
positionsGlobal = cat(2, positionsGlobal, positions);
exemplars.images.labels = cat(2, exemplars.images.labels, lastExemplars.images.labels(positions));
exemplars.images.classes = cat(2, exemplars.images.classes, lastExemplars.images.classes(positions));
if isfield(exemplars.images, 'coarseLabels');
    exemplars.images.coarseLabels = cat(2, exemplars.images.coarseLabels, lastExemplars.images.coarseLabels(positions));
end
exemplars.images.set = cat(2, exemplars.images.set, lastExemplars.images.set(positions));
if isfield(exemplars.meta, 'clusters')
    exemplars.meta.clusters = cat(2, exemplars.meta.clusters, zeros(1, length(positions))-1); % Test images don't have cluster.
end
if isfield(exemplars.images, 'labels_clust')
    exemplars.images.labels_clust = cat(2, exemplars.images.labels_clust, lastExemplars.images.labels_clust(positions));
end
% Concate metadata.
exemplars.meta.classes = cat(2, lastExemplars.meta.classes, imdb_or.meta.classes);
if isfield(exemplars.images, 'coarseLabels');
    exemplars.meta.coarseClasses = cat(2, lastExemplars.meta.coarseClasses, imdb.meta.coarseClasses);
end

sz = size(lastExemplars.images.data);
sz(end) = length(positionsGlobal);
exemplars.images.data = zeros(sz, class(lastExemplars.images.data));
exemplars.images.data(:,:,:,1:length(positionsGlobal)) = lastExemplars.images.data(:,:,:,positionsGlobal);

% Update output
meta.meanval = imdb.meta.dataMean;
meta.meanType = imdb.meta.meanType;
meta.train = opts.train;

opts.net = net;
end

%------------------------------------------------------------------------------------
% INTERNAL FUNCTIONS
%------------------------------------------------------------------------------------
% -------------------------------------------------------------------------
function inputs = getIncBatch(imdb, batch)
% -------------------------------------------------------------------------

images = single(imdb.images.data(:,:,:,batch));

labels = imdb.images.labels(batch) ;
if ~isempty(imdb.opts.gpus)
    images = gpuArray(images) ;
end

j = 1;
jj = 3;
inputs = cell(1, length(imdb.meta.inputs) * 2);
for i=1:length(imdb.meta.inputs)-1
    if strcmp(imdb.meta.inputs{i}, 'image')
        inputs(1:2) = {'image', images};
    else
        inputs(jj:jj+1) = {imdb.meta.inputs{i}, imdb.images.distillationLabels{j}(:, batch)};
        j = j + 1;
        jj = jj + 2;
    end
end

inputs(jj:jj+1) = {imdb.meta.inputs{end}, labels};
end

% -------------------------------------------------------------------------
function inputs = getIncBatchDist(imdb, batch)
% -------------------------------------------------------------------------
images = single(imdb.images.data(:,:,:,batch));

labels = imdb.images.labels(batch) ;
if ~isempty(imdb.opts.gpus)
    images = gpuArray(images) ;
    labels = gpuArray(labels) ;
end

j = 1;
jj = 3;
inputs = cell(1, length(imdb.meta.inputs) * 2);
for i=1:length(imdb.meta.inputs)-1
    if strcmp(imdb.meta.inputs{i}, 'image')
        inputs(1:2) = {'image', images};
    else
        inputs(jj:jj+1) = {imdb.meta.inputs{i}, gpuArray(imdb.images.distillationLabels{j}(:, batch))};
        j = j + 1;
        jj = jj + 2;
    end
end

inputs(jj:jj+1) = {imdb.meta.inputs{end}, labels};
end
