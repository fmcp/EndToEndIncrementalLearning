function [net, info, meta, exemplars] = incremental_training(net, imdb_or, lastExemplars, opts)

if ~isfield(opts, 'freezeWeights')
    opts.freezeWeights = false;
end

% Add new layers.
[net, derOutputs] = fork_resnet(net, ...
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

% Build imdb with new imdb + exemplars.
imdb.meta.exemplars = cat(2, ones(1, length(lastExemplars.images.labels)), zeros(1, length(imdb.images.labels)));
imdb.meta.classes = cat(2, lastExemplars.meta.classes, imdb.meta.classes);
imdb.images.data = cat(4, lastExemplars.images.data, imdb.images.data);
imdb.images.labels = cat(2, lastExemplars.images.labels, imdb.images.labels);
imdb.images.classes = cat(2, lastExemplars.images.classes, imdb.images.classes);
imdb.images.coarseLabels = cat(2, lastExemplars.images.coarseLabels, imdb.images.coarseLabels);
imdb.images.set = cat(2, lastExemplars.images.set, imdb.images.set);

% Data augmentation.
exemplars_ = imdb;
exemplars_.images.data = bsxfun(@plus, exemplars_.images.data, imdb.meta.dataMean);
exemplars_.images.data = exemplars_.images.data * 255;
sz = size(imdb.images.data);
posTraining = find(imdb.images.set == 1);
posTest = find(imdb.images.set == 3);
newSize = (length(posTraining) * 12) + length(posTest);
sz(end) = newSize;
exemplarsFinal = imdb;
exemplarsFinal.images.data = zeros(sz, class(imdb.images.data));
exemplarsFinal.images.labels = zeros(1, newSize, class(imdb.images.labels));
exemplarsFinal.images.classes = zeros(1, newSize, class(imdb.images.classes));
exemplarsFinal.images.coarseLabels = zeros(1, newSize, class(imdb.images.coarseLabels));
exemplarsFinal.images.set = zeros(1, newSize, class(imdb.images.set));
pos = 1;
% Training data with data augmentation.
for i=1:length(posTraining)
    szAux = sz;
    szAux(end) = 12;
    moreImages = zeros(szAux, class(exemplars_.images.data));
    moreImages(:,:,:,1) = exemplars_.images.data(:,:,:,posTraining(i));
    
    % Brightness.
    brightness = unifrnd(-63, 63);
    image = exemplars_.images.data(:,:,:,posTraining(i)) + brightness;
    image(image > 255) = 255;
    image(image < 0) = 0;
    moreImages(:,:,:,2) = image;
    clear('image');
    
    % Contrast.
    contrast = unifrnd(0.2, 1.8);
    m1 = mean(mean(exemplars_.images.data(:,:,1,posTraining(i))));
    m2 = mean(mean(exemplars_.images.data(:,:,2,posTraining(i))));
    m3 = mean(mean(exemplars_.images.data(:,:,3,posTraining(i))));
    image2 = exemplars_.images.data(:,:,1,posTraining(i));
    image2(:,:,1) = (exemplars_.images.data(:,:,1,posTraining(i)) - m1) * contrast + m1;
    image2(:,:,2) = (exemplars_.images.data(:,:,2,posTraining(i)) - m2) * contrast + m2;
    image2(:,:,3) = (exemplars_.images.data(:,:,3,posTraining(i)) - m3) * contrast + m3;
    image2(image2 > 255) = 255;
    image2(image2 < 0) = 0;
    moreImages(:,:,:,3) = image2;
    clear('image2');
    
    % Crop.
    images = moreImages(:,:,:,1:3);
    augmented = padarray(images,[4 4 0 0],'both');
    cropsx = randi(8, 1, 3);
    cropsy = randi(8, 1, 3);
    for jj=1:size(images, 4)
        inx = cropsx(jj);
        enx = inx + size(images, 1) - 1;
        iny = cropsy(jj);
        eny = iny + size(images, 2) - 1;
        images(:,:,:,jj) = augmented(inx:enx, iny:eny, :, jj);
    end
    moreImages(:,:,:,4:6) = images;
    clear('images');
    
    % Mirror.
    moreImages(:,:,:,7:end) = fliplr(moreImages(:,:,:,1:6));
    
    % Cat data.
    exemplarsFinal.images.data(:,:,:,pos:pos+12-1) = moreImages;
    exemplarsFinal.images.labels(pos:pos+12-1) = exemplars_.images.labels(posTraining(i));
    exemplarsFinal.images.classes(pos:pos+12-1) = exemplars_.images.classes(posTraining(i));
    exemplarsFinal.images.coarseLabels(pos:pos+12-1) = exemplars_.images.coarseLabels(posTraining(i));
    exemplarsFinal.images.set(pos:pos+12-1) = exemplars_.images.set(posTraining(i));
    pos = pos + 12;
    clear('moreImages');
end

% Test data.
exemplarsFinal.images.data(:,:,:,pos:end) = exemplars_.images.data(:,:,:,posTest);
exemplarsFinal.images.labels(pos:end) = exemplars_.images.labels(posTest);
exemplarsFinal.images.classes(pos:end) = exemplars_.images.classes(posTest);
exemplarsFinal.images.coarseLabels(pos:end) = exemplars_.images.coarseLabels(posTest);
exemplarsFinal.images.set(pos:end) = exemplars_.images.set(posTest);

exemplarsFinal.images.data = exemplarsFinal.images.data / 255.0;
exemplarsFinal.images.data = bsxfun(@minus, exemplarsFinal.images.data, exemplarsFinal.meta.dataMean);
clear('exemplars_');
imdb = exemplarsFinal;

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

% Train!
fprintf('INFO: training!\n');
[net, ~] = cnn_train_dag_exemplars(net, imdb, @getIncBatch, 'val', find(imdb.images.set == 3), opts.train);
opts.net = net;
aux = unique(lastExemplars.images.labels);
nn = sum(lastExemplars.images.set == 1 & lastExemplars.images.labels == aux(1));
aux2 = opts.maxExemplars;
opts.maxExemplars = opts.maxExemplars + (opts.newtaskdim * nn);
opts.totalClasses = length(unique(lastExemplars.images.labels)) + opts.newtaskdim;
exemplars = build_examplars_set(lastExemplars, imdb_or, opts);
opts.maxExemplars = aux2;
opts.derOutputs = derOutputs;

%% Distillation
% Data augmentation.
exemplars_ = exemplars;
exemplars_.images.data = bsxfun(@plus, exemplars_.images.data, imdb.meta.dataMean);
exemplars_.images.data = exemplars_.images.data * 255;
sz = size(exemplars.images.data);
posTraining = find(exemplars.images.set == 1);
posTest = find(exemplars.images.set == 3);
newSize = (length(posTraining) * 12) + length(posTest);
sz(end) = newSize;
exemplarsFinal = exemplars;
exemplarsFinal.images.data = zeros(sz, class(exemplars.images.data));
exemplarsFinal.images.labels = zeros(1, newSize, class(exemplars.images.labels));
exemplarsFinal.images.classes = zeros(1, newSize, class(exemplars.images.classes));
exemplarsFinal.images.coarseLabels = zeros(1, newSize, class(exemplars.images.coarseLabels));
exemplarsFinal.images.set = zeros(1, newSize, class(exemplars.images.set));
pos = 1;
% Training data.
for i=1:length(posTraining)
    szAux = sz;
    szAux(end) = 12;
    moreImages = zeros(szAux, class(exemplars_.images.data));
    moreImages(:,:,:,1) = exemplars_.images.data(:,:,:,posTraining(i));
    
    % Brightness.
    brightness = unifrnd(-63, 63);
    image = exemplars_.images.data(:,:,:,posTraining(i)) + brightness;
    image(image > 255) = 255;
    image(image < 0) = 0;
    moreImages(:,:,:,2) = image;
    clear('image');
    
    % Contrast.
    contrast = unifrnd(0.2, 1.8);
    m1 = mean(mean(exemplars_.images.data(:,:,1,posTraining(i))));
    m2 = mean(mean(exemplars_.images.data(:,:,2,posTraining(i))));
    m3 = mean(mean(exemplars_.images.data(:,:,3,posTraining(i))));
    image2 = exemplars_.images.data(:,:,1,posTraining(i));
    image2(:,:,1) = (exemplars_.images.data(:,:,1,posTraining(i)) - m1) * contrast + m1;
    image2(:,:,2) = (exemplars_.images.data(:,:,2,posTraining(i)) - m2) * contrast + m2;
    image2(:,:,3) = (exemplars_.images.data(:,:,3,posTraining(i)) - m3) * contrast + m3;
    image2(image2 > 255) = 255;
    image2(image2 < 0) = 0;
    moreImages(:,:,:,3) = image2;
    clear('image2');
    
    % Crop.
    images = moreImages(:,:,:,1:3);
    augmented = padarray(images,[4 4 0 0],'both');
    cropsx = randi(8, 1, 3);
    cropsy = randi(8, 1, 3);
    for jj=1:size(images, 4)
        inx = cropsx(jj);
        enx = inx + size(images, 1) - 1;
        iny = cropsy(jj);
        eny = iny + size(images, 2) - 1;
        images(:,:,:,jj) = augmented(inx:enx, iny:eny, :, jj);
    end
    moreImages(:,:,:,4:6) = images;
    clear('images');
    
    % Mirror.
    moreImages(:,:,:,7:end) = fliplr(moreImages(:,:,:,1:6));
    
    % Cat data.
    exemplarsFinal.images.data(:,:,:,pos:pos+12-1) = moreImages;
    exemplarsFinal.images.labels(pos:pos+12-1) = exemplars_.images.labels(posTraining(i));
    exemplarsFinal.images.classes(pos:pos+12-1) = exemplars_.images.classes(posTraining(i));
    exemplarsFinal.images.coarseLabels(pos:pos+12-1) = exemplars_.images.coarseLabels(posTraining(i));
    exemplarsFinal.images.set(pos:pos+12-1) = exemplars_.images.set(posTraining(i));
    pos = pos + 12;
    clear('moreImages');
end

% Test data.
exemplarsFinal.images.data(:,:,:,pos:end) = exemplars_.images.data(:,:,:,posTest);
exemplarsFinal.images.labels(pos:end) = exemplars_.images.labels(posTest);
exemplarsFinal.images.classes(pos:end) = exemplars_.images.classes(posTest);
exemplarsFinal.images.coarseLabels(pos:end) = exemplars_.images.coarseLabels(posTest);
exemplarsFinal.images.set(pos:end) = exemplars_.images.set(posTest);

exemplarsFinal.images.data = exemplarsFinal.images.data / 255.0;
exemplarsFinal.images.data = bsxfun(@minus, exemplarsFinal.images.data, exemplarsFinal.meta.dataMean);
clear('exemplars_');

imdb = exemplarsFinal;
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
[net, derOutputs] = fork_resnet_remove_distillation(net);
opts.train.derOutputs = derOutputs;

opts.net = net;
exemplars = build_examplars_set(lastExemplars, imdb_or, opts);

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
images = imdb.images.data(:,:,:,batch);
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
images = imdb.images.data(:,:,:,batch);
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
