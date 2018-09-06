% Build CIFAR-100 imdbs following the guidelines of iCaRL: Incremental Classifier and Representation Learning
% https://arxiv.org/abs/1611.07725

max_iters = 5;
nclasses = 100;
batch_sizes = [2 5 10 20 50];
imdb_path = '/home/GAIT_local/Datasets/cifar-100-matlab/train.mat'; % Edit me!
imdbtest_path = '/home/GAIT_local/Datasets/cifar-100-matlab/test.mat'; % Edit me!
metadata = '/home/GAIT_local/Datasets/cifar-100-matlab/meta.mat'; % Edit me!
outdir = '/home/GAIT/SSD/cifar100_incremental'; % Edit me!

%% Build base imdb
train = load(imdb_path);
test = load(imdbtest_path);

% Preapre the imdb structure, returns image train.data with mean image subtracted
train.data = single(permute(reshape(train.data',32,32,3,[]),[2 1 3 4])) ;
test.data = single(permute(reshape(test.data',32,32,3,[]),[2 1 3 4])) ;

data = cat(4, train.data, test.data);
set = cat(2, ones(1, size(train.data, 4)), ones(1, size(test.data, 4))*3);

% Build imdb.
load(metadata);

imdb_.images.data = data ;
imdb_.images.labels = cat(2, single(train.fine_labels)', single(test.fine_labels)') + 1;
imdb_.images.coarseLabels = cat(2, single(train.coarse_labels'), single(test.coarse_labels')) + 1;
imdb_.images.set = set ;
imdb_.images.files = {train.filenames{:} , test.filenames{:}} ;
imdb_.meta.sets = {'train', 'val', 'test'} ;
imdb_.meta.classes = fine_label_names;
imdb_.meta.coarseClasses = coarse_label_names;
imdb_.meta.meanType = 'image';

%% Split into small imdbs
for nit=1:max_iters
    order = randperm(nclasses);
    for i=1:length(batch_sizes) % Batch_sizes
        nParts = nclasses / batch_sizes(i);
        for j=1:nParts
            %% Cut data
            % Find positions
            in = (j-1) * batch_sizes(i) + 1;
            en = in + batch_sizes(i) - 1;
            
            % Build imdb.images
            classes = order(in:en);
            positions = find(ismember(imdb_.images.labels, classes));
            imdb.images.data = imdb_.images.data(:, :, :, positions);
            imdb.images.classes = imdb_.images.labels(positions);
            imdb.images.coarseLabels = imdb_.images.coarseLabels(positions);
            imdb.images.set = imdb_.images.set(positions);
            
            % Data mean.
            imdb.images.data = imdb.images.data / 255.0;
            dataMean = mean(imdb.images.data(:,:,:,imdb.images.set == 1), 4);
            imdb.images.data = bsxfun(@minus, imdb.images.data, dataMean);
            
            % Build imdb.meta
            imdb.meta.sets = imdb_.meta.sets;
            imdb.meta.classes = classes;
            imdb.meta.coarseClasses = unique(imdb.images.coarseLabels);
            imdb.meta.dataMean = dataMean;
            imdb.meta.meanType = imdb_.meta.meanType;
            imdb.meta.whitenData = 0;
            imdb.meta.contrastNormalization = 0;
            
            %% Save imdb
            outname = sprintf('cifar-%d-%02d-%02d-%02d.mat', nclasses, batch_sizes(i), j, nit);
            outpath = fullfile(outdir, outname);
            save(outpath, 'imdb');
            clear imdb;
        end
    end
end
