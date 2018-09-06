function imdbExemplars_ = build_exemplars_set(lastExemplars, imdb, opts)

% New number of exemplars.
if opts.maxExemplars ~= 0
    nExemplars = floor(opts.maxExemplars / opts.totalClasses);
else
    nExemplars = opts.nExemplarsClass;
end

% Compatibility.
if ~isfield(imdb.images, 'labels')
    imdb.images.labels = imdb.images.classes;
end

% Build new imdb.
imdbExemplars = imdb;
imdbExemplars.images.data = [];
imdbExemplars.images.labels = [];
imdbExemplars.images.classes = [];
if isfield(imdb.images, 'coarseLabels');
    imdbExemplars.images.coarseLabels = [];
end
imdbExemplars.images.set = [];
ulabs = unique(imdb.images.labels);
if isfield(imdb.images, 'labels_clust')
    imdbExemplars.images.labels_clust = [];
end

% Add new training exemplars.
for i = 1:length(ulabs)
    opts.n = nExemplars;
    [positions, cluster] = selectPositions(imdb, ulabs(i), 1, opts);
    nExemplars_ = min(length(positions), nExemplars);
    imdbExemplars.images.data = cat(4, imdbExemplars.images.data, imdb.images.data(:,:,:,positions(1:nExemplars_)));
    imdbExemplars.images.labels = cat(2, imdbExemplars.images.labels, imdb.images.labels(positions(1:nExemplars_)));
    imdbExemplars.images.classes = cat(2, imdbExemplars.images.classes, imdb.images.classes(positions(1:nExemplars_)));
    if isfield(imdbExemplars.images, 'coarseLabels')
        imdbExemplars.images.coarseLabels = cat(2, imdbExemplars.images.coarseLabels, imdb.images.coarseLabels(positions(1:nExemplars_)));
    end
    imdbExemplars.images.set = cat(2, imdbExemplars.images.set, imdb.images.set(positions(1:nExemplars_)));
    if ~isempty(cluster)
        if ~isfield(imdbExemplars.meta, 'clusters')
            imdbExemplars.meta.clusters = cluster(1:nExemplars_);
        else
            imdbExemplars.meta.clusters = cat(2, imdbExemplars.meta.clusters, cluster(1:nExemplars_));
        end
    end
    if isfield(imdb.images, 'labels_clust')
        imdbExemplars.images.labels_clust = cat(2, imdbExemplars.images.labels_clust, imdb.images.labels_clust(positions(1:nExemplars_)));
    end
end

% Keep all test exemplars.
positions = find(imdb.images.set == 3);
imdbExemplars.images.data = cat(4, imdbExemplars.images.data, imdb.images.data(:,:,:,positions));
imdbExemplars.images.labels = cat(2, imdbExemplars.images.labels, imdb.images.labels(positions));
imdbExemplars.images.classes = cat(2, imdbExemplars.images.classes, imdb.images.classes(positions));
if isfield(imdbExemplars.images, 'coarseLabels')
    imdbExemplars.images.coarseLabels = cat(2, imdbExemplars.images.coarseLabels, imdb.images.coarseLabels(positions));
end
imdbExemplars.images.set = cat(2, imdbExemplars.images.set, imdb.images.set(positions));
if isfield(imdbExemplars.meta, 'clusters')
    imdbExemplars.meta.clusters = cat(2, imdbExemplars.meta.clusters, zeros(1, length(positions))-1); % Test images don't have cluster.
end
if isfield(imdbExemplars.images, 'labels_clust')
    imdbExemplars.images.labels_clust = cat(2, imdbExemplars.images.labels_clust, imdb.images.labels_clust(positions));
end

% Concat previous exemplars.
if ~isempty(lastExemplars)
    ulabs = unique(lastExemplars.images.labels);
    oldSize = sum(lastExemplars.images.set == 1);
    newSize = length(ulabs) * nExemplars;
    % Remove old exemplars if necessary.
    if newSize ~= oldSize
        for i = 1:length(ulabs)
            opts.n = nExemplars;
            [positions, cluster] = selectPositions(lastExemplars, ulabs(i), 1, opts);
            nExemplars_ = min(nExemplars, length(positions));
            imdbExemplars.images.data = cat(4, imdbExemplars.images.data, lastExemplars.images.data(:,:,:,positions(1:nExemplars_)));
            imdbExemplars.images.labels = cat(2, imdbExemplars.images.labels, lastExemplars.images.labels(positions(1:nExemplars_)));
            imdbExemplars.images.classes = cat(2, imdbExemplars.images.classes, lastExemplars.images.classes(positions(1:nExemplars_)));
            if isfield(imdbExemplars.images, 'coarseLabels')
                imdbExemplars.images.coarseLabels = cat(2, imdbExemplars.images.coarseLabels, lastExemplars.images.coarseLabels(positions(1:nExemplars_)));
            end
            imdbExemplars.images.set = cat(2, imdbExemplars.images.set, lastExemplars.images.set(positions(1:nExemplars_)));
            
            if ~isempty(cluster)
                if ~isfield(imdbExemplars.meta, 'clusters')
                    imdbExemplars.meta.clusters = cluster(1:nExemplars_);
                else
                    imdbExemplars.meta.clusters = cat(2, imdbExemplars.meta.clusters, cluster(1:nExemplars_));
                end
            end
            
            if isfield(imdbExemplars.images, 'labels_clust')
                imdbExemplars.images.labels_clust = cat(2, imdbExemplars.images.labels_clust, lastExemplars.images.labels_clust(positions(1:nExemplars_)));
            end
        end
    else
        positions = find(lastExemplars.images.set == 1);
        imdbExemplars.images.data = cat(4, imdbExemplars.images.data, lastExemplars.images.data(:,:,:,positions));
        imdbExemplars.images.labels = cat(2, imdbExemplars.images.labels, lastExemplars.images.labels(positions));
        imdbExemplars.images.classes = cat(2, imdbExemplars.images.classes, lastExemplars.images.classes(positions));
        if isfield(imdbExemplars.images, 'coarseLabels')
            imdbExemplars.images.coarseLabels = cat(2, imdbExemplars.images.coarseLabels, lastExemplars.images.coarseLabels(positions));
        end
        imdbExemplars.images.set = cat(2, imdbExemplars.images.set, lastExemplars.images.set(positions));
        if isfield(imdbExemplars.meta, 'clusters')
            imdbExemplars.meta.clusters = cat(2, imdbExemplars.meta.clusters, lastExemplars.meta.clusters(positions));
        end
        if isfield(imdbExemplars.images, 'labels_clust')
            imdbExemplars.images.labels_clust = cat(2, imdbExemplars.images.labels_clust, lastExemplars.images.labels_clust(positions));
        end
    end
    
    % Keep all test exemplars.
    positions = find(lastExemplars.images.set == 3);
    imdbExemplars.images.data = cat(4, imdbExemplars.images.data, lastExemplars.images.data(:,:,:,positions));
    imdbExemplars.images.labels = cat(2, imdbExemplars.images.labels, lastExemplars.images.labels(positions));
    imdbExemplars.images.classes = cat(2, imdbExemplars.images.classes, lastExemplars.images.classes(positions));
    if isfield(imdbExemplars.images, 'coarseLabels')
        imdbExemplars.images.coarseLabels = cat(2, imdbExemplars.images.coarseLabels, lastExemplars.images.coarseLabels(positions));
    end
    imdbExemplars.images.set = cat(2, imdbExemplars.images.set, lastExemplars.images.set(positions));
    if isfield(imdbExemplars.meta, 'clusters')
        imdbExemplars.meta.clusters = cat(2, imdbExemplars.meta.clusters, zeros(1, length(positions))-1); % Test images don't have cluster.
    end
    if isfield(imdbExemplars.images, 'labels_clust')
        imdbExemplars.images.labels_clust = cat(2, imdbExemplars.images.labels_clust, lastExemplars.images.labels_clust(positions));
    end
    % Concate metadata.
    imdbExemplars.meta.classes = cat(2, lastExemplars.meta.classes, imdb.meta.classes);
    imdbExemplars.meta.coarseClasses = cat(2, lastExemplars.meta.coarseClasses, imdb.meta.coarseClasses);
end

% Randomize everything.
perm = randperm(size(imdbExemplars.images.data, 4));
imdbExemplars_.images.data = imdbExemplars.images.data(:,:,:,perm);
imdbExemplars_.images.labels = imdbExemplars.images.labels(perm);
imdbExemplars_.images.classes = imdbExemplars.images.classes(perm);
if isfield(imdbExemplars.images, 'coarseLabels')
    imdbExemplars_.images.coarseLabels = imdbExemplars.images.coarseLabels(perm);
end
imdbExemplars_.images.set = imdbExemplars.images.set(perm);
imdbExemplars_.meta = imdbExemplars.meta;
if isfield(imdbExemplars.meta, 'clusters')
    imdbExemplars_.meta.clusters = imdbExemplars.meta.clusters(perm);
end
if isfield(imdbExemplars.images, 'labels_clust')
    imdbExemplars_.images.labels_clust = imdbExemplars.images.labels_clust(perm);
end
end

function [positions, cluster] = selectPositions(imdb, label, set, opts)
cluster = [];
positions = find(imdb.images.labels == label & imdb.images.set == set);
if opts.n < length(positions) % Herding.
    net = opts.net;
    outputs = eval_pool(net, imdb);
    
    % Select positions.
    pos = find(imdb.images.set == set & imdb.images.labels == label);
    outputs = outputs(:, pos);
    
    if length(pos) == 500
        % L2-norm.
        for i = 1:size(outputs, 2)
            outputs(:, i) = outputs(:, i) / norm(outputs(:, i));
        end
        
        % Mean.
        mu = mean(outputs, 2)';
        
        % Ranking.
        alpha_dr_herding = zeros(1, size(outputs, 2));
        iter_herding = 0;
        iter_herding_eff = 0;
        w_t = mu;
        while sum(alpha_dr_herding ~=0) < min(opts.n, 500) && iter_herding_eff < 1000
            tmp_t = w_t * outputs;
            [~, ind_max] = max(tmp_t);
            iter_herding_eff = iter_herding_eff + 1;
            if alpha_dr_herding(ind_max) == 0
                alpha_dr_herding(ind_max) = 1 + iter_herding;
                iter_herding = iter_herding + 1;
            end
            w_t = w_t + mu - outputs(:, ind_max)';
        end
        
        positions = pos(find(alpha_dr_herding > 0 & alpha_dr_herding <= opts.n));
        positions = cat(2, positions, pos(find(alpha_dr_herding == 0)));
    else
        positions = pos(1:opts.n);
    end
end
end

