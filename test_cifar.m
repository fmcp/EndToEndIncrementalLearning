imdbs_dir = ''; % Edit me!
nets_dir = ''; % Edit me!
sufix = '-Herding-000-2000'; % Edit me!

batchs = [2]; % Edit me!
nIters = 1; % Edit me!

if ~exist('gpuId', 'var')
    gpuId = 1; % Edit me!
end

for nbatch_idx=1:length(batchs)
    nblocks = 100 / batchs(nbatch_idx);
    for niter_idx=1:nIters
        for nblock_idx=1:nblocks
            if nblock_idx == 1
		% Initial network.
                net_pattern = sprintf('cifar-resnet-32-batch%02d-block%02d-iter%02d', batchs(nbatch_idx), nblock_idx, niter_idx);
                results_path = fullfile(nets_dir, sprintf('cifar-resnet-32-batch%02d-block%02d-iter%02d-V4', batchs(nbatch_idx), nblock_idx, niter_idx), 'results');
                net_name = 'net-epoch-100.mat';
            else
                net_pattern = sprintf('cifar-resnet-32-batch%02d-block%02d-iter%02d%s', batchs(nbatch_idx), nblock_idx, niter_idx, sufix);
                results_path = fullfile(nets_dir, sprintf('cifar-resnet-32-batch%02d-block%02d-iter%02d%s', batchs(nbatch_idx), nblock_idx, niter_idx, sufix), 'results');
                net_name = 'net-final.mat';
	        %net_name = 'net-epoch-100.mat';
            end
            
            if ~exist(results_path, 'dir')
                mkdir(results_path);
            end
            
            % Load net.
            netPath = fullfile(nets_dir, net_pattern, net_name);
            outpath = fullfile(results_path, 'results.mat');
            if exist(netPath, 'file') && ~exist(outpath, 'file')
                load(netPath);
                net = dagnn.DagNN.loadobj(net);
                net.mode = 'test';
                
		% Parse labels to fit number of classes
		if ~isfield(net.meta, 'eqlabs')
    			net.meta.eqlabs = sort(net.meta.classes.name);
		end

                estim_labels = [];
                labels = [];
                for nimdb_idx=1:nblock_idx
                    % Load imdb.
                    imdb_pattern = sprintf('cifar-100-%02d-%02d-%02d.mat', batchs(nbatch_idx), nimdb_idx, niter_idx);
                    imdbPath = fullfile(imdbs_dir, imdb_pattern);
                    load(imdbPath);
                    imdb.images.labels = imdb.images.classes;
                    
                    % Only test samples.
                    positions = find(imdb.images.set == 3);
                    imdb.images.data = imdb.images.data(:,:,:,positions);
                    imdb.images.labels = imdb.images.labels(1, positions);
                    imdb.images.set = imdb.images.set(1, positions);
                    imdb.images.classes = imdb.images.classes(1, positions);
                    imdb.images.coarseLabels = imdb.images.coarseLabels(1, positions);
                    
                    % Eval imdb.
                    outputs = eval_test(net, imdb);
                    scores = outputs{end};
                    
                    % Compute labels.
                    estim_labels_ = zeros(1, size(scores, 2));
                    for nscore_idx=1:size(scores, 2)
                        [~, index] = max(scores(:, nscore_idx));
                        estim_labels_(nscore_idx) = net.meta.eqlabs(index);
                    end
                    labels = cat(2, labels, imdb.images.labels);
                    estim_labels = cat(2, estim_labels, estim_labels_);
                end
                
                % Compute acc.
                acc = sum(estim_labels == labels);
                acc = acc / length(labels)
                
                results = struct();
                results.acc = acc;
                results.estim_labels = estim_labels;
                results.labels = labels;
                
                % Save model.
                save(outpath, 'results');
            end
        end
    end
end
