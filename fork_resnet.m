function [net, derOutputs] = fork_resnet(net, varargin)
% Based on Zhizhong Li's code
% Medified by F. Castro for End-to-End Incremental Learning. ECCV2018

opts.newtaskdim = 20; % # output of last layer
opts.distillation_temp = 2; % distillation temperature.
opts = vl_argparse(opts, varargin);

opts.mode = 'multiclass'; % type of last layer of the new path
opts.keep_response_loss = 'MI'; % MI for mutual information, L1 for L1-norm. only works when orig_loss is 'for_keep'.
opts.origstyle = 'multiclass';

derOutputs = {};

%% Update loss layer for the old layer. Only the "last new" task is updated.
    % Remove previous softmax_global layer.
    index = strfind({net.layers.name}, 'softmax_global');
    index = find(not(cellfun('isempty', index)));
    global_layer = false;
    for i=1:length(index)
        net.removeLayer(net.layers(index(i)).name);
        global_layer = true;
    end
    
    % Find how many old loss functions we have.
    index = strfind({net.layers.name}, 'loss_old');
    index = find(not(cellfun('isempty', index)));
    last = length(index);
    
    % Remove previous loss layers and add the new ones.
    index = strfind({net.layers.name}, 'loss');
    index = find(not(cellfun('isempty', index)));
    
    % Only for loss (not for loss_taskN)
    inputs = cell(1, length(index));
    for iloss_idx=1:length(index)
        if strcmp(net.layers(index(iloss_idx)).name, 'loss')
            if global_layer
                inputs_ = net.layers(index(iloss_idx)).inputs;
                inputs_{1} = net.layers(index(iloss_idx)-3).name;
                inputs_{2} = sprintf('label_task%d', last+1);
            else
                inputs_ = net.layers(index(iloss_idx)).inputs;
                inputs_{1} = net.layers(index(iloss_idx)-2).name;
            end
            
            % Remove old layer.
            inputs{iloss_idx} = inputs_{1};
            
            outputs_ = net.layers(index(iloss_idx)).outputs;
            for iout = 1:length(outputs_)
                outputs_{iout} = sprintf('%s_old_%02d', outputs_{iout}, last+1);
            end
            net.removeLayer(net.layers(index(iloss_idx)).name);
            
            net.addLayer(sprintf('loss_old_%02d', last+1), dagnn.SoftmaxDiffLoss('mode', opts.keep_response_loss, 'temperature', opts.distillation_temp, 'origstyle', opts.origstyle) , inputs_, outputs_) ;
            derOutputs = [derOutputs {sprintf('loss_old_%02d', last+1), 1}]
        else
            inputs_ = net.layers(index(iloss_idx)).inputs;
            inputs_{1} = net.layers(index(iloss_idx)-2).name;
            inputs{iloss_idx} = inputs_{1};
            derOutputs = [derOutputs {sprintf('loss_old_%02d', iloss_idx), 1}];
        end
    end

%% Add new layers
% Find a previous FC layer to obtain parameters.
index = strfind({net.layers.name}, 'fc');
index = find(not(cellfun('isempty', index)));
last = length(index);

inputs_ = net.layers(index(end)).inputs;
sz_ = net.layers(index(end)).block.size;
sz_(end) = opts.newtaskdim;

% Add new FC layer.
block = dagnn.Conv('size', sz_, 'hasBias', true, 'stride', 1, 'pad', 0);
lName = sprintf('fc_task%d', last+1);
net.addLayer(lName, block, inputs_, lName, {sprintf('%s_f', lName), sprintf('%s_b', lName)});

% Set weights.
params = block.initParams();
net.params(net.layers(net.getLayerIndex(lName)).paramIndexes(1)).value = params{1};
net.params(net.layers(net.getLayerIndex(lName)).paramIndexes(2)).value = params{2};

% Add Softmax.
net.addLayer(sprintf('softmax_task%d', last+1), dagnn.SoftMax(), lName, sprintf('softmax_task%d', last+1));

% Remove previous concatenation layer (if exists).
index = strfind({net.layers.name}, 'concat');
index = find(not(cellfun('isempty', index)));
if ~isempty(index)
    net.removeLayer('concat');
end

% Add concatenation layer.
inputs{end+1} = lName;
block = dagnn.Concat('dim', 3);
net.addLayer('concat', block, inputs, {'concat'}, {});

% Add global classification loss.
net.addLayer('loss', dagnn.Loss('loss', 'softmaxlog'), {'concat', 'global_label'}, 'loss');

% Add Softmax.
net.addLayer('softmax_global', dagnn.SoftMax(), 'concat', 'softmax_global');

index = strfind({net.layers.name}, 'error');
index = find(not(cellfun('isempty', index)));
net.removeLayer(net.layers(index(1)).name);

% Add error layer for the new task.
net.addLayer('error', dagnn.Loss('loss', 'classerror'), {'softmax_global', 'global_label'}, 'error') ;
% Set derOutputs.
derOutputs = [derOutputs {'loss', 1}] % 0.5



