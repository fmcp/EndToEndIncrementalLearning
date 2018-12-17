function [net, derOutputs] = fork_resnet_distillation(net, varargin)
% Based on Zhizhong Li's code
% Modified by F. Castro for End-to-End Incremental Learning. ECCV2018

opts.newtaskdim = 20; % # output of last layer
opts.distillation_temp = 2; % distillation temperature.
opts.derOutputs = {};
opts = vl_argparse(opts, varargin);

opts.mode = 'multiclass'; % type of last layer of the new path
opts.keep_response_loss = 'MI'; % MI for mutual information, L1 for L1-norm. only works when orig_loss is 'for_keep'.
opts.origstyle = 'multiclass';
derOutputs = opts.derOutputs;

%% Update loss layer for the old layer. Only the "last new" task is updated.
if strcmp(opts.orig_loss, 'for_keep')
    switch opts.mode
        case {'multiclass', 'multiclasshinge'}
            layeropts.origstyle = 'multiclass';
        case 'multilabel'
            layeropts.origstyle = 'multilabel';
    end
    
    % Find how many old fc functions we have.
    index = strfind({net.layers.name}, 'fc');
    index = find(not(cellfun('isempty', index)));

    index2 = strfind({net.layers.name}, 'loss_distillation');
    index2 = find(not(cellfun('isempty', index2)));
    if isempty(index2)
        net.addLayer('loss_distillation', dagnn.SoftmaxDiffLoss('mode', opts.keep_response_loss, 'temperature', opts.distillation_temp, 'origstyle', layeropts.origstyle), {net.layers(index(end)).name, 'dist_labels'}, 'loss_distillation') ;
    end
    derOutputs = [derOutputs {'loss_distillation', 1}]
end



