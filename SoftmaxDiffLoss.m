classdef SoftmaxDiffLoss < dagnn.ElementWise
    properties
        opts = {}
        mode = 'MI'
        temperature = 2
        origstyle = 'multiclass'
        opts_vl = struct()
    end
    
    properties (Transient)
        average = 0
        numAveraged = 0
    end
    
    methods
        function outputs = forward(obj, inputs, params)
            outputs{1} = vl_nnsoftmaxdiff(inputs{1}, inputs{2}, [], obj.opts_vl) ;
            n = obj.numAveraged ;
            m = n + size(inputs{1},4) ;
            obj.average = bsxfun(@plus, n * obj.average, gather(outputs{1})) / m ;
            obj.numAveraged = m ;
        end
        
        function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
            derInputs{1} = vl_nnsoftmaxdiff(inputs{1}, inputs{2}, derOutputs{1}, obj.opts_vl) ;
            derInputs{2} = [] ;
            derParams = {} ;
        end
        
        function reset(obj)
            obj.average = 0 ;
            obj.numAveraged = 0 ;
        end
        
        function outputSizes = getOutputSizes(obj, inputSizes, paramSizes)
            outputSizes{1} = [1 1 1 inputSizes{1}(4)] ;
        end
        
        function rfs = getReceptiveFields(obj)
            % the receptive field depends on the dimension of the variables
            % which is not known until the network is run
            rfs(1,1).size = [NaN NaN] ;
            rfs(1,1).stride = [NaN NaN] ;
            rfs(1,1).offset = [NaN NaN] ;
            rfs(2,1) = rfs(1,1) ;
        end
        
        function obj = SoftmaxDiffLoss(varargin)
            obj.load(varargin) ;
            obj.opts_vl.mode = obj.mode;
            obj.opts_vl.temperature = obj.temperature;
            obj.opts_vl.origstyle = obj.origstyle;
        end
    end
end

