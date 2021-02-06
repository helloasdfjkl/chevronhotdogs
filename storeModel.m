classdef storeModel
    properties
        data % table containing data
        numVars % number of total variables
        numData % number of data points
    end
    methods
        function obj = storeModel(dataIn)
            % Initializes storeModel
            if nargin < 1
                dataIn = [];
            end
            obj.data = dataIn;
            [obj.numData, obj.numVars] = size(dataIn);
        end
        function mdl = linReg(obj, modelspec)
            % Performs linear regression, using modelspec to specify
            % further specifications if desired
            if nargin < 2
                modelspec = 'linear';
            end
            mdl = fitlm(obj.data, modelspec, 'Categorical', 1:(obj.numVars-1));
        end
        function tree = treeReg(obj)
            % Performs binary decision tree regression
            tree = fitrtree(obj.data(:, 1:(obj.numVars - 1)), obj.data(:, obj.numVars),'Categorical','all');
        end
        function mdl = stepWiseReg(obj, modelspec)
            % Performs linear regression with interactions, using modelspec to specify
            % further specifications if desired
            if nargin < 2
                modelspec = 'linear';
            end
            mdl = stepwiselm(obj.data, modelspec, 'Categorical', 1:(obj.numVars-1));
        end
        function val = rmse(obj, mdl)
            % Computes RMSE
            % Input: mdl is a LinearModel object
            observed = table2array(obj.data(:, obj.numVars));
            predicted = obj.getPredict(mdl, table2array(obj.data(:, 1:(obj.numVars-1))));
            val = sqrt(mean((observed-predicted).^2));
        end
        function val = rSq(obj, mdl)
            % Computes R^2
            % Input: mdl is a LinearModel object
            observed = table2array(obj.data(:, obj.numVars));
            predicted = obj.getPredict(mdl, table2array(obj.data(:, 1:(obj.numVars-1))));
            r = corrcoef(observed, predicted);
            val = r(2)^2;
        end
        function plotResidual(obj, mdl)
            % Plots residuals
            % Input: mdl is a LinearModel object
            xdata = 1:obj.numData;
            predicted = obj.getPredict(mdl, table2array(obj.data(:, 1:(obj.numVars-1))));
            ydata = table2array(obj.data(:, obj.numVars)) - predicted();
            bar(xdata, ydata)
        end
        function plotObservedVsPredicted(obj, mdl)
            % Plots observed vs predicted vals
            % Input: mdl is a LinearModel object
            xdata = table2array(obj.data(:, obj.numVars));
            ydata = obj.getPredict(mdl, table2array(obj.data(:, 1:(obj.numVars-1))));
            scatter(xdata, ydata, 10, 'filled')
            hold on
            plot([-100 200], [-100 200])
            hold off
        end
        function plotResponse(obj, mdl)
            % Plots responses vs observations
            % Input: mdl is a LinearModel object
            xdata = 1:obj.numData;
            predicted = obj.getPredict(mdl, table2array(obj.data(:, 1:(obj.numVars-1))));
            actual = table2array(obj.data(:, obj.numVars));
            scatter(xdata, actual, 15, 'filled')
            hold on
            scatter(xdata, predicted, 15, 'filled')
            hold off
            legend('Observed', 'Predicted')
        end
        function sales = getPredict(obj, mdl, input)
            % Computes predicted sales given input
            % Input:    mdl is a LinearModel object
            %           input is an array of independent variable values
            sales = predict(mdl, input);
        end
    end
end
