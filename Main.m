%% Setup
clc, clear all
filename = "training_processed.xlsx";
sheetname = "Export";
data = readtable(filename, 'Sheet', sheetname);
model = storeModel(data);

%% Model
clc
regression = model.treeReg()

%% Visualization and validation
clc
rmse = model.rmse(regression);
disp(strcat("RMSE: ", num2str(rmse)))

r2 = model.rSq(regression);
disp(strcat("R^2: ", num2str(r2)))

view(regression,'Mode','graph')

figname = 'Residual plot';
figure('Name', figname)
model.plotResidual(regression);
xlabel('Record number'), ylabel('Residual'),title(figname)

figname = 'Observed vs Predicted';
figure('Name', figname)
model.plotObservedVsPredicted(regression);
xlabel('Observed sales'), ylabel('Predicted sales'),title(figname)
xlim([-10 190]), ylim([-10 190])

figname = 'Response plot';
figure('Name', figname)
model.plotResponse(regression);
xlabel('Record number'), ylabel('Sales'),title(figname)

%% Test model
clc
testfile = "scoring.xlsx";
testSheet = "export";
testData = readtable(testfile, "Sheet", testSheet);
[numData, numVars] = size(testData);
input = testData(:, 1:numVars - 1);
observed = table2array(testData(:, numVars));
predictions = round(model.getPredict(regression, input));

writetable(table(predictions), "Scoring Data Output.xlsx")

%% Testing visualization
clc
rmse = sqrt(mean((observed-predictions).^2));
disp(strcat("RMSE: ", num2str(rmse)))

figname = 'Residual plot (Testing data)';
figure('Name', figname)
xvals = 1:numData;
yvals = observed - predictions;
bar(xvals, yvals)
xlabel('Record number'), ylabel('Residual'),title(figname)

figname = 'Observed vs Predicted (Testing data)';
figure('Name', figname)
scatter(observed, predictions, 10, 'filled')
hold on
plot([-100 200], [-100 200])
hold off
xlabel('Observed sales'), ylabel('Predicted sales'),title(figname)
xlim([-10 190]), ylim([-10 190])

figname = 'Response plot (Testing data)';
figure('Name', figname)
xvals = 1:numData;
scatter(xvals, observed, 15, 'filled')
hold on
scatter(xvals, predictions, 15, 'filled')
hold off
legend('Observed', 'Predicted')
xlabel('Record number'), ylabel('Sales'),title(figname)