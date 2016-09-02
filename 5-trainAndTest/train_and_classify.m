function [ap ] = train_and_classify(TrainData_Kern,TestData_Kern,TrainClass,TestClass)
         nTrain = 1 : size(TrainData_Kern,1);
         TrainData_Kern = [nTrain' TrainData_Kern];         
         nTest = 1 : size(TestData_Kern,1);
         TestData_Kern = [nTest' TestData_Kern];
         C = [0.01 0.1 1 5 10 50 100 500 1000];
         % TODO : Note that here it is best to do the cross validation on training set.
         %warning('It is best to do the cross validation on training set. Skipping cross validation!');
         C = [100];
         model = svmtrain(TrainClass, TrainData_Kern, sprintf('-t 4 -c %1.6f  -q ',C));
         [~, acc, scores] = svmpredict(TestClass, TestData_Kern ,model);                     
         [rc, pr, info] = vl_pr(TestClass, scores(:,1)) ; 
         ap = info.ap;      
end

