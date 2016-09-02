
function [fullvideoname, videoname, classlabel,vocabDir,featDir,actionName,descriptor_path] = getConfig()
    % TODO : Change the paths
     if exist('~/remote/Hollywood2/train_test_split.mat','file') == 2     
        load('~/remote/Hollywood2/train_test_split.mat');
     else
         urlwrite('http://users.cecs.anu.edu.au/~basura/data/train_test_split.mat','train_test_split.mat');
         load('train_test_split.mat');
     end
    
    vocabDir = '~/remote/Data/Vocab'; % Path where dictionary/GMM will be saved.
    featDir = '~/remote/Data/feats'; % Path where features will be saved
    descriptor_path = '~/remote/Data/descriptor/'; % change paths here 
    
    for i = 1 : length(fnames)
        fullvideoname{i,1}=fullfile('~/remote/Hollywood2/AVIClips',sprintf('%s.avi',fnames{i}));
        videoname{i,1} = fnames{i};
    end
    classlabel = labels2;    
    actionName = {'AnswerPhone','DriveCar','Eat','FightPerson','GetOutCar','HandShake','HugPerson','Kiss','Run','SitDown','SitUp','StandUp'};       
end
