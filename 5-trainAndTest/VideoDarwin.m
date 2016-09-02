function W = VideoDarwin(data,CVAL)
    % TODO Add paths
    addpath('~/lib/vlfeat/toolbox');
    vl_setup();
    % TODO Add paths
    % add open cv to LD_LIB Path
    setenv('LD_LIBRARY_PATH','~/usr/lib'); 
    % TODO
    % add lib linear to path
    addpath('~/lib/liblinear/matlab');
    % TODO
    % add lib svm to path
    addpath('~/lib/libsvm/matlab');

    if nargin < 2
	CVAL = 1;
    end	
    OneToN = [1:size(data,1)]';    
    Data = cumsum(data);
    Data = Data ./ repmat(OneToN,1,size(Data,2));
    W_fow = liblinearsvr(getNonLinearity(Data),CVAL,2); clear Data; 			
    order = 1:size(data,1);
    [~,order] = sort(order,'descend');
    data = data(order,:);
    Data = cumsum(data);
    Data = Data ./ repmat(OneToN,1,size(Data,2));
    W_rev = liblinearsvr(getNonLinearity(Data),CVAL,2); 			              
    W = [W_fow ; W_rev]; 
end

function w = liblinearsvr(Data,C,normD)
    if normD == 2
        Data = normalizeL2(Data);
    end    
    if normD == 1
        Data = normalizeL1(Data);
    end    
    N = size(Data,1);
    Labels = [1:N]';
    model = train(double(Labels), sparse(double(Data)),sprintf('-c %1.6f -s 11 -q',C) );
    w = model.w';    
end

function Data = getNonLinearity(Data)
    Data = sign(Data).*sqrt(abs(Data));    
    %Data = vl_homkermap(Data',2,'kchi2');    
end

function x = normalizeL2(x)
    x=x./repmat(sqrt(sum(x.*conj(x),2)),1,size(x,2));
end
