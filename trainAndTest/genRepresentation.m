function W = genRepresentation(data,CVAL)
    OneToN = [1:size(data,1)]';    
    Data = cumsum(data);
    Data = Data ./ repmat(OneToN,1,size(Data,2));
    W_fow = liblinearsvr(getNonLinearity(Data),CVAL,2);             
    order = 1:size(data,1);
    [~,order] = sort(order,'descend');
    data = data(order,:);
    Data = cumsum(data);
    Data = Data ./ repmat(OneToN,1,size(Data,2));
    W_rev = liblinearsvr(getNonLinearity(Data),CVAL,2);                           
    W = [W_fow ; W_rev];
end

function Data = getNonLinearity(Data)
    %Data = sign(Data).*sqrt(abs(Data));
    %Data = vl_homkermap(Data',2,'kchi2');
    %Data =  sqrt(abs(Data));                       
    Data =  sqrt(Data);       
end

function w = liblinearsvr(Data,C,normD)
    if normD == 2
        Data = normalizeL2(Data);
    end
    
    if normD == 1
        Data = normalizeL1(Data);
    end
    % in case it is complex, takes only the real part.  
    N = size(Data,1);
    Labels = [1:N]';
    model = train(double(Labels), sparse(double(Data)),sprintf('-c %d -s 11 -q',C) );
    w = model.w';
end

function X = normalizeL2(X)
    for i = 1 : size(X,1)
        if norm(X(i,:)) ~= 0
            X(i,:) = X(i,:) ./ norm(X(i,:));
        end
    end    
end
