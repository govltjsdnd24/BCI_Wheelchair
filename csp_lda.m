eeglab;
clear; clc;
d = '123_jooyoung.edf';
% d = './records/12345_doyoung.edf';
% d = './records/edf_record-[2023.04.07-20.58.08].edf';
% d = './records/jooyoung_50trials.edf';
% d = './records/sunoong_50trials.edf';
EEG = pop_biosig(d);
%% 전체 CAR 처리
EEG = pop_reref( EEG, []); 
EEG = eeg_checkset( EEG );

%% session 찾기(openvibe에서 이어 붙인 경우 잘라서 bp filter하기 위해)
events = struct2cell(EEG.event);
event_names = squeeze(events(1,1,:));
event_latency = squeeze(events(3,1,:));
% sessions_idx = [1 603];
sessions_idx = [ ...
    find(strcmp(event_names, 'OVTK_StimulationId_ExperimentStart')), ...
    find(strcmp(event_names, 'OVTK_GDF_End_Of_Session'))];
%% session 별로 bp filtering.
bp_wnd = [8 30];
for i=1:size(sessions_idx,1)
    EEG_tmp = pop_select(EEG, 'point', [cell2mat(event_latency(sessions_idx(i,1))), ...
        cell2mat(event_latency(sessions_idx(i,2)))]); EEG_tmp = eeg_checkset( EEG_tmp );
    EEG_tmp = pop_eegfiltnew(EEG_tmp, 'locutoff',bp_wnd(1), ...
        'hicutoff',bp_wnd(2),'plotfreqz',0); EEG_tmp = eeg_checkset( EEG_tmp );
    EEG.data(:,cell2mat(event_latency(sessions_idx(i,1))) ...
        :cell2mat(event_latency(sessions_idx(i,2)))) = EEG_tmp.data; %EEG = eeg_checkset( EEG );
end
%% epoch(0.5~4) 해주기
wnd_size=[0.5 4.0];
rightEEG = pop_epoch( EEG, { 'OVTK_GDF_Right' }, wnd_size);
leftEEG = pop_epoch( EEG, { 'OVTK_GDF_Left' }, wnd_size');
%% baseline correction
epochNum = size(leftEEG.epoch,2);
left = zeros(size(leftEEG.data,2), size(leftEEG.data,1), epochNum);
right = zeros(size(rightEEG.data,2), size(rightEEG.data,1), epochNum);
for i=1:epochNum
    left_tmp = leftEEG.data(:,:,i)';
    right_tmp = rightEEG.data(:,:,i)';
    % figure;
    % plot(right_tmp(:,ch)); hold on;
    left(:,:,i) = left_tmp(:,:) - mean(left_tmp(:,:),1);
    m = mean(right_tmp(:,:),2);
    right(:,:,i) = right_tmp(:,:) - mean(right_tmp(:,:),1);
    % plot(right(:,ch,i)); hold off;
end
%%
clearvars -except right left EEG
%% CSP 적용하기
clc;
chNum = size(left,2);
trialNum = size(left,3);
trial_R = zeros(size(right,2), size(right,1), size(right,3));
trial_L = zeros(size(left,2), size(left,1), size(left,3));
for i=1:size(left,3)
    trial_R(:,:,i) = right(:,:,i)';
    trial_L(:,:,i) = left(:,:,i)';
end
%% Compute the covariance matrices for each class
t = round(0.1 * trialNum);  % 10개로 자를 때 하나 크기 정하기
trainN = 9*t;
testN = t;
featureSize = 2;
acc_total = 0.;
for Rep = 1:10
    acc_sum10 = 0.;
    idxR = randperm(trialNum); % 1~trialNum을 랜덤으로 정렬
    idxL = randperm(trialNum); % 1~trialNum을 랜덤으로 정렬
    
    for rep = 1:10
        idxR_test = idxR(t*(rep-1)+1 : t*(rep));    % t개의 test idx 선택
        idxR_train = setdiff(idxR, idxR_test);      % 나머지를 train idx로 선택
        test_R = trial_R(:,:, idxR_test);
        train_R = trial_R(:,:, idxR_train);
        idxL_test = idxL(t*(rep-1)+1 : t*(rep));    % t개의 test idx 선택
        idxL_train = setdiff(idxL, idxL_test);      % 나머지를 train idx로 선택
        test_L = trial_L(:,:, idxL_test);
        train_L = trial_L(:,:, idxL_train);
        
        % CSP
        sum1 = zeros(chNum, chNum);
        sum2 = zeros(chNum, chNum);
        
        for i=1:trainN
            X1 = train_R(:,:,i)';         % time x channel 형식이어야 해
            X2 = train_L(:,:,i)';         % time x channel 형식이어야 해
        
            C1 = cov(X1);
            C2 = cov(X2);
            sum1 = sum1 + C1;
            sum2 = sum2 + C2;
        end
        
        C1 = sum1./trainN;
        C2 = sum2./trainN;
        
        % Compute the generalized eigenvalue problem
        [V,D,W] = eig(C1,C1+C2);
                
        clearvars V D C1 C2 sum1 sum2
        % LDA 돌릴 준비
        testL=zeros(testN,featureSize * 2);
        trainL=zeros(trainN,featureSize * 2);
        testR=zeros(testN,featureSize * 2);
        trainR=zeros(trainN,featureSize * 2);
                
        % train, test 
        for k=1:featureSize
            filterL=W(:,k);
            filterR=W(:,end-k+1);
            for i=1:trainN
                trialL=squeeze(train_L(:,:,i));
                trialR=squeeze(train_R(:,:,i));
        
                trainL(i,k)=log(filterL'*trialL*trialL'*filterL);
                trainL(i,end-k+1)=log(filterR'*trialL*trialL'*filterR);
        
                trainR(i,k)=log(filterL'*trialR*trialR'*filterL);
                trainR(i,end-k+1)=log(filterR'*trialR*trialR'*filterR);
            
            end
            for i=1:testN
                trialL=squeeze(test_L(:,:,i));
                trialR=squeeze(test_R(:,:,i));
        
                testL(i,k)=log(filterL'*trialL*trialL'*filterL);
                testL(i,end-k+1)=log(filterR'*trialL*trialL'*filterR);
        
                testR(i,k)=log(filterL'*trialR*trialR'*filterL);
                testR(i,end-k+1)=log(filterR'*trialR*trialR'*filterR); 
        
            end
        
        end

        
        
        % LDA 돌리기
        X = cat(1, trainL, trainR);
        Y = cat(1, zeros(trainN,1), ones(trainN,1));
        
        test_X = cat(1, testL, testR);
        test_Y = cat(1, zeros(testN,1), ones(testN,1));
        
        Mdl = fitcdiscr(X,Y);
        result = predict(Mdl, test_X);
        
        data = mean(result==test_Y);
        % disp(data)
        acc_sum10 = acc_sum10 + data;
    end
    disp(strcat(num2str(Rep), ' mean: ', num2str(acc_sum10/10.)));
    acc_total = acc_total + acc_sum10/10;
end
disp(strcat('total acc: ', num2str(acc_total/10.)));