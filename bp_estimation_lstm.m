clear;clc;close all;
load('../LSTM/wave1_0512seluwb_result')
id = 1;
num = 0;
out_new = out;
k = 0;
idd = 1;
idx = 1;
es = [];
acc = [];
truth = [];
for j = 2:size(ids_test,2)
    if y_test(j,1) ~= y_test(j-1,1) || y_test(j,2) ~= y_test(j-1,2)
        truth = [truth;y_test(j-1,:)];
        continue
    end
end
truth = [truth;y_test(j,:)];

for ii = 1:size(ids_test,2)-1
    if y_test(ii,1) == y_test(ii+1,1) && y_test(ii,2) == y_test(ii+1,2)
        continue
    else
        acc = [acc ; [1-mean(abs(out(idx:ii,:)-double(y_test(idx:ii,:)))./double(y_test(idx:ii,:)))]];
        es = [es;mean(out(idx:ii+1,:),1)];
        idx = ii+1; 
    end
end
acc = [acc ; [1-mean(abs(out(idx:ii+1,:)-double(y_test(idx:ii+1,:)))./double(y_test(idx:ii+1,:)))]];
es = [es;mean(out(idx:ii+1,:))];
meanaccc = [];
for j = 1:2:21
    meanaccc = [meanaccc;mean(acc(j:j+1,:),1)];
end
meanacc = mean(meanaccc); %按测量次数

single = 1-mean(abs(out-double(y_test))./double(y_test)); %按单拍
% save('../LSTM/test_0512seluwb','acc','single','meanacc','meanaccc','truth','es')