function sig = maxEnergy(Pure)

Data = Pure;

for iLocation=1:size(Data,2)
    energyOfLocation(iLocation) = sum(Data(:,iLocation).*Data(:,iLocation));
end

dou = energyOfLocation;          %取出 某个位置 所在范围内所有快时间列的能量
[energymax,energymaxi] = max(dou);                         %%通过能量最大找到所需信号
sig = Data(:,energymaxi);

% 
% figure
% plot((1:380)/20,sig)
% xlabel('Time/Seconds')
% ylabel('Amplitude')