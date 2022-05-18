function [ measuredHeartbeat]=VMD_decomposition(HRsignal)
format long g
valueofk= [4,5,7,8,10,12,15,20,25,30];             %K值有变化，是否固定一个值，只分解一次       这样做的原因是   小值可能找不到   大值可能有不必要的
Y = (-2^16/2:2^16/2-1)*(20)/2^16;
w=0;
dyy = [];
for Location=1          %!!!!!!!!!!!!!!!!!!!!!!!!                      %这里单独给Location赋值可以单独得到某个位置的结果
    
    flagheart = 0;   %得到结果则置1
    
    
    DData = HRsignal;
%     
    for j=1:1:size(valueofk,2)                                    %经过几次VMD分解由 valueofk中的K值个数来定
        
        if( flagheart == 0)                       %没有找到心率和呼吸则继续增大K进行VMD分解
            K = valueofk(j);          
            u = VMD_hb(DData,K);
            for i =1:K
                fre(i,:)=abs(fftshift(fft(u(i,:),2^16)))*1000;
                [resultvalue,result] = max(fre(i,:));
                provalue(i) = abs(Y(result)); 
%                 figure%存储所有分解出的频率值
%                 plot(Y,fre(i,:));
%     xlabel('频率/赫兹');
%     ylabel('幅度');
                
            end
            for i = 1:K                           
                if(   provalue(i) > 1 &&  provalue(i) < 1.6   )     
%                 if(   provalue(i) > 0.65 &&  provalue(i) < 1.6   )    
                    energyofu(i) = sum(u(i,:).*u(i,:));
                    w=w+1;               
                    flagheart = 1;
                end
            end
            if flagheart == 1
                 [energyofumax,energyofumaxi] = max(energyofu);  
                 measuredHeartbeat(Location) = provalue(energyofumaxi);
                provalue = [];
                energyofmodule=energyofumax;
            end
        else
            break
        end
    end
    
    if (flagheart==0)
        measuredHeartbeat(Location) = -1;
    end
    provalue=[];
    
end

measuredHeartbeat=measuredHeartbeat.*60;
if (energyofmodule < 10^-9)
    measuredHeartbeat = 0;
end

% for i =1:K
%     figure
%     plot(u(i,:))                                           %存储所有分解出的频率值
%     
% end
% for i =1:K
%     figure                                            %存储所有分解出的频率值
%     plot(Y,fre(i,:))
% end



