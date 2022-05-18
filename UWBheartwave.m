clear;clc;close all;

fileDir='../20220426_slices/';
fileFolder=fullfile(fileDir);
dirOutputUWB=dir(fullfile(fileFolder,'*.mat'));
UWBName={dirOutputUWB.name};
UWBName=sort(UWBName);
e_hr = [];
% for i=1:size(UWBName,2)
%     [fileDir,char(UWBName(i))]
%     dataFile=[fileDir,char(UWBName(i))];
%     load(dataFile);
%     hrSignal1 = maxEnergy(Puredata);
%     hr = VMD_decomposition(hrSignal1);
%     e_hr = [e_hr hr];
% end
% save(['../vmd_hr_0509/','vmd_hr'],'e_hr');

load('../vmd_hr_0504/vmd_hr.mat')
load('./bps3')

ks = [5,7,8,10,12,15,20,25,30,35,40,50,60];
countsno = 0;
countsyes = 0;
corrs=[];
for i=1:size(UWBName,2)
    close all;
    file=[fileDir,char(UWBName(i))]  
    load(file);
    hrSignal1 = maxEnergy(Puredata);
%     hr = VMD_decomposition(hrSignal1);
%     T=size(Puredata,1);
%     peaks=zeros(1,T);tops=zeros(1,T);
%     for pp=1:size(Puredata,1)
%         [vv,tmp]=max(abs(Puredata(pp,:)));
%         peaks(pp)=tmp;  %% position
%         tops(pp)=vv;  %% amplitude
%     end
%     peaks=kalmanfilter_3(peaks);
%     hrSignal1 = [];
%     for n = 1:size(peaks,1)
%         if round(peaks(n,:)) <= size(Puredata,2)
%             hrSignal1 = [hrSignal1;Puredata(n,round(peaks(n,:)))];
%         else
%             hrSignal1 = [hrSignal1;Puredata(n,size(Puredata,2))];
%         end
%     end
    
    flag = 0;
    hr = e_hr(i);
    for k = 1:length(ks)
        if flag~=1
            u = VMD_hb(hrSignal1,ks(k));
            for j=1:size(u,1)
                tmp_fft = abs(fftshift(fft(u(j,:))));
                [~,idx] = max(tmp_fft(101:end));
                f = idx/10*60;
                if abs(f-hr)<8
%                     [ks(k),hr,f]
                    pure_data = sum(u(j:end-1,:));
                    hr_wave = u(j,:);
                    flag = 1;
                    break
                end
            end
        end
    end
%     pure_data = pure_data - mean(pure_data);
    pure_data = mapminmax(pure_data);
    hr_wave = mapminmax(hr_wave);
    mpd_min = mean(-pure_data)+0.15;
    mpd_max = mean(pure_data)+0.15;
%     [minp,minl]=findpeaks(-pure_data,'minpeakheight',mpd_min,'minpeakdistance',10,'NPeaks',25);
%     [maxp,maxl]=findpeaks(pure_data,'minpeakheight',mpd_max,'minpeakdistance',10,'NPeaks',25);
    [minp,minl]=findpeaks(-pure_data,'minpeakheight',mpd_min);
    [maxp,maxl]=findpeaks(pure_data,'minpeakheight',mpd_max);
%     figure
%     plot(pure_data)
%     hold on
%     scatter(minl,-minp)

     %% 调整波形
    m1 = abs(median(pure_data(maxl)));
    m2 = abs(median(pure_data(minl)));
    max1=abs(max(pure_data));
    min1=abs(min(pure_data));

    if((max1-m1)>(1.2*m1))
        alp1=(1.2*m1)/(max1-m1);
    else
        alp1=1;
    end
    
    if((abs(min1)-m2)>(1.2*m2))
        alp2=(1.2*m2)/(abs(min1)-m2);
    else
        alp2=1;
    end
    
    pure_data_tmp = pure_data;
    for j=1:length(pure_data)
        if(pure_data(j)>m1)
            pure_data_tmp(j)=m1+(pure_data(j)-m1)*alp1;
        end
        
%         if (pure_data(j)<m1 && pure_data(j)>0)
%             pure_data_tmp(j)=m1-(m1-pure_data(j))*alp1;
%         end
%         
%         if(-m2<pure_data(j) && pure_data(j)<0)
%             pure_data_tmp(j)=-m2+(pure_data(j)+m2)*alp2;
%         end 
        
        if(pure_data(j)<-m2)
            pure_data_tmp(j)=-m2+(pure_data(j)+m2)*alp2;
        end 
    end

%     figure;
%     plot(pure_data)
%     hold on
%     plot(pure_data_tmp)    
    pure_data = pure_data_tmp;
       
%% 相关性
    [minp_hr,minl_hr]=findpeaks(-hr_wave);
    p = 1;
    pairs = [];
    for j=1:length(minl_hr)
        if j==1 && minl_hr(j) < minl(1)
            continue
        elseif  j==length(minl_hr) && minl_hr(j) > minl(end)
            continue
        else
            for k = p:length(minl)-1
                if minl(k)<=minl_hr(j) && minl(k+1)>=minl_hr(j)
                    [~,idx] = min([abs(minl(k)-minl_hr(j)),abs(minl(k+1)-minl_hr(j))]);
                    pairs = [pairs;minl(idx-1+k),minl_hr(j)];
                    if idx == 1
                        p = k;
                    else
                        p = k+1;
                    end
                    break
                end
            end
        end
    end
%     figure
%     plot(hr_wave)
%     hold on
%     plot(pure_data)
%     scatter(minl,pure_data(minl))
%     scatter(pairs(:,1),pure_data(pairs(:,1)));
%     scatter(pairs(:,2),hr_wave(pairs(:,2)));
    
    for j=1:size(pairs,1)-1
        l = min(pairs(j,:));
        r = max(pairs(j+1,:));
        fit_wave = hr_wave(l:r);
        pure_wave = pure_data(l:r);
        corr=corrcoef(fit_wave,pure_wave);
        if size(corr,1)==2 && corr(1,2)<0.6
            countsno = countsno+1;     
        else
            countsyes = countsyes+1;
        end
%         end
%         figure;
%         plot(fit_wave)
%         hold on
%         plot(pure_wave)
    end
    bp = bps(i,:);
%     mean_bp = mean_bps(i,:)
    save(['../20220510_pure_heartwaves/',char(UWBName(i))],'pairs',...
        'hr_wave','pure_data','hr','bp')
end
countsno
countsyes