clear;clc;close all;
fileDir='../20220504_pure_heartwaves/';
fileFolder=fullfile(fileDir);
dirOutput=dir(fullfile(fileFolder,'*.mat'));
filenames={dirOutput.name};
filenames=sort(filenames);

name = 'gj';
load('bps3');
id = 1;
wave = zeros(1,30);
for i=1:size(filenames,2)
    count = 1;
    file=[fileDir,char(filenames(i))]  
    load(file);
    filename = char(filenames(i));
    f = strsplit(filename,'_');
    fid = char(f(3));
    if ~strcmp(name,fid)
        name = fid;
        id = id + 1;
    end
    for j = 1:size(pairs,1)-1
        if pairs(j,1) == pairs(j+1,1)
            continue
        else
            l = min(pairs(j,:));
            r = max(pairs(j+1,:));
            fit_wave = hr_wave(l:r);
            pure_wave = pure_data(l:r);
            corr=corrcoef(fit_wave,pure_wave);
%             tmpwave = pure_data(pairs(j,1):pairs(j+1,1));
            tmpwave = hr_wave(pairs(j,2):pairs(j+1,2));
            [~,idx] = max(tmpwave);
%             tao = [pairs(j+1,1)- pairs(j,1),pairs(j+1,1)-(idx+pairs(j,1)-1)]/20;
            tao = [pairs(j+1,2)- pairs(j,2),pairs(j+1,2)-(idx+pairs(j,2)-1)]/20;
            if tao(2)>tao(1)
                continue
            else
                if length(tmpwave) < 30
                    wave(1:length(tmpwave)) = tmpwave;
                else
                    wave= tmpwave(1:30);
                end
%                 wave = mapminmax(wave);
                FFT = abs(fft(wave));
                FFT =FFT-mean(FFT);
                FFT = FFT/std(FFT);
                wave =wave-mean(wave);
                wave = wave/std(wave);
                mean_bp = mean_bps(i,:);   
                ['../samples_0512embc/',filename(1:end-4),'_',num2str(count,'%02d')]
                save( ['../samples_0512embc/',filename(1:end-4),'_',num2str(count,'%02d')],'tao','bp','hr','wave','id','corr','mean_bp','FFT')    
                count = count + 1;
            end
        end
    end
end