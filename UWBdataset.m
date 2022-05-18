clear;clc;close all;
fileDir='../samples_0507uwb/';
fileFolder=fullfile(fileDir);
dirOutput=dir(fullfile(fileFolder,'*.mat'));
filenames={dirOutput.name};
filenames=sort(filenames);

name = 'gj';
id = 1;
k_train = 0;
avg_wave_train = [];
avg_bp_train = [];
bps_train = [];
waves_train = [];
ids_train = [];
hr_train = [];
avg_fft_train = [];
fft_train = [];

k_test = 0;
avg_wave_test = [];
avg_bp_test = [];
bps_test = [];
waves_test = [];
ids_test = [];
hr_test = [];
fft_test = [];
avg_fft_test = [];

groups = randperm(10);
% groups(find(groups==1))=[];
groups = groups(1:8);
for i=1:size(filenames,2)
    file=[fileDir,char(filenames(i))]
    load(file);
    filename = char(filenames(i));
    f = strsplit(filename,'_');
    group = char(f(5));
    group = str2num(group);
    fid = char(f(3));
    if ~(size(corr,1)==2 && corr(1,2)<0.6)       
        if ~strcmp(fid,name)
            name = fid;
            k_train = 0;
            id = id + 1;
            avg_wave_train = [];
            avg_bp_train = [];
            avg_fft_train = [];
            avg_fft_test = [];
            k_test = 0;
            avg_wave_test = [];
            avg_bp_test = [];
        end
%         if (group < 6 && id < 13) || (group < 4 && id >= 13)
%         if  group == 1 |(ismember(group,groups) && id < 10) | (ismember(group,groups2) && id >= 10)
        if ismember(group,groups)
            k_train = k_train + 1;
            avg_bp_train = [avg_bp_train;bp];
            avg_wave_train = [avg_wave_train,wave];
%             avg_fft_train = [avg_fft_train,FFT];
            if k_train == 1
                avg_bp_train = mean(avg_bp_train,1);
                waves_train = [waves_train;avg_wave_train];
                bps_train = [bps_train;bp];
                ids_train = [ids_train,id];
                hr_train = [hr_train,hr];
                fft_train = [fft_train;avg_fft_train];
                k_train = 0;
                avg_wave_train  = [];
                avg_bp_train  = [];
                avg_fft_train = [];
            end
        else
            avg_wave_test = [avg_wave_test,wave];
            k_test = k_test + 1;
            avg_bp_test = [avg_bp_test;bp];
%             avg_fft_test = [avg_fft_test,FFT];
            if k_test == 1
                avg_bp_test = mean(avg_bp_test,1);
                waves_test = [waves_test;avg_wave_test];
                bps_test = [bps_test;bp];
                fft_test = [fft_test;avg_fft_test];
                ids_test = [ids_test,id];
                hr_test = [hr_test,hr];
                k_test = 0;
                avg_wave_test  = [];
                avg_bp_test  = [];
                avg_fft_test = [];
            end
        end
    end   
end
save('../LSTM/wave1_05111uwb','waves_test','waves_train','bps_test','bps_train',...
    'ids_test','ids_train','hr_train','hr_test','fft_train','fft_test')