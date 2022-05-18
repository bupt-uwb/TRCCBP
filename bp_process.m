clear;clc;close all;
fid=fopen('../20220426_rawdata/bp.txt');
% ids = {'dy','jhb','lyy','mly','mwy','sjh','wcc','wyl','yyy','yz','zb','zjl','zjw'};
% ids = {'ghb','hjt','lyy','lzj','mly','mwy','sjh','sty','tjs','wcc','wqw','wzh','yyy','yz','zc','zzy'};
ids = {'gj','jhb','lyy','mly','mwy','rjw','wjx','wqw','wxe','yz','zy'};
id_bp = [
    85,126;
    80,130;
    70,102;
    70,111;
    77,117;
    78,118;
    63,97;   
    65,98;
    80,132;
    66,111;
    84,130;
    ];
bps = [];
mean_bps = [];
for i=1:size(id_bp,1)
    for j=1:20
       mean_bps = [mean_bps;id_bp(i,:)];         
    end
end
while ~feof(fid) 
    tline=fgetl(fid); 
    split_line=strsplit(tline);
    idx = find(ismember(ids, char(split_line(1))));
    if idx
        name = char(split_line(1));
        for i = 1:10
            tline=fgetl(fid); 
            split_line=strsplit(tline); 
            for j =1:2
                bps = [bps;str2num(char(split_line(2))) str2num(char(split_line(1)))];
            end
        end
    end
end

save(['bps4'],'bps','mean_bps');