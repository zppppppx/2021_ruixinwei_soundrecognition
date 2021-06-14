clc;clear;close all;
%%
Data = readtable('val_labels.csv');
id = cell2mat(table2array(Data(:,1)));
s = table2array(Data(:,2));
e = table2array(Data(:,3));
tabel = table2array(Data(:,5)) ~= 3;
%% 
audio = input('��������Ƶ�ļ���(����Ҫ������׺��)\n','s');
% audio = '_7oWZq_s_Sk';
begin_time = input('��������ʼʱ��\n');
% begin_time = 0;
finish_time = input('���������ʱ��\n');
% finish_time = 90;

start = find(id==audio,1);
if s(start) ~= 0
    error('label error!');
end

if begin_time < 0 || begin_time > finish_time
    error('������Ϸ���ʼʱ��: 0s �� 900s');
end

if finish_time > 900
    error('������Ϸ�����ʱ��: finish time < 900s');
end
%%
sampling = 16000;
n = 0 : 1/sampling : 900;
begin = find(n == begin_time);
finish = find(n == finish_time);

a = zeros(1,length(n));
for i = 1 : length(n)
    if n(i) >= e(start) && s(start+1) ~= 0
        start = start + 1;
    end
    a(i) = tabel(start);
end
%%
plot(n(begin:finish),a(begin:finish));xlabel('ʱ��');ylabel('��ǩ');


