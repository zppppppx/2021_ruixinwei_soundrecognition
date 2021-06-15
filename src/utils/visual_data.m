clc;clear;close all;
%%  ������
Data = readtable('val_labels.csv');
id = cell2mat(table2array(Data(:,1)));
id = reshape(id',1,[]);
s = table2array(Data(:,2));
e = table2array(Data(:,3));
label = table2array(Data(:,5)) ~= 3;
%%  ��������
audio = input('��������Ƶ�ļ���(����Ҫ������׺��)\n','s');
% audio = '_7oWZq_s_Sk';

a = strfind(id,audio);
if isempty(a) == true
    error('��������ȷ�ļ���');
end

start = (a(1)-1)/11 + 1;
if start ~= round(start) || length(audio) ~= 11
    warning('�����ļ�������������Ϊ��ƥ����������ļ���(��׼�ļ���ӦΪ 11 ���ַ�)');
    start = round((a(1)-1)/11) + 1;
end

% ����ǩ��ʼ�Ƿ�Ϊ0����������²��ᱨ��
% if s(start) ~= 0
%     error('label error!');
% end

% ��ѡ��ʼ��ֹʱ��
begin_time = input('��������ʼʱ��(��λ:s)\n');
% begin_time = 0;
if isnumeric(begin_time) ~= true
    error('��������ֵ����');
end

finish_time = input('���������ʱ��(��λ:s)\n');
% finish_time = 900;
if isnumeric(finish_time) ~= true
    error('��������ֵ����');
end

if begin_time < 0 || begin_time > finish_time
    error('������Ϸ���ʼʱ��: 0s �� 900s');
end

if finish_time > 900
    error('������Ϸ�����ʱ��: finish time < 900s');
end
%%  �������ǩ
sampling = 16000;   % wav ������
n = 0 : 1/sampling : 900;   % 15min ��Ƶ�ļ�
Label = zeros(1,length(n));

for i = 1 : length(n)
    if n(i) >= e(start) && s(start+1) ~= 0
        start = start + 1;
    end
    Label(i) = label(start);
end
%%  ��ͼ
begin = find(n == begin_time);
finish = find(n == finish_time);

plot(n(begin:finish),Label(begin:finish));xlabel('ʱ��');ylabel('��ǩ');axis tight;



