clc;clear;close all;
%%  导入表格
Data = readtable('val_labels.csv');
id = cell2mat(table2array(Data(:,1)));
id = reshape(id',1,[]);
s = table2array(Data(:,2));
e = table2array(Data(:,3));
label = table2array(Data(:,5)) ~= 3;
%%  键盘输入
audio = input('请输入音频文件名(不需要包含后缀名)\n','s');
% audio = '_7oWZq_s_Sk';

a = strfind(id,audio);
if isempty(a) == true
    error('请输入正确文件名');
end

start = (a(1)-1)/11 + 1;
if start ~= round(start) || length(audio) ~= 11
    warning('输入文件名不完整，已为您匹配最相近的文件名(标准文件名应为 11 个字符)');
    start = round((a(1)-1)/11) + 1;
end

% 检查标签起始是否为0，正常情况下不会报错
% if s(start) ~= 0
%     error('label error!');
% end

% 可选起始终止时间
begin_time = input('请输入起始时间(单位:s)\n');
% begin_time = 0;
if isnumeric(begin_time) ~= true
    error('请输入数值类型');
end

finish_time = input('请输入结束时间(单位:s)\n');
% finish_time = 900;
if isnumeric(finish_time) ~= true
    error('请输入数值类型');
end

if begin_time < 0 || begin_time > finish_time
    error('请输入合法起始时间: 0s ～ 900s');
end

if finish_time > 900
    error('请输入合法结束时间: finish time < 900s');
end
%%  采样点标签
sampling = 16000;   % wav 采样率
n = 0 : 1/sampling : 900;   % 15min 音频文件
Label = zeros(1,length(n));

for i = 1 : length(n)
    if n(i) >= e(start) && s(start+1) ~= 0
        start = start + 1;
    end
    Label(i) = label(start);
end
%%  画图
begin = find(n == begin_time);
finish = find(n == finish_time);

plot(n(begin:finish),Label(begin:finish));xlabel('时间');ylabel('标签');axis tight;



