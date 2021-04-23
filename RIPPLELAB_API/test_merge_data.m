tmp = all_result.all_events{1,1}
data_struct = [tmp{3:end}];
rm_fields = {'s_EvMaxFreq','s_CurMaxFreq' };
S = rmfield(data_struct,rm_fields)
% for i =1:numel(tmp)
%     tmp_cat = vercat(tmp.event_time)
% end