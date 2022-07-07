st_HFOSettings.s_FreqIni = 80
st_HFOSettings.s_FreqEnd = 500
st_HFOSettings.s_EpochTime = 600
%st_HFOSettings.s_EpochTime = 540
st_HFOSettings.s_RMSWindow = 3
st_HFOSettings.s_RMSThres = 5
st_HFOSettings.s_MinWind = 6
st_HFOSettings.s_MinTime = 10
st_HFOSettings.s_NumOscMin = 6
st_HFOSettings.s_BPThresh = 3

% st_HFOSettings.s_FreqIni = 80
% st_HFOSettings.s_FreqEnd = 500
% st_HFOSettings.s_EpochTime = 30
% st_HFOSettings.s_RMSWindow = 3
% st_HFOSettings.s_RMSThres = 3
% st_HFOSettings.s_MinWind = 6
% st_HFOSettings.s_MinTime =10
% st_HFOSettings.s_NumOscMin =6
% st_HFOSettings.s_BPThresh =3

st_HFOAnalysis.s_Sampling = 2000; % Channel's FreqSampling %% data sampling

s_algo_state = "all" %"generate_data" %  "all" % "Detector"
