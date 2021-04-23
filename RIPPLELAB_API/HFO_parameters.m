st_FilePath = struct;
%st_FilePath.name = 'AG_ECoG_cleaned.edf'
%st_FilePath.path = '/Users/qiujing/Dropbox/EEG_exp_local/Data/AG/'
st_FilePath.name = 'DL_cleaned.edf'
st_FilePath.path = '/Users/qiujing/Dropbox/EEG_exp_local/Data/DL/'
%st_FilePath.name = 'MG_cleaned.edf'
%st_FilePath.path = '/Users/qiujing/Dropbox/EEG_exp_local/Data/MG/'
%st_FilePath.name = 'NB_cleaned.edf'
%st_FilePath.path = '/Users/qiujing/Dropbox/EEG_exp_local/Data/NB/'
%st_FilePath.name = 'NS_cleaned.edf'
%st_FilePath.path = '/Users/qiujing/Dropbox/EEG_exp_local/Data/NS/'

str_TempFolder  = './Temp/new_data/'; %% folder for saving data
%st_FilePath.name = 'AR.edf'
%st_FilePath.path = '/Users/qiujing/Dropbox/EEG_exp_local/Data/AR/'

%st_FilePath.name = 'HS_removed.edf'
%st_FilePath.path = '/Users/qiujing/Dropbox/EEG_exp_local/Data/HS/'
%st_FilePath.name = 'KC_reorganized.edf'
%st_FilePath.path = '/Users/qiujing/Dropbox/EEG_exp_local/Data/KC/'


%st_HFOSettings.s_FreqIni = 80
st_HFOSettings.s_FreqIni = 10
st_HFOSettings.s_FreqEnd = 500
%st_HFOSettings.s_FreqEnd = 250
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
