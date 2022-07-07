addpath('./Functions/')
addpath('./Icons/')
addpath('./Licensed/')

%% [Variable] - 
% Colormaps
v_Colormap      = {'autumn';'bone';'colorcube';'cool';'copper';'flag';...
                    'gray';'hot';'hsv';'jet';'lines';'pink';'prism';...
                    'spring';'summer';'white';'winter'};

v_TFVisMethod   = {'Original';'Relative';'Normalized';'Z-Score'};

st_Controls.v_Scale     = [0.1;0.2;0.25;0.5;0.6;0.8;1;1.1;1.2;1.25;1.5;...
                            1.6;1.8;2;2.2;2.5;3;4;5;6;8;10];                       
st_Controls.v_TimeWind  = [(0.1:0.1:1)';(1.2:.2:2)'];
st_Controls.v_GridSec   = [2;3;4;5;10;15;20;30;40;50];                    


v_BandLimits            = [60 120 240];

% --- Message Variables ---
st_HandleMsg            = [];

% ---- type evens ------
%v_eventClasses	= f_ListEventClasses();   

%% [Variable] - Lines -
st_Line.SignalWind      = [];
st_Line.SignalEvent     = [];
st_Line.FilterWind      = [];
st_Line.FilterEvent     = [];
st_Line.PowerSpect      = [];
st_Line.TFImage         = [];

%% [Variable] - HFO Info -
st_EvControl.v_EvCurrent   = [];
st_EvControl.v_EvIdxOk     = [];
st_EvControl.v_EvIdxRem    = [];
st_EvControl.s_LstActive   = [];
st_EvControl.v_ListSel     = 1;
st_EvControl.v_ListRem     = [];
st_EvControl.v_EventTime   = [];
st_EvControl.s_PosIni      = [];
st_EvControl.s_PosEnd      = [];

% --- Spec Info ---
st_EvInfo.s_Width       = 4000; %% fixed to 2s
st_EvInfo.v_EveIni      = [];
st_EvInfo.v_EveEnd      = [];
st_EvInfo.v_FiltScale   = 1;
st_EvInfo.v_TimeAxe     = [];
st_EvInfo.v_EventFilter = [];
st_EvInfo.m_Spectrum    = [];
st_EvInfo.v_FreqSpect   = [];
st_EvInfo.v_FreqLimits  = [];
st_EvInfo.s_EvMaxFreq   = [];
st_EvInfo.s_CurMaxFreq  = [];
st_EvInfo.s_EvenTime    = [];

%% [Variable] - Load -

st_Load         = [];
st_FileData     = [];
st_ElecData     = [];
 
st_LoadInfo.s_Open          = 'true';
st_LoadInfo.str_RootPath    = './Analysis/';
st_LoadInfo.str_PathName    = './Analysis/';
st_LoadInfo.v_ElecNames     = [];
st_LoadInfo.s_ElecIdx       = 0;
    
% --- Signal Info ---
                        
st_SigData.v_Signal     = [];
st_SigData.str_SigLabel = [];
st_SigData.str_SigName  = [];

%% [Variable] - Cursor  -

st_Cursors.s_hCursor1   = [];
st_Cursors.s_hCursor2   = [];
st_Cursors.v_hCurLine1  = [];
st_Cursors.v_hCurLine2  = [];
st_Cursors.s_SizeCur1   = 0.015;
st_Cursors.s_SizeCur2   = 0.015;
st_Cursors.s_PosCur1    = 0;
st_Cursors.s_PosCur2    = 1;
st_Cursors.s_ColorCur1  = [.749 0 .749];
st_Cursors.s_ColorCur2  = [1 .4 0];
st_Cursors.s_CurrentCur = [];
st_Cursors.s_ElecCursor = [];
st_Cursors.s_IdxCur1    = [];
st_Cursors.s_IdxCur2    = [];
st_Cursors.s_KeyStep    = 0.005;

data_dir = '/mnt/SSD4/qiujing_data/EEG_data'
data_folder = dir(data_dir)
data_folder = {data_folder(1:end).name}
%data_folder = {'Pt4_HS'}
for p = 1:numel(data_folder)
    patient_name = data_folder(p)
    
    patient_name= patient_name{1}
    if p > 10
        continue
    end
    if strcmp(patient_name, 'NS_long') | strcmp(patient_name, '.') | strcmp(patient_name, '..')
        continue
    end
%     if  ~strcmp(patient_name,'Pt17_SM')
%         continue
%     end
    if contains(patient_name, '_')
        pure_name =  split(patient_name,'_');
        pure_name =  pure_name(end)
        pure_name = pure_name{1}
    else
        pure_name =  patient_name
    end
    
    bank_name = ['A'; 'B']
    foler_name = fullfile(data_dir, patient_name, 'HFO events')
    for j=1:2
        file_name = [pure_name '_'  bank_name(j)  '_ave_original_STE.rhfe' ]
        path_name = fullfile(foler_name, file_name);
        if isfile(path_name)
            st_LoadInfo.str_FileName = file_name
            st_LoadInfo.str_PathName = foler_name;
            f_LoadAnalysis(st_EvControl,st_EvInfo, st_Load, st_LoadInfo)
        else
            disp(['didnt find file' path_name])
        end
    end
       
end


% In this section are indicated the nested functions for Independent Icons 
%::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
function f_LoadAnalysis(st_EvControl,st_EvInfo, st_Load, st_LoadInfo)
        %Load Analysis
        %st_LoadInfo.str_FileName = 'AG_A_ave_original_STE_TEST.rhfe';
        %st_LoadInfo.str_PathName = '/mnt/SSD2/qiujing_data/EEG_data/AG/HFO events/';
        
        st_LoadInfo.str_FullPath = fullfile(st_LoadInfo.str_PathName, st_LoadInfo.str_FileName);
        st_Load         = load(st_LoadInfo.str_FullPath,'-mat');
        st_FileData     = st_Load.st_FileData;
                                        
        st_Load         = rmfield(st_Load,{'st_FileData'});
                                
        st_LoadInfo.v_ElecNames = fieldnames(st_Load);
        %st_LoadInfo.v_ElecNames = sort(st_LoadInfo.v_ElecNames);
        all_result.channel_names = st_LoadInfo.v_ElecNames
        %channel_names = st_LoadInfo.v_ElecNames
                
 
        % Load Electrode
        number_electrodes = numel(st_LoadInfo.v_ElecNames) 
       
        all_hfos_info = cell(number_electrodes,1);
        for i = 1: number_electrodes
             all_events{i} = f_LoadElectrode(st_EvControl,st_EvInfo, st_Load, st_LoadInfo, i);
        end
        
        all_result.all_events = all_events
        
        str_name	= [st_LoadInfo.str_PathName '/' str_FileName(1:end-5) '_hfo_spike_250_info.mat'];
        save(str_name, 'all_result', '-v7.3');
        
end
% function [st_EvInfo, st_EvControl] = f_loadHFO(st_ElecData, st_EvControl,st_EvInfo, st_Load, st_LoadInfo)
%         
%         
% end



function channel_HFO_info = f_LoadElectrode(st_EvControl,st_EvInfo, st_Load, st_LoadInfo, sel_channel_indx)
        % Load Data
                       
        if isempty(st_LoadInfo.v_ElecNames)
            warndlg('No event was detected','Warning')
            return
        end
                            
        %st_LoadInfo.s_ElecIdx   = get(st_Info.ElecList,'Value');
        st_LoadInfo.s_ElecIdx   = sel_channel_indx
        st_ElecData             = getfield(st_Load,...
                                st_LoadInfo.v_ElecNames{...
                                st_LoadInfo.s_ElecIdx}); %#ok<*GFLD>
        
%         if isempty(st_ElecData.v_Intervals)
%             warndlg('No event was detected','Warning')
%             return
%         end
        
         % Find Event Initial Time
        v_TimeIntervIni         = st_ElecData.st_HFOInfo.m_EvtLims(:,1);
        v_TimeIntervIni         = v_TimeIntervIni./...
                                st_ElecData.st_HFOInfo.s_Sampling;
        
        % Set Interval Position Variable
        st_EvControl.v_EvIdxOk   = 1:numel(v_TimeIntervIni);
        st_EvControl.v_EvIdxRem  = [];
                % Set List of Events
        st_EvControl.v_EventTime = cell(numel(v_TimeIntervIni),1);
        
        for kk=1:numel(v_TimeIntervIni)
            st_EvControl.v_EventTime(kk,1)  = {sprintf(...
                                            '%02.0f:%02.0f:%05.2f',...
                                            f_Secs2hms(v_TimeIntervIni(kk)))};
        end
        
        st_EvControl.s_LstActive = 'detected';
 
        
        for j= 1:numel(v_TimeIntervIni)
            st_EvInfo.event_time = st_ElecData.st_HFOInfo.m_EvtLims(j,:);
            st_EvControl.v_ListSel  = j;
            st_EvControl.v_EvCurrent = st_EvControl.v_EvIdxOk(...
                                st_EvControl.v_ListSel); 
           st_EvInfo.v_EveIni    = st_ElecData.st_HFOInfo.m_Rel2IntLims(...
                                st_EvControl.v_EvCurrent,1);
        st_EvInfo.v_EveEnd    = st_ElecData.st_HFOInfo.m_Rel2IntLims(...
                                st_EvControl.v_EvCurrent,2);
            
        s_middleSample    	= round(mean(st_ElecData.st_HFOInfo.m_Rel2IntLims(...
                            st_EvControl.v_EvCurrent,:)));
                        
        st_EvInfo.v_TimeAxe	= ((1:numel(st_ElecData.v_Intervals{...
                            st_EvControl.v_EvCurrent,1})) - ...
                            s_middleSample)./...
                            st_ElecData.st_HFOInfo.s_Sampling;                   
                            
        if st_EvInfo.v_EveIni < 1
            st_EvInfo.v_EveIni	= 1;
        end
        
        if st_EvInfo.v_EveEnd > numel(st_EvInfo.v_TimeAxe)
            st_EvInfo.v_EveEnd	= numel(st_EvInfo.v_TimeAxe);
        end
        
        % Get the mean index of event selected
        s_EventMean             = round(mean([...
                                st_EvInfo.v_EveEnd(end) ...
                                st_EvInfo.v_EveIni(1)]));
                                
        s_WindowMean            = round(st_EvInfo.s_Width / 2);
                        
        st_EvControl.s_PosIni    = s_EventMean - s_WindowMean;
        st_EvControl.s_PosEnd    = s_EventMean + s_WindowMean;
        
        if st_EvControl.s_PosIni < 1
            st_EvControl.s_PosIni    = 1;
            st_EvControl.s_PosEnd    = st_EvInfo.s_Width;
        end
        
        if st_EvControl.s_PosEnd > ...
                numel(st_ElecData.v_Intervals{st_EvControl.v_ListSel,1})
            st_EvControl.s_PosIni    = numel(...
                                    st_ElecData.v_Intervals{...
                                    st_EvControl.v_ListSel,1}) - ...
                                                       st_EvInfo.s_Width;
            st_EvControl.s_PosEnd    = numel(...
                                    st_ElecData.v_Intervals{...
                                    st_EvControl.v_ListSel,1});
        end
                                
        st_EvInfo.s_EvenTime   = (st_EvInfo.v_EveEnd(end) - ...
                                st_EvInfo.v_EveIni(1)) / ...
                                st_ElecData.st_HFOInfo.s_Sampling * 1000;
                                
        st_EvControl.s_PosIni    = round(st_EvControl.s_PosIni);
        st_EvControl.s_PosEnd    = round(st_EvControl.s_PosEnd);
        
        
        

        
        % Place Electrone Analysis Info
        st_EvInfo.v_FreqLimits  = [st_ElecData.st_HFOSetting.s_FreqIni ...
                                    st_ElecData.st_HFOSetting.s_FreqEnd];          
        %st_EvInfo, st_EvControl = f_loadHFO(st_ElecData, st_EvControl,st_EvInfo, st_Load, st_LoadInfo) ;
            
        st_EvInfo.v_EventFilter = f_FilterSet(st_ElecData, st_EvControl,st_EvInfo, st_Load, st_LoadInfo);
            
        [st_EvInfo.m_Spectrum , st_EvInfo.v_FreqSpect] = f_ScalogramSet(st_ElecData, st_EvControl,st_EvInfo, st_Load, st_LoadInfo);

        channel_HFO_info{j}= st_EvInfo;
        clear  st_EvInfo.m_Spectrum  st_EvInfo.v_FreqSpect  st_EvInfo.v_EventFilter st_EvInfo.v_TimeAxe st_EvInfo.event_time
        end
       
        % Draw Lines
                          
end
    
%% [Functions] Plot Functions: fetch spectrom and gamma info
% In this section are indicated the nested functions for plotting 
%::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
function  [channel_HFO_info] = f_DisplayProcess(st_ElecData, st_EvControl,st_EvInfo, st_Load, st_LoadInfo)
        % Process to draw lines
        
        if isempty(st_Load)
            return
        end
        
        if isempty(st_EvControl.v_EvIdxOk) && ...
                strcmp(st_EvControl.s_LstActive,'detected')
            return
            
        elseif isempty(st_EvControl.v_EvIdxRem) && ...
                strcmp(st_EvControl.s_LstActive,'rejected')
            
            st_EvControl.s_LstActive = 'detected';
            f_DisplayProcess(st_Load, st_EvControl);
            disp('rejected to detected')
            return
        end
        
        % Filter original signal
        
                
    end
   %% [Functions] Signal Processing
% In this section are indicated the nested functions for signal processing 
%::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    function v_SignalFilt = f_FilterSet(st_ElecData, st_EvControl,st_EvInfo, st_Load, st_LoadInfo)
        % This function filter the signal in the band selected
        
        if isempty(st_Load)
            return
        end
        
        st_EvInfo.v_EventFilter = zeros(...
                                size(st_ElecData.v_Intervals{...
                                    st_EvControl.v_ListSel,1}));
                                                               
        s_Filter                = f_DesignIIRfilter(...
                                st_ElecData.st_HFOInfo.s_Sampling,...
                                st_EvInfo.v_FreqLimits,...
                                [st_EvInfo.v_FreqLimits(1)-0.5 ...
                                st_EvInfo.v_FreqLimits(2)+0.5]);
                                                       
        v_SignalFilt            = f_FilterIIR(...
                                st_ElecData.v_Intervals{...
                                st_EvControl.v_EvCurrent,1},...
                                s_Filter);
                            
        %st_EvInfo.v_EventFilter = v_SignalFilt;
              
    end
%::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    function [m_GaborWT, v_FreqAxis] = f_ScalogramSet(st_ElecData, st_EvControl,st_EvInfo, st_Load, st_LoadInfo)
        
        if isempty(st_Load)
            return
        end
                 
        s_FreqSeg       = 512; 
        s_StDevCycles   = 3; 
        s_Magnitudes    = 1;
        s_SquaredMag    = 0;
        s_MakeBandAve   = 0;
        s_MinFreqHz     = st_EvInfo.v_FreqLimits(1);
        s_MaxFreqHz     = st_EvInfo.v_FreqLimits(2);
        s_ColorLevels   = 256;
       
                       
        v_SigInterv     = st_ElecData.v_Intervals{...
                        st_EvControl.v_EvCurrent,1}(...
                        st_EvControl.s_PosIni:st_EvControl.s_PosEnd);
        
        s_WrpWnd    = round(numel(v_SigInterv)*0.5)+1;
        v_IniWind   = v_SigInterv(1:s_WrpWnd);
        s_First     = v_IniWind(1);
        v_EndWind   = v_SigInterv(end-s_WrpWnd:end);
        s_Last      = v_EndWind(end);
        
        v_IniWind   = v_IniWind(:) - s_First;
        v_EndWind   = v_EndWind(:) - s_Last;
        
        v_IniWind   = flipud(-v_IniWind) + s_First;
        v_EndWind   = flipud(-v_EndWind) + s_Last;
        
        v_IniWind   = v_IniWind(1:end-1);
        v_EndWind   = v_EndWind(2:end);
        
        v_SigInterv = vertcat(v_IniWind,v_SigInterv,v_EndWind);        
        
        [m_GaborWT, v_TimeAxis, v_FreqAxis] = ...
                            f_GaborTransformWait(...
                            v_SigInterv,...
                            st_ElecData.st_HFOInfo.s_Sampling,...
                            s_MinFreqHz, ...
                            s_MaxFreqHz, ...
                            s_FreqSeg, ...
                            s_StDevCycles, ...
                            s_Magnitudes, ...
                            s_SquaredMag, ...
                            s_MakeBandAve);
        
        % instead of offering visualization options: original, relative,
        % normalized, z-score, we only consider original one 
                
        m_GaborWT   = m_GaborWT(:,numel(v_IniWind)+1:...
                        numel(v_SigInterv)-numel(v_EndWind)); 
                
        clear v_EndWind v_IniWind
        

%         linkaxes([st_hAxes.Signal,st_hAxes.Signal],'x')
  
                                
%         st_Line.Spectrogram     = f_ImageMatrix(m_GaborWT, v_TimeAxis, ...
%                                 v_FreqAxis,v_Lims,str_ColorMap,...
%                                 s_ColorLevels, 0);
               
%         st_EvInfo.m_Spectrum    = m_GaborWT;
%         st_EvInfo.v_FreqSpect   = v_FreqAxis;
             
    end
%::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
 
 
    
    