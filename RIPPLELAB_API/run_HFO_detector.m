% figure out path setting
% Set HFO parameters
run('HFO_parameters_detector')
st_FilePath = struct;
str_TempFolder  = './Temp/new_data/'; %% folder for saving data
input_data_dir =  'Test_EEG_data/'
data_folder = dir(input_data_dir)
data_folder(ismember( {data_folder.name}, {'.', '..'})) = [];

st_FilePath_sub_dir_list= {}
st_FilePath_name_list = {}


for n = 1 : numel(data_folder)
    
    new_files = dir(fullfile(input_data_dir, data_folder(n).name, '*.edf'))
    new_files(ismember( {new_files.name}, {'.', '..'})) = [];
    % for same patient with multiple edfs
    for j=1: numel(new_files)
        st_FilePath_sub_dir_list = [st_FilePath_sub_dir_list ; fullfile(input_data_dir, data_folder(n).name)]
        st_FilePath_name_list = [st_FilePath_name_list ; new_files(j).name]
    end
    
end
%%

tmp_size = size(st_FilePath_sub_dir_list)
number_patient = tmp_size(1)
for n = 1 : number_patient
    
    st_FilePath.name = st_FilePath_name_list(n,:)%'DL_cleaned.edf'
    st_FilePath.name = st_FilePath.name{1}
    f_Ini(1,[]);
    %st_FilePath.path =  [st_FilePath_dir st_FilePath_sub_dir_list(n,:)]
    st_FilePath.path = st_FilePath_sub_dir_list(n,:);
    st_FilePath.path = st_FilePath.path{1}
    st_FilePath.full = {[st_FilePath.path st_FilePath.name]};
    rhfe_str_SaveFileName            = strcat(st_FilePath.name(1:end-4),'_hfo.rhfe');
    rhfe_path = fullfile(st_FilePath.path , rhfe_str_SaveFileName);
    st_FilePath
    
    st_ChInfo = f_FileRead_modified(st_FilePath); %st_ChInfo.st_FileInfo

    st_ChInfo
    st_ChInfo.str_ChName	= [];
    st_ChInfo.s_Sampling    = [];
    st_ChInfo.s_TotalTime = [];
    st_ChInfo = f_InfoChAdd_modified(st_ChInfo);
    st_Data.v_Labels	= st_ChInfo.str_ChName;
    st_Data.s_Sampling	= st_ChInfo.s_Sampling(1);
    disp("loaded and updated channel information!")

    num_channel = numel(st_ChInfo.str_ChName);
    channel_name = st_ChInfo.str_ChName
    %save([str_TempFolder st_FilePath.name '_channel_name.mat'], 'channel_name');
    %save('AR_channel_name.mat', 'channel_name');
    % 
    % % %%%%%%%%%%%%%%%%%%%%%################################
    if s_algo_state == "generate_data" | s_algo_state == "all"
        v_ListIdx  = 1:num_channel;
        st_Data = f_ChannelLoad_modified(st_ChInfo, st_Data, v_ListIdx, str_TempFolder)
    end
     % Select data to save in HFO file                            
    %st_HFOData.str_FileName     = st_FileInfo.str_FileName;        
%     st_HFOData.v_TimeLims       = st_Data.v_TimeLims;
%     st_HFOData.str_ChtLabels  = st_Data.v_Labels(...
%                                 st_HFOControl.v_ChannelIdx);
%  
    % %%%%%%%%%%%%%%%%%%%%%################################
    if s_algo_state == "Detector" | s_algo_state == "all"
        all_events = cell(num_channel,1);
        if ~isempty(which(rhfe_path))
            delete(rhfe_path)
        end
        
        st_FileData     = st_ChInfo.st_FileInfo{1}; 
        save(rhfe_path, 'st_FileData')
        for i = 1:num_channel
            st_ChInfo.str_ChName(i)

            tmp_st_analysis = run_HFO_method(str_TempFolder, st_FilePath, st_Data, st_HFOSettings,st_HFOAnalysis, i);
            all_events{i} = tmp_st_analysis.m_EvtLims;
            tmp_st_analysis.str_ChLabel = st_Data.v_Labels(i)
            str_TempFile = tmp_st_analysis.str_TempFile
            tmp_st_analysis	= rmfield(tmp_st_analysis, 'str_TempFile');
            [st_HFOAnalysis] = f_HFOSetChInfo(st_Data, tmp_st_analysis, i)
            f_HFOSaveCh(str_TempFile, st_HFOAnalysis,st_HFOSettings,  i, rhfe_path)
        % Set Channel Info Cell
        end
        % disp(["Detected results are saved " str_name])
    end
end

 function [st_HFOAnalysis] = f_HFOSetChInfo(st_Data, st_HFOAnalysis, selected_channel_index)
        % Set Channel Info Cell
        
        st_HFOAnalysis.s_ChIdx          = selected_channel_index
            
       if isempty(st_HFOAnalysis.m_EvtLims)
%             st_HFOData.v_ChHFOInfo{...
%                 st_HFOControl.s_IdxCycle}  = st_HFOAnalysis;            
%             st_HFOAnalysis                  = struct;
            return
        end
        
        st_HFOAnalysis.m_IntervLims = zeros(size(st_HFOAnalysis.m_EvtLims));
        st_HFOAnalysis.m_Rel2IntLims = zeros(size(st_HFOAnalysis.m_EvtLims));

        s_IntWidth      = 2;    % Save 2 seconds from signal Interval
        s_IntWidth      = s_IntWidth .* st_HFOAnalysis.s_Sampling;
        s_IntMean       = round(s_IntWidth / 2);
        
        for kk = 1:size(st_HFOAnalysis.m_EvtLims,1)
            
            s_EvtMean   = round(mean(st_HFOAnalysis.m_EvtLims(kk,:)));
            
        	s_PosIni    = s_EvtMean - s_IntMean;
        	s_PosEnd    = s_EvtMean + s_IntMean;
            
            if s_PosIni < 1
                s_PosIni    = 1;
                s_PosEnd    = s_IntWidth;
            elseif s_PosEnd > numel(st_Data.v_Time)
                s_PosIni    = numel(st_Data.v_Time) - s_IntWidth;
                s_PosEnd    = numel(st_Data.v_Time);
            end
            
            st_HFOAnalysis.m_IntervLims(kk,:)   = [s_PosIni,s_PosEnd];
            st_HFOAnalysis.m_Rel2IntLims(kk,:)  = st_HFOAnalysis.m_EvtLims(kk,:)...
                                                - s_PosIni + 1;
        end
        
            
 end

function v_Intervals = f_GetHFOIntervals(pstr_SignalPath,ps_SignalIdx,pm_Lims)

    load(pstr_SignalPath)
    pv_Signal       = m_Data(:,ps_SignalIdx); 
    clear m_Data

    v_Intervals     = cell(size(pm_Lims,1),1);

    for kk = 1:size(pm_Lims,1)        
        v_Intervals(kk) = {pv_Signal(pm_Lims(kk,1):pm_Lims(kk,2))};
    end
end
function f_HFOSaveCh(str_TempFile, st_HFOAnalysis, st_HFOSettings,  selected_channel_index, str_SavePath)
    % Save channel HFO Info


    if isempty(st_HFOAnalysis.m_EvtLims)
        return
    end
    
    st_Ch.st_HFOSetting = st_HFOSettings;
    st_Ch.st_HFOInfo    = st_HFOAnalysis;
    st_Ch.v_Intervals   = f_GetHFOIntervals(str_TempFile,...
                       selected_channel_index,...
                        st_Ch.st_HFOInfo.m_IntervLims); 

    st_Ch.st_HFOInfo.v_EvType	= ones(...
                                size(st_Ch.st_HFOInfo.m_EvtLims,1),1);

    str_Channel = st_Ch.st_HFOInfo.str_ChLabel{1,1};
    str_Channel = strrep(str_Channel,'-','_');
    str_Channel = strrep(str_Channel,' ','_');
    st_save     = struct;

    if ~isnan(str2double(str_Channel))
        str_Channel = strcat('Ch',str_Channel);
    end

    st_save     = setfield(st_save,str_Channel,st_Ch); %#ok<NASGU,SFLD>

    save(str_SavePath, '-struct', 'st_save', '-append')
end 



function f_Ini(s_Indicator,pst_Path)
        % Set Paths for all the folders
        
        if isempty(pst_Path)
            st_Path = struct;
        end
        
        if s_Indicator
            dbstop if error
            clc;clear;close all
            %     warning off all
            st_Path.path    = which('run_HFO_detector');
            st_Path.path    = fileparts(st_Path.path);
            st_Path.pathold = cd(st_Path.path);
            
            addpath(genpath(fullfile('.','Functions')));
            addpath(genpath(fullfile('.','Auxiliar-GUI')));
            addpath(genpath(fullfile('.','Icons')));
            addpath(genpath(fullfile('.','External')));
            addpath(genpath(fullfile('.','Memory')));
            addpath(genpath(fullfile('.','Temp')));
        else
            rmpath(genpath(fullfile('.','Functions')));
            rmpath(genpath(fullfile('.','Auxiliar-GUI')));
            rmpath(genpath(fullfile('.','Icons')));
            rmpath(genpath(fullfile('.','External')));
            rmpath(genpath(fullfile('.','Memory')));
            rmpath(genpath(fullfile('.','Temp')));
            
            st_Path.pathold = cd(st_Path.pathold);
            
        end
end
% 
function [st_HFOAnalysis] = run_HFO_method(str_TempFolder, st_FilePath, st_Data, st_HFOSettings, st_HFOAnalysis, selected_channel_index)
%     str_TempFolder      = './Temp/';
%     str_TempFile        = [];
    str_TmpFName	= st_FilePath.name;
    str_TmpFName(str_TmpFName=='.') = '-'; 
    str_TempFile = [str_TempFolder  str_TmpFName '~tmp.mat'];
    load(str_TempFile)
     
    st_Data.m_Data  = m_Data;
    

    st_HFOAnalysis.m_EvtLims	= f_findHFOxSTE(...
                                str_TempFile,...
                                selected_channel_index,...
                                st_HFOSettings,...
                                st_HFOAnalysis.s_Sampling); 
     st_HFOAnalysis.str_TempFile = str_TempFile
            
end



    
function [st_ChInfo] = f_FileRead_modified(st_FilePath)
    % Read signal header for analysis modified for only reading in one
    %file
    m_CellStr	= cell(1,1);

    st_ChInfo.st_FileInfo	= cell(1);

    st_FileInfo = f_GetHeader(st_FilePath);
    

    if ~(st_FileInfo.s_error || st_FileInfo.s_Check)

        st_FilePath.name    = [];
        st_FilePath.path	= [];
        st_FilePath.full	= [];
        return
    end
    % Check if the file is a valid file
    if st_FileInfo.s_error && st_FileInfo.s_Check

        st_FilePath.name    = [];
        st_FilePath.path	= [];
        st_FilePath.full	= [];
        errordlg(st_FileInfo.st_Custom,'Reading Warning');

        return
    end
    
    if numel(st_FileInfo.v_Labels) ~= numel(st_FileInfo.s_Samples)
        st_FileInfo.s_Samples   = repmat(st_FileInfo.s_Samples,...
                                size(st_FileInfo.v_Labels));
    end

    if numel(st_FileInfo.v_Labels) ~= numel(st_FileInfo.v_SampleRate)
        st_FileInfo.v_SampleRate	= repmat(st_FileInfo.v_SampleRate,...
                                    size(st_FileInfo.v_Labels));
    end

    st_FileInfo.s_Time	= str2double(...
                        mat2str(st_FileInfo.s_Time(1)));

    st_ChInfo.st_FileInfo(1)	= {st_FileInfo};

    if ismember('',st_FileInfo.v_Labels)            
        v_NumChanStr            = cellstr(strcat(repmat('chan',...
                                numel(st_FileInfo.s_Samples),1),...
                                num2str((...
                                1:numel(st_FileInfo.s_Samples))')));
        st_FileInfo.v_Labels	= v_NumChanStr;                
    end

    m_CellStr{1,1} = st_FileInfo.v_Labels;
    m_CellStr{2,1} = st_FileInfo.v_SampleRate(:);

    if numel(st_FileInfo.s_Samples) == 1
        st_FileInfo.s_Samples   = repmat(st_FileInfo.s_Samples,...
                                size(st_FileInfo.v_SampleRate(:)));
    else
        st_FileInfo.s_Samples   = st_FileInfo.s_Samples(:);

    end
    m_CellStr{3,1} = st_FileInfo.s_Samples;

    v_CommonCh  = cell(1);
    v_CommonFs  = cell(1);
    v_CommonNel	= cell(1);
    kk = 1;
    v_CommonCh	= vertcat(v_CommonCh,m_CellStr{1,kk}); 
    v_CommonFs	= vertcat(v_CommonFs,m_CellStr{2,kk});
    v_CommonNel = vertcat(v_CommonNel,m_CellStr{3,kk}); %#ok<AGROW> 

    v_CommonCh	= v_CommonCh(2:end);
    v_CommonFs	= cell2mat(v_CommonFs);
    v_CommonNel	= cell2mat(v_CommonNel);

    [v_CommonCh,v_Id]	= unique(v_CommonCh);
    v_CommonFs          = v_CommonFs(v_Id);
    v_CommonNel         = v_CommonNel(v_Id);
    [~,v_Id]            = sort(v_Id);
    v_CommonCh          = v_CommonCh(v_Id);
    v_CommonFs          = v_CommonFs(v_Id);
    v_CommonNel         = v_CommonNel(v_Id);

    m_CommonTable       = zeros(numel(v_CommonCh),1);
    m_CommonTable(:,kk) = ismember(v_CommonCh,m_CellStr{1,kk});

    m_CommonTable = sum(m_CommonTable,2);  

    st_ChInfo.str_CommonCh	= v_CommonCh(m_CommonTable == 1);
    st_ChInfo.v_CommonFs      = v_CommonFs(m_CommonTable == 1);
    st_ChInfo.v_CommonNel     = v_CommonNel(m_CommonTable == 1);

    if numel(m_CommonTable) ~= numel(st_ChInfo.str_CommonCh)
        warndlg(['There are channels not common' ...
                ', these channels will be ignored'],'!! Warning !!');
    end
    %st_ChInfo.str_CommMont = st_ChInfo.str_CommonCh;
                
end

function [st_ChInfo] = f_InfoChAdd_modified(st_ChInfo)
    % Add a channel to read or load
    %str_File    = st_FileInfo.str_FileName
    s_Sampling  = st_ChInfo.v_CommonFs;
    s_TotalTime = st_ChInfo.v_CommonNel/...
                            (s_Sampling * 60);
          

    if  isempty(st_ChInfo.str_ChName)
        st_ChInfo.str_ChName    = st_ChInfo.str_CommonCh
    else
        st_ChInfo.str_ChName	= vertcat(st_ChInfo.str_ChName,...
                                st_ChInfo.str_CommonCh);
    end

    if isempty(st_ChInfo.s_Sampling)
        st_ChInfo.s_Sampling	= s_Sampling; 
    else
        st_ChInfo.s_Sampling	= vertcat(st_ChInfo.s_Sampling,...
                                s_Sampling);
    end

    if isempty(st_ChInfo.s_TotalTime)
        st_ChInfo.s_TotalTime	= s_TotalTime;        
    else                                        
        st_ChInfo.s_TotalTime	= vertcat(st_ChInfo.s_TotalTime,...
                                s_TotalTime);
    end

    [~,v_Idx]	= unique(st_ChInfo.str_ChName)
    v_Idx       = sort(v_Idx,'ascend');

    st_ChInfo.str_ChName    = st_ChInfo.str_ChName(v_Idx); 
    st_ChInfo.s_Sampling    = st_ChInfo.s_Sampling(v_Idx)
    st_ChInfo.s_TotalTime   = st_ChInfo.s_TotalTime(v_Idx);
    
end  

function [st_Data] = f_ChannelLoad_modified(st_ChInfo, st_Data, v_ChIdx, str_TempFolder)
    %Load the channels selected        

    % Load selected channels  
    st_FileInfo = st_ChInfo.st_FileInfo{1}
    st_Data.v_TimeLims = [0  st_FileInfo.s_Time]
    
    st_Dat	= f_GetData(st_FileInfo, st_Data.v_TimeLims, v_ChIdx);

    if isempty(st_Dat.m_Data)
        disp('Error Loading Data:')
        toc(s_ticID)

        st_ChInfo.FlagError = true;

        return
    end
    m_Data  = single(st_Dat.m_Data(:,v_ChIdx));
    st_Data.v_Time = st_Dat.v_Time;
    
    % Save Original Data
    str_TmpFName	= st_FileInfo.str_FileName;
    str_TmpFName(str_TmpFName=='.') = '-'; 
    str_TempFile	= [str_TempFolder  str_TmpFName '~tmp.mat'];

    if ~isempty(which(str_TempFile))
        delete(str_TempFile)
    end
    save(str_TempFile,'m_Data','-v7.3')
    clear m_Data st_Dat
end

 function f_TimeFreqSetSpectrum()
        % Set Scalogram
        s_miTimexPlot   = 0.25;
        if ~logical(get(st_Select.TimFrqPanel,'Value')) || ...
                any(isnan(st_Spectrum.v_FreqLims))
            return
        end
        
        if ~isfield(st_Data,'m_Data')
            return
        end
                
        if st_SpectrumOpt.s_PlotOK
            return
        end
        
        if sum(st_Spectrum.v_WindLims == ...
            [st_Position.s_IdxIni st_Position.s_IdxEnd]) == 2 && ...
            sum(get(st_hAxes.TimeFreq,'YLim') == ...
                st_Spectrum.v_FreqLims)	== 2
            return
        end
        
        if toc(st_Position.s_TFdisplayTime) < s_miTimexPlot
            return
        end
        pause(0.01)
        
        st_Spectrum.s_FreqSeg       = st_Spectrum.s_FreqRes; 
        st_Spectrum.s_StDevCycles   = 3; 
        st_Spectrum.s_Magnitudes    = 1;
        st_Spectrum.s_SquaredMag    = 0;
        st_Spectrum.s_MakeBandAve   = 0;
        st_Spectrum.s_MinFreqHz     = st_Spectrum.v_FreqLims(1);
        st_Spectrum.s_MaxFreqHz     = st_Spectrum.v_FreqLims(2);
        st_Spectrum.s_Phases        = 0;
        st_Spectrum.s_TimeStep      = 1/st_Spectrum.v_FreqLims(2);
        
        v_DataTF = st_Data.m_Data(...
                st_Position.s_IdxIni:...
                st_Position.s_IdxEnd,...
                st_Spectrum.st_FilterData.v_NewChIdx);
                        
        
        
        if st_Spectrum.s_TFSampling ~= st_Data.s_Sampling && ...
                logical(get(st_TimeFreq.AliasingChk,'Value'))
            
            s_Filter        = f_GetIIRFilter(st_Data.s_Sampling,...
                                st_Spectrum.s_TFSampling,[],'low',[]);
                            
            v_DataTF        = f_IIRBiFilter(v_DataTF,s_Filter);
            
            if logical(mod(st_Data.s_Sampling,st_Spectrum.s_TFSampling))
                
                v_Time          = (0:numel(v_DataTF) - 1)./ ...
                                    st_Spectrum.s_TFSampling;
                v_TimeAxisAux   = 0:1/st_Spectrum.s_TFSampling:v_Time(end);
                
                v_DataTF        = interp1(v_Time, v_DataTF, v_TimeAxisAux,...
                                'pchip')';
                            
                clear v_Time
                
            else
                v_TimeAxisAux   = (0:numel(v_DataTF) - 1)./ ...
                                    st_Spectrum.s_TFSampling;
                s_SampleStep    = round(st_Data.s_Sampling /...
                                                st_Spectrum.s_TFSampling);
                v_Ind           = 1:s_SampleStep:numel(v_DataTF);
                v_TimeAxisAux   = v_TimeAxisAux(v_Ind);
                v_DataTF        = v_DataTF(v_Ind);
            end
        else
            v_TimeAxisAux = (0:numel(v_DataTF) - 1)./ ...
                                st_Spectrum.s_TFSampling;
        end
                   
        %::::::::::: Time Axis Auxiliar ::::::::::  
        
        v_DataTF    = v_DataTF(:);

        s_SampAve = round(st_Spectrum.s_TimeStep * st_Data.s_Sampling);
        if s_SampAve < 1
            s_SampAve = 1;
        end
    
        if s_SampAve > 1
            v_IndSamp = 1:s_SampAve:numel(v_TimeAxisAux);
            v_TimeAxisAux = v_TimeAxisAux(v_IndSamp);
        end            
                    
        s_Len       = round(numel(v_DataTF) * 0.1);
        v_DataTF    = v_DataTF(:);
        v_DataTF    = [flipud(v_DataTF(2:s_Len + 1));...
                        v_DataTF;flipud(v_DataTF(end - s_Len:end - 1))];         
                           
        %::::::::::: Calculate Spectrogram :::::::::: 
        [st_Spectrum.m_GaborWT,...
            st_Spectrum.v_TimeAxis,...
                st_Spectrum.v_FreqAxis] = ...
                            f_GaborTransformWait(...
                            single(v_DataTF),...
                            st_Spectrum.s_TFSampling,...
                            st_Spectrum.s_MinFreqHz, ...
                            st_Spectrum.s_MaxFreqHz, ...
                            st_Spectrum.s_FreqSeg, ...
                            st_Spectrum.s_StDevCycles, ...
                            st_Spectrum.s_Magnitudes, ...
                            st_Spectrum.s_SquaredMag, ...
                            st_Spectrum.s_MakeBandAve, ...
                            st_Spectrum.s_Phases,...
                            st_Spectrum.s_TimeStep,0);
        
        %::::::::::: Reshape Time Axis ::::::::::
        s_Time1	= v_TimeAxisAux(end) - v_TimeAxisAux(1);
        s_Time2 = st_Spectrum.v_TimeAxis(end) - ....
                st_Spectrum.v_TimeAxis(1);
        s_Time1 = s_Time2 - s_Time1;
        s_Time1 = s_Time1 / 2;
        s_Time1 = find(st_Spectrum.v_TimeAxis >= s_Time1, 1);
        s_Time1 = s_Time1 + 1;
        
        st_Spectrum.m_GaborWT   = st_Spectrum.m_GaborWT(:,...
                                s_Time1 + 1:s_Time1 + numel(v_TimeAxisAux));
        st_Spectrum.v_TimeAxis  = v_TimeAxisAux;
                               

        st_Spectrum.s_Exists    = true;
        f_TimeFreqPlotSpectrum()
        
        st_Position.s_TFdisplayTime  = tic;
        
end
    
function f_TimeFreqPlotSpectrum()
        % Plot the epectrogram
            
        st_Spectrum.str_ColorMap    = v_Colormap{get(...
                                            st_TimeFreq.Colormap,'Value')};
        st_Spectrum.s_ColorLevels   = 256;
        
        m_GaborTemp                 = st_Spectrum.m_GaborWT;
        
        switch get(st_TimeFreq.MethodMenu,'Value')
            case 2
                m_GaborTemp    = f_Matrix2RelAmplitud(m_GaborTemp);
                
            case 3
                m_GaborTemp    = f_Matrix2Norm(m_GaborTemp);
            
            case 4
                
                m_GaborTemp    = f_Matrix2ZScore(m_GaborTemp);
                
            otherwise
        end
        
        v_Lims(1)   = min(m_GaborTemp(:));
        v_Lims(2)   = max(m_GaborTemp(:))*st_SpectrumOpt.s_TimeFreqScale;
        
        if v_Lims(1) > v_Lims(2)
            v_Lims(2)  = v_Lims(1);
        end
                            
        if exist('st_Line','var') && ishandle(st_Line.TFImage)   
            delete(st_Line.TFImage)
        end
        
        pause(0.5)
        
        set(st_hFigure.main,'CurrentAxes',st_hAxes.TimeFreq)
        axes(st_hAxes.TimeFreq); 
        
        [m_GaborTemp,v_TimeTemp,v_FreqTemp] = f_PixelScale(m_GaborTemp,...
                                        st_Spectrum.v_TimeAxis,...
                                        st_Spectrum.v_FreqAxis);
        
        f_ImageMatrix(m_GaborTemp, v_TimeTemp,v_FreqTemp, v_Lims, ...
                            st_Spectrum.str_ColorMap, ...
                                st_Spectrum.s_ColorLevels, 0);
        
        clear m_GaborTemp v_TimeTemp v_FreqTemp
               
        ht_YlabelTF     = get(st_hAxes.TimeFreq,'YLabel');
        
        delete(ht_YlabelTF)
                        
        ht_YlabelTF = ylabel(...
                    {cell2mat(st_Data.v_Labels(...
                    st_Spectrum.st_FilterData.v_NewChIdx)),...
                                                      'Frequency (Hz)'},...
                    'Parent',st_hAxes.TimeFreq,...
                    'FontSize',st_Letter.toolcontrol);
                
        set(st_hAxes.TimeFreq,...
            'XTick',[],...
            'XTickLabel',[],...
            'YLabel',ht_YlabelTF,...
            'YLim',[st_Spectrum.s_MinFreqHz st_Spectrum.s_MaxFreqHz]) 
        
        st_Spectrum.v_WindLims     = [st_Position.s_IdxIni ...
                                    st_Position.s_IdxEnd];
                                
        f_CursorRebuildLineTF()
                                
end