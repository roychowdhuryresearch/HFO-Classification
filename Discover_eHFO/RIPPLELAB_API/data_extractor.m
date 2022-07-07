% figure out path setting
run('HFO_parameters')
f_Ini(1,[]);
st_FilePath.full = {[st_FilePath.path st_FilePath.name]};
st_ChInfo = f_FileRead_modified(st_FilePath); %st_ChInfo.st_FileInfo



st_ChInfo
st_ChInfo.str_ChName	= [];
st_ChInfo.s_Sampling    = [];
st_ChInfo.s_TotalTime = [];
%st_ChInfo = f_InfoChAdd_modified(st_ChInfo);
%st_Data.v_Labels	= st_ChInfo.str_ChName;
%st_Data.s_Sampling	= st_ChInfo.s_Sampling(1);
%disp("loaded and updated channel information!")

num_channel = numel(st_ChInfo.str_ChName);


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

% 