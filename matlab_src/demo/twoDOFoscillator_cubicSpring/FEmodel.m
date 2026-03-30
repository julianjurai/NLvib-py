%========================================================================
% DESCRIPTION:
% Class representing a linear 3D finite element (FE) model. Besides mass 
% and stiffness matrices, the degree-of-freedom (DOF) map (DOFmap) is set 
% up, which assigns a node (second column) and respective nodal DOF (third 
% column) to the respective column/row of M and K (first column). Also, 
% the mesh is imported in terms of node and element definitions. Finally, 
% one can animate a time series (method 'animate'), relying on CalculiX.
% 
% The primary current purpose of this class is to set up FE models before
% application of component mode synthesis within the tool NLstep 
% (https://github.com/maltekrack/).
% 
% In this simple version, the FE model is set up from files exported by 
% the open source FE tool CalculiX (https://www.dhondt.de/). Further, every 
% node in the DOFmap must have exactly three degrees of freedom, namely 
% the translations aligned with a global cartesian coordinate system 
% (i.e., generalized coordinates such as Lagrange multipliers, or the 
% application of constraints to individual translations is not allowed 
% here), identified by the nodal DOF value 1, 2 and 3, corresponding to
% global x, y and z direction, respectively. 
%========================================================================
% This file is part of NLvib.
%
% If you use NLvib, please refer to the book:
%   M. Krack, J. Gross: Harmonic Balance for Nonlinear Vibration
%   Problems. Springer, 2019. https://doi.org/10.1007/978-3-030-14023-6.
%
% COPYRIGHT AND LICENSING:
% NLvib Copyright (C) 2025
%   Malte Krack (malte.krack@ila.uni-stuttgart.de)
%   Johann Gross (johann.gross@ila.uni-stuttgart.de)
%   University of Stuttgart
% This program comes with ABSOLUTELY NO WARRANTY.
% NLvib is free software, you can redistribute and/or modify it under the
% GNU General Public License as published by the Free Software Foundation,
% either version 3 of the License, or (at your option) any later version.
% For details on license and warranty, see http://www.gnu.org/licenses
% or gpl-3.0.txt.
%========================================================================
classdef FEmodel < MechanicalSystem

    properties
        DOFmap              % DOF map
        nodes               % node definitions
        elements            % element definitions
    end

    methods
        function obj = FEmodel(modelFileName)
            % Read mass and stiffness matrices
            M = FEmodel.readMatrixCalculiX([modelFileName '.mas']);
            K = FEmodel.readMatrixCalculiX([modelFileName '.sti']);

            % Call parent constructor
            % NOTE: An FE model has no inherent damping, so we set D=0*M.
            obj = obj@MechanicalSystem(M,0*M,K,[],[]);

            % Read DOF map and mesh
            obj.DOFmap = ...
                FEmodel.readDOFmapCalculiX([modelFileName '.dof']);
            [obj.nodes,obj.elements] = ...
                FEmodel.readMeshCalculix([modelFileName '.msh']);
        end
        function IND = getIndex2NodalDOF(obj,nodeSet,DOF)
            idx = false(size(obj.DOFmap,1),1);
            for ss = 1:length(nodeSet)
                idx(obj.DOFmap(:,2)==nodeSet(ss)) = true;
            end
            if nargin>2 && ~isempty(DOF)
                if numel(DOF)~=1 || ~ismember(DOF,[1 2 3])
                    error('DOF specifier must be scalar 1, 2 or 3.');
                end
                idx = idx & obj.DOFmap(:,3) == DOF;
            end
            IND = find(idx);
        end
        function pos = getNodePosition(obj,inode)
            node = obj.nodes([obj.nodes.in]==inode);
            pos = node.pos;
        end
        function animate(obj,u,animation_file)
            %% Animate response
            if nargin<3 || isempty(animation_file)
                animation_file = 'anim';
            end

            % Displacement available for all nodes in DOFmap
            nodu = FEmodel.getnodes(obj.nodes,obj.DOFmap(1:3:end,2));

            % Open Calculix result file for writing
            fid = FEmodel.fopen_cgx(animation_file);

            % Write File header and Node Definition Block
            FEmodel.print_nod(obj.nodes,fid);

            % Write Element Definition Block
            FEmodel.print_ele(obj.elements,fid);

            % Write Nodal Results Block
            FEmodel.print_results(nodu,fid,u);

            % Specify end of data and close file
            FEmodel.fclose_cgx(fid);
        end
    end
    methods (Static)
        function matrix = readMatrixCalculiX(filename)
            %% Function reading a symmetric matrix from an ASCII file 
            % containing an array, where the first column corresponds to 
            % the row index, the second to the column index and the third 
            % to the value. Not specified elements are presumed as zero. 
            % The matrix is stored as sparse.

            % Load raw data
            A = load(filename);

            % Only proceed with nonzero entries
            A = A(A(:,3)~=0,:);

            % Setup sparse matrix
            m = max(A(:,1));
            n = m;
            nzmax = 2*length(A(:,3))-m;
            matrix = sparse(A(:,1),A(:,2),A(:,3),m,n,nzmax);

            % Retain symmetry by only considering upper triangular part
            matrix = transpose(triu(matrix)) + triu(matrix) - ...
                diag(diag(matrix));
        end
        function DOFmap = readDOFmapCalculiX(filename)
            %% Function reading DOF map from ASCII file
            filestr = fileread(filename);
            filestr = strrep(filestr,'.',' '); % ?
            tmp = textscan(filestr,'%u %u');
            DOFmap = zeros(size(tmp{1},1),3);
            DOFmap(:,1) = 1:size(tmp{1},1); % system DOF index
            DOFmap(:,2) = double(tmp{1});   % node index
            DOFmap(:,3) = double(tmp{2});   % nodal DOF index
        end
        function [nodes,elements] = readMeshCalculix(filename)
            %% Function reading mesh from ASCII file
            nodes = []; elements = [];
            % Scan CalculiX input deck for node and element definitions
            fid = fopen(filename);
            tline = fgetl(fid); ntline = 1;
            while ischar(tline)
                if isempty(tline)
                    % Empty line
                    tline = fgetl(fid); ntline = ntline+1;
                    continue;
                end
                [keyword,pos] = textscan(tline,'*%s',1,'Delimiter',',');
                if ~isempty(keyword) && ~isempty(keyword{1})
                    keyword = keyword{1}{1};
                    if ~isempty(tline(pos+1:end))
                        params = ...
                            textscan(tline(pos+1:end),'%s','Delimiter',',');
                        params = params{1};
                    else
                        params = {};
                    end
                else
                    % Unexpected input, treat as comment and read next line
                    tline = fgetl(fid); ntline = ntline+1;
                    continue;
                end

                switch upper(keyword)
                    case 'NODE'
                        [dat,tline,ntline] = FEmodel.get_data(fid,...
                            ntline,'%u','%f','%f','%f');
                        clear nodes;
                        nodes(1:size(dat,1)) = struct('in',[],'pos',[]);
                        for inod=1:size(dat,1)
                            nodes(inod).in = dat(inod,1);
                            nodes(inod).pos = dat(inod,2:4);
                        end
                        continue;
                    case 'ELEMENT'
                        % Elements definition
                        etype = '';
                        for i=1:length(params)
                            [name,val] = FEmodel.get_param(params{i});
                            switch upper(name)
                                case 'TYPE'
                                    etype = strtrim(val);
                                case 'ELSET'
                                otherwise
                                    warning(['Invalid parameter name ' name ...
                                        ' for keyword ' upper(keyword) '!']);
                            end
                        end

                        % Define elements according to type
                        switch upper(etype)
                            case 'C3D4'
                                [dat,tline,ntline] = FEmodel.get_data(fid,ntline,'%u',...
                                    '%f','%f','%f','%f');
                                ip = length(elements);
                                if ip==0
                                    clear elements;
                                end
                                elements(ip+(1:size(dat,1))) = struct('in',[],'type',[],'inode',[]);
                                for iel=1:size(dat,1)
                                    elements(ip+iel).in = dat(iel,1);
                                    elements(ip+iel).type = 'C3D4';
                                    elements(ip+iel).inode = dat(iel,2:end);
                                end
                            case 'C3D8'
                                [dat,tline,ntline] = FEmodel.get_data(fid,ntline,'%u',...
                                    '%f','%f','%f','%f',...
                                    '%f','%f','%f','%f');
                                ip = length(elements);
                                if ip==0
                                    clear elements;
                                end
                                elements(ip+(1:size(dat,1))) = struct('in',[],'type',[],'inode',[]);
                                for iel=1:size(dat,1)
                                    elements(ip+iel).in = dat(iel,1);
                                    elements(ip+iel).type = 'C3D8';
                                    elements(ip+iel).inode = dat(iel,2:end);
                                end
                            case {'C3D10','C3D10R'}
                                [dat,tline,ntline] = FEmodel.get_data(fid,ntline,'%u',...
                                    '%f','%f','%f','%f','%f',...
                                    '%f','%f','%f','%f','%f');
                                ip = length(elements);
                                if ip==0
                                    clear elements;
                                end
                                elements(ip+(1:size(dat,1))) = struct('in',[],'type',[],'inode',[]);
                                for iel=1:size(dat,1)
                                    elements(ip+iel).in = dat(iel,1);
                                    elements(ip+iel).type = 'C3D10';
                                    elements(ip+iel).inode = dat(iel,2:end);
                                end
                            case 'C3D15'
                                [dat,tline,ntline] = FEmodel.get_data(fid,ntline,'%u',...
                                    '%f','%f','%f','%f','%f',...
                                    '%f','%f','%f','%f','%f',...
                                    '%f','%f','%f','%f','%f');
                                ip = length(elements);
                                if ip==0
                                    clear elements;
                                end
                                elements(ip+(1:size(dat,1))) = struct('in',[],'type',[],'inode',[]);
                                for iel=1:size(dat,1)
                                    elements(ip+iel).in = dat(iel,1);
                                    elements(ip+iel).type = 'C3D15';
                                    elements(ip+iel).inode = dat(iel,2:end);
                                end
                            case {'C3D20','C3D20R'}
                                [dat,tline,ntline] = FEmodel.get_data(fid,ntline,'%u',...
                                    '%f','%f','%f','%f','%f',...
                                    '%f','%f','%f','%f','%f',...
                                    '%f','%f','%f','%f','%f',...
                                    '%f','%f','%f','%f','%f');
                                ip = length(elements);
                                if ip==0
                                    clear elements;
                                end
                                elements(ip+(1:size(dat,1))) = struct('in',[],'type',[],'inode',[]);
                                for iel=1:size(dat,1)
                                    elements(ip+iel).in = dat(iel,1);
                                    elements(ip+iel).type = 'C3D20';
                                    elements(ip+iel).inode = dat(iel,2:end);
                                end
                            otherwise
                                error(['Unknown element type ' etype '.']);
                        end
                        continue;
                end
                % Read next line
                tline = fgetl(fid); ntline = ntline+1;
            end
        end
        function [name,val] = get_param(parameter)
            %% Function required by 'readMeshCalculiX'
            if iscell(parameter)
                parameter = parameter{1};
            end
            paramparts = textscan(parameter,'%s','Delimiter','=');
            paramparts = [paramparts{:}];
            name = paramparts{1};
            if length(paramparts)>1
                val = paramparts{2};
                switch class(val)
                    case 'char'
                        % Already char, do nothing
                        %                 val = val;
                    case {'int','double'}
                        val = str2num(val);
                    otherwise
                        % Assume char, do nothing
                end
            else
                val = [];
            end
        end
        function [dat,tline,ntline] = get_data(fid,ntline,varargin)
            %% Function required by 'readMeshCalculiX'
            % Get data in subsequent lines according to format specified in
            % varargin

            % Maximum number of data lines
            nmax = 1e6;

            % Test input data type
            if any(ismember(varargin,{'%s'}))
                % String input data
                dat = cell(nmax,length(varargin));
            else
                dat = zeros(nmax,length(varargin));
            end

            % Collect data
            idat = 0; format = [varargin{:}];
            for i=1:nmax
                tline = fgetl(fid); ntline = ntline+1;
                if isempty(tline) || (isnumeric(tline)&&tline==-1)
                    break;
                end
                if length(varargin)>11
                    % Data has to be read from multiple lines!
                    if length(varargin)>21
                        error('Too many input parmameters');
                    end
                    if length(varargin)~=16
                        dati1 = textscan(tline,[varargin{1:11}],'Delimiter',',',...
                            'CommentStyle','**');
                        tline = fgetl(fid); ntline = ntline+1;
                        dati2 = textscan(tline,[varargin{12:end}],'Delimiter',',',...
                            'CommentStyle','**');
                    else
                        dati1 = textscan(tline,[varargin{1:10}],'Delimiter',',',...
                            'CommentStyle','**');
                        tline = fgetl(fid); ntline = ntline+1;
                        dati2 = textscan(tline,[varargin{11:end}],'Delimiter',',',...
                            'CommentStyle','**');
                    end

                    dati = horzcat(dati1,dati2);
                else
                    % Data is contained in single line
                    dati = textscan(tline,format,'Delimiter',',',...
                        'CommentStyle','**');
                end

                if isempty(dati) || isempty(dati{1})
                    % No data contained in line, check whether line is comment line
                    if strcmp(tline(1:2),'**')
                        % Line is comment, proceed with next line
                        continue;
                    else
                        % Line probably contains next keyword
                        break;
                    end
                else
                    % If a string input is expected, check whether the current line
                    % is actually a new keyword
                    dobreak = 0;
                    if iscell(dati{1})
                        tmp = dati{1}{1};
                        if strcmp(tmp(1),'*')
                            dobreak = 1;
                        end
                    end
                    if dobreak
                        % Line contains next keyword, break
                        break;
                    else
                        % Regular input line, increment data pointer and store
                        % input data
                        idat = idat+1;
                        if iscell(dat)
                            % Store string data
                            dat(idat,:) = dati;
                        else
                            % Store numeric data
                            for id=1:length(dati)
                                if ~isempty(dati{id})
                                    dati{id} = double(dati{id});
                                else
                                    dati{id} = NaN;
                                end
                            end
                            dati = cellfun(@double,dati);
                            dat(idat,:) = dati;
                        end
                    end
                end
            end

            % Erase unused, allocated data entries
            dat = dat(1:idat,:);
        end

        function fid = fopen_cgx(data_file,nodes,elements)
            fid = fopen([data_file '.frd'],'w');

            % Write Model Header Record
            fprintf(fid,'    1C\n');

            if nargin>=3
                % Write File header and Node Definition Block
                print(nodes,fid);

                % Write Element Definition Block
                print(elements,fid);
            end
        end

        function fclose_cgx(fid)

            [fname,~,~,~] = fopen(fid);
            [~,data_file,~] = fileparts(fname);

            % Specify end of data and close file
            fprintf(fid,' 9999');
            fclose(fid);
            %% Correct number of digits in the exponent in case of Windows
            % notation
            filestr = fileread(fname);

            % Save in file
            fid = fopen([data_file '.frd'],'w');
            fprintf(fid,filestr);
            fclose(fid);
        end

        function nodes = getnodes(Nodes,nset)
            %% Find nodes with node number 'nset' in Nodes
            nset = nset(:).';

            % Find node for each element of the node set list
            orig = [Nodes.in]; orig = orig(:); new = nset;
            I = zeros(length(nset),1);
            for i=1:length(nset)
                indi = find(orig==new(i));
                if isempty(indi)
                    error(['Node ' num2str(new(i)) ...
                        ' not in considered set of ' num2str(length(orig)) ' nodes.']);
                end
                I(i) = indi(1);
            end
            nodes = Nodes(I);
        end

        function print_nod(nod,fid)
            % Prepare FE Interface definition
            nodes_n = [nod.in]';
            nodes_pos = reshape([nod.pos],3,[])';
            nodes_pos(abs(nodes_pos)<1e-9) = 0;

            % Write Nodal Point Coordinate Block
            fprintf(fid,'    2C                                                                   1\n');
            fprintf(fid,' -1%10u% 6.5E% 6.5E% 6.5E\n',[nodes_n nodes_pos]');
            fprintf(fid,' -3\n');
        end

        function print_results(nodes,fid,u)
            % Avoid difficulties with small numbers
            Ntd = size(u,2);
            nodes_n = [nodes.in]';
            u(abs(u)<1e-9) = 0;

            for ii=1:Ntd

                % Write Parameter Header Block
                fprintf(fid,'    1PSTEP %13u\n',ii);

                % Write Nodal Results Block
                fprintf(fid,'  100CL%5u% 6.5E %11u %21u %4u%s %6u\n',...
                    100+ii,ii/Ntd,length(nodes),1,ii,'NANIM',1);
                fprintf(fid,' -4  DISP        4    1\n');
                fprintf(fid,' -5  D1          1    2    1    0\n');
                fprintf(fid,' -5  D2          1    2    2    0\n');
                fprintf(fid,' -5  D3          1    2    3    0\n');
                fprintf(fid,' -5  ALL         1    2    0    0    1ALL\n');
                fprintf(fid,' -1%10u% 6.5E% 6.5E% 6.5E\n',...
                    [nodes_n';full(reshape(u(:,ii),3,[]))]);
                fprintf(fid,' -3\n');
            end
        end

        function print_ele(obj,fid)
            %% Write Element Definition Block
            fprintf(fid,'    3C                                                                   1\n');
            for iel=1:length(obj)

                % Write Element header
                [nel,ntype] = FEmodel.get_info_ele(obj(iel));
                fprintf(fid,' -1 %9u %4u %4u %4u\n',iel,ntype,0,0);
                %                 fprintf(fid,' -1 %9u %4u %4u %4u\n',obj(iel).in,ntype,0,0);

                % Write nodes of element
                while ~isempty(nel)
                    fprintf(fid,' -2');
                    fprintf(fid,' %9u',nel(1:min(10,length(nel))));
                    fprintf(fid,'\n');
                    nel(1:min(10,length(nel))) = [];
                end
            end
            fprintf(fid,' -3\n');
        end

        function [nel,ntype] = get_info_ele(obj)
            switch obj.type
                case 'C3D4'
                    ntype = 3;
                    nel = obj.inode(1:4);
                case 'C3D8'
                    ntype = 1;
                    nel = obj.inode(1:8);
                case 'C3D10'
                    ntype = 6;
                    nel = obj.inode(1:10);
                case 'C3D15'
                    ntype = 5;
                    nel = obj.inode(1:15);
                    % Account for different element definition CGX vs.
                    % CCX (last two blocks of 3 nodes are exchanged)
                    nel = [nel(1:9) nel(13:15) nel(10:12)];
                case 'C3D20'
                    ntype = 4;
                    nel = obj.inode(1:20);

                    % Account for different element definition CGX vs.
                    % CCX (last two blocks of 4 nodes are exchanged)
                    nel = [nel(1:12) nel(17:20) nel(13:16)];
                case 'C3D6'
                    ntype = 2;
                    nel = obj.inode(1:6);
                case 'C2D3'
                    ntype = 7;
                    nel = obj.inode(1:3);
                case 'C2D6'
                    ntype = 8;
                    nel = obj.inode(1:6);
                case 'C2D4'
                    ntype = 9;
                    nel = obj.inode(1:4);
                case 'C2D8'
                    ntype = 10;
                    nel = obj.inode(1:8);
                otherwise
                    error('Unknown element');
            end
        end
    end
end