function elementlist=ReadDicomElementList(fname)
% This function ReadDicomElementList will read all (raw) tags from a 
% dicom file. Output is a struct will all tags including there group,
% element number, type, data, length and other info.
%
%   Elements = ReadDicomElementList(Filename)
%
%
% inputs,
%    Filename : Name of the dicom file (optional)
%
% outputs,
%    Elements : A struct with the raw dicom tags, containing :
%             Elements(*).name
%             Elements(*).data
%             Elements(*).group
%             Elements(*).number
%             Elements(*).type
%             Elements(*).length
%             Elements(*).explicit
%             Elements(*).location
%
% example,
%   Elements = ReadDicomElementList('images\example.dcm');
%
% Function is written by D.Kroon University of Twente (October 2010)

% Display a file choose box if not provide as function input
if(nargin<1)
    [fn, dn] = uigetfile('.dcm', 'Select a dicom file'); fname=[dn fn];
end

% Open the dicom file
f=fopen(fname,'r', 'ieee-le');
if(f<0), 
    error('ReadDicomElementList:input',['could not open file' fname]);
end

% Get the File size
fseek(f,0,'eof'); 
fsize = ftell(f); 
fseek(f,0,'bof');

% Read the Dicom header Prefix
fseek(f, 128, 'bof');
Prefix=fread(f, 4, 'uint8=>char')';
if(~strcmpi(Prefix,'DICM'))
    fseek(f, 8, 'bof');
    Prefix=fread(f, 3, 'uint8=>char')';
    if(~strcmpi(Prefix,'ISO'))
        warning('DicomInfo2:input','not a valid dicom header');
    end
    fseek(f, 0, 'bof');
end

% Load Dicom Tag Library
functionname='ReadDicomElementList.m';
functiondir=which(functionname);
functiondir=functiondir(1:end-length(functionname));
load([functiondir 'Dictonary/DicomTagDictionary.mat']); 

% Read all tags until, the PixelData is reached
ntags=0; elementlist=struct;
while(ftell(f)<fsize)
    element=ReadDicomElement(f,dcmdic);
    ntags=ntags+1;
    elementlist(ntags).name=element.name;
    elementlist(ntags).data=element.data;
    elementlist(ntags).group=element.group;
    elementlist(ntags).number=element.number;
    elementlist(ntags).type=element.type;
    elementlist(ntags).length=element.length;
    elementlist(ntags).info=element.info;
    elementlist(ntags).explicit=element.explicit;
    elementlist(ntags).location=element.location;
end
fclose('all');


function element=ReadDicomElement(f,dcmdic)
element=struct();
element.location=ftell(f);
element.group=lower(hexstr(fread(f, 2, 'uint8')));
element.number=lower(hexstr(fread(f, 2, 'uint8')));

field1=['tag' element.group element.number];
field2=['tag' element.group(1:3) 'x' element.number];
field3=['tag' element.group(1:3) 'x' element.number];
if(isfield(dcmdic,field1))
    taginfo=dcmdic.(field1);
elseif(isfield(dcmdic,field2))
    taginfo=dcmdic.(field2);
elseif(isfield(dcmdic,field3))
    taginfo=dcmdic.(field3);
else
    taginfo.name=['Private_' element.group '_' element.number];
    taginfo.type='US';
end

VRtest=fread(f,1,'uint16'); fseek(f,-2,0);

% http://www.leadtools.com/SDK/medical/dicom-spec1.htm
% http://www.leadtools.com/SDK/medical/dicom-spec5.htm
Explicit=(VRtest>255)&&(VRtest~=65535); 
if(Explicit)
    element.type=fread(f,2,'char=>char')';
    element.length=fread(f,1,'uint16');
    if(element.length==0)
       switch(element.type)
       case {'OB','OW','UN','SQ'}
            element.length=fread(f,1,'uint32');
            Explicit=Explicit+1;
       end
    end
    element.info='Type Included';
else
    element.type=taginfo.type;
    element.length=fread(f,1,'uint32');
    element.info='Type Dictonary';
end
element.explicit=Explicit;

% Note: VRs of UT may not have an Undefined Length, ie. a Value Length of
% FFFFFFFFH.
if(element.length==2^32-1),  element.length=-1; end

element.name=taginfo.name;
switch(element.type(1:2))
    case 'OB' %   OB -     |single trailing 0x00 to make even number of bytes. Transfer Syntax determines len
        if( strcmp(element.number,'0001')&&strcmp(element.group,'0002'))
            element.data='';  
            s1=ftell(f);
            VRtest=0; while (VRtest==0||VRtest==2), VRtest=fread(f,1,'uint8'); end
            s2=ftell(f); 
            fseek(f,s1,'bof');
            element.length=s2-s1;
        end
        element.data=fread(f, element.length, 'uint8')';
    case 'SH' %   SH 16    |Short String. may be padded
        element.data=fread(f, element.length, 'char=>char')';
    case 'SQ' %   SQ  -    |Sequence of zero or more items
        element.data=[];
    case 'UI' %   UI 64    |Unique Identifier (delimiter = .) 0-9 only, trailing space to make even #
        element.data=fread(f, element.length, 'char=>char')';
    case 'UL' %   UL 4     |Unsigned long integer
        element.data=fread(f,element.length/4,'ulong')';
    case 'US' %   US 2     |Unsigned short integer (word)
        element.data=fread(f,element.length/2,'ushort')';
    case 'SS' %   SS 2     |signed short integer (word)
        element.data=fread(f,element.length/2,'short')';
    case 'SL' %   SL 4     |signed long integer
        element.data=fread(f,element.length/4,'long')';
    case 'FL' %   FL 4     |Single precision floating pt number (float)
        element.data=fread(f,element.length/4,'float')';
    case 'FD' %   FD 16    |Double precision floating pt number (double)
        element.data=fread(f,element.length/8,'double')';
    case 'AE' %   AE 16    |Application Name
        element.data=fread(f, element.length, 'char=>char')';  
    case 'OF' %   OF -     |Other Float String. floats
        element.data=fread(f, element.length, 'char=>char')';  
    case 'OW' %   OW -     |Other Word String. words
        element.data=fread(f, element.length, 'uint8')';  
    case 'PN' %   PN -     |Person's Name 64byte max per component. 5 components. delimiter = ^
        element.data=fread(f, element.length, 'char=>char')';  
    case 'ST' %   ST 1024  |Short Text of chars
        element.data=fread(f, element.length, 'char=>char')';  
    case 'TM' %   TM 16    |Time hhmmss.frac (or older format: hh:mm:ss.frac)
        element.data=fread(f, element.length, 'char=>char')';  
    case 'UT' %   UT -     |Unlimited Text. trailing spaces ignored
        element.data=fread(f, element.length, 'char=>char')';  
    case 'AS' %   AS 4     |Age String: nnnW or nnnM or nnnY
        element.data=fread(f, element.length, 'char=>char')';  
    case 'AT' %   AT 4     |Attribute Tag gggg,eeee
        element.data=fread(f, element.length, 'char=>char')';  
    case 'CS' %   CS 16    |Code String
        element.data=fread(f, element.length, 'char=>char')';  
    case 'DA' %   DA 8     |Date yyyymmdd (check for yyyy.mm.dd also and convert)
        element.data=fread(f, element.length, 'char=>char')';
    case 'DS' %   DS 16    |Decimal String may start with + or - and may be padded with l or t space
        element.data=fread(f, element.length, 'char=>char')';        
    case 'DT' %   DT 26    |Date Time YYYYMMDDHHMMSS.FFFFFF&ZZZZ (&ZZZ is optional & = + or -)
        element.data=fread(f, element.length, 'char=>char')';        
    case 'IS' %   IS 12    |Integer encoded as string. may be padded
        element.data=fread(f, element.length, 'char=>char')';       
    case 'LO' %   LO 64    |Character string. can be padded. cannot contain \ or any control chars except ESC
        element.data=fread(f, element.length, 'char=>char')';       
    case 'LT' %   LT 10240 |Long Text. Leading spaces are significant. trailing spaces arent
        element.data=fread(f, element.length, 'char=>char')';            
    otherwise
        if(element.length>0)
            element.data=fread(f, element.length, 'char=>char')';
        else
            element.data=[];
        end
end

function str=removepadding(str)
while((~isempty(str))&&(str(end)==' ')), str=str(1:end-1); end
while((~isempty(str))&&(str(1)==' ')), str=str(2:end); end

function valstr=hexstr(val)
valstr='';
for i=length(val):-1:1,
    if(val(i)<16),
        valstr=[valstr '0' num2str(dec2hex(val(i)))]; 
    else
        valstr=[valstr num2str(dec2hex(val(i)))]; 
    end
end



