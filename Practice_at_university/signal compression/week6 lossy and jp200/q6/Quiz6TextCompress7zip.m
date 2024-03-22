% function  Quiz6_BWT_TextCompress7zip.m
% See the previous settings used in the Quiz 5 for Stages I,II

%% 0 Check that you have the program 7zip installed in your computer
% The simplest check is to run
dos('7z')
% the output of the above command prints to screen about 60 lines of usage
% instructions
% A better help on the parameters of 7zip can be found e.g., at
% https://sevenzip.osdn.jp/chm/cmdline/switches/method.htm
% On the Windows computers you might have the program at this path:
dos('D:\GRAM\MasterProgramme\Tampere\signal compression\week4 adaptive model of arithetic code\FilesForQuiz5TextCompress7zip')
crt_dir = pwd
copyfile('D:\GRAM\MasterProgramme\Tampere\signal compression\week4 adaptive model of arithetic code\FilesForQuiz5TextCompress7zip',crt_dir)
% Save the files from the Quiz6 archive to  current matlab working
% directory

%% Stage I. The input text to be compressed is in the file 'book1.txt' that is located in the
%% archived folder for this quiz
% You should place the file book1.txt in the current folder where you run 
% matlab for this task
InputFileName = 'book1.txt'
sss=dir(InputFileName);
SizeInBytes = sss.bytes
NumberOfSymbols = SizeInBytes 

%% Stage II. Prepare some space where to store compressed versions by various methods
% create a temporary directory for working space
warning off
mkdir('./temp_dir')
delete('./temp_dir/*');
ArchiveName2 = './temp_dir/Arch2.7z';

%%  Experiment with the coder "Burrows-Wheeler Transform" (BWT = bzip2)

% (BWT was presented in Topic 6, slides 121-132)
% The bzip2 coder is implemented in the archiver 7zip, as one of the main
% coding options, called by setting the method to ' -m0=Bzip2'
A = imread('BZIP2_parameters.png'); figure(1),imagesc(A),colormap(gray) % shows the parameters of bzip2
% For further reading on Bzip2 implementation you can follow the introduction and the links at 
% https://en.wikipedia.org/wiki/Bzip2

%% Stage III: ENCODING: A first simple call of Bzip2 for compressing the file InputFileName='book1.txt', using defalut parameters
% command a = append archive

% create the command line to be passed to the operating system for
% execution: '7z a ./temp_dir/Arch1.7z book1.txt -m0=Bzip2'

tic
ComBzip2 = ['7z a ' ArchiveName2 ' ' InputFileName ' -m0=Bzip2' ];
delete(ArchiveName2) % be sure ArchiveName2 is empty
% ask the operating system to execute the command line ComBzip2
[status, cmdout] = dos(ComBzip2);
EncTimeBzip2 = toc; % the time taken by the encoding process

% get info about ArchiveName2 = './temp_dir/Arch2.7z' which is the encoded 
% file created by 7zip
sssa=dir(ArchiveName2);

% Store the compression rate 
BrateBzip2 = sssa.bytes*8/NumberOfSymbols % bits/symbol
RateBzip2 = sssa.bytes*8  % total bits

%% Q6.1.1: The initial size (in bits) of the file book1.txt is 6150168
%% Q6.1.2 The size of the bzip2 compreesed file, in bits is: 1861736
%% Q6.1.3 The compression ratio achieved by bzip2 for this file is 2.22170 bits per symbol
%% Q6.1.4 For this file, the BurroesWheleer bzip2 compressor is worse than the gzip compressor (implemented in matlab, see Quiz 5)

%% Stage IV.  DECODING

% Repeat the encoding with default parameters of the file book1.txt
InputFileName = 'book1.txt';
delete(ArchiveName2)
ComBzip2 = ['7z a ' ArchiveName2 ' ' InputFileName ' -m0=Bzip2' ];
[status, cmdout] = dos(ComBzip2);

%% Now perform decoding (extract archive)

% Prepare a directory where to decode the file from the encoded file
mkdir('./temp_dir/decoded/');
delete('./temp_dir/decoded/*'); % be sure that the directory is empty

% create the command line to be passed to the operating system for
% execution: ComBzip2Extr = '7z e ./temp_dir/Arch2.7z -o./temp_dir/decoded'

ComBzip2Extr = ['7z e ' ArchiveName2 ' -o./temp_dir/decoded' ]

% ask the operating system to execute the command line ComPPmd
tic
status = dos(ComBzip2Extr);
DecTimeBzip2 = toc % the time taken by the decoding process

% get info about './temp_dir/decoded/book1.txt' which is the decoded
% file created by 7zip
ssse=dir(['./temp_dir/decoded/'  InputFileName])

%% First sanity check: is the size of the decoded file equal to the size of the input file?
sss=dir(InputFileName);
% The size in bytes of the file
SizeInBytes = sss.bytes
SizesAreEqual = SizeInBytes == ssse.bytes

%% Now check that their contents are identical
fid=fopen('./book1.txt','r');
txt_str = fread(fid,'uchar');
fclose(fid);

fid=fopen('./temp_dir/decoded/book1.txt','r');
txt_str_Bzip2 = fread(fid,'uchar');
fclose(fid);

[sum( txt_str(:)~= txt_str_Bzip2(:) ) sum( txt_str(:) == txt_str_Bzip2(:) )]

%% Q.6.2.1 The reconstruction obtained by bzip2 is identical to the initial file book1.txt 
%% Q.6.2.2 A lossy text is acceptable reconstruction of a book, since we are more interested in saving bits than in reconstructing exactly the words

%% Stage V. Further options for the Bzip2 method are: x (level of compression), d (dictionary size) pass (number of passes)
% (from % https://sevenzip.osdn.jp/chm/cmdline/switches/method.htm)
%  

% x=[1 | 3 | 5 | 7 | 9 ]
% Sets level of compression
%     Level 	Dictionary 	NumPasses 	Description
%     1         100000          1       Fastest
%     3         500000          1       Fast
%     5         900000          1       Normal
%     7         900000          2       Maximum
%     9         900000          7       Ultra
    
% d={Size}[b|k|m]
% Sets the Dictionary size for BZip2. You must specify the size in bytes, kilobytes, or megabytes. The maximum value for the Dictionary size is 900000b. If you do not specify any symbol from set [b|k|m], dictionary size will be calculated as DictionarySize = 2^Size bytes.

% pass={NumPasses}
% Sets the number of passes. It can be in the range from 1 to 10. The default value is 1 for normal mode, 2 for maximum mode and 7 for ultra mode. A bigger number can give a little bit better compression ratio and a slower compression process.

% mt=[off | on | {N}]
% Sets multithread mode. If you have a multiprocessor or multicore system, you can get a speed increase with this switch. If you specify {N}, for example mt=4, 7-Zip tries to use 4 threads.

% Note that x, pass and d are connected. You can experiment various settings
delete(ArchiveName2)
ComBzip2 = ['7z a ' ArchiveName2 ' ' InputFileName ' -m0=Bzip2 -mx=1 -mpass=1 -md=9000000b' ]
[status, cmdout] = dos(ComBzip2)
sssa = dir(ArchiveName2)
BRateBzip2 = sssa.bytes*8/sss.bytes
        
% Let's make a run for getting the effect of parameters x and mt
if(1 ==0)
    InputFileName = 'book1.txt'
    sss=dir(InputFileName);
    SizeInBytes = sss.bytes
    NumberOfSymbols = SizeInBytes
else
    InputFileName = 'TEXTBOOK_SPECTRAL.pdf'
    sss=dir(InputFileName);
    SizeInBytes = sss.bytes
    NumberOfSymbols = SizeInBytes
end
ArchiveName2 = './temp_dir/Arch2.7z';
for i_compr_lev = [1 2 3 4 5 6 7 8 9]
    for imt = 1:2
        if(imt == 1)
            mt = 'on';
        else
            mt = 'off';
        end
        tic
        delete(ArchiveName2);
        ComBzip2 = ['7z a ' ArchiveName2 ' ' InputFileName ' -m0=Bzip2 -mx=' num2str(i_compr_lev) ' -mmt=' mt ];
        [status, cmdout] = dos(ComBzip2);
        sssa=dir(ArchiveName2);
        BRateBzip2( i_compr_lev,imt )  = sssa.bytes*8/sss.bytes;
        EncTimeBzip2( i_compr_lev,imt ) = toc;
        
        % Check extraction
        mkdir('./temp_dir/decoded/');
        delete('./temp_dir/decoded/*');
        tic
        ComBzip2Extr = ['7z e ' ArchiveName2 ' -o./temp_dir/decoded' ];
        status = dos(ComBzip2Extr);
        DecTimeBzip2( i_compr_lev,imt ) = toc;
        ssse=dir(['./temp_dir/decoded/'  InputFileName]);
        % check that the size of the reconstructed file is identical to the
        % size of the original
        EqualFilesBzip2( i_compr_lev,imt )  = (ssse.bytes==sss.bytes);
    end
end
BRateBzip2
EncTimeBzip2
DecTimeBzip2
EqualFilesBzip2

        
%% Q6.3.1: For some tested parameters the reconstruction was incorrect
%% Q6.3.2: The best  compression ratio is obtained with the dictionary size 900000
%% Q6.3.3: The parameters for the best encoding time are  x = 1 and mt = off  

InputFileName = 'TEXTBOOK_SPECTRAL.pdf'
sss=dir(InputFileName);
SizeInBytes = sss.bytes
NumberOfSymbols = SizeInBytes 


%% Stage V. Change the input file to be the pdf file 'TEXTBOOK_SPECTRAL.pdf' (present in the archive for this quiz) and rerun the Stage IV
%
%% Q6.4.1: The reconstruction of 'TEXTBOOK_SPECTRAL.pdf', was not correct for all tested sets of parameters 
%% Q6.4.2: The best parameters for compression ratio is x = 2 
%% Q6.4.3: The best parameters for encoding time are  x = 1 and mt = on 
%% Q6.4.4: The compression ratio achieved by Bzip2 on  the file 'TEXTBOOK_SPECTRAL.pdf' is almost the same as the best compression ratio,  6.765, achieved by PPM in Quiz5 
%% Q6.4.5: The ratio between the slowest encoding time and the fastest encoding time, for the same value of mt, is larger than 1.5, but the compression ratio doesn't change very significantly
