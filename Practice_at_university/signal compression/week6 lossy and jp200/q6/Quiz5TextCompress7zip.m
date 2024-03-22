% function  Quiz5TextCompress7zip.m

%% Save the book1.txt and TEXTBOOK_SPECTRAL.pdf from the FilesForQuiz5.zip to the working directory

%% 0 Check that you have the program 7zip installed in your computer
clear all;
clc;
% The simplest check is to run
dos('7z')
% the output of the above command prints to screen about 60 lines of usage
% instructions
% A better help on the parameters of 7zip can be found e.g., at
% https://sevenzip.osdn.jp/chm/cmdline/switches/method.htm
% On the Windows computers you might have the program at this path:
dos('D:\GRAM\MasterProgramme\Tampere\signal compression\week6 lossy and jp200\q6')
crt_dir = pwd
copyfile('D:\GRAM\MasterProgramme\Tampere\signal compression\week6 lossy and jp200\q6\7z.exe',crt_dir)
% Copy the 7z.exe on the current directory for running the examples
% If you don't have the program installed, you could download it, as
% explained at https://www.7-zip.org/


%% Stage I. The input text to be compressed is in the file 'book1.txt' that is located in the
%% archived folder for this quiz
% You should place the file book1.txt in the current folder where you run 
% matlab for this task
% We read the info about the file from disk (but don't read the text to
% matlab, since matlab is just an interface for running system comands)

InputFileName = 'book1.txt'
sss=dir(InputFileName);
% The size in bytes of the file
SizeInBytes = sss.bytes
% The file has one alphanumeric symbol (letter, or digit, or 
% punctuation signs) in each byte, hence the size of the text is
NumberOfSymbols = SizeInBytes 

%% Stage II. Prepare some space where to store compressed versions by various methods
% create a temporary directory for working space
warning off
mkdir('./temp_dir')
ArchiveName1 = './temp_dir/Arch1.7z';
% make sure that the archives ar empty (otherwise the program 7z will 
% concatenate the newly compressed files to the existent content of the
% archives)
delete(ArchiveName1)


%%  Experiment with the context coder "Prediction by partial matching" (PPM)

% (presented in Topic 5, slides 116-119)
% The PPM coder is implemented in the archiver 7zip, as one of the main
% coding options, called by setting the method to ' -m0=PPmd' (this
% will use PPM with the ESC method D, (see the lecture slides)
% For further reading on PPM you can follow the introduction and links at 
% https://en.wikipedia.org/wiki/Prediction_by_partial_matching

%% Stage III: ENCODING: A first simple call of PPM for compressing the file InputFileName='book1.txt', using defalut parameters
% command a = append archive

% create the command line to be passed to the operating system for
% execution: '7z a ./temp_dir/Arch1.7z book1.txt -m0=PPmd'
ComPPmd = ['7z a ' ArchiveName1 ' ' InputFileName ' -m0=PPmd' ];

% ask the operating system to execute the command line ComPPmd
tic
[status, cmdout] = dos(ComPPmd);
EncTimePPmd = toc % the time taken by the encoding process

% get info about ArchiveName1 = './temp_dir/Arch1.7z' which is the encoded 
% file created by 7zip
sssa=dir(ArchiveName1);

% Store the compressed bitrate 
BratePPmd = sssa.bytes*8/NumberOfSymbols % bits/symbol
RatePPmd = sssa.bytes*8; % total bits

%% Q5.1.1: The initial size of the file in bits is 6150168
%% Q5.1.2 The size of the compreesed file, in bytes is: 209951
%% Q5.1.3 The compressed bitrate achieved is 2.28 bits per symbol


%% Stage IV.  DECODING

% Repeat the encoding with default parameters of the file book1.txt
InputFileName = 'book1.txt';
delete(ArchiveName1)
ComPPmd = ['7z a ' ArchiveName1 ' ' InputFileName ' -m0=PPmd' ];
[status, cmdout] = dos(ComPPmd);

%% Now perform decoding (extract archive)

% Prepare a directory where to decode the file from the encoded file
mkdir('./temp_dir/decoded/');
delete('./temp_dir/decoded/*'); % be sure that the directory is empty

% create the command line to be passed to the operating system for
% execution: ComPPmdExtr = '7z e ./temp_dir/Arch1.7z -o./temp_dir/decoded'

ComPPmdExtr = ['7z e ' ArchiveName1 ' -o./temp_dir/decoded' ]

% ask the operating system to execute the command line ComPPmd
tic
status = dos(ComPPmdExtr);
DecTimePPmd = toc % the time taken by the decoding process

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
txt_str_PMD = fread(fid,'uchar');
fclose(fid);

[sum( txt_str(:)~= txt_str_PMD(:) ) sum( txt_str(:) == txt_str_PMD(:) )]

%% Q.5.2.1 The size of the initial file and the size of the decoded file are the same
%% Q5.2.2 There are 0 decoding errors

%% Stage V. Further options for the PPM method are: memory size and model order
% (from % https://sevenzip.osdn.jp/chm/cmdline/switches/method.htm)
%  
% mem={Size}[b|k|m|g]
%     Sets the size of memory used for PPMd. You must specify the size in bytes, 
%       kilobytes, or megabytes. The maximum value is 2GB = 2^31 bytes. 
%       The default value is 24 (16MB). If you do not specify any symbol from 
%       the set [b|k|m|g], the memory size will be calculated as (2^Size) bytes. 
%       PPMd uses the same amount of memory for compression and decompression.
%     if [b|k|m|g] are not specified, the memory is taken as 2^Size in bytes
%     default value is Size = 24 (16Mbytes)
%
% o={Size}
%  Sets the model order for PPMd. The size must be in the range [2,32]. The default value is 6.

InputFileName = 'book1.txt';
% InputFileName = 'TEXTBOOK_SPECTRAL.pdf';
sss=dir(InputFileName);
SizeInBytes = sss.bytes;
NumberOfSymbols = SizeInBytes;
clear BratePPmd EncTimePPmd DecTimePPmd EqualFiles
for mem1 = 16:28
    for order = 2:10
        tic
        delete(ArchiveName1)
        ComPPmd = ['7z a ' ArchiveName1 ' ' InputFileName ' -m0=PPmd' ' -mmem=' num2str(mem1)  ' -mo=' num2str(order)];
        [status, cmdout] = dos(ComPPmd);
        sssa=dir(ArchiveName1);
        % sssa.bytes*8/sss.bytes
        
        BratePPmd(mem1,order) = sssa.bytes*8/NumberOfSymbols; % bits/char
        RatePPmd = sssa.bytes*8; % total bits
        EncTimePPmd(mem1,order) = toc;
        
        % Check extraction
        mkdir('./temp_dir/decoded/');
        delete('./temp_dir/decoded/*');
        tic
        ComPPmdExtr = ['7z e ' ArchiveName1 ' -o./temp_dir/decoded' ];
        status = dos(ComPPmdExtr);
        DecTimePPmd(mem1,order) = toc;
        ssse=dir(['./temp_dir/decoded/'  InputFileName]);
        % check that the size of the reconstructed file is identical to the
        % size of the original
        EqualFiles(mem1,order) = ssse.bytes==sss.bytes;
    end
end
BratePPmd
EqualFiles
EncTimePPmd
DecTimePPmd

format long
minPPmd = min(min(BratePPmd(16:28,2:10)))
[i_mem_min,i_ord_min] = find(BratePPmd(16:28,2:10)==minPPmd)
BratePPmd(16:28,2:10)==minPPmd

%% Q5.3.1: For none of the tested parameters the reconstruction was correct
%% Q5.3.2: The best parameters are  mem >= 2^23 byes and ord = 5
%% Q5.3.3: The default parameters used in ComPPmd = ['7z a ' ArchiveName1 ' ' InputFileName ' -m0=PPmd' ]
%% achieve the best compression for the text file book1.txt


%% Stage V. Change the input file to be the pdf file 'TEXTBOOK_SPECTRAL.pdf' (present in the archive for this quiz)
%
%% Q5.4.1: The reconstruction of For 'TEXTBOOK_SPECTRAL.pdf', was not correct for some of the tested sets of parameters (for mem1 = 16:28   for order = 2:10)
%% Q5.4.2: The best parameters from all tested above are mem >= 2^25 byes and ord = 10
%% Q5.4.3: The default parameters used in ComPPmd = ['7z a ' ArchiveName1 ' ' InputFileName ' -m0=PPmd' ]
%% achieve the best compression for the file 'TEXTBOOK_SPECTRAL.pdf'
%% Q5.4.4: The best compressed bitrate is about 6.7 bits/sample
%% Q5.4.5 The file 'TEXTBOOK_SPECTRAL.pdf' is less compressible than book1.txt, one reason being that it has more 
%% mathematical equations

