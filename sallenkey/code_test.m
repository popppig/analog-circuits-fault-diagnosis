clc
clear
for i=1:10
    label=(1:1:10)'
    [Yc, codebook, old_codebook] = code(label, 'code_MOC');
    Yc
    Yd = code(Yc, old_codebook,[], codebook);
    Yd
end