

gray2 = imread('Image_1.png');
figure(1),imagesc(gray2)

     %% calic
    result_calic = calic(gray2);
    res_MID(j,3) = result_calic(1);
    res_MID(j,7) = result_calic(5);
    
    %% loco
    result_loco = loco(gray2);
    res_MID(j,4) = result_loco(1);
    res_MID(j,8) = result_loco(5);