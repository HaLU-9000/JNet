openDir ="/home/haruhiko/Documents/JNet/_results_for_paper/fig2_a/no_noise/original/";
saveDir ="/home/haruhiko/Documents/JNet/_results_for_paper/fig2_a/no_noise/deconv/"; 
list = getFileList(openDir);
Array.show(list);
for (i=0; i<list.length; i++){
    image = "-image file " + openDir+list[i];
    psf = " -psf file /home/haruhiko/Downloads/PSF_GL0.tif";
    algorithm = " -algorithm RLTV 10 0.1000";
    run("DeconvolutionLab2 Run", image + psf + algorithm);
    wait(1800000);
    selectImage("Final Display of RLTV");
    saveAs("Tiff", saveDir + list[i]);
    close();
}
//
//openDir ="/home/haruhiko/Documents/JNet/_var_num_realisticblur6_without_noise/";
//saveDir ="/home/haruhiko/Documents/JNet/_results_for_paper/fig2_o/no_noise/deconv/"; 
//list = getFileList(openDir);
//Array.show(list);
//for (i=0; i<list.length; i++){
//    image = "-image file " + openDir+list[i];
//    psf = " -psf file /home/haruhiko/Downloads/PSF_GL0.tif";
//    algorithm = " -algorithm RLTV 10 0.1000";
//    run("DeconvolutionLab2 Run", image + psf + algorithm);
//    wait(1200000);
//    selectImage("Final Display of RLTV");
//    saveAs("Tiff", saveDir + list[i]);
//    close();
//}

