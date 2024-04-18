openDir ="/home/haruhiko/Documents/JNet/_var_num_beadsblur2/";
saveDir ="/home/haruhiko/Downloads/RLTV_results/"; 
list = getFileList(openDir);
Array.show(list);
for (i=0; i<list.length; i++){
    image = "-image file " + openDir+list[i];
    psf = " -psf file /home/haruhiko/Downloads/PSF_GL0.tif";
    algorithm = " -algorithm RLTV 10 0.1000";
    run("DeconvolutionLab2 Run", image + psf + algorithm);
    wait("Final Display of RLTV");
    selectImage("Final Display of RLTV");
    saveAs("Tiff", saveDir + list[i]);
    close();
}
