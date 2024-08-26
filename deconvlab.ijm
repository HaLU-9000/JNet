openDir ="/home/haruhiko/Documents/JNet/_20231208_tsuji_beads_roi_stackreged_tif/";
saveDir ="/home/haruhiko/Documents/JNet/_results_for_paper/fig4/deconv/"; 
list = getFileList(openDir);
Array.show(list);
for (i=0; i<list.length; i++){
    image = "-image file " + openDir+list[i];
    psf = " -psf file /home/haruhiko/Documents/JNet/PSF_GL_beads_lowres.tif";
    algorithm = " -algorithm RLTV 10 0.1000";
    run("DeconvolutionLab2 Run", image + psf + algorithm);
    wait(10000);
    selectImage("Final Display of RLTV");
    saveAs("Tiff", saveDir + list[i]);
    close();
}
