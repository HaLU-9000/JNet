// Set the folder containing the images
inputFolder = getDirectory("Choose a Folder with Images");

// Set the output folder for the ZIP files
outputFolder = inputFolder; // Save in the same folder as the images

// Get a list of all .tif files in the folder
fileList = getFileList(inputFolder);
for (i = 0; i < fileList.length; i++) {
    if (endsWith(fileList[i], ".tif")) {
        // Open the image
        open(inputFolder + fileList[i]);

        // Run 3D Manager
        run("3D Manager");
        Ext.Manager3D_Segment(127, 255);

        // Add the image to 3D Manager
        Ext.Manager3D_AddImage();

        // Measure and save
        outputFileName = replace(fileList[i], ".tif", ".csv");
        Ext.Manager3D_Measure();
        saveAs("Results",outputFolder + outputFileName);
        
        // Save the 3D segmentation result as a ZIP file
        outputFileName = replace(fileList[i], ".tif", ".zip");
        Ext.Manager3D_Save(outputFolder + outputFileName);

        // Close the image and 3D Manager for the next iteration
        close();
        Ext.Manager3D_Reset();
    }
}
