openDir ="/home/haruhiko/Downloads/Set_03/";
saveDir ="/home/haruhiko/Downloads/Set_03_stackreg_z/"; 
list = getFileList(openDir);
Array.show(list);
for (i=0; i<list.length; i++){
    run("Bio-Formats Windowless Importer", "open="+openDir+list[i]);
    run("Split Channels");
    selectImage("C1-"+list[i]);
    run("StackReg ", "transformation=[Rigid Body]");
    name = getTitle();
	dotIndex = lastIndexOf(name,".");
	title = substring(name,3,dotIndex);
    newname = title + ".tif";
    saveAs("Tiff", saveDir + newname);
    close();
    selectImage("C2-"+list[i]);
    close();
}