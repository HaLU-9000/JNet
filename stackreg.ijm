openDir ="/home/haruhiko/Documents/JNet/_wakelabdata/microglia_tumor/";
saveDir ="/home/haruhiko/Documents/JNet/_wakelabdata/microglia_tumor_separated/"; 
list = getFileList(openDir);
Array.show(list);
for (i=0; i<list.length; i++){
    run("Bio-Formats Windowless Importer", "open="+openDir+list[i]);
    run("Split Channels");
    selectImage("C2-"+list[i]);
    //run("StackReg ", "transformation=[Rigid Body]");
    name = getTitle();
	dotIndex = lastIndexOf(name,".");
	title = substring(name,3,dotIndex);
    newname = title + "C2.tif";
    saveAs("Tiff", saveDir + newname);
    close();
    selectImage("C1-"+list[i]);
    name = getTitle();
    dotIndex = lastIndexOf(name,".");
    title = substring(name,3,dotIndex);
    newname = title + "C1.tif";
    saveAs("Tiff", saveDir + newname);
    close();
}
