
open("/home/haruhiko/Documents/JNet/_apply_JNet_609/MD495_1G2_D14_FINC1-T04x512_768_y512_768.tif");
selectImage("MD495_1G2_D14_FINC1-T04x512_768_y512_768.tif");
run("3D Manager Options", "volume surface compactness fit_ellipse integrated_density mean_grey_value std_dev_grey_value minimum_grey_value maximum_grey_value centroid_(pix) centroid_(unit) distance_to_surface centre_of_mass_(pix) centre_of_mass_(unit) bounding_box radial_distance closest distance_between_centers=10 distance_max_contact=1.80 drawing=Contour display");
run("3D Manager");
selectImage("MD495_1G2_D14_FINC1-T04x512_768_y512_768.tif-3Dseg");
Ext.Manager3D_Segment(128, 255);
Ext.Manager3D_AddImage();
Ext.Manager3D_Save("/home/haruhiko/Documents/JNet/_apply_JNet_609/Roi3D.zip");
