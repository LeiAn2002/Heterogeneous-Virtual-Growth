// Poly 1
/* Options for the meshing: */
General.Verbosity   = 0;
Mesh.MshFileVersion = 2;
Mesh.Algorithm      = 1;
Mesh.Algorithm3D    = 4;
Mesh.Binary         = 1;
Mesh.SaveAll = 1;
Mesh.OptimizeNetgen = 1;

Merge "./tmp127587-0-surf.msh";
Surface Loop (1) = {1};
Volume (1) = {1};
Mesh.CharacteristicLengthMax = 3.611111111111;
