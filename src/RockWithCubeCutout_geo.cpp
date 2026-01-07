#include "DD4hep/DetFactoryHelper.h"
#include <XML/Helper.h>

using namespace dd4hep;

static Ref_t createDetector(Detector& desc, xml_h handle, SensitiveDetector /*sens*/) {
  xml_det_t detElem = handle;
  std::string detName = detElem.nameStr();
  int detID = detElem.id();

  // Dimensions of the rock block
  xml_dim_t dim = detElem.dimensions();
  double dx = dim.x();
  double dy = dim.y();
  double dz = dim.z();

  // Placement position
  xml_dim_t dim_pos = detElem.position();
    
    std::string matName = "Rock";  // default
    if (detElem.hasAttr(_Unicode(material))) {
      matName = detElem.attr<std::string>(_Unicode(material));
    }
    
    Material rock = desc.material(matName);
    
  // Material rock = desc.material("Rock");

  // Base rock block
  Box rockBox(dx/2., dy/2., dz/2.);
  Solid finalSolid = rockBox;

  // Subtract all cubes defined in XML
  for (xml_coll_t c(detElem, _Unicode(cube)); c; ++c) {
    xml_comp_t x_cube = c;
    xml_dim_t cube_dim = x_cube.dimensions();
    xml_dim_t cube_pos = x_cube.position();

    Box cube(cube_dim.x()/2., cube_dim.y()/2., cube_dim.z()/2.);
    finalSolid = SubtractionSolid(finalSolid, cube,
                  Position(cube_pos.x()-dim_pos.x(), cube_pos.y()-dim_pos.y(), cube_pos.z()-dim_pos.z()));
    std::cout << "Cube position: ("
              << cube_pos.x() << ", "
              << cube_pos.y() << ", "
              << cube_pos.z() << ")"
              << std::endl;
  }
    
  for (xml_coll_t s(detElem, _Unicode(sphere)); s; ++s) {
    xml_comp_t x_sphere = s;
    xml_dim_t sphere_dim = x_sphere.dimensions();
    xml_dim_t sphere_pos = x_sphere.position();
    double radius = sphere_dim.r();
    Sphere sphere(0.0, radius, 0.0, M_PI, 0.0, 2.0 * M_PI);
    finalSolid = SubtractionSolid(finalSolid, sphere,
                  Position(sphere_pos.x()-dim_pos.x(), sphere_pos.y()-dim_pos.y(), sphere_pos.z()-dim_pos.z()));
    std::cout << "Sphere position: ("
              << sphere_pos.x() << ", "
              << sphere_pos.y() << ", "
              << sphere_pos.z() << ")"
              << std::endl;
  }

  // Rock volume with holes
  Volume rockVol(detName, finalSolid, rock);
  rockVol.setVisAttributes(desc.visAttributes("AnlGray"));



  // Place rock volume in the mother volume
  DetElement det(detName, detID);
  Volume motherVol = desc.pickMotherVolume(det);

  PlacedVolume phv = motherVol.placeVolume(
    rockVol,
    Position(dim_pos.x(), dim_pos.y(), dim_pos.z())
  );
  phv.addPhysVolID("system", detID);
  det.setPlacement(phv);

  return det;
}

DECLARE_DETELEMENT(RockWithCubeCutout, createDetector)
