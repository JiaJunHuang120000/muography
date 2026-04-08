// SPDX-License-Identifier: LGPL-3.0-or-later

#include "DD4hep/DetFactoryHelper.h"
#include <XML/Helper.h>

using namespace dd4hep;

static Ref_t createDetector(Detector& desc, xml_h handle, SensitiveDetector /*sens*/) {
  xml_det_t detElem = handle;
  std::string detName = detElem.nameStr();
  int detID = detElem.id();

  // =========================
  // Main detector dimensions
  // =========================
  xml_dim_t dim = detElem.dimensions();
  double dx = dim.x();
  double dy = dim.y();
  double dz = dim.z();

  // =========================
  // Position & rotation
  // =========================
  xml_dim_t dim_pos = detElem.position();
  xml_dim_t dim_rot = detElem.rotation();

  // =========================
  // Material
  // =========================
  std::string matName = "Rock";
  if (detElem.hasAttr(_Unicode(material))) {
    matName = detElem.attr<std::string>(_Unicode(material));
  }
  Material rock = desc.material(matName);

  // =========================
  // Base rock volume
  // =========================
  Box rockBox(dx / 2., dy / 2., dz / 2.);
  Solid finalSolid = rockBox;

  // =========================
  // Subtract cubes
  // =========================
  for (xml_coll_t c(detElem, _Unicode(cube)); c; ++c) {
    xml_comp_t x_cube = c;

    xml_dim_t cube_dim = x_cube.dimensions();
    xml_dim_t cube_pos = x_cube.position();
    xml_dim_t cube_rot = x_cube.rotation();

    Box cube(cube_dim.x() / 2., cube_dim.y() / 2., cube_dim.z() / 2.);

    Transform3D cube_tr(
      RotationZYX(cube_rot.z(), cube_rot.y(), cube_rot.x()),
      Position(
        cube_pos.x() - dim_pos.x(),
        cube_pos.y() - dim_pos.y(),
        cube_pos.z() - dim_pos.z()
      )
    );

    finalSolid = SubtractionSolid(finalSolid, cube, cube_tr);

    std::cout << "Cube position: ("
              << cube_pos.x() << ", "
              << cube_pos.y() << ", "
              << cube_pos.z() << ")"
              << std::endl;
  }

  // =========================
  // Subtract spheres
  // =========================
  for (xml_coll_t s(detElem, _Unicode(sphere)); s; ++s) {
    xml_comp_t x_sphere = s;

    xml_dim_t sphere_dim = x_sphere.dimensions();
    xml_dim_t sphere_pos = x_sphere.position();
    xml_dim_t sphere_rot = x_sphere.rotation();

    double radius = sphere_dim.r();
    Sphere sphere(0.0, radius, 0.0, M_PI, 0.0, 2.0 * M_PI);

    Transform3D sphere_tr(
      RotationZYX(sphere_rot.z(), sphere_rot.y(), sphere_rot.x()),
      Position(
        sphere_pos.x() - dim_pos.x(),
        sphere_pos.y() - dim_pos.y(),
        sphere_pos.z() - dim_pos.z()
      )
    );

    finalSolid = SubtractionSolid(finalSolid, sphere, sphere_tr);

    std::cout << "Sphere position: ("
              << sphere_pos.x() << ", "
              << sphere_pos.y() << ", "
              << sphere_pos.z() << ")"
              << std::endl;
  }

  // =========================
  // Create volume
  // =========================
  Volume rockVol(detName, finalSolid, rock);
  rockVol.setVisAttributes(desc.visAttributes("AnlGray"));

  // =========================
  // Placement in world
  // =========================
  DetElement det(detName, detID);
  Volume motherVol = desc.pickMotherVolume(det);

  Transform3D tr(
    RotationZYX(dim_rot.z(), dim_rot.y(), dim_rot.x()),
    Position(dim_pos.x(), dim_pos.y(), dim_pos.z())
  );

  PlacedVolume phv = motherVol.placeVolume(rockVol, tr);
  phv.addPhysVolID("system", detID);
  det.setPlacement(phv);

  return det;
}

DECLARE_DETELEMENT(RockWithCubeCutout, createDetector)