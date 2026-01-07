// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2023

//==========================================================================
//  Simple sphere detector example
//--------------------------------------------------------------------------
//  Author: adapted for demo
//==========================================================================

#include "DD4hep/DetFactoryHelper.h"
#include <XML/Helper.h>

using namespace dd4hep;

static Ref_t createDetector(Detector& desc, xml_h handle, SensitiveDetector /*sens*/) {
  xml_det_t detElem   = handle;
  std::string detName = detElem.nameStr();
  int detID           = detElem.id();

  // Read dimensions from XML
  xml_dim_t dim = detElem.dimensions();
  double radius = dim.r();  // Sphere radius
  xml_dim_t pos = detElem.position(); // Center position
  xml_dim_t rot = detElem.rotation(); // Rotation if needed

  // Default material
  Material air = desc.material("CarbonFiber_25percent");

  // Define a simple sphere
  Sphere sphere(0.0, radius, 0.0, M_PI, 0.0, 2.0 * M_PI);

  // Create volume for sphere
  Volume sphereVol(detName, sphere, air);

  // Apply attributes from XML
  sphereVol.setAttributes(desc, detElem.regionStr(), detElem.limitsStr(), detElem.visStr());

  // Create detector element
  DetElement det(detName, detID);

  // Place inside world
  Volume motherVol = desc.pickMotherVolume(det);
  auto tr = Transform3D(RotationZYX(rot.z(), rot.y(), rot.x()),
                        Position(pos.x(), pos.y(), pos.z()));

  PlacedVolume phv = motherVol.placeVolume(sphereVol, tr);
  phv.addPhysVolID("system", detID);
  det.setPlacement(phv);

  return det;
}

// Register detector
DECLARE_DETELEMENT(SimpleSphere, createDetector)
