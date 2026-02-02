import subprocess
from pathlib import Path

# Load TARGETS from config.sh
cmd = ["bash", "-c", "source $DETECTOR_PATH/bash/config.sh && printf '%s\n' \"${TARGETS[@]}\""]
lines = subprocess.check_output(cmd, text=True).splitlines()

xml_blocks = []
xml_blocks1 = []

template = Path("soil_target.template.xml").read_text()

# Inject geometry
final_xml = template.replace("{{TARGETS}}", "".join(xml_blocks))
final_xml = final_xml.replace("{{TARGETS1}}", "".join(xml_blocks1))


# Write final XML
Path("soil_free.xml").write_text(final_xml)
print("✅ soil_free.xml generated successfully")

for line in lines:
    parts = line.split()
    shape = parts[0]

    params = dict(p.split("=") for p in parts[1:])

    if shape == "sphere":
        block = f"""
  <sphere>
    <dimensions r="{params['r']}" />
    <position x="{params['x']}" y="{params['y']}" z="{params['z']}" />
  </sphere>
"""
    elif shape == "cube":
        block = f"""
  <cube>
    <dimensions x="{params['xdim']}" y="{params['ydim']}" z="{params['zdim']}" />
    <position x="{params['x']}" y="{params['y']}" z="{params['z']}" />
  </cube>
"""
    else:
        raise ValueError(f"Unknown shape: {shape}")

    xml_blocks.append(block)
    

i = 1
for line in lines:
    
    parts = line.split()
    shape = parts[0]
    material = parts[-1]

    params = dict(p.split("=") for p in parts[1:])
    if shape == "sphere" and params['material'] not in  ["Air", "Vacuum"]:
        block = f"""
<detector id="{2000+i}" name="SimpleSphere{i}" type="SimpleSphere" vis="AnlGray" material="{params['material']}">
  <dimensions r="{params['r']}" />
  <rotation x="0" y="0" z="0" />
  <position x="{params['x']}" y="{params['y']}" z="{params['z']}" />
</detector> 
"""
        i += 1
        xml_blocks1.append(block)
        
    elif shape == "cube" and params['material'] not in  ["Air", "Vacuum"]:
        block = f"""
<detector id="{2000+i}" name="ExtraCube{i}" type="RockWithCubeCutout" vis="AnlGray" material="{params['material']}">
    <dimensions x="{params['xdim']}" y="{params['ydim']}" z="{params['zdim']}" />
    <position x="{params['x']}" y="{params['y']}" z="{params['z']}" />
</detector> 
"""
        i += 1

        xml_blocks1.append(block)

# Load template
template = Path("soil_target.template.xml").read_text()

# Inject geometry
final_xml = template.replace("{{TARGETS}}", "".join(xml_blocks))
final_xml = final_xml.replace("{{TARGETS1}}", "".join(xml_blocks1))


# Write final XML
Path("soil_target.xml").write_text(final_xml)

print("✅ soil_target.xml generated successfully")
