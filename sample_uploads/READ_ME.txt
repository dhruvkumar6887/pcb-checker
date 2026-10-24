PCB CHECKER SAMPLE UPLOADS

These are synthetic demonstration images, not actual phone photographs.

Open http://127.0.0.1:8000/ and upload:
  Correct PCB:       01_correct_pcb.png
  Phone PCB:         02_phone_missing_parts.png
  KiCad schematic:   03_reference.kicad_sch
Then click Compare images.

Ground truth: R2 (10k resistor) and C1 (10uF capacitor) were erased.
Simple comparison highlights changes with V labels; it does not infer part names.
Use 04_phone_correct_board.png instead to try a board with no parts removed.

All four ArUco tags are already included. Keep the default holder dimensions:
86.25 mm horizontal / 61.25 mm vertical between marker centers, 10 mm tags.

Optional advanced files:
  bom.csv                 Part identities and values
  schematic.net           XML netlist reference
  placement.csv           Approximate placement proposals, not final body boxes
  components.json         Exact ground-truth component boxes for this fixture
  marker_distances.json   Ground-truth center distances to marker IDs 0,1,2,3

The KiCad schematic is a self-contained component reference sheet with named
nets. It is not a build-ready electrical circuit or PCB manufacturing layout.
Schematic symbol positions do not describe physical PCB or tag locations.
