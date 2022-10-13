from mp_api import *
from mp_api.client import MPRester

from pymatgen.io.cif import CifWriter

import csv
import random

#Contact materials project with api key and download all material
#structures and material ids of crytalline structures (those with
#some defined crystal system)

#We also choose several target for regression:band_gap, formation_energy_per_atom, energy_above_hull;
#and for classification: is_metal
#fields = ['band_gap','formation_energy_per_atom', 'energy_above_hull','is_metal']

                #Right now we search for any material with one of the seven crystal systems,
                #though oxides may be an interesting subset in the future 

fields = ['material_id', 'structure']

print("Contacting materials project database...")
with MPRester("M3npHaL9jodYuqvW6q9faTs4koRBRB9G") as mpr:
    print(f"Success, downloading crystalline structure fields: {fields}")
    crystals = mpr.materials.search(crystal_system=["Triclinic", "Monoclinic", "Orthorhombic", "Tetragonal", "Trigonal", "Hexagonal", "Cubic"], fields=fields)  
    
print(f"Found {len(crystals)} crystalline structures")


#Print .cif files of all downloaded structures via CifWriter and
#associated write file property

direc = 'crystal_data'

for crys in crystals:
    crys_cif=CifWriter(crys.structure, significant_figures=4)
    crys_cif.write_file(f'./{direc}/{crys.material_id}.cif')
    print(f'Generated cif file for {crys.material_id}')
print("\n\nCifs stored in {direc}")

#Create list of material_ids for further property searches

        #Apparently the materials project partitions different
        #types of material properties into different databases.
        #Hence, we may need a different call for every database


material_ids = []
for crys in crystals:
    material_ids.append(crys.material_id)
print(material_ids)

#search thermo for energy_above_hull and formation_energy_per_atom from list of material_ids (done iteratively due to generally large list)

material_ids_thermo = [] #This is just to confirm we are matching the material_ids
formation_energy_per_atom = []
energy_above_hull = []

with MPRester("M3npHaL9jodYuqvW6q9faTs4koRBRB9G") as mpr:
    for i in material_ids:
        crystals = mpr.thermo.search(material_ids = [i], fields=['formation_energy_per_atom', 'material_id', 'energy_above_hull']);
        material_ids_thermo.append(crystals[0].material_id)
        formation_energy_per_atom.append(crystals[0].formation_energy_per_atom)
        energy_above_hull.append(crystals[0].energy_above_hull)


#This should always be true but saves headaches later if not for some reason

assert len(material_ids_thermo) == len(energy_above_hull)
assert len(material_ids_thermo) == len(formation_energy_per_atom)


#Print material id and property into id_prop_field.csv for each field
# (for now, as based on cgcnn model)

prop_list = formation_energy_per_atom


with open(f"./crystal_data/id_prop.csv", mode="w", newline="") as id_prop:
    id_prop = csv.writer(id_prop, delimiter=",")
    for material_id,prop in zip(material_ids_thermo,prop_list):
        id_prop.writerow([material_id,prop])

print(f'Generated "./crystal_data/id_prop.csv"')
print('Done with id_props for formation_energy_per_atom')
