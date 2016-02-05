# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 11:19:36 2015

@author: erlean
"""
__tissues = {1: 'hardbone',
               2: 'skin',
               3: 'muscle',
               4: 'soft',
               5: 'redbonemarrow',
               6: 'yellowbonemarrow',
               7: 'adipose',
               8: 'lung',
               9: 'air',
               10: 'water',
               }

__jo_organs = """1;Adrenal, left;4
                2;Adrenal, right;4
                3;Anterior nasal passage (ET1);9
                4;Posterior nasal passage down to larynx (ET2);4
                5;Oral mucosa, tongue;3
                6;Oral mucosa, lips and cheeks;7
                7;Trachea;4
                8;Bronchi;8
                9;Blood vessels, head;5
                10;Blood vessels, trunk;5
                11;Blood vessels, arms;5
                12;Blood vessels, legs;5
                13;Humeri, proximal end, cortical bone;1
                14;Humeri, upper half, spongiosa;1
                15;Humeri, upper half, medullary cavity;1
                16;Humeri, lower half, cortical;1
                17;Humeri, lower half, spongiosa;1
                18;Humeri, lower half, medullary cavity;1
                19;Ulnae and radii, cortical;1
                20;Ulnae and radii, spongiosa;1
                21;Ulnae and radii, medullary cavity;1
                22;Wrists and hand bones, cortical;1
                23;Wrists and hand bones, spongiosa;1
                24;Clavicles, cortical;1
                25;Clavicles, spongiosa;1
                26;Cranium, cortical;1
                27;Cranium, spongiosa;1
                28;Femora, upper half, cortical;1
                29;Femora, upper half, spongiosa;1
                30;Femora, upper half, medullary cavity;1
                31;Femora, lower half, cortical;1
                32;Femora, lower half, spongiosa;1
                33;Femora, lower half, medullary cavity;1
                34;Tibiae, fibulae and patellae:;1
                35;Tibiae, fibulae and patellae, spongiosa;1
                36;Tibiae, fibulae and patellae, medullary cavity;1
                37;Ankles and foot bones:;1
                38;Ankles and foot bones, spongiosa;1
                39;Mandible, cortical;1
                40;Mandible, spongiosa;1
                41;Pelvis, cortical;1
                42;Pelvis, spongiosa;1
                43;Ribs, cortical;1
                44;Ribs, spongiosa;1
                45;Scapulae, cortical;1
                46;Scapulae, spongiosa;1
                47;Cervical spine, cortical;1
                48;Cervical spine, spongiosa;1
                49;Thoracic spine, cortical;1
                50;Thoracic spine, spongiosa;1
                51;Lumbar spine, cortical;1
                52;Lumbar spine, spongiosa;1
                53;Sacrum, cortical;1
                54;Sacrum, spongiosa;1
                55;Sternum, cortical;1
                56;Sternum, spongiosa;1
                57;Cartilage, head;1
                58;Cartilage, trunk;1
                59;Cartilage, arms;1
                60;Cartilage, legs;1
                61;Brain;7
                62;Breast, left, adipose tissue;7
                63;Breast, left, glandular tissue;4
                64;Breast, right, adipose tissue;7
                65;Breast, right, glandular tissue;4
                66;Eye lens, left;4
                67;Eye bulb, left;4
                68;Eye lens, right;4
                69;Eye bulb, right;4
                70;Gall bladder wall;4
                71;Gall bladder contents;10
                72;Stomach wall;4
                73;Stomach contents;4
                74;Small intestine wall;4
                75;Small intestine contents;4
                76;Ascending colon wall;4
                77;Ascending colon contents;4
                78;Transverse colon wall, right;4
                79;Transverse colon contents, right;4
                80;Transverse colon wall, left;4
                81;Transverse colon contents, left;4
                82;Descending colon wall;4
                83;Descending colon contents;4
                84;Sigmoid colon wall;4
                85;Sigmoid colon contents;4
                86;Rectum wall;4
                87;Heart wall;3
                88;Heart contents (blood);5
                89;Kidney, left, cortex;4
                90;Kidney, left, medulla;4
                91;Kidney, left, pelvis;4
                92;Kidney, right, cortex;4
                93;Kidney, right, medulla;4
                94;Kidney, right, pelvis;4
                95;Liver;4
                96;Lung, left, blood;5
                97;Lung, left, tissue;8
                98;Lung, right, blood;5
                99;Lung, right, tissue;8
                100;Lymphatic nodes, extrathoracic airways;4
                101;Lymphatic nodes, thoracic airways;4
                102;Lymphatic nodes, head;4
                103;Lymphatic nodes, trunk;4
                104;Lymphatic nodes, arms;4
                105;Lymphatic nodes, legs;4
                106;Muscle, head;3
                107;Muscle, trunk;3
                108;Muscle, arms;3
                109;Muscle, legs;3
                110;Oesophagus (wall);3
                111;Ovary, left;4
                112;Ovary, right;4
                113;Pancreas;4
                114;Pituitary gland;4
                115;Prostate;4
                116;Residual tissue, head;7
                117;Residual tissue, trunk;7
                118;Residual tissue, arms;7
                119;Residual tissue, legs;7
                120;Salivary glands, left;4
                121;Salivary glands, right;4
                122;Skin, head;2
                123;Skin, trunk;2
                124;Skin, arms;2
                125;Skin, legs;2
                126;Spinal cord;6
                127;Spleen;4
                128;Teeth;1
                129;Testis, left;4
                130;Testis, right;4
                131;Thymus;4
                132;Thyroid;4
                133;Tongue (inner part);3
                134;Tonsils;4
                135;Ureter, left;4
                136;Ureter, right;4
                137;Urinary bladder wall;4
                138;Urinary bladder contents;4
                139;Uterus / Cervix;4
                140;Air inside body;9
                141;Skin at top and bottom;2"""


def jo_organs():
    lg = __jo_organs.replace('\n', ';').replace('\r', '').split(';')
    while len(lg) > 0:
        org = lg.pop(0)
        desc = lg.pop(0)
        tiss = lg.pop(0)
        yield int(org), desc.strip(), __tissues[int(tiss)]


__baby_organs = """0;Air;9
                    1;Adrenals;4
                    2;Bladder wall;4
                    3;Brain;7
                    4;Eyes;4
                    5;Eye lenses;4
                    6;Heart;3
                    8;Small intestine wall;4
                    10;Left kidney;4
                    11;Right kidney;4
                    12;Liver;4
                    14;Left lung;8
                    15;Right lung;8
                    16;Ovaries;4
                    17;Pancreas;4
                    19;Head skin;2
                    20;Trunk skin;2
                    21;Leg skin;2
                    22;Spinal cord;4
                    23;Spleen;4
                    24;Stomach wall;4
                    25;Testes;4
                    26;Thymus;4
                    27;Thyroid;4
                    29;Head tissue;4
                    30;Trunk tissue;4
                    31;Leg tissue;4
                    32;Uterus;4
                    34;Upper left arm bone;1
                    35;Lower left arm bones;1
                    36;Left hand bones;1
                    37;Upper right arm bone;1
                    38;Lower right arm bones;1
                    39;Right hand bones;1
                    40;Clavicles;1
                    41;Facial skeleton;1
                    43;Upper left leg bone;1
                    44;Lower left leg bones;1
                    45;Left foot bones;1
                    46;Upper right leg bone;1
                    47;Lower right leg bones;1
                    48;Right foot bones;1
                    49;Pelvis;1
                    50;Ribs;1
                    51;Scapulae;1
                    52;Skull;1
                    54;Cervical spine;1
                    55;Thoracic spine;1
                    56;Lumbar spine;1
                    57;Sternum;1
                    93;Stomach contents;4
                    94;Small intestine contents;4
                    95;UknownERROR;4
                    96;Bladder contents;10
                    97;UknownERROR;4
                    103;Upper large intestine (ascending+transverse colon) wall;4
                    104;Lower large intestine (descending+sigmoid colon) wall;4
                    105;Upper large intestine contents;4
                    106;Lower large intestine contents;4
                    107;Arm skin;2
                    108;Arm tissue;4
                    109;Breast;7
                    110;Oesophagus;4
                    111;Gall bladder wall;4
                    112;Gall bladder contents;4
                    113;Muscle tissue, head;3
                    114;Muscle tissue, trunk;3
                    115;Muscle tissue, arms;3
                    116;Muscle tissue, legs;3"""
def baby_organs():
    lg = __baby_organs.replace('\n', ';').replace('\r', '').split(';')
    while len(lg) > 0:
        org = lg.pop(0)
        desc = lg.pop(0)
        tiss = lg.pop(0)
        yield int(org), desc.strip(), __tissues[int(tiss)]

__child_organs="""0;Air;9
                    1;Adrenals;4
                    2;Bladder wall;4
                    3;Brain;7
                    4;Eyes;4
                    5;Eye lenses;4
                    6;Heart;3
                    8;Colon ascending + transverse (wall);4
                    9;Colon descending (wall);4
                    10;Small intestine (wall + contents);4
                    12;Left kidney;4
                    13;Right kidney;4
                    14;Liver;4
                    16;Left lung;8
                    17;Right lung;8
                    18;Ovaries;4
                    19;Pancreas;4
                    21;Head skin;2
                    22;Trunk skin;2
                    23;Arm skin;2
                    24;Leg skin;2
                    25;Spinal cord;4
                    26;Spleen;4
                    27;Stomach wall;4
                    28;Thymus;4
                    29;Thyroid;4
                    31;Head tissue;4
                    32;Trunk tissue;4
                    33;Left arm tissue;3
                    34;Right arm tissue;3
                    35;Left leg tissue;3
                    36;Right leg tissue;3
                    37;Trachea;4
                    38;Uterus;4
                    40;Upper left arm bone;1
                    41;Lower left arm bones;1
                    42;Left hand bones;1
                    43;Upper right arm bone;1
                    44;Lower right arm bones;1
                    45;Right hand bones;1
                    46;Clavicles;1
                    47;Facial skeleton;1
                    49;Upper left leg bone;1
                    50;Lower left leg bones;1
                    51;Left foot bones;1
                    52;Upper right leg bone;1
                    53;Lower right leg bones;1
                    54;Right foot bones;1
                    55;Pelvis;1
                    56;Ribs;1
                    61;Scapulae;1
                    62;Skull;1
                    64;Cervical spine;1
                    65;Thoracic spine;1
                    66;Lumbar spine;1
                    67;Sacrum;1
                    68;Sternum;1
                    110;Mucous membrane (head);4
                    111;Bladder contents;10
                    112;Colon contents;4
                    113;Stomach contents;4
                    114;Testes;4"""
def child_organs():
    lg = __child_organs.replace('\n', ';').replace('\r', '').split(';')
    while len(lg) > 0:
        org = lg.pop(0)
        desc = lg.pop(0)
        tiss = lg.pop(0)
        yield int(org), desc.strip(), __tissues[int(tiss)]

__katja_organs="""0;Air;9
                1;Adrenal, left;4
                2;Adrenal, right;4
                3;Anterior nasal passage (ET1);2
                4;Posterior nasal passage down to larynx (ET2);2
                5;Oral mucosa, tongue;4
                6;Oral mucosa, lips and cheeks;4
                7;Trachea;4
                8;Bronchi;4
                9;Blood vessels, head;4
                10;Blood vessels, trunk;4
                11;Blood vessels, arms;4
                12;Blood vessels, legs;4
                13;Humeri, proximal end, cortical bone;1
                14;Humeri, upper half, spongiosa;1
                15;Humeri, upper half, medullary cavity;1
                16;Humeri, lower half, cortical;1
                17;Humeri, lower half, spongiosa;1
                18;Humeri, lower half, medullary cavity;1
                19;Ulnae and radii, cortical;1
                20;Ulnae and radii, spongiosa;1
                21;Ulnae and radii, medullary cavity;1
                22;Wrists and hand bones, cortical;1
                23;Wrists and hand bones, spongiosa;1
                24;Clavicles, cortical;1
                25;Clavicles, spongiosa;1
                26;Cranium, cortical;1
                27;Cranium, spongiosa;1
                28;Femora, upper half, cortical;1
                29;Femora, upper half, spongiosa;1
                30;Femora, upper half, medullary cavity;1
                31;Femora, lower half, cortical;1
                32;Femora, lower half, spongiosa;1
                33;Femora, lower half, medullary cavity;1
                34;Tibiae, fibulae and patellae, cortical;1
                35;Tibiae, fibulae and patellae, spongiosa;1
                36;Tibiae, fibulae and patellae, medullary cavity;1
                37;Ankles and foot bones, cortical;1
                38;Ankles and foot bones, spongiosa;1
                39;Mandible, cortical;1
                40;Mandible, spongiosa;1
                41;Pelvis, cortical;1
                42;Pelvis, spongiosa;1
                43;Ribs, cortical;1
                44;Ribs, spongiosa;1
                45;Scapulae, cortical;1
                46;Scapulae, spongiosa;1
                47;Cervical spine, cortical;1
                48;Cervical spine, spongiosa;1
                49;Thoracic spine, cortical;1
                50;Thoracic spine, spongiosa;1
                51;Lumbar spine, cortical;1
                52;Lumbar spine, spongiosa;1
                53;Sacrum, cortical;1
                54;Sacrum, spongiosa;1
                55;Sternum, cortical;1
                56;Sternum, spongiosa;1
                57;Cartilage, head;1
                58;Cartilage, trunk;1
                59;Cartilage, arms;1
                60;Cartilage, legs;1
                61;Brain;4
                62;Breast, left, adipose tissue ;7
                63;Breast, left, glandular tissue;4
                64;Breast, right, adipose tissue;7
                65;Breast, right, glandular tissue;4
                66;Eye lens, left;4
                67;Eye bulb, left;4
                68;Eye lens, right;4
                69;Eye bulb, right;4
                70;Gall bladder wall;4
                71;Gall bladder contents;4
                72;Stomach wall;4
                73;Stomach contents;4
                74;Small intestine wall;4
                75;Small intestine contents;4
                76;Ascending colon wall;4
                77;Ascending colon contents;4
                78;Transverse colon wall, right;4
                79;Transverse colon contents, right;4
                80;Transverse colon wall, left;4
                81;Transverse colon contents, left;4
                82;Descending colon wall;4
                83;Descending colon contents;4
                84;Sigmoid colon wall;4
                85;Sigmoid colon contents;4
                86;Rectum wall;4
                87;Heart wall;3
                88;Heart contents (blood);4
                89;Kidney, left, cortex;4
                90;Kidney, left, medulla;4
                91;Kidney, left, pelvis;4
                92;Kidney, right, cortex;4
                93;Kidney, right, medulla;4
                94;Kidney, right, pelvis;4
                95;Liver;4
                96;Lung, left, blood;4
                97;Lung, left, tissue;8
                98;Lung, right, blood;4
                99;Lung, right, tissue;8
                100;Lymphatic nodes, extrathoracic airways;4
                101;Lymphatic nodes, thoracic airways;4
                102;Lymphatic nodes, head;4
                103;Lymphatic nodes, trunk;4
                104;Lymphatic nodes, arms;4
                105;Lymphatic nodes, legs;4
                106;Muscle, head;3
                107;Muscle, trunk;3
                108;Muscle, arms;3
                109;Muscle, legs;3
                110;Oesophagus (wall);4
                111;Ovary, left;4
                112;Ovary, right;4
                113;Pancreas;4
                114;Pituitary gland;4
                115;Prostate;4
                116;Residual tissue, head;7
                117;Residual tissue, trunk;7
                118;Residual tissue, arms;7
                119;Residual tissue, legs;7
                120;Salivary glands, left;4
                121;Salivary glands, right;4
                122;Skin, head;2
                123;Skin, trunk;2
                124;Skin, arms;2
                125;Skin, legs;2
                126;Spinal cord;2
                127;Spleen;4
                128;Teeth;1
                129;Testis, left;4
                130;Testis, right;4
                131;Thymus;4
                132;Thyroid;4
                133;Tongue (inner part);4
                134;Tonsils;4
                135;Ureter, left;4
                136;Ureter, right;4
                137;Urinary bladder wall;4
                138;Urinary bladder contents;4
                139;Uterus / Cervix;4
                140;Air inside body;9
                141;Skin at top and bottom;2
                210;Placenta;4
                211;Umbilical cord;4
                212;Amniotic fluid;4
                213;Fetus Head tissue;4
                214;Fetus cerebrospinal fluid;10
                215;Fetus Spine;1
                220;Fetus Skin;2
                221;Fetus Trunk;4
                222;Fetus Arms;4
                223;Fetus Legs;4
                224;Fetus Brain;4
                225;Fetus Eyes;4
                226;Fetus Eye lenses;4
                227;Fetus Spinal cord;4
                228;Fetus Lungs;4
                229;Fetus Heart;3
                230;Fetus Kidneys;4
                231;Fetus Liver;4
                232;Fetus Stomach;4
                233;Fetus Gall bladder;4
                240;Fetus Cranium;1"""

def katja_organs():
    lg = __katja_organs.replace('\n', ';').replace('\r', '').split(';')
    while len(lg) > 0:
        org = lg.pop(0)
        desc = lg.pop(0)
        tiss = lg.pop(0)
        yield int(org), desc.strip(), __tissues[int(tiss)]


__adam_organs="""0;Air;9
                1;LEFT ADRENAL;4
                2;RIGHT ADRENAL;4
                3;ADRENALS;4
                4;BLADDER;4
                5;CONTENTS;4
                6;BRAIN;7
                7;G.I. STOMACH;4
                8;CONTENTS;4
                9;G.I. U.L.I.;4
                10;CONTENTS;4
                11;G.I. L.L.I.;4
                12;CONTENTS;4
                13;SM.INT.+CONTS;4
                14;GENITALIA;1
                15;HEART;3
                16;LEFT KIDNEY;4
                17;RIGHT KIDNEY;4
                18;KIDNEYS;4
                19;LIVER;4
                20;LEFT LUNG;8
                21;RIGHT LUNG;8
                22;LUNGS;8
                23;RM L.ARM;5
                24;RM R.ARM;5
                25;RM CLAVICLES;5
                26;RM UP.L.LEG;5
                27;RM UP.R.LEG;5
                28;RM PELVIS;5
                29;RM RIBS;5
                30;RM SCAPULAE;5
                31;RM CRANIUM;5
                32;RM FAC.SKEL.;5
                33;RM LOW.SPINE;5
                34;RM MID.SPINE;5
                35;RM UPP.SPINE;5
                36;RM MID.REGION;5
                37;RM LOW.REGION;5
                38;RM HEAD;5
                39;RED MARROW;5
                40;CR UP.L.ARM;1
                42;CR UP.R.ARM;1
                44;CR CLAVICLES;1
                45;CR UP.L.LEG;1
                47;CR UP.R.LEG;1
                49;CR PELVIS;1
                50;CR RIBS;1
                51;CR SCAPULAE;1
                52;CR CRANIUM;1
                53;CR FAC.SKEL.;1
                54;CR LOW.SPINE;1
                55;CR MID.SPINE;1
                56;CR UPP.SPINE;1
                57;CR MID.REGION;1
                58;CR LOW.REGION;1
                59;CR HEAD;1
                60;ORNL   MARROW;6
                64;GABLAD SURF;4
                65;GABLAD CONT;4
                66;GABLAD TOTAL;4
                67;PANCREAS;4
                68;LEFT ARM BONE;1
                69;RT. ARM BONE;1
                70;CLAVICLES;1
                71;LEFT LEG BONE;1
                72;RT. LEG BONE;1
                73;PELVIS;1
                74;RIBS;1
                75;SCAPULAE;1
                76;CRANIUM;1
                77;SPINE;1
                78;SKELETON;1
                79;TRUNK SKIN;2
                80;LEG SKIN;2
                81;HEAD SKIN;2
                82;TOTAL SKIN;2
                83;SPLEEN;4
                84;LEFT TESTIS;4
                85;RIGHT TESTIS;4
                86;TESTES;4
                87;THYMUS;4
                88;THYROID;4
                89;TRUNK TISSUE;4
                90;LEG TISSUE;4
                91;HEAD TISSUE;4
                92;TOTAL TISSUE;4
                93;FAC.SKELETON;1
                94;TRUNK;4
                95;LEGS;4
                96;HEAD;4
                97;TOTAL BODY;4
                98;SKIN BACKSIDE;2
                99;SKIN FRONTAL;2
                100;PERS. DOSE;9
                101;SURFACE ENTR;9
                102;SURFACE EXIT;9
                103;MARR RIBS PO;1
                104;MARR RIBS AN;1
                105;SURFACE ROT.;9
                106;LENSE OF EYE;4
                107;COLON (9+11);4
                108;OESOPHAGUS;4"""


def adam_organs():
    lg = __adam_organs.replace('\n', ';').replace('\r', '').split(';')
    while len(lg) > 0:
        org = lg.pop(0)
        desc = lg.pop(0)
        tiss = lg.pop(0)
        yield int(org), desc.strip(), __tissues[int(tiss)]

__eva_organs = """0;Air;9;
                    1;LEFT ADRENAL;4;
                    2;RIGHT ADRENAL;4;
                    3;ADRENALS;4;
                    4;BLADDER;4;
                    5;BLADDER CONTENTS;4;
                    6;BRAIN;7;
                    7;G.I. STOMACH;4;
                    8;CONTENTS;4;
                    9;G.I. U.L.I.;4;
                    10;CONTENTS;4;
                    11;G.I. L.L.I.;4;
                    12;CONTENTS;4;
                    13;SM.INT.+CONTS;4;
                    14;FAC.SKELETON;1;
                    15;HEART;3;
                    16;LEFT KIDNEY;4;
                    17;RIGHT KIDNEY;4;
                    18;KIDNEYS;4;
                    19;LIVER;4;
                    20;LEFT LUNG;8;
                    21;RIGHT LUNG;8;
                    22;LUNGS;8;
                    23;RM L.ARM;5;
                    24;RM R.ARM;5;
                    25;RM CLAVICLES;5;
                    26;RM UP.L.LEG;5;
                    27;RM UP.R.LEG;5;
                    28;RM PELVIS;5;
                    29;RM RIBS;5;
                    30;RM SCAPULAE;5;
                    31;RM CRANIUM;5;
                    32;RM FAC.SKEL.;5;
                    33;RM LOW.SPINE;5;
                    34;RM MID.SPINE;5;
                    35;RM UPP.SPINE;5;
                    39;RED MARROW;5;
                    64;LEFT OVARY;4;
                    65;RIGHT OVARY;4;
                    66;OVARIES;4;
                    67;PANCREAS;4;
                    68;LEFT ARM BONE;1;
                    69;RT. ARM BONE;1;
                    70;CLAVICLES;1;
                    71;LEFT LEG BONE;1;
                    72;RT. LEG BONE;1;
                    73;PELVIS;1;
                    74;RIBS;1;
                    75;SCAPULAE;1;
                    76;CRANIUM;1;
                    77;SPINE;1;
                    78;SKELETON;1;
                    79;TRUNK SKIN;2;
                    80;LEG SKIN;2;
                    81;HEAD SKIN;2;
                    82;TOTAL SKIN;2;
                    83;SPLEEN;4;
                    84;GABLAD SURF;4;
                    85;GABLAD CONT;4;
                    86;GABLAD TOTAL;4;
                    87;THYMUS;4;
                    88;THYROID;4;
                    89;TRUNK TISSUE;4;
                    90;LEG TISSUE;4;
                    91;HEAD TISSUE;4;
                    92;TOTAL TISSUE;4;
                    93;UTERUS;4;
                    94;TRUNK;4;
                    95;LEGS;4;
                    96;HEAD;4;
                    97;TOTAL BODY;4;
                    98;SKIN BACKSIDE;2;
                    99;SKIN FRONTAL;2;
                    100;PERS. DOSE;9;
                    101;SURFACE ENTR;9;
                    102;SURFACE EXIT;9;
                    105;SURFACE ROT.;9;
                    106;FEM.BREAST;7;
                    107;RIGHT BREAST;7;
                    108;LEFT BREAST;7;
                    109;SKIN R.BRE.;2;
                    110;SKIN L.BREA.;2;
                    111;LENSE OF EYE;4;
                    112;COLON (9+11);4;
                    113;OESOPHAGUS;4;"""
def eva_organs():
    lg = __eva_organs.replace('\n', ';').replace('\r', '').split(';')
    while len(lg) > 0:
        org = lg.pop(0)
        desc = lg.pop(0)
        tiss = lg.pop(0)
        lg.pop(0)
        yield int(org), desc.strip(), __tissues[int(tiss)]

__donna_helga__irene_frank_organs = """0;Air;9
                    1;Adipose tissue arms;7
                    2;Adipose tissue rest of body;7
                    3;Adrenals;4
                    4;Bladder wall;4
                    5;Bladder contents;4
                    6;Blood vessels in arms;4
                    7;Blood vessels in rest of body;4
                    8;Arm bones;1
                    9;Leg bones;1
                    10;Bones in rest of body;1
                    11;Brain;7
                    12;Breast, adipose tissue ;7
                    13;Breast, glandular tissue;4
                    14;Bronchi;4
                    15;Cartilage in arms;4
                    16;Cartilage in rest of body;4
                    17;Caecum;4
                    18;Ascending colon;4
                    19;Transverse colon;4
                    20;Descending colon;4
                    21;Sigmoid colon;4
                    22;Rectum;4
                    23;Caecum contents;4
                    24;Ascending colon contents;4
                    25;Transverse colon contents;4
                    26;Descending colon contents;4
                    27;Sigmoid colon contents;4
                    28;Rectum contents;4
                    29;Eye lenses;4
                    30;Eyes;4
                    31;Anterior nasal passage (ET1);2
                    32;Posterior nasal passage down to larynx (ET2);2
                    33;Gall bladder wall;4
                    34;Gall bladder contents;4
                    35;Heart wall;3
                    36;Heart contents (blood);4
                    37;Kidneys;4
                    38;Liver;4
                    39;Lungs;8
                    40;Muscle in arms;3
                    41;Muscle in rest of body;3
                    42;Oesophagus;4
                    43;Ovaries;4
                    44;Pancreas;4
                    45;Prostate;4
                    46;Salivary glands;4
                    47;Skin at arms;2
                    48;Skin at rest of body;2
                    49;Duodenum;4
                    50;Ileum;4
                    51;Duodenum contents;4
                    52;Ileum contents;4
                    53;Spinal cord;1
                    54;Spleen;4
                    55;Stomach wall;4
                    56;Stomach contents;4
                    57;Teeth;1
                    58;Testes;4
                    59;Thymus;4
                    60;Thyroid;4
                    61;Tongue;4
                    62;Trachea;4
                    63;Ureters;4
                    64;Uterus;4
                    65;Patient table;4"""
def donna_organs():
    lg = __donna_helga__irene_frank_organs.replace('\n', ';').replace('\r', '').split(';')
    while len(lg) > 0:
        org = lg.pop(0)
        desc = lg.pop(0)
        tiss = lg.pop(0)
        yield int(org), desc.strip(), __tissues[int(tiss)]

def helga_organs():
    for val in donna_organs():
        yield val

def irene_organs():
    for val in donna_organs():
        yield val
def frank_organs():
    for val in donna_organs():
        yield val


__vishum_organs = """0;Air;9;
                     1;Head adipose;7;
                     2;Trunk adipose;7;
                     3;Left arm adipose;7;
                     4;Right arm adipose;7;
                     5;Left leg adipose;7;
                     6;Right leg adipose;7;
                     7;Adrenals;4;
                     8;Bladder wall;4;
                     9;Bladder contents;4;
                     10;Blood vessels;4;
                     11;Grey matter;7;
                     12;White matter;7;
                     13;Ventricles;10;
                     14;Adipose tissue, left breast;7;
                     15;Adipose tissue, right breast;7;
                     16;Glandular tissue, left breast;4;
                     17;Glandular tissue, right breast;4;
                     18;Bronchi;4;
                     19;Ascending colon;4;
                     20;Transverse colon;4;
                     21;Descending colon;4;
                     22;Sigmoid colon;4;
                     23;Rectum;4;
                     24;Colon contents;4;
                     25;Eye lenses;4;
                     26;Eyes;4;
                     27;Anterior nasal passage;2;
                     28;Posterior nasal passage;2;
                     29;Pharynx (nasal part);2;
                     30;Pharynx (oral part);2;
                     31;Larynx;2;
                     32;Gall bladder wall;4;
                     33;Gall bladder contents;4;
                     34;Heart;4;
                     35;Left kidney;4;
                     36;Right kidney;4;
                     37;Liver;4;
                     38;Left lung;8;
                     39;Right lung;8;
                     40;Head muscle;3;
                     41;Trunk muscle;3;
                     42;Left arm muscle;3;
                     43;Right arm muscle;3;
                     44;Left leg muscle;3;
                     45;Right leg muscle;3;
                     46;Oesophagus;4;
                     47;Ovaries;4;
                     48;Pancreas;4;
                     49;Prostate;4;
                     50;Salivary glands;4;
                     51;Upper left arm bone;1;
                     52;Lower left arm bones;1;
                     53;Left hand bones;1;
                     54;Upper right arm bone;1;
                     55;Lower right arm bones;1;
                     56;Right hand bones;1;
                     57;Clavicles;1;
                     58;Upper left leg bone;1;
                     59;Lower left leg bones;1;
                     60;Left foot bones;1;
                     61;Upper right leg bone;1;
                     62;Lower right leg bones;1;
                     63;Left foot bones;1;
                     64;Mandible;1;
                     65;Pelvis;1;
                     66;Left rib 1;1;
                     67;Left rib 2;1;
                     68;Left rib 3;1;
                     69;Left rib 4;1;
                     70;Left rib 5;1;
                     71;Left rib 6 ;1;
                     72;Left rib 7 ;1;
                     73;Left rib 8 ;1;
                     74;Left rib 9 ;1;
                     75;Left rib 10 ;1;
                     76;Left rib 11 ;1;
                     77;Left rib 12 ;1;
                     78;Right rib 1 ;1;
                     79;Right rib 2 ;1;
                     80;Right rib 3 ;1;
                     81;Right rib 4 ;1;
                     82;Right rib 5 ;1;
                     83;Right rib 6 ;1;
                     84;Right rib 7 ;1;
                     85;Right rib 8 ;1;
                     86;Right rib 9 ;1;
                     87;Right rib 10 ;1;
                     88;Right rib 11 ;1;
                     89;Right rib 12 ;1;
                     90;Scapulae;1;
                     91;Skull;1;
                     92;Cervical vertebra 1;1;
                     93;Cervical vertebra 2 ;1;
                     94;Cervical vertebra 3 ;1;
                     95;Cervical vertebra 4 ;1;
                     96;Cervical vertebra 5 ;1;
                     97;Cervical vertebra 6 ;1;
                     98;Cervical vertebra 7 ;1;
                     99;Thoracic vertebra 1 ;1;
                    100;Thoracic vertebra 2 ;1;
                    101;Thoracic vertebra 3 ;1;
                    102;Thoracic vertebra 4 ;1;
                    103;Thoracic vertebra 5 ;1;
                    104;Thoracic vertebra 6 ;1;
                    105;Thoracic vertebra 7 ;1;
                    106;Thoracic vertebra 8 ;1;
                    107;Thoracic vertebra 9 ;1;
                    108;Thoracic vertebra 10 ;1;
                    109;Thoracic vertebra 11 ;1;
                    110;Thoracic vertebra 12 ;1;
                    111;Lumbar vertebra 1;1;
                    112;Lumbar vertebra 2 ;1;
                    113;Lumbar vertebra 3 ;1;
                    114;Lumbar vertebra 4 ;1;
                    115;Lumbar vertebra 5 ;1;
                    116;Sacrum;1;
                    117;Sternum;1;
                    118;Head skin;2;
                    119;Trunk skin;2;
                    120;Left arm skin;2;
                    121;Right arm skin;2;
                    122;Left leg skin;2;
                    123;Right leg skin;2;
                    124;Small intestine wall;4;
                    125;Small intestine contents;4;
                    126;Spinal cord;4;
                    127;Spleen;4;
                    128;Stomach wall;4;
                    129;Stomach contents;4;
                    130;Teeth;1;
                    131;Testes;4;
                    132;Thymus;4;
                    133;Thyroid;4;
                    134;Tongue;4;
                    135;Trachea;4;
                    136;Uterus;4;
                    137;Patient table;4;"""


def vishum_organs():
    lg = __vishum_organs.replace('\n', ';').replace('\r', '').split(';')

    while len(lg) > 0:
        org = lg.pop(0)
        desc = lg.pop(0)
        tiss = lg.pop(0)
        lg.pop(0)
        yield int(org), desc.strip(), __tissues[int(tiss)]


__golem_organs = """0;Air;9
                    2;Adipose tissue head;7
                    3;Adipose tissue trunk;7
                    4;Adipose tissue left arm;7
                    5;Adipose tissue right arm;7
                    6;Adipose tissue left leg;7
                    7;Adipose tissue right leg;7
                    8;Adrenals;4
                    9;Bladder wall;4
                    27;Brain;4
                    28;Anterior nasal passage;2
                    29;Posterior nasal passage;2
                    30;Pharynx (nasal part);2
                    31;Pharynx (oral part);2
                    32;Larynx;2
                    33;Bronchial tree;4
                    34;Eyes;4
                    35;Eye lenses;4
                    36;Heart;4
                    38;Left kidney;4
                    39;Right kidney;4
                    41;Liver;4
                    43;Left lung;8
                    44;Right lung;8
                    46;Muscle tissue head;3
                    47;Muscle tissue trunk;3
                    48;Muscle tissue left arm;3
                    49;Muscle tissue right arm;3
                    50;Muscle tissue left leg;3
                    51;Muscle tissue right leg;3
                    52;Oesophagus;4
                    54;Pancreas;4
                    55;Penis;4
                    135;Upper left arm bone;1
                    136;Lower left arm bones;1
                    137;Left hand bones;1
                    138;Upper right arm bone;1
                    139;Lower right arm bones;1
                    140;Right hand bones;1
                    141;Clavicles;1
                    143;Upper left leg bone;1
                    144;Lower left leg bones;1
                    145;Left foot bones;1
                    146;Upper right leg bone;1
                    147;Lower right leg bones;1
                    148;Right foot bones;1
                    149;Mandible;1
                    150;Pelvis;1
                    152;Rib 1 left side;1
                    153;Rib 2 left side;1
                    154;Rib 3 left side;1
                    155;Rib 4 left side;1
                    156;Rib 5 left side;1
                    157;Rib 6 left side;1
                    158;Rib 7 left side;1
                    159;Rib 8 left side;1
                    160;Rib 9 left side;1
                    161;Rib 10 left side;1
                    162;Rib 11 left side;1
                    163;Rib 12 left side;1
                    164;Rib 1 right side;1
                    165;Rib 2 right side;1
                    166;Rib 3 right side;1
                    167;Rib 4 right side;1
                    168;Rib 5 right side;1
                    169;Rib 6 right side;1
                    170;Rib 7 right side;1
                    171;Rib 8 right side;1
                    172;Rib 9 right side;1
                    173;Rib 10 right side;1
                    174;Rib 11 right side;1
                    175;Rib 12 right side;1
                    176;Scapulae;1
                    177;Skull;1
                    180;Cervical vertebra 1;1
                    181;Cervical vertebra 2;1
                    182;Cervical vertebra 3;1
                    183;Cervical vertebra 4;1
                    184;Cervical vertebra 5;1
                    185;Cervical vertebra 6;1
                    186;Cervical vertebra 7;1
                    188;Thoracic vertebra 1;1
                    189;Thoracic vertebra 2;1
                    190;Thoracic vertebra 3;1
                    191;Thoracic vertebra 4;1
                    192;Thoracic vertebra 5;1
                    193;Thoracic vertebra 6;1
                    194;Thoracic vertebra 7;1
                    195;Thoracic vertebra 8;1
                    196;Thoracic vertebra 9;1
                    197;Thoracic vertebra 10;1
                    198;Thoracic vertebra 11;1
                    199;Thoracic vertebra 12;1
                    201;Lumbar vertebra 1;1
                    202;Lumbar vertebra 2;1
                    203;Lumbar vertebra 3;1
                    204;Lumbar vertebra 4;1
                    205;Lumbar vertebra 5;1
                    206;Sacrum;1
                    207;Sternum;1
                    211;Head skin;2
                    212;Trunk skin;2
                    214;Left arm skin;2
                    215;Right arm skin;2
                    217;Left leg skin;2
                    218;Right leg skin;2
                    219;Small intestine (wall + contents);4
                    220;Spinal cord;4
                    221;Spleen;4
                    222;Stomach wall;4
                    223;Teeth;1
                    224;Testes;4
                    225;Thymus;4
                    226;Thyroid;4
                    227;Trachea;4
                    242;Bladder contents;4
                    243;Large intestine contents;4
                    244;Stomach contents;4
                    245;Bed + pillow;4
                    250;Ascending + transverse colon;4
                    251;Descending + sigmoid colon;4
                    252;Gall bladder;4
                    253;Prostate;4"""
def golem_organs():
    lg = __golem_organs.replace('\n', ';').replace('\r', '').split(';')

    while len(lg) > 0:
        org = lg.pop(0)
        desc = lg.pop(0)
        tiss = lg.pop(0)
        yield int(org), desc.strip(), __tissues[int(tiss)]
