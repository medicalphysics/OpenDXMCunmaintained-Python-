# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 11:19:36 2015

@author: erlean
"""
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

    tissues = {1: 'hardbone',
               2: 'skin',
               3: 'muscle',
               4: 'soft',
               5: 'redbonemarrow',
               6: 'yellowbonemarrow',
               7: 'adipose',
               8: 'lung',
               9: 'air',
               }


    while len(lg) > 0:
        org = lg.pop(0)
        desc = lg.pop(0)
        tiss = lg.pop(0)
        yield int(org), desc.strip(), tissues[int(tiss)]
