# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 11:12:55 2015

@author: erlean
"""

from PyQt4 import QtGui
import logging
logger = logging.getLogger('OpenDXMC')

gray = [[int(i) for i in range(256)] for _ in range(3)]

hot_iron = [[0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34,
             36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64, 66, 68,
             70, 72, 74, 76, 78, 80, 82, 84, 86, 88, 90, 92, 94, 96, 98, 100,
             102, 104, 106, 108, 110, 112, 114, 116, 118, 120, 122, 124, 126,
             128, 130, 132, 134, 136, 138, 140, 142, 144, 146, 148, 150, 152,
             154, 156, 158, 160, 162, 164, 166, 168, 170, 172, 174, 176, 178,
             180, 182, 184, 186, 188, 190, 192, 194, 196, 198, 200, 202, 204,
             206, 208, 210, 212, 214, 216, 218, 220, 222, 224, 226, 228, 230,
             232, 234, 236, 238, 240, 242, 244, 246, 248, 250, 252, 254, 255,
             255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
             255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
             255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
             255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
             255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
             255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
             255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
             255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
             255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
             255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 4, 6,
             8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40,
             42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64, 66, 68, 70, 72, 74,
             76, 78, 80, 82, 84, 86, 88, 90, 92, 94, 96, 98, 100, 102, 104, 106,
             108, 110, 112, 114, 116, 118, 120, 122, 124, 126, 128, 130, 132,
             134, 136, 138, 140, 142, 144, 146, 148, 150, 152, 154, 156, 158,
             160, 162, 164, 166, 168, 170, 172, 174, 176, 178, 180, 182, 184,
             186, 188, 190, 192, 194, 196, 198, 200, 202, 204, 206, 208, 210,
             212, 214, 216, 218, 220, 222, 224, 226, 228, 230, 232, 234, 236,
             238, 240, 242, 244, 246, 248, 250, 252, 255],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 8, 12, 16, 20,
             24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88,
             92, 96, 100, 104, 108, 112, 116, 120, 124, 128, 132, 136, 140, 144,
             148, 152, 156, 160, 164, 168, 172, 176, 180, 184, 188, 192, 196,
             200, 204, 208, 212, 216, 220, 224, 228, 232, 236, 240, 244, 248,
             252, 255]
            ]

pet = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 3, 5, 7, 9, 11, 13,
        15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49,
        51, 53, 55, 57, 59, 61, 63, 65, 67, 69, 71, 73, 75, 77, 79, 81, 83, 85,
        86, 88, 90, 92, 94, 96, 98, 100, 102, 104, 106, 108, 110, 112, 114,
        116, 118, 120, 122, 124, 126, 128, 130, 132, 134, 136, 138, 140, 142,
        144, 146, 148, 150, 152, 154, 156, 158, 160, 162, 164, 166, 168, 170,
        171, 173, 175, 177, 179, 181, 183, 185, 187, 189, 191, 193, 195, 197,
        199, 201, 203, 205, 207, 209, 211, 213, 215, 217, 219, 221, 223, 225,
        227, 229, 231, 233, 235, 237, 239, 241, 243, 245, 247, 249, 251, 253,
        255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
        255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
        255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
        255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
        255, 255, 255, 255, 255, 255, 255, 255],
       [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36,
        38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 65, 67, 69, 71, 73,
        75, 77, 79, 81, 83, 85, 87, 89, 91, 93, 95, 97, 99, 101, 103, 105, 107,
        109, 111, 113, 115, 117, 119, 121, 123, 125, 128, 126, 124, 122, 120,
        118, 116, 114, 112, 110, 108, 106, 104, 102, 100, 98, 96, 94, 92, 90,
        88, 86, 84, 82, 80, 78, 76, 74, 72, 70, 68, 66, 64, 63, 61, 59, 57, 55,
        53, 51, 49, 47, 45, 43, 41, 39, 37, 35, 33, 31, 29, 27, 25, 23, 21, 19,
        17, 15, 13, 11, 9, 7, 5, 3, 1, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20,
        22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56,
        58, 60, 62, 64, 66, 68, 70, 72, 74, 76, 78, 80, 82, 84, 86, 88, 90, 92,
        94, 96, 98, 100, 102, 104, 106, 108, 110, 112, 114, 116, 118, 120, 122,
        124, 126, 128, 130, 132, 134, 136, 138, 140, 142, 144, 146, 148, 150,
        152, 154, 156, 158, 160, 162, 164, 166, 168, 170, 172, 174, 176, 178,
        180, 182, 184, 186, 188, 190, 192, 194, 196, 198, 200, 202, 204, 206,
        208, 210, 212, 214, 216, 218, 220, 222, 224, 226, 228, 230, 232, 234,
        236, 238, 240, 242, 244, 246, 248, 250, 252, 255],
       [0, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35,
        37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59, 61, 63, 65, 67, 69, 71,
        73, 75, 77, 79, 81, 83, 85, 87, 89, 91, 93, 95, 97, 99, 101, 103, 105,
        107, 109, 111, 113, 115, 117, 119, 121, 123, 125, 127, 129, 131, 133,
        135, 137, 139, 141, 143, 145, 147, 149, 151, 153, 155, 157, 159, 161,
        163, 165, 167, 169, 171, 173, 175, 177, 179, 181, 183, 185, 187, 189,
        191, 193, 195, 197, 199, 201, 203, 205, 207, 209, 211, 213, 215, 217,
        219, 221, 223, 225, 227, 229, 231, 233, 235, 237, 239, 241, 243, 245,
        247, 249, 251, 253, 255, 252, 248, 244, 240, 236, 232, 228, 224, 220,
        216, 212, 208, 204, 200, 196, 192, 188, 184, 180, 176, 172, 168, 164,
        160, 156, 152, 148, 144, 140, 136, 132, 128, 124, 120, 116, 112, 108,
        104, 100, 96, 92, 88, 84, 80, 76, 72, 68, 64, 60, 56, 52, 48, 44, 40,
        36, 32, 28, 24, 20, 16, 12, 8, 4, 0, 4, 8, 12, 16, 20, 24, 28, 32, 36,
        40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 85, 89, 93, 97, 101, 105,
        109, 113, 117, 121, 125, 129, 133, 137, 141, 145, 149, 153, 157, 161,
        165, 170, 174, 178, 182, 186, 190, 194, 198, 202, 206, 210, 214, 218,
        222, 226, 230, 234, 238, 242, 246, 250, 255]
       ]

hot_metal_blue = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, 3, 6, 9, 12, 15, 18, 21, 24, 26, 29, 32,
                   35, 38, 41, 44, 47, 50, 52, 55, 57, 59, 62, 64, 66, 69, 71,
                   74, 76, 78, 81, 83, 85, 88, 90, 93, 96, 99, 102, 105, 108,
                   111, 114, 116, 119, 122, 125, 128, 131, 134, 137, 140, 143,
                   146, 149, 152, 155, 158, 161, 164, 166, 169, 172, 175, 178,
                   181, 184, 187, 190, 194, 198, 201, 205, 209, 213, 217, 221,
                   224, 228, 232, 236, 240, 244, 247, 251, 255, 255, 255, 255,
                   255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
                   255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
                   255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
                   255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
                   255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
                   255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
                   255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   2, 4, 6, 8, 9, 11, 13, 15, 17, 19, 21, 23, 24, 26, 28, 30,
                   32, 34, 36, 38, 40, 41, 43, 45, 47, 49, 51, 53, 55, 56, 58,
                   60, 62, 64, 66, 68, 70, 72, 73, 75, 77, 79, 81, 83, 85, 87,
                   88, 90, 92, 94, 96, 98, 100, 102, 104, 105, 107, 109, 111,
                   113, 115, 117, 119, 120, 122, 124, 126, 128, 130, 132, 134,
                   136, 137, 139, 141, 143, 145, 147, 149, 151, 152, 154, 156,
                   158, 160, 162, 164, 166, 168, 169, 171, 173, 175, 177, 179,
                   181, 183, 184, 186, 188, 190, 192, 194, 196, 198, 200, 201,
                   203, 205, 207, 209, 211, 213, 215, 216, 218, 220, 222, 224,
                   226, 228, 229, 231, 233, 235, 237, 239, 240, 242, 244, 246,
                   248, 250, 251, 253, 255],
                  [0, 2, 4, 6, 8, 10, 12, 14, 16, 17, 19, 21, 23, 25, 27, 29,
                   31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59,
                   61, 63, 65, 67, 69, 71, 73, 75, 77, 79, 81, 83, 84, 86, 88,
                   90, 92, 94, 96, 98, 100, 102, 104, 106, 108, 110, 112, 114,
                   116, 117, 119, 121, 123, 125, 127, 129, 131, 133, 135, 137,
                   139, 141, 143, 145, 147, 149, 151, 153, 155, 157, 159, 161,
                   163, 165, 167, 169, 171, 173, 175, 177, 179, 181, 183, 184,
                   186, 188, 190, 192, 194, 196, 198, 200, 197, 194, 191, 188,
                   185, 182, 179, 176, 174, 171, 168, 165, 162, 159, 156, 153,
                   150, 144, 138, 132, 126, 121, 115, 109, 103, 97, 91, 85, 79,
                   74, 68, 62, 56, 50, 47, 44, 41, 38, 35, 32, 29, 26, 24, 21,
                   18, 15, 12, 9, 6, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 3, 6, 9, 12, 15, 18, 21, 24, 26, 29, 32, 35,
                   38, 41, 44, 47, 50, 53, 56, 59, 62, 65, 68, 71, 74, 76, 79,
                   82, 85, 88, 91, 94, 97, 100, 103, 106, 109, 112, 115, 118,
                   121, 124, 126, 129, 132, 135, 138, 141, 144, 147, 150, 153,
                   156, 159, 162, 165, 168, 171, 174, 176, 179, 182, 185, 188,
                   191, 194, 197, 200, 203, 206, 210, 213, 216, 219, 223, 226,
                   229, 232, 236, 239, 242, 245, 249, 252, 255]
                  ]
hot_metal_green= [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, 3, 6, 9, 12, 15, 18, 21, 24, 26, 29, 32,
                   35, 38, 41, 44, 47, 50, 52, 55, 57, 59, 62, 64, 66, 69, 71,
                   74, 76, 78, 81, 83, 85, 88, 90, 93, 96, 99, 102, 105, 108,
                   111, 114, 116, 119, 122, 125, 128, 131, 134, 137, 140, 143,
                   146, 149, 152, 155, 158, 161, 164, 166, 169, 172, 175, 178,
                   181, 184, 187, 190, 194, 198, 201, 205, 209, 213, 217, 221,
                   224, 228, 232, 236, 240, 244, 247, 251, 255, 255, 255, 255,
                   255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
                   255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
                   255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
                   255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
                   255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
                   255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
                   255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
                  [0, 2, 4, 6, 8, 10, 12, 14, 16, 17, 19, 21, 23, 25, 27, 29,
                   31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59,
                   61, 63, 65, 67, 69, 71, 73, 75, 77, 79, 81, 83, 84, 86, 88,
                   90, 92, 94, 96, 98, 100, 102, 104, 106, 108, 110, 112, 114,
                   116, 117, 119, 121, 123, 125, 127, 129, 131, 133, 135, 137,
                   139, 141, 143, 145, 147, 149, 151, 153, 155, 157, 159, 161,
                   163, 165, 167, 169, 171, 173, 175, 177, 179, 181, 183, 184,
                   186, 188, 190, 192, 194, 196, 198, 200, 197, 194, 191, 188,
                   185, 182, 179, 176, 174, 171, 168, 165, 162, 159, 156, 153,
                   150, 144, 138, 132, 126, 121, 115, 109, 103, 97, 91, 85, 79,
                   74, 68, 62, 56, 50, 47, 44, 41, 38, 35, 32, 29, 26, 24, 21,
                   18, 15, 12, 9, 6, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 3, 6, 9, 12, 15, 18, 21, 24, 26, 29, 32, 35,
                   38, 41, 44, 47, 50, 53, 56, 59, 62, 65, 68, 71, 74, 76, 79,
                   82, 85, 88, 91, 94, 97, 100, 103, 106, 109, 112, 115, 118,
                   121, 124, 126, 129, 132, 135, 138, 141, 144, 147, 150, 153,
                   156, 159, 162, 165, 168, 171, 174, 176, 179, 182, 185, 188,
                   191, 194, 197, 200, 203, 206, 210, 213, 216, 219, 223, 226,
                   229, 232, 236, 239, 242, 245, 249, 252, 255],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   2, 4, 6, 8, 9, 11, 13, 15, 17, 19, 21, 23, 24, 26, 28, 30,
                   32, 34, 36, 38, 40, 41, 43, 45, 47, 49, 51, 53, 55, 56, 58,
                   60, 62, 64, 66, 68, 70, 72, 73, 75, 77, 79, 81, 83, 85, 87,
                   88, 90, 92, 94, 96, 98, 100, 102, 104, 105, 107, 109, 111,
                   113, 115, 117, 119, 120, 122, 124, 126, 128, 130, 132, 134,
                   136, 137, 139, 141, 143, 145, 147, 149, 151, 152, 154, 156,
                   158, 160, 162, 164, 166, 168, 169, 171, 173, 175, 177, 179,
                   181, 183, 184, 186, 188, 190, 192, 194, 196, 198, 200, 201,
                   203, 205, 207, 209, 211, 213, 215, 216, 218, 220, 222, 224,
                   226, 228, 229, 231, 233, 235, 237, 239, 240, 242, 244, 246,
                   248, 250, 251, 253, 255],
                  ]
gist_earth = [[0, 0, 1, 1, 2, 3, 3, 4, 5, 5, 6, 7, 7, 8, 9, 9, 10, 11, 11, 12, 13, 13, 14, 15, 15, 16, 17, 17, 18, 19, 19, 20, 21, 21, 22, 23, 23, 24, 25, 25, 26, 27, 27, 28, 29, 29, 30, 31, 31, 32, 33, 33, 34, 35, 35, 36, 37, 37, 38, 39, 39, 40, 41, 41, 42, 43, 43, 44, 45, 45, 46, 47, 47, 48, 48, 49, 49, 50, 50, 51, 51, 52, 52, 53, 53, 54, 54, 55, 55, 56, 56, 56, 57, 57, 58, 58, 59, 59, 60, 60, 61, 61, 62, 62, 63, 63, 64, 64, 64, 65, 65, 66, 66, 67, 67, 68, 68, 69, 71, 73, 75, 78, 80, 82, 84, 87, 89, 91, 93, 95, 98, 100, 102, 104, 107, 109, 111, 113, 115, 118, 120, 121, 123, 125, 126, 128, 130, 131, 133, 135, 136, 138, 140, 141, 143, 145, 146, 148, 150, 151, 153, 154, 156, 158, 159, 161, 163, 164, 166, 168, 169, 171, 173, 174, 176, 178, 179, 181, 182, 183, 183, 184, 184, 185, 185, 185, 186, 186, 187, 187, 188, 188, 188, 189, 189, 190, 190, 190, 191, 191, 192, 192, 193, 194, 195, 197, 198, 199, 200, 201, 202, 203, 204, 206, 207, 208, 209, 210, 211, 212, 213, 214, 216, 217, 218, 219, 220, 221, 222, 223, 225, 226, 227, 228, 229, 230, 231, 232, 233, 235, 236, 237, 238, 239, 240, 241, 242, 244, 245, 246, 247, 248, 249, 250, 251, 253],
              [0, 0, 0, 0, 0, 0, 0, 0, 2, 4, 6, 9, 11, 13, 16, 18, 20, 22, 25, 27, 29, 32, 34, 36, 39, 41, 43, 45, 48, 50, 52, 54, 56, 58, 60, 62, 64, 66, 69, 71, 73, 75, 77, 79, 81, 83, 84, 86, 88, 90, 92, 94, 96, 97, 99, 101, 102, 104, 105, 107, 109, 110, 112, 113, 115, 116, 118, 120, 121, 123, 124, 126, 127, 128, 129, 129, 130, 130, 131, 132, 132, 133, 133, 134, 134, 135, 136, 136, 137, 137, 138, 138, 139, 140, 140, 141, 141, 142, 142, 143, 144, 144, 145, 145, 146, 147, 147, 148, 148, 149, 149, 150, 151, 151, 152, 152, 153, 153, 154, 155, 155, 156, 156, 157, 157, 158, 159, 159, 160, 160, 161, 161, 162, 163, 163, 163, 164, 164, 165, 165, 166, 166, 167, 167, 167, 168, 168, 169, 169, 170, 170, 171, 171, 171, 172, 172, 173, 173, 174, 174, 174, 175, 175, 176, 176, 177, 177, 178, 178, 178, 179, 179, 180, 180, 181, 181, 181, 182, 182, 181, 181, 180, 179, 178, 177, 176, 175, 175, 174, 173, 172, 171, 170, 169, 169, 168, 167, 166, 165, 164, 163, 163, 163, 163, 164, 164, 165, 166, 166, 167, 168, 169, 170, 171, 172, 173, 173, 174, 175, 176, 177, 178, 179, 181, 182, 183, 185, 186, 188, 189, 191, 193, 195, 197, 199, 201, 203, 205, 207, 209, 211, 213, 215, 217, 220, 222, 224, 227, 230, 233, 236, 239, 242, 245, 248, 250],
              [0, 43, 56, 67, 78, 88, 99, 110, 115, 116, 116, 116, 116, 116, 117, 117, 117, 117, 117, 117, 118, 118, 118, 118, 118, 119, 119, 119, 119, 119, 119, 120, 120, 120, 120, 120, 121, 121, 121, 121, 121, 121, 122, 122, 122, 122, 122, 123, 123, 123, 123, 123, 123, 124, 124, 124, 124, 124, 125, 125, 125, 125, 125, 125, 126, 126, 126, 126, 126, 127, 127, 127, 127, 126, 125, 123, 122, 121, 120, 119, 117, 116, 115, 114, 112, 111, 110, 109, 108, 106, 105, 104, 103, 101, 100, 99, 98, 97, 95, 94, 93, 92, 90, 89, 88, 87, 85, 84, 83, 82, 81, 79, 78, 77, 76, 74, 73, 72, 71, 70, 70, 71, 71, 72, 72, 73, 74, 74, 75, 75, 76, 77, 77, 78, 78, 79, 79, 80, 81, 81, 82, 82, 82, 82, 83, 83, 83, 84, 84, 84, 85, 85, 85, 86, 86, 86, 87, 87, 87, 88, 88, 88, 88, 89, 89, 89, 90, 90, 90, 91, 91, 91, 92, 92, 92, 93, 93, 93, 94, 94, 94, 95, 95, 95, 95, 96, 96, 96, 97, 97, 97, 98, 98, 98, 99, 99, 99, 100, 100, 100, 101, 103, 105, 108, 110, 113, 115, 118, 120, 123, 125, 127, 130, 132, 135, 137, 140, 142, 145, 147, 150, 152, 154, 157, 159, 162, 164, 167, 169, 172, 175, 178, 181, 184, 187, 190, 193, 196, 199, 202, 205, 208, 211, 214, 217, 220, 223, 226, 229, 232, 235, 238, 241, 244, 247, 250]]

gist_rainbow = [[255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
               255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
               255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
               255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
               255, 255, 255, 254, 248, 243, 237, 232, 227, 221, 216, 210, 205,
               199, 194, 189, 183, 178, 172, 167, 162, 156, 151, 145, 140, 135,
               129, 124, 118, 113, 108, 102,  97,  91,  86,  81,  75,  70,  64,
                59,  54,  48,  43,  37,  32,  27,  21,  16,  10,   5,   0,   0,
                 0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
                 0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
                 0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
                 0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
                 0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
                 0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
                 0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
                 0,   0,   3,   8,  14,  19,  25,  30,  36,  41,  47,  52,  57,
                63,  68,  74,  79,  85,  90,  95, 101, 106, 112, 117, 123, 128,
               133, 139, 144, 150, 155, 161, 166, 172, 177, 182, 188, 193, 199,
               204, 210, 215, 220, 226, 231, 237, 242, 248, 253, 255, 255, 255,
               255, 255, 255, 255, 255, 255, 255, 255, 255],
               [  0,   0,   0,   0,   0,   0,   0,   0,   1,   7,  12,  18,  23,
                28,  34,  39,  45,  50,  55,  61,  66,  72,  77,  82,  88,  93,
                99, 104, 110, 115, 120, 126, 131, 137, 142, 147, 153, 158, 164,
               169, 174, 180, 185, 191, 196, 201, 207, 212, 218, 223, 228, 234,
               239, 245, 250, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
               255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
               255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
               255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
               255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
               255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
               255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
               255, 255, 255, 255, 255, 255, 255, 251, 246, 241, 235, 230, 224,
               219, 213, 208, 202, 197, 192, 186, 181, 175, 170, 164, 159, 154,
               148, 143, 137, 132, 126, 121, 116, 110, 105,  99,  94,  88,  83,
                77,  72,  67,  61,  56,  50,  45,  39,  34,  29,  23,  18,  12,
                 7,   1,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
                 0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
                 0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
                 0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
                 0,   0,   0,   0,   0,   0,   0,   0,   0],
                 [ 40,  35,  30,  24,  19,  14,   8,   3,   0,   0,   0,   0,   0,
                 0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
                 0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
                 0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
                 0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
                 0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
                 0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
                 0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   5,
                10,  16,  21,  26,  32,  37,  43,  48,  53,  59,  64,  69,  75,
                80,  86,  91,  96, 102, 107, 112, 118, 123, 129, 134, 139, 145,
               150, 155, 161, 166, 172, 177, 182, 188, 193, 198, 204, 209, 215,
               220, 225, 231, 236, 241, 247, 252, 255, 255, 255, 255, 255, 255,
               255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
               255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
               255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
               255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
               255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
               255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
               255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 251, 245, 240,
               234, 229, 223, 218, 212, 207, 202, 196, 191]]
jet = [[  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
         0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
         0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
         0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
         0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
         0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
         0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   2,
         6,   9,  12,  15,  19,  22,  25,  28,  31,  35,  38,  41,  44,
        48,  51,  54,  57,  60,  64,  67,  70,  73,  77,  80,  83,  86,
        90,  93,  96,  99, 102, 106, 109, 112, 115, 119, 122, 125, 128,
       131, 135, 138, 141, 144, 148, 151, 154, 157, 160, 164, 167, 170,
       173, 177, 180, 183, 186, 190, 193, 196, 199, 202, 206, 209, 212,
       215, 219, 222, 225, 228, 231, 235, 238, 241, 244, 248, 251, 254,
       255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
       255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
       255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
       255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
       255, 255, 255, 255, 255, 255, 255, 250, 246, 241, 237, 232, 228,
       223, 218, 214, 209, 205, 200, 196, 191, 187, 182, 178, 173, 168,
       164, 159, 155, 150, 146, 141, 137, 132, 128],
       [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
         0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
         0,   0,   0,   0,   0,   0,   0,   4,   8,  12,  16,  20,  24,
        28,  32,  36,  40,  44,  48,  52,  56,  60,  64,  68,  72,  76,
        80,  84,  88,  92,  96, 100, 104, 108, 112, 116, 120, 124, 128,
       132, 136, 140, 144, 148, 152, 156, 160, 164, 168, 172, 176, 180,
       184, 188, 192, 196, 200, 204, 208, 212, 216, 220, 224, 228, 232,
       236, 240, 244, 248, 252, 255, 255, 255, 255, 255, 255, 255, 255,
       255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
       255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
       255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
       255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
       255, 255, 255, 255, 255, 255, 255, 255, 252, 248, 245, 241, 237,
       234, 230, 226, 222, 219, 215, 211, 208, 204, 200, 196, 193, 189,
       185, 182, 178, 174, 171, 167, 163, 159, 156, 152, 148, 145, 141,
       137, 134, 130, 126, 122, 119, 115, 111, 108, 104, 100,  96,  93,
        89,  85,  82,  78,  74,  71,  67,  63,  59,  56,  52,  48,  45,
        41,  37,  34,  30,  26,  22,  19,  15,  11,   8,   4,   0,   0,
         0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
         0,   0,   0,   0,   0,   0,   0,   0,   0],
        [128, 132, 137, 141, 146, 150, 155, 159, 164, 168, 173, 178, 182,
       187, 191, 196, 200, 205, 209, 214, 218, 223, 227, 232, 237, 241,
       246, 250, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
       255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
       255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
       255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
       255, 255, 255, 255, 255, 255, 255, 255, 255, 254, 251, 248, 244,
       241, 238, 235, 231, 228, 225, 222, 219, 215, 212, 209, 206, 202,
       199, 196, 193, 190, 186, 183, 180, 177, 173, 170, 167, 164, 160,
       157, 154, 151, 148, 144, 141, 138, 135, 131, 128, 125, 122, 119,
       115, 112, 109, 106, 102,  99,  96,  93,  90,  86,  83,  80,  77,
        73,  70,  67,  64,  60,  57,  54,  51,  48,  44,  41,  38,  35,
        31,  28,  25,  22,  19,  15,  12,   9,   6,   2,   0,   0,   0,
         0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
         0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
         0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
         0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
         0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
         0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
         0,   0,   0,   0,   0,   0,   0,   0,   0] ]
cool = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 32, 34, 35, 36, 36, 38, 39, 40, 40, 42, 43, 44, 44, 46, 47, 48, 48, 50, 51, 52, 52, 54, 55, 56, 56, 58, 59, 60, 60, 62, 63, 64, 65, 65, 67, 68, 69, 70, 71, 72, 73, 73, 75, 76, 77, 78, 79, 80, 81, 81, 83, 84, 85, 86, 87, 88, 89, 89, 91, 92, 93, 94, 95, 96, 97, 97, 99, 100, 101, 102, 103, 104, 105, 105, 107, 108, 109, 110, 111, 112, 113, 113, 115, 116, 117, 118, 119, 120, 121, 121, 123, 124, 125, 126, 127, 128, 129, 130, 131, 131, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 147, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 163, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 179, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 195, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 211, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 227, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 243, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255],
        [255, 254, 253, 252, 251, 250, 249, 248, 247, 246, 245, 244, 243, 242, 241, 240, 239, 238, 237, 236, 235, 234, 233, 232, 231, 230, 229, 228, 227, 226, 225, 224, 223, 222, 221, 220, 219, 218, 217, 216, 215, 214, 213, 211, 211, 210, 209, 208, 207, 206, 205, 204, 203, 202, 201, 200, 199, 198, 197, 195, 195, 194, 193, 192, 191, 190, 189, 188, 187, 186, 185, 184, 183, 182, 181, 179, 179, 178, 177, 176, 175, 174, 173, 172, 171, 170, 169, 168, 167, 166, 165, 163, 163, 162, 161, 160, 159, 158, 157, 156, 155, 154, 153, 152, 151, 150, 149, 147, 147, 146, 145, 144, 143, 142, 141, 140, 139, 138, 137, 136, 135, 134, 133, 131, 131, 130, 129, 128, 127, 126, 125, 124, 123, 121, 121, 120, 119, 118, 117, 116, 114, 113, 113, 112, 111, 110, 109, 108, 107, 105, 105, 104, 103, 102, 101, 100, 98, 97, 97, 96, 95, 94, 93, 92, 91, 89, 89, 88, 87, 86, 85, 84, 82, 81, 81, 80, 79, 78, 77, 76, 75, 73, 73, 72, 71, 70, 69, 68, 66, 65, 65, 64, 63, 62, 61, 60, 59, 57, 56, 56, 55, 54, 53, 52, 50, 49, 48, 48, 47, 46, 45, 44, 43, 41, 40, 40, 39, 38, 37, 36, 34, 33, 32, 32, 31, 30, 29, 28, 27, 25, 24, 24, 23, 22, 21, 20, 18, 17, 16, 16, 15, 14, 13, 12, 11, 9, 8, 8, 7, 6, 5, 4, 2, 1, 0, 0],
        [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255]
       ]
#lut_table = {'gray': [QtGui.QColor(i, i, i).rgb() for i in range(256)]}
_names = ['gray', 'hot_metal_blue', 'pet', 'hot_iron', 'gist_rainbow', 'jet', 'cool', 'hot_metal_green', 'gist_earth']
_tables = [gray, hot_metal_blue, pet, hot_iron, gist_rainbow, jet, cool, hot_metal_green, gist_earth]

def get_lut_raw(name):
    if not name in _names:
        logger.warning('Incorrect LUT name. Could not find lut {0}, using gray'.format(name))
        name = 'gray'
    ind = _names.index(name)
    return _tables[ind]
    

def get_lut(name, alpha=255):
    if not name in _names:
        logger.warning('Incorrect LUT name. Could not find lut {0}, using gray'.format(name))
        name = 'gray'
    ind = _names.index(name)
    tb = _tables[ind]
    try:
        n_alpha = len(alpha)
    except TypeError:
        pass
    else:
        if n_alpha < 256:
            raise AssertionError('Lenght of alpha must be grater or equal 256')
        return [QtGui.QColor(tb[0][i], tb[1][i], tb[2][i], alpha[i]).rgba()
                for i in range(256)]

    return [QtGui.QColor(tb[0][i], tb[1][i], tb[2][i], alpha).rgba()
            for i in range(256)]


