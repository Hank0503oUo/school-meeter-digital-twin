# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 16:19:17 2019

@author: Steve
"""

import numpy as np
import pandas as pd
#iddfile = "C:/Users/Steve/Anaconda3/Lib/site-packages/eppy/resources/iddfiles/Energy+V8_9_0.idd"
#fname1 = "C:/Users/Steve/Desktop/master_0917/Master_dissertation/model/model_v6.idf"
#IDF.setiddname(iddfile)
#idf1 = eppy.modeleditor.IDF(fname1)
def wall_coord(floor_area,c_ht,aspect_ratio,off_rat):
     '''
     標號方式
     ENWC
     E表示外牆
     NW表示西北方
     C表示ceiling
     ISEC
     I表示內牆
     SE表示東南方
     C
     '''
     #WWR = 0.5#窗牆比
     #floor_area = 5000#基地面積
     #floor_num = 10
     #c_ht = 3#樓高
     #aspect_ratio = 2#長寬比
     #off_rat = 0.7#居室面積比
     
     '''
     index編號方式:;東北方開始，順時間編號，先編天花板再編樓板
     '''
     #座標命名
     wid = (floor_area/aspect_ratio)**0.5
     leng = wid*aspect_ratio
     IE = wid - (wid-(wid*(1-off_rat)**0.5))/2
     IW = (wid-(wid*(1-off_rat)**0.5))/2
     IN = leng - (leng-(leng*(1-off_rat)**0.5))/2
     IS = (leng-(leng*(1-off_rat)**0.5))/2
     f_ht = 0
     EE = wid
     EW = 0
     EN = leng
     ES = 0
     
     #座標命名
     Ori_index = ['ENEC','ENEF','INEC','INEF','ISEC','ISEF','ESEC','ESEF','ESWC','ESWF','ISWC','ISWF','INWC','INWF','ENWC','ENWF']
     Axis_columns = ['x','y','z']
     coord = pd.DataFrame(np.ones((len(Ori_index),len(Axis_columns))),index = Ori_index,columns = Axis_columns)
     coord.loc['ENEC'] = np.array([EE,EN,c_ht])
     coord.loc['ENEF'] = np.array([EE,EN,f_ht])
     coord.loc['INEC'] = np.array([IE,IN,c_ht])
     coord.loc['INEF'] = np.array([IE,IN,f_ht])
     coord.loc['ISEC'] = np.array([IE,IS,c_ht])
     coord.loc['ISEF'] = np.array([IE,IS,f_ht])
     coord.loc['ESEC'] = np.array([EE,ES,c_ht])
     coord.loc['ESEF'] = np.array([EE,ES,f_ht])
     coord.loc['ESWC'] = np.array([EW,ES,c_ht])
     coord.loc['ESWF'] = np.array([EW,ES,f_ht])
     coord.loc['ISWC'] = np.array([IW,IS,c_ht])
     coord.loc['ISWF'] = np.array([IW,IS,f_ht])
     coord.loc['INWC'] = np.array([IW,IN,c_ht])
     coord.loc['INWF'] = np.array([IW,IN,f_ht])
     coord.loc['ENWC'] = np.array([EW,EN,c_ht])
     coord.loc['ENWF'] = np.array([EW,EN,f_ht])
     '''
     宣告
     '''
     wall_index = ['E_ExtWall','S_ExtWall','W_ExtWall','N_ExtWall',
                   'E_Ceiling','S_Ceiling','W_Ceiling','N_Ceiling','C_Ceiling',
                   'E_Floor','S_Floor','W_Floor','N_Floor','C_Floor',
                   'E_IntWall_S','S_IntWall_E',
                   'S_IntWall_W','W_IntWall_S',
                   'W_IntWall_N','N_IntWall_W',
                   'N_IntWall_E','E_IntWall_N',
                   'C_IntWall_E','E_IntWall_C',
                   'C_IntWall_S','S_IntWall_C',
                   'C_IntWall_W','W_IntWall_C',
                   'C_IntWall_N','N_IntWall_C',]
     wall_columns = ['x1','y1','z1','x2','y2','z2','x3','y3','z3','x4','y4','z4']
     wall_coord = pd.DataFrame(np.ones((len(wall_index),len(wall_columns))),index = wall_index,columns = wall_columns)
     '''
     給定座標
     '''
     wall_coord.loc['E_ExtWall'] = np.hstack([coord.loc['ESEC'],coord.loc['ESEF'],coord.loc['ENEF'],coord.loc['ENEC']])
     wall_coord.loc['S_ExtWall'] = np.hstack([coord.loc['ESWC'],coord.loc['ESWF'],coord.loc['ESEF'],coord.loc['ESEC']])
     wall_coord.loc['W_ExtWall'] = np.hstack([coord.loc['ENWC'],coord.loc['ENWF'],coord.loc['ESWF'],coord.loc['ESWC']])
     wall_coord.loc['N_ExtWall'] = np.hstack([coord.loc['ENEC'],coord.loc['ENEF'],coord.loc['ENWF'],coord.loc['ENWC']])
     wall_coord.loc['E_Ceiling'] = np.hstack([coord.loc['ISEC'],coord.loc['ESEC'],coord.loc['ENEC'],coord.loc['INEC']])
     wall_coord.loc['S_Ceiling'] = np.hstack([coord.loc['ISWC'],coord.loc['ESWC'],coord.loc['ESEC'],coord.loc['ISEC']])
     wall_coord.loc['W_Ceiling'] = np.hstack([coord.loc['INWC'],coord.loc['ENWC'],coord.loc['ESWC'],coord.loc['ISWC']])
     wall_coord.loc['N_Ceiling'] = np.hstack([coord.loc['INEC'],coord.loc['ENEC'],coord.loc['ENWC'],coord.loc['INWC']])
     wall_coord.loc['C_Ceiling'] = np.hstack([coord.loc['INWC'],coord.loc['ISWC'],coord.loc['ISEC'],coord.loc['INEC']])
     wall_coord.loc['E_Floor'] = np.hstack([coord.loc['INEF'],coord.loc['ENEF'],coord.loc['ESEF'],coord.loc['ISEF']])
     wall_coord.loc['S_Floor'] = np.hstack([coord.loc['ISEF'],coord.loc['ESEF'],coord.loc['ESWF'],coord.loc['ISWF']])
     wall_coord.loc['W_Floor'] = np.hstack([coord.loc['ISWF'],coord.loc['ESWF'],coord.loc['ENWF'],coord.loc['INWF']])
     wall_coord.loc['N_Floor'] = np.hstack([coord.loc['INWF'],coord.loc['ENWF'],coord.loc['ENEF'],coord.loc['INEF']])
     wall_coord.loc['C_Floor'] = np.hstack([coord.loc['INEF'],coord.loc['ISEF'],coord.loc['ISWF'],coord.loc['INWF']])
     wall_coord.loc['E_IntWall_S'] = np.hstack([coord.loc['ISEC'],coord.loc['ISEF'],coord.loc['ESEF'],coord.loc['ESEC']])
     wall_coord.loc['S_IntWall_E'] = np.hstack([coord.loc['ESEC'],coord.loc['ESEF'],coord.loc['ISEF'],coord.loc['ISEC']])
     wall_coord.loc['S_IntWall_W'] = np.hstack([coord.loc['ISWC'],coord.loc['ISWF'],coord.loc['ESWF'],coord.loc['ESWC']])
     wall_coord.loc['W_IntWall_S'] = np.hstack([coord.loc['ESWC'],coord.loc['ESWF'],coord.loc['ISWF'],coord.loc['ISWC']])
     wall_coord.loc['W_IntWall_N'] = np.hstack([coord.loc['INWC'],coord.loc['INWF'],coord.loc['ENWF'],coord.loc['ENWC']])
     wall_coord.loc['N_IntWall_W'] = np.hstack([coord.loc['ENWC'],coord.loc['ENWF'],coord.loc['INWF'],coord.loc['INWC']])
     wall_coord.loc['N_IntWall_E'] = np.hstack([coord.loc['INEC'],coord.loc['INEF'],coord.loc['ENEF'],coord.loc['ENEC']])
     wall_coord.loc['E_IntWall_N'] = np.hstack([coord.loc['ENEC'],coord.loc['ENEF'],coord.loc['INEF'],coord.loc['INEC']])
     wall_coord.loc['C_IntWall_E'] = np.hstack([coord.loc['ISEC'],coord.loc['ISEF'],coord.loc['INEF'],coord.loc['INEC']])
     wall_coord.loc['E_IntWall_C'] = np.hstack([coord.loc['INEC'],coord.loc['INEF'],coord.loc['ISEF'],coord.loc['ISEC']])
     wall_coord.loc['C_IntWall_S'] = np.hstack([coord.loc['ISWC'],coord.loc['ISWF'],coord.loc['ISEF'],coord.loc['ISEC']])
     wall_coord.loc['S_IntWall_C'] = np.hstack([coord.loc['ISEC'],coord.loc['ISEF'],coord.loc['ISWF'],coord.loc['ISWC']])
     wall_coord.loc['C_IntWall_W'] = np.hstack([coord.loc['INWC'],coord.loc['INWF'],coord.loc['ISWF'],coord.loc['ISWC']])
     wall_coord.loc['W_IntWall_C'] = np.hstack([coord.loc['ISWC'],coord.loc['ISWF'],coord.loc['INWF'],coord.loc['INWC']])
     wall_coord.loc['C_IntWall_N'] = np.hstack([coord.loc['INEC'],coord.loc['INEF'],coord.loc['INWF'],coord.loc['INWC']])
     wall_coord.loc['N_IntWall_C'] = np.hstack([coord.loc['INWC'],coord.loc['INWF'],coord.loc['INEF'],coord.loc['INEC']])
     
     return wall_coord
#print(idf1.idfobjects['BUILDINGSURFACE:DETAILED'][0] )
#print(wall_coord.loc['W_IntWall_C'])
def window_coord(WWR,floor_area,aspect_ratio,c_ht):

     #     WWR = 0.5 
     #     floor_area = 5000
     #     aspect_ratio = 2
     #     c_ht = 3
     wid = (floor_area/aspect_ratio)**0.5
     leng = wid*aspect_ratio
     hd = c_ht*(1-WWR)/2
     EE = wid
     EW = 0
     EN = leng
     ES = 0
     h = c_ht
     Ori_index = ['WENH','WENL','WESH','WESL','WWSH','WWSL','WWNH','WWNL']
     Axis_columns = ['x','y','z']
     coord = pd.DataFrame(np.ones((len(Ori_index),len(Axis_columns))),index = Ori_index,columns = Axis_columns)
     window_index = ['E_Window','S_Window','N_Window','W_Window']
     window_columns = ['x1','y1','z1','x2','y2','z2','x3','y3','z3','x4','y4','z4']
     coord.loc['WENH'] = [EE,EN,h-hd]
     coord.loc['WENL'] = [EE,EN,hd]
     coord.loc['WESH'] = [EE,ES,h-hd]
     coord.loc['WESL'] = [EE,ES,hd]
     coord.loc['WWSH'] = [EW,ES,h-hd]
     coord.loc['WWSL'] = [EW,ES,hd]
     coord.loc['WWNH'] = [EW,EN,h-hd]
     coord.loc['WWNL'] = [EW,EN,hd]
     window_coord = pd.DataFrame(np.ones((len(window_index),len(window_columns))),index = window_index,columns = window_columns)
     window_coord.loc['E_Window'] = np.hstack([coord.loc['WESH'],coord.loc['WESL'],coord.loc['WENL'],coord.loc['WENH']])
     window_coord.loc['S_Window'] = np.hstack([coord.loc['WWSH'],coord.loc['WWSL'],coord.loc['WESL'],coord.loc['WESH']])
     window_coord.loc['N_Window'] = np.hstack([coord.loc['WENH'],coord.loc['WENL'],coord.loc['WWNL'],coord.loc['WWNH']])
     window_coord.loc['W_Window'] = np.hstack([coord.loc['WWNH'],coord.loc['WWNL'],coord.loc['WWSL'],coord.loc['WWSH']])
     return window_coord

def shading_coord(WWR,floor_area,aspect_ratio,shading_ratio,c_ht):
     
     
     
     wid = (floor_area/aspect_ratio)**0.5
     leng = wid*aspect_ratio 
     hd = c_ht*(1-WWR)/2
     l = (c_ht - hd*2) * shading_ratio
     shading_index = ['E_Window_shading','S_Window_shading','N_Window_shading','W_Window_shading']
     Axis_columns = ['depth']
     shading_coord = pd.DataFrame(np.ones((len(shading_index),len(Axis_columns))),index = shading_index,columns = Axis_columns)
     shading_coord.loc['E_Window_shading'] = l
     shading_coord.loc['S_Window_shading'] = l
     shading_coord.loc['N_Window_shading'] = l
     shading_coord.loc['W_Window_shading'] = l
     return shading_coord

     






