# les bibliothèques
import os
import struct
import numpy as np 
import cv2
from math import pi 
import math
import time
from pylab import *
import matplotlib.pyplot as plt
import streamlit as st
from streamlit.logger import get_logger
import pandas as pd
from tempfile import TemporaryFile
import tempfile
import pickle
from PIL import Image

logger = get_logger(__name__)
#fonction pour supprimer le pading "le cadre"
def supp_pad(s,V):
    for i in range(V):
        s=np.delete(s, (0), axis=0)
        s=np.delete(s, (-1), axis=0)
        s=np.delete(s, (-1), axis=1)
        s=np.delete(s, (0), axis=1)
    return(s)

def call1 ( i , j , mat):
    d1=distance(i,j,i,0)
    d2=distance(i,j,0,j)
    d3=distance(i,j,mat.shape[0],j)
    d4=distance(i,j,i,mat.shape[1])
    if (d1<=d2)and(d1<=d3)and(d1<=d4):
        return((i,0))
    else :
        if (d2<=d1)and(d2<=d3)and(d2<=d4):
            return((0,j))
        else :
            if (d3<=d1)and(d3<=d2)and(d3<=d4):
                return((mat.shape[0]-1,j))
            else :
                return((i,mat.shape[1]-1))
#fonction call retourne le minimum des distance
def call ( i , j , mat):
    d1=distance(i,j,i,0)
    d2=distance(i,j,0,j)
    d3=distance(i,j,mat.shape[0],j)
    d4=distance(i,j,i,mat.shape[1])
    return(min(d1,d2,d3,d4))
#la fonction re pour trouvez les résidus de somme =0
def re (r,n,img,listt): 
    #créé une matrice  vide avec zeros ( image )
    cc=np.zeros((img.shape[0],img.shape[1])).astype('f')
    kernel=np.zeros((r*2+1,r*2+1)).astype('f')
    i=r 
    b=False
    while i<img.shape[0]-r:
        j=r
        while j <img.shape[1]-r:
            if img[i,j] !=0:
                kernel=np.copy(img[i-r:r+i+1,j-r:r+1+j])    
                s=np.sum(kernel)
                if s==0:
                    b=False
                    for lign in kernel :
                        for col in lign:
                            if col!=0:
                                b=True
                    if b:
                        listt.append(i)
                        listt.append(j)
                        n+=1
                        
                        img[i-r:r+1+i,j-r:r+1+j]=0
                        k=0
                        while k< kernel.shape[0]:
                            l=0
                            while l<kernel.shape[1]:
                                if kernel[k,l]!=0:
                                    cc[i+k-r,j+l-r]=n
                                l+=1
                            k+=1
            j+=1    
        i+=1    
    return(n,cc,listt)
def re1 (img1,cc1,listt1,maxxx):
    for i in range ( 0,img1.shape[0]):
        j=0
        while j< img1.shape[1]:
            if img1[i,j]==1:
                b=False
                v1=False
                v2=False
                a1=i
                while( a1 <img1.shape[0])and (not b):
                    a2=0
                    while (a2<img1.shape[1])and ( not b):
                        if (img1[a1,a2]==-1)&(not b):
                            X2=a1
                            Y2=a2
                            dd=distance(i,j,X2,Y2)
                            for a3 in range(a1,img1.shape[0]):
                                for a4 in range(0,img1.shape[1]):
                                    if img1[a3,a4]==-1:
                                        ddd=distance(i,j,a3,a4)
                                        if ddd<dd:
                                            X2=a3
                                            Y2=a4
                            dmin1=distance( i ,j ,listt1[0],listt1[1])
                            dmin2=distance( X2 ,Y2 ,listt1[0],listt1[1])
                            for ln in range (0,len( listt1),2):
                                d3=distance(i ,j ,listt1[ln] ,listt1[ln+1])
                                d4=distance(X2 ,Y2 ,listt1[ln] ,listt1[ln+1])
                                if d3<dmin1:
                                    dmin1=d3
                                    v1=True
                                    centrr1=cc1[listt1[ln],listt1[ln+1]]
                                if d4 < dmin2:
                                    dmin2=d4
                                    v2=True
                                    centrr2=cc1[listt1[ln],listt1[ln+1]]
                            if dmin1<dmin2:
                                minn=dmin1
                                if v1:
                                    c=centrr1
                                else:
                                    c=cc1[listt1[0],listt1[1]]
                                b=True
                            else :
                                minn=dmin2
                                if v2:
                                    c=centrr2
                                else:
                                    c=cc1[listt1[0],listt1[1]]
                                b=True
                        a2+=1
                    a1+=1
                if  b:    
                    lm1=call(i,j,img1)
                    lm2=call(X2,Y2,img1)
                    if lm1<dmin1:
                        img1[i,j]=0
                        cc1[i,j]=-10
                    elif lm2<dmin2:
                        img1[X2,Y2]=0
                        cc1[X2,Y2]=-10
                        cc1[i,j]=-10
                        img1[i,j]=0
                    else :
                        lazam=distance(i,j,X2,Y2)
                        minn=min(lm1,lm2)
                        if lazam*10<minn:
                            cc1[i,j]=c
                            cc1[X2,Y2]=c
                            img1[i,j]=0
                            img1[X2,Y2]=0
                        elif lazam < minn:
                            maxxx+=1
                            img1[X2,Y2]=0
                            cc1[X2,Y2]=maxxx            
                            cc1[i,j]=maxxx
                            img1[i,j]=0
                        else :
                            img1[X2,Y2]=0
                            cc1[X2,Y2]=-10
                            cc1[i,j]=-10
                            img1[i,j]=0
            
            elif img1[i,j]==-1:
                b=False
                v1=False
                v2=False
                a1=i
                while (a1 < img1.shape[0]) and (not b ):
                    a2=0
                    while (a2 <img1.shape[1])and (not b):
                        if (img1[a1,a2]==1)&(not b):
                            X2=a1
                            Y2=a2
                            dd=distance(i,j,X2,Y2)
                            for a3 in range(a1,img1.shape[0]):
                                for a4 in range(0,img1.shape[1]):
                                    if img1[a3,a4]==1:
                                        ddd=distance(i,j,a3,a4)
                                        if ddd<dd:
                                            X2=a3
                                            Y2=a4
                            dmin1=distance( i ,j ,listt1[0],listt1[1])
                            dmin2=distance( X2 ,Y2 ,listt1[0],listt1[1])
                            for ln in range (2,len( listt1),2):
                                d3=distance(i ,j ,listt1[ln] ,listt1[ln+1])
                                d4=distance(X2 ,Y2 ,listt1[ln] ,listt1[ln+1])
                                if d3<dmin1:
                                    dmin1=d3
                                    centrr1=cc1[listt1[ln],listt1[ln+1]]
                                    v1=True
                                if d4 < dmin2:
                                    dmin2=d4
                                    v2=True
                                    centrr2=cc1[listt1[ln],listt1[ln+1]]
                            if dmin1<dmin2:
                                minn=dmin1
                                if v1:
                                    c=centrr1
                                else:
                                    c=cc1[listt1[0],listt1[1]]
                                b=True
                            else :
                                minn=dmin2
                                if v2:
                                    c=centrr2
                                else:
                                    c=cc1[listt1[0],listt1[1]]
                                b=True
                        a2+=1
                    a1+=1
                if b :
                    lm1=call(i,j,img1)
                    lm2=call(X2,Y2,img1)
                    if lm1<dmin1:
                        img1[i,j]=0
                        cc1[i,j]=-10
                    elif lm2<dmin2:
                        img1[X2,Y2]=0
                        cc1[X2,Y2]=-10
                        cc1[i,j]=-10
                        img1[i,j]=0
                    else :
                        lazam=distance(i,j,X2,Y2)
                        minn=min(lm1,lm2)
                        if lazam*10<minn:
                            cc1[i,j]=c
                            cc1[X2,Y2]=c
                            img1[i,j]=0
                            img1[X2,Y2]=0
                        elif lazam<minn:
                            maxxx+=1
                            img1[X2,Y2]=0
                            cc1[X2,Y2]=maxxx
                            cc1[i,j]=maxxx
                            img1[i,j]=0
                        else :
                            img1[X2,Y2]=0
                            cc1[X2,Y2]=-10
                            cc1[i,j]=-10
                            img1[i,j]=0
            j+=1
    
    for i in range(img1.shape[0]):
        for j in range(img1.shape[1]):
            if img1[i,j]!=0:
                cc1[i,j]=-10
    return(cc1)  

def resultat (img,V):
    img=np.pad(img, (V, V), 'constant', constant_values=( 0, 0))
    #calcule des residus 
    img=img/255
    i=r
    while i <img.shape[0]-V :
        j=r;
        while j<img.shape[1]-V:                                                 
            s1=img[i][j+1]-img[i][j]                                            
            if s1 < -0.5 :
                s1=s1+1
            else:
                if s1 >0.5:
                    s1=s1-1 
            s2=img[i+1][j+1]-img[i][j+1]
            if s2 < -0.5 :
                s2=s2+1
            else:
                if s2 >0.5:
                    s2=s2-1
                    
            
            s3=img[i+1][j]-img[i+1][j+1]
            if s3 < -0.5 :
                s3=s3+1
            else:
                if s3 >0.5:
                    s3=s3-1
            s4=img[i][j]-img[i+1][j]
            if s4 < -0.5 :
                s4=s4+1
            else:
                if s4 >0.5:
                    s4=s4-1
            img[i][j]=s1+s2+s3+s4
            if (img[i][j]>-0.05)&(img[i][j]<0.05) :
                img[i][j]=0
            if img[i][j]>0.90 :
                img[i][j]=1                
            if img[i][j]<-0.90:
                img[i][j]=-1
            j+=1
        i+=1    
    image=np.copy(img)
    image=supp_pad(image,V)
    cc=np.zeros((img.shape[0],img.shape[1])).astype('f')
    kernel=np.zeros((V*2+1,V*2+1)).astype('f')
    listt=[]
    f=False
    n=0
    (n,cc1,listt)=re(V,n,img,listt)
    cc=cc+cc1
    for k1 in range(1,10): 
        f=False
        while (not f) :
            s=0
            (n,cc1,listt)=re(k1,n,img,listt)
            cc=cc+cc1
            for i in cc1:
                for j in i:
                    s=s+j
            if s==0:
                f=True
    cc =re1(img,cc,listt,n)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i,j]!=0:
                cc[i,j]=-10
    cc=supp_pad(cc,V)
    for kk in range(len(listt)):
        listt[kk]=listt[kk]-V
    return(cc,listt,int(np.max(cc)),image)
def distance(p0,p1,pp0,pp1):
    return math.sqrt((p0 - pp0)**2 + (p1 - pp1)**2)
def inter (mat  ,rrr):
    l2=[]
    j=0
    for i in range (0, len(rrr),2):
        b=False
        x1=rrr[i]
        y1=rrr[i+1]
        j+=1
        mm=0
        moy=0
        for k in range( 0,mat.shape[0]):
            for l in range( mat.shape[1]):
                if mat[k,l]==j:
                    d=distance(x1,y1,k,l)
                    if d!=0:
                        mm+=1
                        moy=moy+d
                        b=True
        if b : 
            moy=moy/mm
            l2.append(moy)
    m=0
    s=0
    for i in l2:
        s=s+i
        m+=1
    moy=s/m
    return(moy)
def intra (lst):
    dis_intra=[]
    l1=[]
    s=0
    for i in range(0,len(lst),2):
        for j in range(0,len(lst),2):
            d=distance(lst[i],lst[i+1],lst[j],lst[j+1])
            s=s+d
    m=len(lst)/2
    moy=s/m
    return(moy)
def minim (x,y,liste) :
    t1=liste[0]
    t2=liste[1]
    dmin=distance(x,y,t1,t2)
    z1=t1
    z2=t2
    for i in range(2,len(liste),2):
        t1=liste[i]
        t2=liste[i+1]
        d=distance(x,y,t1,t2)
        if d<=dmin:
            dmin=d
            z1=t1
            z2=t2
    return(z1,z2)
def supp (x1,y1,lst):
    for p in range (0,len(lst),2):
        p1=lst[p]
        p2=lst[p+1]
        if (x1==p1) and (y1==p2):
            lst.pop(p)
            lst.pop(p)
            break
    return(lst)
def nombre_group (mat,nbr ):
    mat1=np.copy(mat)
    for i in range(1,nbr+1):
        cc1=np.zeros((mat.shape[0],mat.shape[1])).astype('f')
        l_1=[]
        for a1 in range(0,mat.shape[0]):
            for a2 in range(0,mat.shape[1]):
                if (int(mat[a1,a2])==i):
                    l_1.append(a1)
                    l_1.append(a2)
        if l_1:
            s1=[]
            s1.append(l_1[0])
            s1.append(l_1[1])
            mat1=matching(l_1,i,mat1,s1,cc1) 
    return(mat1)

def matching(lst , rrrr , mat , s1,cc1):
    mat1=np.copy(mat)
    while s1 :
        lst=supp(s1[0],s1[1],lst)
        if lst:
            (x,y)=minim(s1[0],s1[1],lst)
            mat1=get_line(mat1,rrrr,(s1[0],s1[1]),(x,y))
            s1.append(x)
            s1.append(y)
            s1=supp(s1[0],s1[1],s1)
        else :
            s1=[]
    return(mat1)
def get_line(cc,cod,start, end):
    matricu=np.copy(cc)
    x1, y1 = start
    x2, y2 = end
    dx = x2 - x1
    dy = y2 - y1
    is_steep = abs(dy) > abs(dx)
    if is_steep:
        x1, y1 = y1, x1
        x2, y2 = y2, x2
    swapped = False
    if x1 > x2:
        x1, x2 = x2, x1
        y1, y2 = y2, y1
        swapped = True
    dx = x2 - x1
    dy = y2 - y1
    error = int(dx / 2.0)
    ystep = 1 if y1 < y2 else -1
    y = y1
    points = []
    for x in range(x1, x2 + 1):
        coord = (y, x) if is_steep else (x, y)
        points.append(coord)
        error -= abs(dy)
        if error < 0:
            y += ystep
            error += dx
    if swapped:
        points.reverse()
    for i in points:
        (x1,y1)=i
        matricu[x1,y1]=cod
    return (matricu)
def khir (tup , lst):
    dmin=distance(tup[0],tup[1],lst[0][0][0],lst[0][0][1])
    t=(lst[0][0][0],lst[0][0][1])
    n=lst[0][1]
    if dmin==0:
        dmin=1000
    for k1 in lst :
        d=distance(tup[0],tup[1],k1[0][0],k1[0][1])
        if d!=0:
            if d < dmin:
                t=(k1[0][0],k1[0][1])
                n=k1[1]
    return(t,n)
def khir1 (n,lst):
    i=0
    for k in lst :
        if k[1]==n:
            lst.pop(i)
        i+=1
    return(lst)
def selection(dicto,dicto1,nbr):
    lst_1=[]
    lst_2=[]
    lll=[]
    l2=[]
    lst=[]
    for r in dicto.keys():
        moy_inter=inter(dicto[r],dicto1[r]) # à minimiser
        moy_intra=intra(dicto1[r]) # à maximiser
        lst_1.append((moy_inter,r))
        lst_2.append((moy_intra,r))
        l2.append(((moy_inter,moy_intra),r))
    for k in l2 :
        x=k[0][0]
        y=k[0][1]
        n=k[1]
        cpt=0
        for k1 in l2:
            if (k1[0][0]<x)and (k1[0][1]>y):
                cpt+=1
        lst.append((n,cpt))
    som=0
    rang=0
    lstt=[]
    ls=[]
    ls1=[]
    for k in lst:
        if som<nbr:
            som1=0
            for k1 in lst:
                if k1[1]==rang:
                    som1+=1
            if som1==0:
                rang+=1
            else:
                if (som1+som)<=nbr:
                    for k1 in lst:
                        if k1[1]==rang:
                            lstt.append(k1[0])
                    som=som+som1
                    rang+=1
                else :
                    monq=nbr-som
                    if som1<3:
                        for k1 in lst :
                            if k1[1]==rang:
                                if monq>0:
                                    lstt.append(k1[0])
                                    monq-=1
                        break
                    else:
                        mx=0
                        for k1 in lst:
                            if k1[1]==rang:
                                ls.append(k1[0])
                        for k1 in l2 :
                            if k1[1] in ls :
                                ls1.append(k1)
                                if mx<k1[0][1]:
                                    mx=k1[0][1]
                                    nb=k1[1]
                        for k1 in ls1:
                            if k1[1]==nb:
                                tup=k1[0]
                        (tt,nn)=khir(tup,ls1)
                        ls1=khir1(nb,ls1)
                        nb=nn
                        for k1 in range(len(ls)-2):
                            x1=tup[0]
                            y1=tup[1]
                            (tt1,nn1)=khir(tt,ls1)
                            x2=tt1[0]
                            y2=tt1[1]
                            m=abs(x1-x2)*(y1-y2)
                            lll.append((m,tt))
                            tup=tt
                            tt=tt1
                            ls1=khir1(nb,ls1)
                            nb=nn1
                        for k1 in range(monq):
                            mx=0
                            for k2 in lll:
                                if k2[0]>mx:
                                    mx=k2[0]
                                    tt=k2[1]
                            for k2 in l2 :
                                if k2[0]==tt:
                                    lstt.append(k2[1])
                            hhhh=0
                            for k2 in lll:
                                if k2[1]==tt:
                                    lll.pop(hhhh)
                                hhhh+=1
                            tt=0
                        break        
    return(lstt,lst)
def centre_grp(img,nb): # 'nb' est le nombre de groupe de l'image
    lst=[]
    a=b=0
    for k in range(1,nb+1):
        cl=ln=cp=0
        lst1=[]
        for i in range(0,img.shape[0]):
            for j in range(0,img.shape[1]):
                bol=False
                if img[i,j]==k:
                    lst1.append((i,j))
                    if cp<=0:
                        a=i
                        b=j
                        cp+=1
                    if j<b:
                        b=j
                    bol=True
                    cp+=1
                    if j>cl:
                        cl=j
                    if i>ln:
                        ln=i
        x=int((ln+a)/2)
        y=int((cl+b)/2)
        if lst1:
            (xx,yy)=centre_proche(x,y,lst1)
            lst.append(xx)
            lst.append(yy)
    return(lst)

def centre_proche(x,y,liste) :# co_lst ne doit pas contenir ni x ni y
    t1=liste[0][0]
    t2=liste[0][1]
    b=False
    dmin=distance(x,y,t1,t2)
    for i in range(1,len(liste)):
        t1=liste[i][0]
        t2=liste[i][1]
        d=distance(x,y,t1,t2)
        if (d<=dmin):
            dmin=d
            (x1,x2)=(t1,t2)
            b=True
    if b:
        return((x1,x2))
    else:
        return((t1,t2))
def traitement(s,lst):
    debut= int(s.shape[0]/2)
    mx=int(np.max(s))
    l=[]               
    img1=np.zeros((image.shape[0],image.shape[1])).astype('f')
    mnn=mx
    for k1 in range(debut,s.shape[0]) :
        for k2 in range(s.shape[1]) :
            if s[k1,k2]>0 and s[k1,k2]<mnn:
                mnn=s[k1,k2]
    mnnn=mnn              
    mxx=0
    for k1 in range(0,debut) :
        for k2 in range(s.shape[1]) :
            if s[k1,k2]>mxx and s[k1,k2]<mnn:
                mxx=s[k1,k2]
    k=int(np.max(s[debut:,:]))
    mxx=k+70
    while k>=int(mnn):    
        for k1 in range(debut,s.shape[0]):
            for k2 in range(s.shape[1]):
                if s[k1,k2]==k:
                    s[k1,k2]=mxx
        mxx-=1
        k-=1
    mnn=int(np.max(s[debut:,:]))
    k=int(np.max(s[0:debut,:]))+70
    mxx=k
    while k >=int(mnnn):
        mxx+=1
        for k1 in range(0,debut):
            for k2 in range(s.shape[1]):       
                if s[k1,k2]==k:
                    s[k1,k2]=mxx
        k-=1
    ll=[]
    maxxx=np.max(s)
    sommmmm=0
    for b in range(1,int(maxxx)):
        bomm=False
        for b1 in s :
            for b2 in b1 :
                if( b2==b )and (not bomm):
                    bomm=True
        if (not bomm):
            sommmmm+=1
            ll.append(b)
    ll.reverse()
    for b in ll:
        for b1 in range(s.shape[0]) :
            for b2 in  range(s.shape[1]):
                if (s[b1,b2] >b ):
                    s[b1,b2]=s[b1,b2]-1
    nbr=int(maxxx-sommmmm)
    for i in range(int(mx)):
        som=0
        for k1 in range(s.shape[0]):
            for k2 in range(s.shape[1]):
                if s[k1,k2]==i:
                    som=som+image[k1,k2]
        if som!=0:
            l.append((i,som))
    for i in l:
        f=False
        pp=i[1]
        while (not f):
            if i[1]<0:
                boom=False
                lign=0
                while ( lign < s.shape[0]) and (not boom):
                    col=0
                    while (col < s.shape[1])and(not boom):
                        if (s[lign,col]==i[0])and (image[lign,col]==-1):
                            s[lign,col]=0
                            img1[lign,col]=image[lign,col]
                            boom=True
                        col+=1
                    lign+=1
                pp+=1
                if pp==0:
                    f=True
            else:
                boom=False
                lign=0
                while ( lign < s.shape[0]) and (not boom):
                    col=0
                    while (col < s.shape[1])and(not boom):
                        if (s[lign,col]==i[0])and (image[lign,col]==1):
                            s[lign,col]=0
                            img1[lign,col]=image[lign,col]
                            boom=True
                        col+=1
                    lign+=1
                pp-=1
                if pp==0:
                    f=True
    
    for k1 in range(image.shape[0]):
        for k2 in range(image.shape[1]) :
            if (image[k1,k2]!=0)and (s[k1,k2]==0):
                img1[k1,k2]=image[k1,k2]
    
    s=re1(img1,s,lst,mx)
    return(s)
def croi ( dicto,dicto1,cr,u):
    for i in cr :
        s1=np.copy(dicto[i[0]])
        s2=np.copy(dicto[i[1]])
        debut= int(s1.shape[0]/2)
        tmp=s1[debut:,:]
        s1[debut:,:]=s2[debut:,:]
        s2[debut:,:]=tmp
        s1=traitement(s1,dicto1[i[0]])
        mx=int(np.max(s1))
        l_1=centre_grp(s1,int(np.max(s1)))
        s2=traitement(s2,dicto1[i[1]])
        mx=int(np.max(s2))
        l_2=centre_grp(s2,int(np.max(s2)))
        dicto[u+1]=s1
        dicto1[u+1]=l_1
        dicto[u+2]=s2
        dicto1[u+2]=l_2
        u+=2
    return(dicto,dicto1,u)
def mutation(mat,lst):
    l=[]
    for i in range(0,len (lst)-2,2):
        dmin=distance(lst[i],lst[i+1],lst[i+2],lst[i+3])
        k1=mat[lst[i],lst[i+1]]
        k2=mat[lst[i+2],lst[i+3]]
        for j in range(i+4,len(lst),2):
            d=distance(lst[i],lst[i+1],lst[j],lst[j+1])
            if d < dmin :
                dmin=d
                k1=mat[lst[i],lst[i+1]]
                k2=mat[lst[j],lst[j+1]]
        k=(k1,k2)
        l.append((k,dmin))    
    mn=l[0][1]
    nb=l[0][0][0]
    c=l[0][0][1]
    for i in l :
        if i[1]<mn:
            mn=i[1]
            nb=i[0][0]
            c=i[0][1]
    for i in range(mat.shape[0]) :
        for j in range(mat.shape[1]):
            if mat[i,j]==nb:
                mat[i,j]=c
    return(mat)            
def algo_gen(dicto,dicto1,gnr1,prm ,prc,v):
    for i in range(gnr1):
        croise=[]
        dictto=dict(dicto)
        dictto1=dict(dicto1)
        (t,tt)=selection(dictto,dictto1,4)
        for k in range(0, len(t),2) :
            croise.append((t[k],t[k+1]))
        (dicto,dicto1,v)=croi(dicto,dicto1,croise,v)
        if prm>0.5:
            dicto[v]=mutation(dicto[v],dicto1[v])    
    return(tt)
def derlm1 (C,D):
    ln=C.shape[0]
    cl=C.shape[1]
    uw= np.zeros((ln,cl)).astype('f')
    B= np.zeros((ln,cl)).astype('f')
    A= np.zeros((ln,cl)).astype('f')
    K= np.zeros((ln,cl)).astype('f')
    CC= np.zeros((ln,cl)).astype('f')
    CC[:,:]=0.0
    K[:,:]=-2000
    for i in range(ln):
        for j in range(cl):
            if D[i,j]!=0:
                D[i,j]=1
    for i in range(ln):
        for j in range(cl):
            if D[i,j]==1:
                C[i,j]=-100
                A[i,j]=C[i,j]                
    for i in range(ln):
        for j in range(cl):
            if C[i,j]==-100:
                CC[i,j]=-100
                K[i,j]=-1000               
    g2=0
    g1=0
    g3=0
    for i in CC:
        for j in i :
            if j ==0:
                g1+=1
    bom=False
    for i in range(5,ln-5):
        if bom:
            break
        for j in range(5,cl-5):
            if CC[i,j]==0:
                m=0
                m=np.sum(CC[i-1:i+2,j-1:j+2] )
                if m==0:
                    CC[i,j]=-150
                    K[i,j]=20
                    bom=True
                    break
    
    while(1):
        g2=g1            
        for i in range(1,ln-1):
            for j in range(1,cl-1):
                if CC[i,j]==0:
                    if CC[i,j-1]==-150:
                        a2=j-1
                        a1=i
                        df1= abs ( (C[i,j]+(2*3.14*(K[a1,a2]+1)))-(C[a1,a2]+(2*3.14*K[a1,a2]))  )
                        df2= abs ( (C[i,j]+(2*3.14*(K[a1,a2]-1)))-(C[a1,a2]+(2*3.14*K[a1,a2]))  ) 
                        df3= abs ( (C[i,j]+(2*3.14*K[a1,a2]))-(C[a1,a2]+(2*3.14*K[a1,a2]))      ) 
                        if (df1<df2)and ( df1<df3):
                            K[i,j]=K[a1,a2]+1
                        elif (df2<df1)and ( df2<df3):
                            K[i,j]=K[a1,a2]-1
                        else :
                            K[i,j]=K[a1,a2]
                        if ( abs (K[i,j]-K[a1,a2])<=1):
                            CC[i,j]=-150
                        continue# normlmnt ndirro ta3 li tadina la 2em 
                    if CC[i,j+1]==-150:
                        a2=j+1
                        a1=i
                        df1= abs ( (C[i,j]+(2*3.14*(K[a1,a2]+1)))-(C[a1,a2]+(2*3.14*K[a1,a2]))  ) 
                        df2= abs ( (C[i,j]+(2*3.14*(K[a1,a2]-1)))-(C[a1,a2]+(2*3.14*K[a1,a2]))  ) 
                        df3= abs ( (C[i,j]+(2*3.14*K[a1,a2]))-(C[a1,a2]+(2*3.14*K[a1,a2]))      ) 
                        if (df1<df2)and ( df1<df3):
                            K[i,j]=K[a1,a2]+1
                        elif (df2<df1)and ( df2<df3):
                            K[i,j]=K[a1,a2]-1
                        else :
                            K[i,j]=K[a1,a2]
                        if ( abs (K[i,j]-K[a1,a2])<=1):
                            CC[i,j]=-150
                        continue
                    if CC[i-1,j]==-150:
                        a2=j
                        a1=i-1
                        df1= abs ( (C[i,j]+(2*3.14*(K[a1,a2]+1)))-(C[a1,a2]+(2*3.14*K[a1,a2]))  ) ;
                        df2= abs ( (C[i,j]+(2*3.14*(K[a1,a2]-1)))-(C[a1,a2]+(2*3.14*K[a1,a2]))  ) ;
                        df3= abs ( (C[i,j]+(2*3.14*K[a1,a2]))-(C[a1,a2]+(2*3.14*K[a1,a2]))      ) ;
                        if (df1<df2)and ( df2<df3):
                            K[i,j]=K[a1,a2]+1
                        elif (df2<df1)and ( df2<df3):
                            K[i,j]=K[a1,a2]-1
                        else :
                            K[i,j]=K[a1,a2]
                        if ( abs (K[i,j]-K[a1,a2])<=1):
                            CC[i,j]=-150
                        continue
                    if CC[i+1,j]==-150:
                        a2=j
                        a1=i+1
                        df1= abs ( (C[i,j]+(2*3.14*(K[a1,a2]+1)))-(C[a1,a2]+(2*3.14*K[a1,a2]))  ) ;
                        df2= abs ( (C[i,j]+(2*3.14*(K[a1,a2]-1)))-(C[a1,a2]+(2*3.14*K[a1,a2]))  ) ;
                        df3= abs ( (C[i,j]+(2*3.14*K[a1,a2]))-(C[a1,a2]+(2*3.14*K[a1,a2]))      ) ;
                        if (df1<df2)and ( df2<df3):
                            K[i,j]=K[a1,a2]+1
                        elif (df2<df1)and ( df2<df3):
                            K[i,j]=K[a1,a2]-1
                        else :
                            K[i,j]=K[a1,a2]
                        if ( abs (K[i,j]-K[a1,a2])<=1):
                            CC[i,j]=-150
                        continue
        g1=0;
        for i in range(1,ln-1):
            for j in range(1,cl-1):
                if CC[i,j]==0:
                    g1=g1+1
        if g1 ==0 :
            break 
        elif g1==g2:
            break
        
    # for i in range (ln-1):
        # for j in range(cl-1):
            # if (K[i,j]!=-1000)and(K[i,j]!=-2000):
                # K[i,j]=C[i,j]+(2*3.14*K[i,j])
    return(K)

def drlm2(CC,D): 
    ln=CC.shape[0]
    cl=CC.shape[1]
    cohs= np.zeros((ln,cl)).astype('f')
    coh= np.zeros((ln,cl)).astype('f')
    cut= np.zeros((ln,cl)).astype('f')
    uwd= np.zeros((ln,cl)).astype('f')
    c= np.zeros((ln,cl)).astype('f')
    K= np.zeros((ln,cl)).astype('f')
    intt= np.zeros((ln,cl)).astype('f')
    ssin= np.zeros((ln,cl)).astype('f')
    scos= np.zeros((ln,cl)).astype('f')
  
    for i in range(1,ln-1):
        for j in range(1,cl-1):
            ssin[i,j]=0
            scos[i,j]=0
            for b1 in range(-1,2):
                for a1 in range(-1,2):
                    ssin[i,j]=math.sin(CC[i+a1,j+b1])+ssin[i,j]
                    scos[i,j]=math.cos(CC[i+a1,j+b1])+scos[i,j]
            coh[i,j]=(math.sqrt(   (ssin[i,j]*ssin[i,j])+(scos[i,j]*scos[i,j])))/9

    K=np.copy(D)
    
    for i in range(ln):
        for j in range(cl):
            if (D[i,j]==-2000)or(D[i,j]==-1000):
                c[i,j]=-100
            else:
                c[i,j]=-150
                
    mincoh=np.min(coh)
    maxcoh=np.max(coh)
    g2=0
    g1=0
    g3=0
    s=maxcoh
    pas=0.1
    while s >=(-1*pas):
        for i in range(ln-1):
            for j in range(cl-1):
                if (coh[i,j]>=s)and(c[i,j]==-100):
                    c[i,j]=0
        while(1):
            g2=g1
            for i in range(ln-1):
                for j in range(cl-1):
                    if c[i,j]==0:
                        if c[i,j-1]==-150:
                            a1=i
                            a2=j-1
                            df1= abs ( (CC[i,j]+(2*3.14*(K[a1,a2]+1)))-(CC[a1,a2]+(2*3.14*K[a1,a2]))  )
                            df2= abs ( (CC[i,j]+(2*3.14*(K[a1,a2]-1)))-(CC[a1,a2]+(2*3.14*K[a1,a2]))  ) 
                            df3= abs ( (CC[i,j]+(2*3.14*K[a1,a2]))-(CC[a1,a2]+(2*3.14*K[a1,a2]))      ) 
                            if (df1<df2)and ( df1<df3):
                                K[i,j]=K[a1,a2]+1
                            elif (df2<df1)and ( df2<df3):
                                K[i,j]=K[a1,a2]-1
                            else :
                                K[i,j]=K[a1,a2]
                            if ( abs (K[i,j]-K[a1,a2])<=1):
                                c[i,j]=-150
                            continue   
                               
                        if c[i,j+1]==-150:
                            a2=j+1
                            a1=i
                            df1= abs ( (CC[i,j]+(2*3.14*(K[a1,a2]+1)))-(CC[a1,a2]+(2*3.14*K[a1,a2]))  ) 
                            df2= abs ( (CC[i,j]+(2*3.14*(K[a1,a2]-1)))-(CC[a1,a2]+(2*3.14*K[a1,a2]))  ) 
                            df3= abs ( (CC[i,j]+(2*3.14*K[a1,a2]))-(CC[a1,a2]+(2*3.14*K[a1,a2]))      ) 
                            if (df1<df2)and ( df1<df3):
                                K[i,j]=K[a1,a2]+1
                            elif (df2<df1)and ( df2<df3):
                                K[i,j]=K[a1,a2]-1
                            else :
                                K[i,j]=K[a1,a2]
                            if ( abs (K[i,j]-K[a1,a2])<=1):
                                c[i,j]=-150
                            continue
                        if c[i-1,j]==-150:
                            a2=j
                            a1=i-1
                            df1= abs ( (CC[i,j]+(2*3.14*(K[a1,a2]+1)))-(CC[a1,a2]+(2*3.14*K[a1,a2]))  ) ;
                            df2= abs ( (CC[i,j]+(2*3.14*(K[a1,a2]-1)))-(CC[a1,a2]+(2*3.14*K[a1,a2]))  ) ;
                            df3= abs ( (CC[i,j]+(2*3.14*K[a1,a2]))-(CC[a1,a2]+(2*3.14*K[a1,a2]))      ) ;
                            if (df1<df2)and ( df2<df3):
                                K[i,j]=K[a1,a2]+1
                            elif (df2<df1)and ( df2<df3):
                                K[i,j]=K[a1,a2]-1
                            else :
                                K[i,j]=K[a1,a2]
                            if ( abs (K[i,j]-K[a1,a2])<=1):
                                c[i,j]=-150
                            continue
                        if c[i+1,j]==-150:
                            a2=j
                            a1=i+1
                            df1= abs ( (CC[i,j]+(2*3.14*(K[a1,a2]+1)))-(CC[a1,a2]+(2*3.14*K[a1,a2]))  ) ;
                            df2= abs ( (CC[i,j]+(2*3.14*(K[a1,a2]-1)))-(CC[a1,a2]+(2*3.14*K[a1,a2]))  ) ;
                            df3= abs ( (CC[i,j]+(2*3.14*K[a1,a2]))-(CC[a1,a2]+(2*3.14*K[a1,a2]))      ) ;
                            if (df1<df2)and ( df2<df3):
                                K[i,j]=K[a1,a2]+1
                            elif (df2<df1)and ( df2<df3):
                                K[i,j]=K[a1,a2]-1
                            else :
                                K[i,j]=K[a1,a2]
                            if ( abs (K[i,j]-K[a1,a2])<=1):
                                c[i,j]=-150
                            continue
            g1=0;
            for i in range(1,ln-1):
                for j in range(1,cl-1):
                    if c[i,j] ==0:
                        g1=g1+1;
            if g1 ==0 :
                break 
            elif g1==g2:
                break
        s=s-pas;
    for i in range(ln-1):
        for j in range(cl-1):
            uwd[i,j]=CC[i,j]+(K[i,j]*2*3.14)

    return(uwd)

def lp_norm(intt ,uw):
    ln=intt.shape[0]
    cl=intt.shape[1]
    a=0.0
    a1=0.0
    b1=0.0
    k1=0.0
    for i in range(3,ln-3):
        k2=0.0
        for j in range(3,cl-3):
            ax=intt[i,j+1]-intt[i,j]
            ay=intt[i+1,j]-intt[i,j]
            if ax>3.14 :
                ax=ax-3.14
            if ax<(3.14*(-1)):
                ax=ax+3.14
            if ay>3.14 :
                ay=ay-3.14
            if ay<3.14 :
                ay=ay+3.14
            a1=a1+abs((uw[i,j+1]-uw[i,j])-ax)
            b1=b1+abs((uw[i+1,j]-uw[i,j])-ay)
            k2+=1
        k1+=1
    a1=a1/(k1*k2)
    b1=b1/(k1*k2)
    a=a1+b1
    return(a)
def talwin1(im):
    y1 = np.zeros((img.shape[0], img.shape[1], 3)).astype('f')
    for i in range ( im.shape[0]):
        for j in range ( im.shape[1]):
            if im[i,j]==0:
                y1[i,j]=[230,10,90]
            else:
                y1[i,j]=[255,255,0]
    return(y1)
def talwin(im):
    y1 = np.zeros((img.shape[0], img.shape[1], 3)).astype('f')
    for i in range ( im.shape[0]):
        for j in range ( im.shape[1]):
            if im[i,j]==0:
                y1[i,j]=[250,230,170]
            elif im[i,j]==1:
                y1[i,j]=[255,0,0]
            else :
                y1[i,j]=[0,0,255]
                
    return(y1)
lt=time.ctime()
print(lt)
x=[]
y=[]
dico={}
dico1={}

st.sidebar.header("Paramètres")
st.title(" Déroulement De Phase Interférometrique ")
st.header("Description")
st.write("L'inter-férométrie radar InSAR est une méthode basée sur la mesure de la différence de phase entre deux images radars. Cette méthode permet de détecter et de quantifier les déformations de surface et les mouvements du sol. La différence de phases, due au trajet aller-retour de l'onde envoyée par le radar, entre deux images entraine une ambigüité modulo . Afin de lever cette ambigüité on doit procéder au déroulement de phase. Plusieurs méthodes de déroulement de phase ont été proposées dans la littérature.")
st.header("Tutoriel")
if st.checkbox(''):
    st.write("Le processus d’exploitation de cette application se décompose en deux étapes, une première étape qui concerne le paramétrage du NSGA-II sur le côté gauche de l'interface, où on dispose de 5 paramètres à configurer, et une deuxième étape d’exécution et de visualisation des résultats qui débute dès que le bouton ' Commencer Déroulement' en bas à gauche est appuyée. En haut du côté droit de l’interface s’affiche une icône d’un menu déroulant contenant davantage options. Une fois l’exécution est terminée, les résultats s’affichent les uns au-dessous des autres en commençant avec la population initiale en arrivant aux nouvelles générations.")


pop= st.sidebar.slider("Population Initiale", 1, 30, 2, 1)
gnr= st.sidebar.slider("Nombre de Génération", 1, 20, 2, 1)
prm= st.sidebar.slider("Probabilité de Mutation", 0., .2, .05, .05)
prc = st.sidebar.slider("Probabilité de Croisement", 0., 1., .75, .05)

#st.set_option('deprecation.showfileUploaderEncoding', False)
data_file = st.file_uploader("Vous pouvez charger les images au format brut ou PNG")

p=["","head","isola","knee","noise","spiral","shear","longs"]
option1 = st.sidebar.selectbox(
    'Veuillez sélectionner une image à partir du menu déroulant ci-dessous:',
        p
        )

if (not option1 ) and (not data_file):
    st.warning("Veuillez entrer votre image svp !!!")
elif option1 and data_file :
    st.error("Vous avez entré plusieurs images, veuillez entrer une seule image seulement  !!!")
else:
    st.success("Choix accepté, vous pouvez lancer le déroulement") 
    

st.subheader(" echantillion:")
        

i1 = cv2.imread("head.png",0)
st.image(i1,width=i1.shape[1],caption="head")
i2 = cv2.imread("isola.png")
st.image(i2,width=i2.shape[1],caption="isola")
i3 = cv2.imread("knee.png")
st.image(i3,width=i3.shape[1],caption="knee")
i4 = cv2.imread("noise.png")
st.image(i4,width=i4.shape[1],caption="noise")
i5 = cv2.imread("spiral.png")
st.image(i5,width=i5.shape[1],caption="spiral")
i6 = cv2.imread("shear.png")
st.image(i6,width=i6.shape[1],caption="shear")
i7 = cv2.imread("longs.png")
st.image(i7,width=i7.shape[1],caption="longs")


if st.sidebar.button("commmmmmmmm"):
    if option1 or data_file:
        if option1:
            img = cv2.imread(option1+".png",0).astype('f')
            'Vous avez choisit: ',option1
            sousou=cv2.imread(option1+".png")
        elif data_file:
            img1=Image.open(data_file)
            sousou=np.array(img1,ndmin=3)
            img=np.array(img1,ndmin=2).astype('f')
        r=0
        for i in range(2):
            r+=1
            img=cv2.imread(option1+".png",0).astype('f')
            (cccccc,listoo,n,image)=resultat(img,r)
            dico[r]=cccccc
            if len(listoo)!=0:
                dico1[r]=listoo
        st.title("Affichage Des Résultats")
        st.subheader(" Interférogramme Choisit:")
        st.image(sousou,width=None)
        ima=talwin(image)
        st.subheader(" Image Des Résidus:")
        cv2.imwrite('image_des_reseduuuuuu.png',ima)
        hlm=cv2.imread( "image_des_reseduuuuuu.png")
        st.image(hlm,width=None,caption="Image des résidus.")
        cc=nombre_group(dico[r],int(np.max(dico[r]))) 
        v=0
        bom=False
        for i in range(0,cc.shape[0]-1) :
            for j in range(0,cc.shape[1]-1)  :
                if cccccc[i,j]==-10:
                    v+=1
                    t=call1(i,j,cc)
                    cc=get_line(cc,255,(i,j),t)
                    bom=True
        for k1 in range(cc.shape[0]):
                for k2 in range(cc.shape[1]):
                    if cc[k1,k2]!=0:
                        cc[k1,k2]=1
        imma=talwin1(cc)
        st.subheader(" Image Des Coupes De Branches:")
        cv2.imwrite('image_des_cuts.png',imma)
        hlmm=cv2.imread( "image_des_cuts.png")
        st.image(hlmm,width=None,caption="Image des coupes de branches.")
        select=algo_gen(dico,dico1,gnr,prm ,prc,r)
        ll=[]
        for k1 in select:
            if k1[1]==0:
                ll.append(k1[0])
        for k1 in ll:
            ccc=nombre_group(k1,int(np.max(k1))) 
            v=0
            bom=False
            for i in range(0,ccc.shape[0]-1) :
                for j in range(0,ccc.shape[1]-1)  :
                    if cccccc[i,j]==-10:
                        v+=1
                        t=call1(i,j,cc)
                        cc=get_line(ccc,255,(i,j),t)
                        bom=True
            for k1 in range(ccc.shape[0]):
                    for k2 in range(ccc.shape[1]):
                        if ccc[k1,k2]!=0:
                            ccc[k1,k2]=1
            sousou=sousou/255
            sousou=sousou*2*3.14
            kk=derlm1(sousou,ccc)
            kk1=drlm2(sousou,kk)
            cv2.imwrite('image_final',imma)
            hlmm=cv2.imread( "image_final.png")
            st.image(hlmm,width=None,caption="Image de deroulement ")
        lt=time.ctime()
        print(lt)
        
        if st.button("Celebrate !"):
            st.balloons()
        