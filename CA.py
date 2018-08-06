# -*- coding: utf-8 -*-
# -------------------------------------------
# ------ © Nader Trabelsi - April 2018 ------
# -------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.linalg import svd
from scipy.stats import chi2_contingency as chi2_op

## Le tableau de contingence -effectif-: (la variable tab_CE) ##
tab_CE = np.array([[650, 100, 102],
             [146, 337, 37],
             [220, 98, 310]])
tab_CE_pur=tab_CE.copy() 
tab_CF_pur=tab_CE_pur/tab_CE_pur.sum() # A utiliser après plus loin pour le test chi2

Lig = pd.Index(["Mercedes","BMW","Audi"], "rows")
Col = pd.Index(["Bonne", "Moyenne", "Basse"],"cols")
tab_CE = pd.DataFrame(tab_CE,
                  index=Lig,
                  columns=Col)

tab_CE_nuage=tab_CE.copy() # A utiliser après plus loin pour dessiner le nuage

tab_CE['Total'] = tab_CE.sum(axis=1)
tab_CE.loc['Total', :] = tab_CE.sum(axis=0)


## Le tableau de contingence -fréquence-: (la variable tab_CF) ##
tab_CF= tab_CE/(tab_CE.iloc[3,3])


## Le tableau de profils-lignes: (la varialbe tab_PL) ##
tab_PL = tab_CF.copy()
tab_PL.rename(index={'Total': 'Profil Moyen'},inplace=True)
for i in range(0,4):
    for j in range(0,4):
        tab_PL.iloc[i,j]=(tab_PL.iloc[i,j]/tab_PL.iloc[i,3])*100


## Le tableau de profils-colonnes: (la variable tab_PC) ##
tab_PC = tab_CF.copy()
tab_PC.rename(columns={'Total': 'Profil Moyen'},inplace=True)
for i in range(0,4):
    for j in range(0,4):
        tab_PC.iloc[i,j]=(tab_PC.iloc[i,j]/tab_PC.iloc[3,j])*100
        

## Le tableau de fréquences théoriques: (la variable tab_FT) ##
tab_FT=tab_CE.copy()
tab_FT=tab_FT*0 # Pour initiliser le tableau tout en conservant les noms des lignes et colonnes

tab_FT['Total']=(tab_PC['Profil Moyen'].values)/100
tab_FT.loc['Total',:]=tab_PL.loc['Profil Moyen',:].values/100
for i in range(0,3):
    for j in range(0,3):
        tab_FT.iloc[i,j]=tab_FT.iloc[i,3]*tab_FT.iloc[3,j]


## Test de dépendence/indépendence: ##
chi2 , p , degLib , freq_theo = chi2_op(tab_CF_pur)
if p<0.05:
    print("L'hypothèse d'indépendence est vraie")
else:
    print("L'hypothèse de dépendence est vraie")


## Construction des nuages: ##
lignes = tab_CE_nuage.index.values 
colonnes = tab_CE_nuage.columns.values
matCont = np.matrix(tab_CE_nuage, dtype=float) # Matrice de contingence
matCorr = matCont / matCont.sum() # Matrice de correspondance

# Les vecteurs de Total l et c, respectivement, par lignes et par colonnes
l = matCorr.sum(axis=1)
c = matCorr.sum(axis=0).T

# Matrice diagonale des sommes lignes/colonnes
D_l_rsq = np.diag(1. / np.sqrt(l.A1))
D_c_rsq = np.diag(1. / np.sqrt(c.A1))

# Matrice des erreurs pour les différentes observations
matErr = D_l_rsq * (matCorr - l * c.T) * D_c_rsq

# Factorisation de la matrice
matGauc, matDiag, matDroi = svd(matErr, full_matrices=False)
matDiag = np.asmatrix(np.diag(matDiag))
matDroi = matDroi.T


corPrinLig = D_l_rsq * matGauc * matDiag # Coordonnées principales des lignes
corPrinCol = D_c_rsq * matDroi * matDiag # Coordonnées principales des colonnes
corStLig = D_l_rsq * matGauc # Coordonnées standard des lignes
corStCol = D_c_rsq * matDroi # Coordonnées standard des colonnes

# Total de la variance de la matrice des données
inertie = sum([(matCorr[i,j] - l[i,0] * c[j,0])**2 / (l[i,0] * c[j,0])
                   for i in range(matCont.shape[0])
                   for j in range(matCont.shape[1])])

# Affichage
corPrinLig = corPrinLig.A
corPrinCol = corPrinCol.A
corStLig = corStLig.A
corStCol = corStCol.A

plt.figure(1)

xmini,xmaxi,ymini,ymaxi=np.zeros(4)

for i, t in enumerate(lignes):
    x, y = corPrinLig[i,0], corPrinLig[i,1]
    plt.text(x, y, t, va='center', ha='center', color='r')
    plt.plot(x,y,'k^')
    xmini = min(x, xmini if xmini else x)
    xmaxi = max(x, xmaxi if xmaxi else x)
    ymini = min(y, ymini if ymini else y)
    ymaxi = max(y, ymaxi if ymaxi else y)

   
for i, t in enumerate(colonnes):
    x, y = corPrinCol[i,0], corPrinCol[i,1]
    plt.text(x, y, t, va='center', ha='center', color='b')
    plt.plot(x,y,'go') 
    xmini = min(x, xmini if xmini else x)
    xmaxi = max(x, xmaxi if xmaxi else x)
    ymini = min(y, ymini if ymini else y)
    ymaxi = max(y, ymaxi if ymaxi else y)

if xmini and xmaxi:
     pas = (xmaxi - xmini) * 0.1
     plt.xlim(xmini - pas, xmaxi + pas)
if ymini and ymaxi:
     pas = (ymaxi - ymini) * 0.1
     plt.ylim(ymini - pas, ymaxi + pas)

plt.grid()
plt.xlabel('Facteur 1')
plt.ylabel('Facteur 2')
plt.show()